from __future__ import annotations

import json
import re
from dataclasses import dataclass
from hashlib import sha1
from math import ceil
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np
from PIL import Image

from app.core.config import settings
from app.core.errors import AppError, ErrorCode
from app.models.schemas import ImageRGBBands
from app.services.state_store import ImageContext, RuntimeStore
from app.utils.spectral_math import nearest_band_index, scale_to_unit_reflectance

try:
    from spectral import open_image as spectral_open_image
except ImportError:  # pragma: no cover - optional runtime dependency
    spectral_open_image = None

try:
    import rasterio
    from rasterio.windows import Window
except ImportError:  # pragma: no cover - optional runtime dependency
    rasterio = None
    Window = None


@dataclass(frozen=True)
class LoadedImageData:
    lines: int
    samples: int
    bands: int
    interleave: str
    dtype: str
    wavelengths: np.ndarray
    wavelength_unit: str
    fwhm_arr: np.ndarray | None
    scale_factor: float | None
    data_ignore_value: float | None
    cube: np.ndarray | None
    data_backend: str


PREVIEW_MAX_DIM = 1800
PREVIEW_STRETCH_LOW = 2.0
PREVIEW_STRETCH_HIGH = 98.0
PREVIEW_SAMPLE_UPPER = 800_000
PREVIEW_ALGO_VERSION = "envi_quicklook_v2"


def _extract_envi_value(text: str, key: str) -> str | None:
    pattern = re.compile(
        rf"(?im)^\s*{re.escape(key)}\s*=\s*(\{{[^}}]*\}}|[^\r\n]+)\s*$",
        flags=re.MULTILINE,
    )
    m = pattern.search(text)
    if not m:
        return None
    return m.group(1).strip()


def _find_companion_hdr(path: Path) -> Path | None:
    candidates: list[Path] = []
    if path.suffix.lower() == ".hdr":
        candidates.append(path)
    else:
        candidates.append(path.with_suffix(".hdr"))
        candidates.append(Path(str(path) + ".hdr"))
        candidates.append(path.parent / f"{path.stem}.hdr")
    seen: set[str] = set()
    for cand in candidates:
        key = str(cand.resolve()) if cand.exists() else str(cand)
        if key in seen:
            continue
        seen.add(key)
        if cand.exists() and cand.is_file():
            return cand
    return None


def _load_metadata_from_hdr(hdr_path: Path) -> dict[str, Any]:
    try:
        text = hdr_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return {}
    out: dict[str, Any] = {}
    wave_raw = _extract_envi_value(text, "wavelength")
    fwhm_raw = _extract_envi_value(text, "fwhm")
    unit_raw = _extract_envi_value(text, "wavelength units")
    scale_raw = _extract_envi_value(text, "reflectance scale factor") or _extract_envi_value(text, "reflectance_scale_factor")
    ignore_raw = _extract_envi_value(text, "data ignore value") or _extract_envi_value(text, "data_ignore_value")
    if wave_raw is not None:
        out["wavelength"] = parse_header_array(wave_raw)
    if fwhm_raw is not None:
        out["fwhm"] = parse_header_array(fwhm_raw)
    if unit_raw is not None:
        out["wavelength_unit"] = str(unit_raw).strip().strip("{}").strip() or "nm"
    if scale_raw is not None:
        out["scale_factor"] = parse_scale_factor(scale_raw)
    if ignore_raw is not None:
        out["ignore_value"] = parse_ignore_value(ignore_raw)
    return out


def _first_valid_number(values: list[Any]) -> float | None:
    for v in values:
        if v is None:
            continue
        try:
            n = float(v)
        except Exception:
            continue
        if np.isfinite(n):
            return n
    return None


def parse_header_array(value: Any) -> np.ndarray:
    if value is None:
        return np.array([], dtype=np.float32)
    if isinstance(value, np.ndarray):
        return value.astype(np.float32, copy=False)
    if isinstance(value, (list, tuple)):
        return np.asarray([float(v) for v in value], dtype=np.float32)
    if isinstance(value, str):
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value)
        return np.asarray([float(v) for v in nums], dtype=np.float32)
    return np.array([], dtype=np.float32)


def parse_scale_factor(value: Any) -> float | None:
    arr = parse_header_array(value)
    if arr.size == 0:
        return None
    scale = float(arr[0])
    if not np.isfinite(scale) or scale <= 0:
        return None
    return scale


def parse_ignore_value(value: Any) -> float | None:
    arr = parse_header_array(value)
    if arr.size == 0:
        return None
    v = float(arr[0])
    if not np.isfinite(v):
        return None
    return v


def normalize_cube_shape(cube: np.ndarray, lines: int, samples: int, bands: int) -> np.ndarray:
    if cube.ndim != 3:
        raise AppError(ErrorCode.IMAGE_PARSE_FAILED, "Unsupported cube dimensions.", status_code=500)
    shape = cube.shape
    if shape == (lines, samples, bands):
        return cube
    if shape == (bands, lines, samples):
        return np.transpose(cube, (1, 2, 0))
    if shape == (lines, bands, samples):
        return np.transpose(cube, (0, 2, 1))
    raise AppError(
        ErrorCode.IMAGE_PARSE_FAILED,
        f"Unsupported cube shape: {shape}, expected permutations of (lines, samples, bands).",
        status_code=500,
    )


def _linear_stretch_envi_style(
    band: np.ndarray,
    scale_factor: float | None,
    ignore_value: float | None,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    chan = np.asarray(band, dtype=np.float32)
    chan = scale_to_unit_reflectance(chan, scale_factor=scale_factor)

    valid = np.isfinite(chan)
    if valid_mask is not None:
        valid &= np.asarray(valid_mask, dtype=bool)
    if ignore_value is not None and np.isfinite(ignore_value):
        scaled_ignore = float(scale_to_unit_reflectance(np.array([ignore_value], dtype=np.float32), scale_factor=scale_factor)[0])
        valid &= np.abs(chan - np.float32(scaled_ignore)) > 1e-6

    if np.count_nonzero(valid) == 0:
        return np.zeros(chan.shape, dtype=np.uint8)

    values = chan[valid]
    # ENVI-like robust linear percent stretch: clip using 2%-98% after removing extreme tails.
    if values.size > PREVIEW_SAMPLE_UPPER:
        step = max(1, values.size // PREVIEW_SAMPLE_UPPER)
        values = values[::step]
    lo_guard, hi_guard = np.percentile(values, [0.2, 99.8])
    guard_valid = valid & (chan >= lo_guard) & (chan <= hi_guard)
    if np.count_nonzero(guard_valid) < 64:
        guard_valid = valid
    lo, hi = np.percentile(chan[guard_valid], [PREVIEW_STRETCH_LOW, PREVIEW_STRETCH_HIGH])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(chan[valid]))
        hi = float(np.nanmax(chan[valid]))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros(chan.shape, dtype=np.uint8)

    stretched = np.clip((chan - lo) / (hi - lo), 0.0, 1.0)
    stretched[~valid] = 0.0
    return (stretched * 255.0).astype(np.uint8)


def create_preview_rgb_from_bands(
    bands_rgb: tuple[np.ndarray, np.ndarray, np.ndarray],
    scale_factor: float | None,
    ignore_value: float | None,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    out_channels = []
    for band in bands_rgb:
        out_channels.append(
            _linear_stretch_envi_style(
                band,
                scale_factor=scale_factor,
                ignore_value=ignore_value,
                valid_mask=valid_mask,
            )
        )
    return np.stack(out_channels, axis=-1)


def pick_rgb_bands(wavelengths: np.ndarray, display_mode: str) -> tuple[int, int, int]:
    if wavelengths.size == 0:
        return 0, 1, 2
    if display_mode == "false_color":
        targets = (850.0, 650.0, 550.0)
    else:
        targets = (650.0, 550.0, 450.0)
    return tuple(nearest_band_index(wavelengths, t) for t in targets)


class ImageService:
    def __init__(self, store: RuntimeStore) -> None:
        self.store = store
        self._preview_index_path = settings.previews_dir / "preview_index.json"
        self._preview_index_lock = RLock()

    def _load_preview_index(self) -> dict[str, str]:
        if not self._preview_index_path.exists():
            return {}
        try:
            raw = json.loads(self._preview_index_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(raw, dict):
            return {}
        out: dict[str, str] = {}
        for k, v in raw.items():
            if isinstance(k, str) and isinstance(v, str):
                out[k] = v
        return out

    def _save_preview_index(self, index: dict[str, str]) -> None:
        tmp = self._preview_index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self._preview_index_path)

    @staticmethod
    def _build_preview_cache_key(
        hdr: Path,
        display_mode: str,
        rgb_bands: tuple[int, int, int],
    ) -> str:
        stat = hdr.stat()
        raw = (
            f"{str(hdr)}|{int(stat.st_mtime_ns)}|{int(stat.st_size)}|"
            f"{display_mode}|{rgb_bands[0]}-{rgb_bands[1]}-{rgb_bands[2]}|{PREVIEW_ALGO_VERSION}"
        )
        return sha1(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _preview_stride(lines: int, samples: int) -> tuple[int, int]:
        step = max(1, int(ceil(max(lines, samples) / PREVIEW_MAX_DIM)))
        return step, step

    def _read_preview_bands(
        self,
        loaded: LoadedImageData,
        image_path: Path,
        rgb_bands: tuple[int, int, int],
    ) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray | None]:
        step_y, step_x = self._preview_stride(loaded.lines, loaded.samples)
        r_idx, g_idx, b_idx = rgb_bands
        if loaded.data_backend == "spy":
            if loaded.cube is None:
                raise AppError(ErrorCode.IMAGE_PARSE_FAILED, "spy cube not available", status_code=500)
            cube = loaded.cube
            bands = (
                np.asarray(cube[::step_y, ::step_x, r_idx], dtype=np.float32),
                np.asarray(cube[::step_y, ::step_x, g_idx], dtype=np.float32),
                np.asarray(cube[::step_y, ::step_x, b_idx], dtype=np.float32),
            )
            return bands, None

        if rasterio is None or Window is None:
            raise AppError(ErrorCode.IMAGE_PARSE_FAILED, "rasterio not available", status_code=500)
        with rasterio.open(image_path) as ds:
            out_h = max(1, int(ceil(loaded.lines / step_y)))
            out_w = max(1, int(ceil(loaded.samples / step_x)))
            read_kwargs = dict(
                out_shape=(out_h, out_w),
                out_dtype=np.float32,
                resampling=rasterio.enums.Resampling.nearest,
            )
            r = ds.read(r_idx + 1, **read_kwargs)
            g = ds.read(g_idx + 1, **read_kwargs)
            b = ds.read(b_idx + 1, **read_kwargs)
            masks = ds.read_masks(
                [r_idx + 1, g_idx + 1, b_idx + 1],
                out_shape=(3, out_h, out_w),
                resampling=rasterio.enums.Resampling.nearest,
            )
            valid_mask = np.all(masks > 0, axis=0)
        return (
            np.asarray(r, dtype=np.float32),
            np.asarray(g, dtype=np.float32),
            np.asarray(b, dtype=np.float32),
        ), valid_mask

    def _get_or_create_preview(
        self,
        loaded: LoadedImageData,
        image_path: Path,
        rgb_bands: tuple[int, int, int],
        preview_key: str,
    ) -> Path:
        with self._preview_index_lock:
            index = self._load_preview_index()
            existing_file = index.get(preview_key)
            if existing_file:
                existing_path = settings.previews_dir / existing_file
                if existing_path.exists():
                    return existing_path

            preview_file = f"pv_{preview_key[:20]}.png"
            preview_path = settings.previews_dir / preview_file
            if not preview_path.exists():
                bands_rgb, valid_mask = self._read_preview_bands(
                    loaded=loaded,
                    image_path=image_path,
                    rgb_bands=rgb_bands,
                )
                preview = create_preview_rgb_from_bands(
                    bands_rgb=bands_rgb,
                    scale_factor=loaded.scale_factor,
                    ignore_value=loaded.data_ignore_value,
                    valid_mask=valid_mask,
                )
                Image.fromarray(preview, mode="RGB").save(preview_path, format="PNG", optimize=True)

            index[preview_key] = preview_file
            self._save_preview_index(index)
            return preview_path

    @staticmethod
    def _validate_xy(ctx: ImageContext, x: int, y: int) -> None:
        if x < 0 or y < 0 or x >= ctx.samples or y >= ctx.lines:
            raise AppError(
                ErrorCode.PIXEL_OUT_OF_RANGE,
                f"pixel out of range: ({x}, {y})",
                status_code=400,
            )

    @staticmethod
    def _apply_ignore_value(curves: np.ndarray, ignore_value: float | None) -> np.ndarray:
        if ignore_value is None or not np.isfinite(ignore_value):
            return curves
        arr = np.asarray(curves, dtype=np.float32)
        arr[np.abs(arr - np.float32(ignore_value)) <= 1e-6] = np.nan
        return arr

    def _read_points_with_rasterio(
        self,
        image_path: Path,
        xs: np.ndarray,
        ys: np.ndarray,
    ) -> np.ndarray:
        if rasterio is None or Window is None:
            raise AppError(ErrorCode.IMAGE_PARSE_FAILED, "rasterio not available", status_code=500)

        x0 = int(np.min(xs))
        x1 = int(np.max(xs))
        y0 = int(np.min(ys))
        y1 = int(np.max(ys))
        width = x1 - x0 + 1
        height = y1 - y0 + 1

        with rasterio.open(image_path) as ds:
            data = ds.read(window=Window(x0, y0, width, height), out_dtype=np.float32)
            mask = ds.dataset_mask(window=Window(x0, y0, width, height))
        local_x = (xs - x0).astype(np.int64)
        local_y = (ys - y0).astype(np.int64)
        # data shape: (bands, height, width) -> (N, bands)
        curves = np.asarray(data[:, local_y, local_x].T, dtype=np.float32)
        valid = np.asarray(mask[local_y, local_x] > 0, dtype=bool)
        if curves.ndim == 1:
            curves = curves.reshape(1, -1)
            valid = valid.reshape(1)
        curves[~valid, :] = np.nan
        return curves

    def read_pixel_spectrum(self, ctx: ImageContext, x: int, y: int) -> np.ndarray:
        self._validate_xy(ctx, x, y)
        if ctx.data_backend == "spy":
            if ctx.cube is None:
                raise AppError(ErrorCode.IMAGE_PARSE_FAILED, "spy cube not available", status_code=500)
            spec = np.asarray(ctx.cube[y, x, :], dtype=np.float32).copy()
            return self._apply_ignore_value(spec, ctx.data_ignore_value)

        curves = self._read_points_with_rasterio(
            image_path=ctx.image_path,
            xs=np.asarray([x], dtype=np.int32),
            ys=np.asarray([y], dtype=np.int32),
        )
        spec = curves[0].copy()
        return self._apply_ignore_value(spec, ctx.data_ignore_value)

    def read_point_spectra(self, ctx: ImageContext, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        xs = np.asarray(xs, dtype=np.int64)
        ys = np.asarray(ys, dtype=np.int64)
        if xs.size == 0 or ys.size == 0 or xs.shape != ys.shape:
            raise AppError(ErrorCode.MATCH_FAILED, "invalid selected pixels", status_code=422)
        if np.min(xs) < 0 or np.min(ys) < 0 or np.max(xs) >= ctx.samples or np.max(ys) >= ctx.lines:
            raise AppError(ErrorCode.PIXEL_OUT_OF_RANGE, "selected pixels out of range", status_code=400)

        if ctx.data_backend == "spy":
            if ctx.cube is None:
                raise AppError(ErrorCode.IMAGE_PARSE_FAILED, "spy cube not available", status_code=500)
            curves = np.asarray(ctx.cube[ys, xs, :], dtype=np.float32)
            if curves.ndim == 1:
                curves = curves.reshape(1, -1)
            return self._apply_ignore_value(curves, ctx.data_ignore_value)

        curves = self._read_points_with_rasterio(
            image_path=ctx.image_path,
            xs=xs.astype(np.int32),
            ys=ys.astype(np.int32),
        )
        return self._apply_ignore_value(curves, ctx.data_ignore_value)

    def load_image(self, image_path: str, display_mode: str) -> ImageContext:
        hdr = Path(image_path).expanduser().resolve()
        if not hdr.exists():
            raise AppError(ErrorCode.IMAGE_FILE_NOT_FOUND, f"image file not found: {hdr}")
        if spectral_open_image is None and rasterio is None:
            raise AppError(
                ErrorCode.IMAGE_PARSE_FAILED,
                "Neither spectral nor rasterio is installed.",
                status_code=500,
            )

        suffix = hdr.suffix.lower()
        prefer_rasterio = suffix in {".tif", ".tiff", ".geotiff"}

        try:
            if prefer_rasterio and rasterio is not None:
                loaded = self._load_with_rasterio(hdr)
            else:
                loaded = self._load_with_spy(hdr) if spectral_open_image is not None else self._load_with_rasterio(hdr)
        except Exception as primary_exc:
            if rasterio is None:
                raise AppError(
                    ErrorCode.IMAGE_PARSE_FAILED,
                    f"failed to parse image: {primary_exc}",
                    status_code=500,
                ) from primary_exc
            try:
                loaded = self._load_with_rasterio(hdr)
            except Exception as fallback_exc:
                raise AppError(
                    ErrorCode.IMAGE_PARSE_FAILED,
                    f"failed to parse image with both SPy and rasterio: {fallback_exc}",
                    status_code=500,
                ) from fallback_exc

        rgb_bands = pick_rgb_bands(loaded.wavelengths, display_mode)
        preview_key = self._build_preview_cache_key(hdr=hdr, display_mode=display_mode, rgb_bands=rgb_bands)
        image_id = f"img_{preview_key[:12]}"
        preview_path = self._get_or_create_preview(
            loaded=loaded,
            image_path=hdr,
            rgb_bands=rgb_bands,
            preview_key=preview_key,
        )

        ctx = ImageContext(
            image_id=image_id,
            image_path=hdr,
            lines=loaded.lines,
            samples=loaded.samples,
            bands=loaded.bands,
            interleave=loaded.interleave,
            dtype=loaded.dtype,
            wavelengths=loaded.wavelengths.astype(np.float32),
            wavelength_unit=loaded.wavelength_unit,
            fwhm=loaded.fwhm_arr.astype(np.float32) if loaded.fwhm_arr is not None else None,
            reflectance_scale_factor=loaded.scale_factor,
            data_ignore_value=loaded.data_ignore_value,
            preview_path=preview_path,
            rgb_bands=rgb_bands,
            data_backend=loaded.data_backend,
            cube=loaded.cube,
        )
        self.store.upsert_image(ctx)
        return ctx

    @staticmethod
    def _normalize_wave_unit(
        wavelengths: np.ndarray,
        fwhm: np.ndarray,
        wavelength_unit: str,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        unit_lower = wavelength_unit.lower()
        if unit_lower in {"um", "μm", "micrometer", "micrometers"}:
            wavelengths = wavelengths * 1000.0
            if fwhm.size:
                fwhm = fwhm * 1000.0
            return wavelengths, fwhm, "nm"
        return wavelengths, fwhm, wavelength_unit

    def _load_with_spy(self, hdr: Path):
        if spectral_open_image is None:
            raise RuntimeError("spectral not available")
        spy_img = spectral_open_image(str(hdr))
        meta = getattr(spy_img, "metadata", {}) or {}
        lines = int(meta.get("lines", getattr(spy_img, "nrows", 0)))
        samples = int(meta.get("samples", getattr(spy_img, "ncols", 0)))
        bands = int(meta.get("bands", getattr(spy_img, "nbands", 0)))
        interleave = str(meta.get("interleave", "unknown")).lower()
        dtype = str(meta.get("data type", "unknown"))
        wavelengths = parse_header_array(meta.get("wavelength"))
        fwhm = parse_header_array(meta.get("fwhm"))
        scale_factor = parse_scale_factor(
            meta.get("reflectance scale factor", None)
            if "reflectance scale factor" in meta
            else meta.get("reflectance_scale_factor", None)
        )
        ignore_value = parse_ignore_value(
            meta.get("data ignore value", None)
            if "data ignore value" in meta
            else meta.get("data_ignore_value", None)
        )
        wavelength_unit = str(meta.get("wavelength units", "nm"))

        if wavelengths.size == 0:
            wavelengths = np.arange(bands, dtype=np.float32)
            wavelength_unit = "index"
        if wavelengths.size != bands:
            wavelengths = wavelengths[:bands] if wavelengths.size > bands else np.pad(
                wavelengths,
                (0, bands - wavelengths.size),
                mode="edge",
            )
        wavelengths, fwhm, wavelength_unit = self._normalize_wave_unit(wavelengths, fwhm, wavelength_unit)
        if fwhm.size == 0:
            fwhm_arr: np.ndarray | None = None
        elif fwhm.size >= bands:
            fwhm_arr = fwhm[:bands]
        else:
            fwhm_arr = np.pad(fwhm, (0, bands - fwhm.size), mode="edge")

        cube = spy_img.open_memmap()
        cube_lsb = normalize_cube_shape(np.asarray(cube), lines, samples, bands)
        return LoadedImageData(
            lines,
            samples,
            bands,
            interleave,
            dtype,
            wavelengths.astype(np.float32),
            wavelength_unit,
            fwhm_arr.astype(np.float32) if fwhm_arr is not None else None,
            scale_factor,
            ignore_value,
            cube_lsb,
            "spy",
        )

    def _load_with_rasterio(self, hdr: Path):
        if rasterio is None:
            raise RuntimeError("rasterio not available")
        with rasterio.open(hdr) as ds:
            lines = int(ds.height)
            samples = int(ds.width)
            bands = int(ds.count)
            interleave = str(ds.profile.get("interleave", "unknown")).lower()
            dtype = str(ds.dtypes[0]) if ds.dtypes else "unknown"
            tags = ds.tags() or {}
            tags_envi = ds.tags(ns="ENVI") or {}
            wavelengths = parse_header_array(tags.get("wavelength") or tags_envi.get("wavelength"))
            fwhm = parse_header_array(tags.get("fwhm") or tags_envi.get("fwhm"))
            scale_factor = parse_scale_factor(
                tags.get("reflectance scale factor", None)
                if "reflectance scale factor" in tags
                else tags.get("reflectance_scale_factor", None) or tags_envi.get("reflectance scale factor") or tags_envi.get("reflectance_scale_factor")
            )
            ignore_value = parse_ignore_value(
                tags.get("data ignore value", None)
                if "data ignore value" in tags
                else tags.get("data_ignore_value", None) or tags_envi.get("data ignore value") or tags_envi.get("data_ignore_value")
            )
            nodata_from_ds = _first_valid_number([ds.nodata, *(ds.nodatavals or ())])
            if ignore_value is None:
                ignore_value = nodata_from_ds

            wavelength_unit = str(tags.get("wavelength units") or tags_envi.get("wavelength units") or "nm")

            need_hdr = wavelengths.size == 0 or fwhm.size == 0
            if need_hdr:
                companion_hdr = _find_companion_hdr(hdr)
                if companion_hdr is not None:
                    from_hdr = _load_metadata_from_hdr(companion_hdr)
                    if wavelengths.size == 0:
                        wavelengths = np.asarray(from_hdr.get("wavelength", np.array([], dtype=np.float32)), dtype=np.float32)
                    if fwhm.size == 0:
                        fwhm = np.asarray(from_hdr.get("fwhm", np.array([], dtype=np.float32)), dtype=np.float32)
                    if wavelength_unit.strip().lower() in {"", "unknown", "index"} and from_hdr.get("wavelength_unit"):
                        wavelength_unit = str(from_hdr["wavelength_unit"])
                    if scale_factor is None and "scale_factor" in from_hdr:
                        scale_factor = from_hdr["scale_factor"]
                    if ignore_value is None and "ignore_value" in from_hdr:
                        ignore_value = from_hdr["ignore_value"]

            if wavelengths.size == 0 or fwhm.size == 0:
                raise AppError(
                    ErrorCode.IMAGE_PARSE_FAILED,
                    (
                        "failed to obtain required wavelength/fwhm information from image metadata; "
                        "please provide a companion .hdr file with wavelength and fwhm."
                    ),
                    status_code=422,
                )

            if wavelengths.size != bands:
                wavelengths = wavelengths[:bands] if wavelengths.size > bands else np.pad(
                    wavelengths,
                    (0, bands - wavelengths.size),
                    mode="edge",
                )
            wavelengths, fwhm, wavelength_unit = self._normalize_wave_unit(wavelengths, fwhm, wavelength_unit)
            if fwhm.size == 0:
                fwhm_arr: np.ndarray | None = None
            elif fwhm.size >= bands:
                fwhm_arr = fwhm[:bands]
            else:
                fwhm_arr = np.pad(fwhm, (0, bands - fwhm.size), mode="edge")

        return LoadedImageData(
            lines,
            samples,
            bands,
            interleave,
            dtype,
            wavelengths.astype(np.float32),
            wavelength_unit,
            fwhm_arr.astype(np.float32) if fwhm_arr is not None else None,
            scale_factor,
            ignore_value,
            None,
            "rasterio",
        )

    def extract_pixel_spectrum(self, image_id: str, x: int, y: int) -> tuple[np.ndarray, ImageContext]:
        ctx = self.store.get_image(image_id)
        if ctx is None:
            raise AppError(
                ErrorCode.IMAGE_CONTEXT_NOT_FOUND,
                f"image context not found: {image_id}",
                status_code=404,
            )
        spectrum = self.read_pixel_spectrum(ctx=ctx, x=x, y=y)
        return spectrum, ctx

    @staticmethod
    def rgb_model(r: int, g: int, b: int) -> ImageRGBBands:
        return ImageRGBBands(r=r, g=g, b=b)
