from __future__ import annotations

import re
import time
from dataclasses import dataclass

import numpy as np

from app.core.config import settings
from app.core.errors import AppError, ErrorCode
from app.models.schemas import (
    MatchContextData,
    MatchResultItem,
    PixelMatchData,
    QuerySpectrumData,
    RegionSelection,
    SpectralMaskRange,
    SignatureInfo,
)
from app.services.cache_service import SignatureCacheService
from app.services.image_service import ImageService
from app.services.library_service import LibraryService
from app.services.state_store import ImageContext
from app.utils.signature import build_signature_hash
from app.utils.spectral_math import (
    build_custom_range_mask,
    merge_mask_ranges,
    normalize_vector,
    pearson_r,
    sanitize_reflectance,
    scale_to_unit_reflectance,
)


@dataclass
class PreparedSignature:
    signature_hash: str
    status: str
    progress: int
    current_step: str
    cache_exists: bool


@dataclass
class QuerySelectionResult:
    spectrum: np.ndarray
    center_x: int
    center_y: int
    mode: str
    pixel_count: int


VARIANT_MODE_SUFFIX_RE = re.compile(
    r"_(ASD(?:FR|HR|NG)[A-Za-z0-9]*|AVIRIS[A-Za-z0-9]*|BECK[A-Za-z0-9]*|NIC4[A-Za-z0-9]*)_(AREF|RREF|RTGC|TRAN)$",
    flags=re.IGNORECASE,
)


def normalize_display_name(raw_name: str) -> str:
    s = raw_name.strip()
    if s.lower().startswith("splib07b_"):
        s = s[9:]
    s = VARIANT_MODE_SUFFIX_RE.sub("", s)
    return s.replace("_", " ").strip() or raw_name


def parse_variant_mode(raw_name: str) -> tuple[str | None, str | None]:
    m = VARIANT_MODE_SUFFIX_RE.search(raw_name)
    if not m:
        return None, None
    return m.group(1).upper(), m.group(2).upper()


class MatchService:
    def __init__(
        self,
        image_service: ImageService,
        cache_service: SignatureCacheService,
        library_service: LibraryService,
    ) -> None:
        self.image_service = image_service
        self.cache_service = cache_service
        self.library_service = library_service

    def prepare_signature(
        self,
        image_id: str,
        ignore_water_bands: bool,
        build_async: bool = True,
    ) -> PreparedSignature:
        ctx = self.image_service.store.get_image(image_id)
        if ctx is None:
            raise AppError(
                ErrorCode.IMAGE_CONTEXT_NOT_FOUND,
                f"image context not found: {image_id}",
                status_code=404,
            )
        signature_hash = build_signature_hash(
            image_waves=ctx.wavelengths,
            image_fwhm=ctx.fwhm,
            ignore_water_bands=ignore_water_bands,
            resample_algo_version=settings.resample_algo_version,
            clean_rules_version=settings.clean_rules_version,
        )

        current = self.cache_service.status(signature_hash)
        if current["status"] == "not_found" and build_async:
            self.cache_service.start_build_async(
                signature_hash=signature_hash,
                img_waves=ctx.wavelengths,
                img_fwhm=ctx.fwhm,
                ignore_water_bands=ignore_water_bands,
            )
            current = self.cache_service.status(signature_hash)

        return PreparedSignature(
            signature_hash=signature_hash,
            status=str(current["status"]),
            progress=int(current["progress"]),
            current_step=str(current["current_step"]),
            cache_exists=str(current["status"]) == "ready",
        )

    @staticmethod
    def to_signature_info(prepared: PreparedSignature, ignore_water_bands: bool) -> SignatureInfo:
        return SignatureInfo(
            hash=prepared.signature_hash,
            ignore_water_bands=ignore_water_bands,
            cache_exists=prepared.cache_exists,
            build_status=prepared.status,  # type: ignore[arg-type]
        )

    @staticmethod
    def _clip_xy(ctx: ImageContext, x: float, y: float) -> tuple[int, int]:
        ix = int(np.clip(int(round(x)), 0, ctx.samples - 1))
        iy = int(np.clip(int(round(y)), 0, ctx.lines - 1))
        return ix, iy

    @staticmethod
    def _normalize_custom_mask_ranges(
        ranges: list[SpectralMaskRange] | None,
    ) -> list[tuple[float, float]]:
        if not ranges:
            return []
        return merge_mask_ranges([(float(r.start), float(r.end)) for r in ranges])

    @staticmethod
    def _compute_nan_aware_sam_batch(
        batch: np.ndarray,
        query_clean: np.ndarray,
        valid_mask: np.ndarray,
        min_valid_bands: int,
    ) -> np.ndarray:
        rows = np.asarray(batch, dtype=np.float32)
        if rows.size == 0:
            return np.empty((0,), dtype=np.float32)

        q = np.asarray(query_clean, dtype=np.float32)
        valid_cols = np.asarray(valid_mask, dtype=bool)
        q_valid = valid_cols & np.isfinite(q)
        if int(np.count_nonzero(q_valid)) < int(min_valid_bands):
            return np.full((rows.shape[0],), np.float32(np.pi), dtype=np.float32)

        q_filled = np.where(q_valid, q, 0.0).astype(np.float32, copy=False)
        q_sq = (q_filled * q_filled).astype(np.float32, copy=False)

        row_valid = np.isfinite(rows) & q_valid.reshape(1, -1)
        row_filled = np.where(row_valid, rows, 0.0).astype(np.float32, copy=False)

        valid_counts = np.count_nonzero(row_valid, axis=1).astype(np.int32, copy=False)
        numer = row_filled @ q_filled
        den_row = np.sqrt(np.sum(row_filled * row_filled, axis=1).astype(np.float32, copy=False))
        den_q = np.sqrt((row_valid.astype(np.float32, copy=False) @ q_sq).astype(np.float32, copy=False))
        den = den_row * den_q

        scores = np.full((rows.shape[0],), np.float32(np.pi), dtype=np.float32)
        good = (valid_counts >= int(min_valid_bands)) & np.isfinite(den) & (den > 1e-8)
        if np.any(good):
            cosine = np.clip(numer[good] / den[good], -1.0, 1.0)
            scores[good] = np.arccos(cosine).astype(np.float32, copy=False)
        return scores

    def _score_all_candidates_nan_aware(
        self,
        signature_hash: str,
        total: int,
        query_clean: np.ndarray,
        valid_mask: np.ndarray,
        min_valid_bands: int,
    ) -> np.ndarray:
        chunk = 2048
        scores = np.empty((total,), dtype=np.float32)
        for start in range(0, total, chunk):
            end = min(start + chunk, total)
            batch = self.cache_service.load_resampled_slice(signature_hash, start, end)
            if batch.size == 0:
                scores[start:end] = np.float32(np.pi)
                continue
            scores[start:end] = self._compute_nan_aware_sam_batch(
                batch=batch,
                query_clean=query_clean,
                valid_mask=valid_mask,
                min_valid_bands=min_valid_bands,
            )
        return scores

    def _score_candidate_subset_nan_aware(
        self,
        signature_hash: str,
        row_indices: np.ndarray,
        query_clean: np.ndarray,
        valid_mask: np.ndarray,
        min_valid_bands: int,
    ) -> np.ndarray:
        idx = np.asarray(row_indices, dtype=np.int64)
        if idx.size == 0:
            return np.empty((0,), dtype=np.float32)
        rows = self.cache_service.load_resampled_rows(signature_hash, idx)
        if rows.size == 0:
            return np.full((idx.shape[0],), np.float32(np.pi), dtype=np.float32)
        return self._compute_nan_aware_sam_batch(
            batch=rows,
            query_clean=query_clean,
            valid_mask=valid_mask,
            min_valid_bands=min_valid_bands,
        )

    def _mean_curve_from_pixels(
        self,
        ctx: ImageContext,
        xs: np.ndarray,
        ys: np.ndarray,
    ) -> QuerySelectionResult:
        if xs.size == 0 or ys.size == 0:
            raise AppError(ErrorCode.MATCH_FAILED, "selection contains no pixel", status_code=422)

        curves = self.image_service.read_point_spectra(ctx=ctx, xs=xs, ys=ys)
        if curves.ndim == 1:
            curves = curves.reshape(1, -1)
        mean_curve = np.nanmean(curves, axis=0)
        if not np.isfinite(mean_curve).any():
            raise AppError(
                ErrorCode.MATCH_FAILED,
                "selection has no valid reflectance values",
                status_code=422,
            )

        cx = int(np.clip(int(np.round(float(np.mean(xs)))), 0, ctx.samples - 1))
        cy = int(np.clip(int(np.round(float(np.mean(ys)))), 0, ctx.lines - 1))
        return QuerySelectionResult(
            spectrum=mean_curve.astype(np.float32),
            center_x=cx,
            center_y=cy,
            mode="pixel",
            pixel_count=int(xs.size),
        )

    @staticmethod
    def _points_inside_polygon(
        xs: np.ndarray,
        ys: np.ndarray,
        poly_x: np.ndarray,
        poly_y: np.ndarray,
    ) -> np.ndarray:
        inside = np.zeros(xs.shape[0], dtype=bool)
        j = poly_x.shape[0] - 1
        for i in range(poly_x.shape[0]):
            xi, yi = poly_x[i], poly_y[i]
            xj, yj = poly_x[j], poly_y[j]
            intersects = ((yi > ys) != (yj > ys)) & (
                xs < (xj - xi) * (ys - yi) / ((yj - yi) + 1e-12) + xi
            )
            inside ^= intersects
            j = i
        return inside

    def _extract_query_selection(
        self,
        ctx: ImageContext,
        x: int,
        y: int,
        selection: RegionSelection | None,
    ) -> QuerySelectionResult:
        if selection is None or selection.mode == "pixel":
            curve = self.image_service.read_pixel_spectrum(ctx=ctx, x=x, y=y)
            return QuerySelectionResult(
                spectrum=curve,
                center_x=x,
                center_y=y,
                mode="pixel",
                pixel_count=1,
            )

        mode = selection.mode
        if mode == "box":
            if None in (selection.x0, selection.y0, selection.x1, selection.y1):
                raise AppError(ErrorCode.INVALID_REQUEST, "box selection requires x0,y0,x1,y1", status_code=422)
            x0, x1 = sorted((int(selection.x0), int(selection.x1)))
            y0, y1 = sorted((int(selection.y0), int(selection.y1)))
            x0 = int(np.clip(x0, 0, ctx.samples - 1))
            x1 = int(np.clip(x1, 0, ctx.samples - 1))
            y0 = int(np.clip(y0, 0, ctx.lines - 1))
            y1 = int(np.clip(y1, 0, ctx.lines - 1))
            if x1 < x0 or y1 < y0:
                raise AppError(ErrorCode.MATCH_FAILED, "invalid box selection", status_code=422)
            xs = np.arange(x0, x1 + 1, dtype=np.int32)
            ys = np.arange(y0, y1 + 1, dtype=np.int32)
            gx, gy = np.meshgrid(xs, ys)
            result = self._mean_curve_from_pixels(ctx=ctx, xs=gx.ravel(), ys=gy.ravel())
            result.mode = "box"
            return result

        if mode == "circle":
            if selection.cx is None or selection.cy is None or selection.radius is None:
                raise AppError(ErrorCode.INVALID_REQUEST, "circle selection requires cx,cy,radius", status_code=422)
            radius = float(selection.radius)
            if not np.isfinite(radius) or radius <= 0:
                raise AppError(ErrorCode.INVALID_REQUEST, "circle radius must be > 0", status_code=422)
            cx = float(selection.cx)
            cy = float(selection.cy)
            x0 = int(np.clip(int(np.floor(cx - radius)), 0, ctx.samples - 1))
            x1 = int(np.clip(int(np.ceil(cx + radius)), 0, ctx.samples - 1))
            y0 = int(np.clip(int(np.floor(cy - radius)), 0, ctx.lines - 1))
            y1 = int(np.clip(int(np.ceil(cy + radius)), 0, ctx.lines - 1))
            xs = np.arange(x0, x1 + 1, dtype=np.int32)
            ys = np.arange(y0, y1 + 1, dtype=np.int32)
            gx, gy = np.meshgrid(xs, ys)
            mask = ((gx.astype(np.float64) - cx) ** 2 + (gy.astype(np.float64) - cy) ** 2) <= (radius ** 2)
            result = self._mean_curve_from_pixels(ctx=ctx, xs=gx[mask].ravel(), ys=gy[mask].ravel())
            result.mode = "circle"
            return result

        if mode == "lasso":
            points = selection.points or []
            if len(points) < 3:
                raise AppError(ErrorCode.INVALID_REQUEST, "lasso selection requires at least 3 points", status_code=422)
            poly_x = np.asarray([float(p.x) for p in points], dtype=np.float64)
            poly_y = np.asarray([float(p.y) for p in points], dtype=np.float64)
            poly_x = np.clip(poly_x, 0.0, float(ctx.samples - 1))
            poly_y = np.clip(poly_y, 0.0, float(ctx.lines - 1))
            x0 = int(np.floor(np.min(poly_x)))
            x1 = int(np.ceil(np.max(poly_x)))
            y0 = int(np.floor(np.min(poly_y)))
            y1 = int(np.ceil(np.max(poly_y)))
            xs = np.arange(x0, x1 + 1, dtype=np.int32)
            ys = np.arange(y0, y1 + 1, dtype=np.int32)
            gx, gy = np.meshgrid(xs, ys)
            flat_x = gx.ravel().astype(np.float64) + 0.5
            flat_y = gy.ravel().astype(np.float64) + 0.5
            inside = self._points_inside_polygon(flat_x, flat_y, poly_x, poly_y)
            result = self._mean_curve_from_pixels(ctx=ctx, xs=gx.ravel()[inside], ys=gy.ravel()[inside])
            result.mode = "lasso"
            return result

        raise AppError(ErrorCode.INVALID_REQUEST, f"unsupported selection mode: {mode}", status_code=422)

    def match_pixel(
        self,
        image_id: str,
        x: int,
        y: int,
        top_n: int,
        ignore_water_bands: bool,
        min_valid_bands: int | None,
        return_candidate_curves: bool,
        selection: RegionSelection | None = None,
        custom_masked_ranges: list[SpectralMaskRange] | None = None,
    ) -> PixelMatchData:
        started = time.perf_counter()
        ctx = self.image_service.store.get_image(image_id)
        if ctx is None:
            raise AppError(
                ErrorCode.IMAGE_CONTEXT_NOT_FOUND,
                f"image context not found: {image_id}",
                status_code=404,
            )
        query_selection = self._extract_query_selection(ctx=ctx, x=x, y=y, selection=selection)
        query_spec = query_selection.spectrum
        prepared = self.prepare_signature(image_id=image_id, ignore_water_bands=ignore_water_bands, build_async=False)
        if prepared.status in {"not_found", "failed"}:
            self.cache_service.build_sync(
                signature_hash=prepared.signature_hash,
                img_waves=ctx.wavelengths,
                img_fwhm=ctx.fwhm,
                ignore_water_bands=ignore_water_bands,
            )
        status_after = self.cache_service.status(prepared.signature_hash)
        if status_after["status"] == "building":
            ready = self.cache_service.wait_until_ready(
                prepared.signature_hash,
                timeout_sec=settings.cache_build_wait_timeout_sec,
                poll_ms=settings.cache_build_wait_poll_ms,
            )
            if not ready:
                raise AppError(
                    ErrorCode.CACHE_BUILDING,
                    f"signature cache is building: {prepared.signature_hash}",
                    status_code=409,
                )
            status_after = self.cache_service.status(prepared.signature_hash)
        if status_after["status"] != "ready":
            raise AppError(
                ErrorCode.CACHE_FAILED,
                f"signature cache is not ready: {prepared.signature_hash}",
                status_code=500,
            )

        active = self.cache_service.load_active_signature(prepared.signature_hash)
        base_valid_mask = active.valid_mask.astype(bool)
        normalized_custom_ranges = self._normalize_custom_mask_ranges(custom_masked_ranges)
        self.image_service.store.set_image_mask_ranges(image_id, normalized_custom_ranges)
        dynamic_mask = build_custom_range_mask(ctx.wavelengths, normalized_custom_ranges)
        valid_mask = base_valid_mask & dynamic_mask
        query_scaled = scale_to_unit_reflectance(
            query_spec,
            scale_factor=ctx.reflectance_scale_factor,
        )
        query_clean = sanitize_reflectance(
            query_scaled,
            clip_min=settings.clip_reflectance_min,
            clip_max=settings.clip_reflectance_max,
        )
        effective_min_valid_bands = int(
            min_valid_bands if min_valid_bands is not None else settings.min_valid_bands
        )
        effective_min_valid_bands = max(1, min(effective_min_valid_bands, int(ctx.bands)))
        bands_used_mask = valid_mask & np.isfinite(query_clean)
        bands_used = int(np.count_nonzero(bands_used_mask))
        if bands_used < effective_min_valid_bands:
            raise AppError(
                ErrorCode.MATCH_FAILED,
                f"not enough valid bands: {bands_used} < {effective_min_valid_bands}",
                status_code=422,
            )

        query_norm = normalize_vector(query_clean, valid_mask=valid_mask)
        total = int(active.spectra_norm.shape[0])
        candidate_idx: np.ndarray
        if normalized_custom_ranges:
            candidate_idx = np.arange(total, dtype=np.int64)
            sam_scores = self._score_all_candidates_nan_aware(
                signature_hash=prepared.signature_hash,
                total=total,
                query_clean=query_clean,
                valid_mask=valid_mask,
                min_valid_bands=effective_min_valid_bands,
            )
        else:
            if active.faiss_index is not None and total >= 50000:
                top_k = min(settings.match_candidate_topk, total)
                scores, idx = active.faiss_index.search(query_norm.reshape(1, -1).astype(np.float32), top_k)
                _ = scores
                candidate_idx = idx[0]
                candidate_idx = candidate_idx[candidate_idx >= 0].astype(np.int64)
            else:
                candidate_idx = np.arange(total, dtype=np.int64)

            if candidate_idx.size == 0:
                raise AppError(ErrorCode.MATCH_FAILED, "no candidates returned by retrieval", status_code=500)

            if candidate_idx.shape[0] == total:
                sam_scores = self._score_all_candidates_nan_aware(
                    signature_hash=prepared.signature_hash,
                    total=total,
                    query_clean=query_clean,
                    valid_mask=valid_mask,
                    min_valid_bands=effective_min_valid_bands,
                )
            else:
                sam_scores = self._score_candidate_subset_nan_aware(
                    signature_hash=prepared.signature_hash,
                    row_indices=candidate_idx,
                    query_clean=query_clean,
                    valid_mask=valid_mask,
                    min_valid_bands=effective_min_valid_bands,
                )

        if sam_scores.size == 0:
            raise AppError(ErrorCode.MATCH_FAILED, "no candidates returned by retrieval", status_code=500)

        order = np.argsort(sam_scores)
        top_count = min(top_n, order.shape[0])
        top_local_idx = order[:top_count]
        top_candidate_idx = candidate_idx[top_local_idx]
        top_sam = sam_scores[top_local_idx]
        candidate_count = int(candidate_idx.shape[0])

        row_indices = top_candidate_idx.astype(np.int64)
        top_curves = self.cache_service.load_resampled_rows(prepared.signature_hash, row_indices)
        spectrum_ids = active.meta_ids[row_indices].astype(int)
        meta_map = self.library_service.fetch_metadata(spectrum_ids.tolist())

        results: list[MatchResultItem] = []
        for rank, (row_idx, sid, sam_score) in enumerate(zip(row_indices, spectrum_ids, top_sam, strict=True), start=1):
            curve = top_curves[rank - 1] if top_curves.size else np.array([], dtype=np.float32)
            pearson = pearson_r(query_clean, curve, valid_mask=valid_mask) if curve.size else float("nan")
            meta = meta_map.get(
                int(sid),
                None,
            )
            raw_name = meta.name if meta is not None else f"Spectrum {sid}"
            sensor_type = meta.instrument_variant if meta is not None else None
            measure_mode = meta.measure_mode if meta is not None else None
            if sensor_type is None or measure_mode is None:
                parsed_sensor, parsed_mode = parse_variant_mode(raw_name)
                sensor_type = sensor_type or parsed_sensor
                measure_mode = measure_mode or parsed_mode

            name = normalize_display_name(raw_name)
            class_name = meta.class_name if meta is not None else None
            source = meta.source if meta is not None else None
            results.append(
                MatchResultItem(
                    rank=rank,
                    spectrum_id=int(sid),
                    name=name,
                    class_name=class_name,
                    sensor_type=sensor_type,
                    measure_mode=measure_mode,
                    source=source,
                    sam_score=float(sam_score),
                    pearson_r=float(pearson) if np.isfinite(pearson) else None,
                    curve=curve.astype(float).tolist() if return_candidate_curves and curve.size else None,
                )
            )

        elapsed = int((time.perf_counter() - started) * 1000)
        return PixelMatchData(
            query=QuerySpectrumData(
                x=query_selection.center_x,
                y=query_selection.center_y,
                bands_total=ctx.bands,
                bands_used=bands_used,
                selection_mode=query_selection.mode,  # type: ignore[arg-type]
                selected_pixels=query_selection.pixel_count,
                wavelengths=ctx.wavelengths.astype(float).tolist(),
                spectrum=query_clean.astype(float).tolist(),
            ),
            match_context=MatchContextData(
                signature_hash=prepared.signature_hash,
                metric="sam",
                ignore_water_bands=ignore_water_bands,
                min_valid_bands=effective_min_valid_bands,
                custom_masked_ranges=[
                    SpectralMaskRange(start=float(start), end=float(end))
                    for start, end in normalized_custom_ranges
                ],
                candidate_count=candidate_count,
                elapsed_ms=elapsed,
            ),
            results=results,
        )

    def extract_spectrum(self, image_id: str, x: int, y: int) -> tuple[np.ndarray, np.ndarray]:
        spectrum, ctx = self.image_service.extract_pixel_spectrum(image_id=image_id, x=x, y=y)
        spectrum_scaled = scale_to_unit_reflectance(
            spectrum,
            scale_factor=ctx.reflectance_scale_factor,
        )
        spectrum = sanitize_reflectance(
            spectrum_scaled,
            clip_min=settings.clip_reflectance_min,
            clip_max=settings.clip_reflectance_max,
        )
        return ctx.wavelengths.astype(np.float32), spectrum.astype(np.float32)
