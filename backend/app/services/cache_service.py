from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, Thread

import h5py
import numpy as np

from app.core.config import settings
from app.core.errors import AppError, ErrorCode
from app.services.library_service import LibraryService
from app.services.state_store import ActiveSignatureData, RuntimeStore
from app.utils.spectral_math import (
    build_valid_mask,
    normalize_rows,
    resample_gaussian_fwhm,
    resample_linear,
    sanitize_reflectance,
)

try:
    import faiss
except ImportError:  # pragma: no cover - optional runtime dependency
    faiss = None


class SignatureCacheService:
    def __init__(self, store: RuntimeStore, library_service: LibraryService) -> None:
        self.store = store
        self.library_service = library_service
        self._build_locks: dict[str, Lock] = {}
        self._locks_lock = Lock()

    def cache_dir(self, signature_hash: str) -> Path:
        return settings.signatures_dir / signature_hash

    def status(self, signature_hash: str) -> dict[str, str | int]:
        in_mem = self.store.get_signature_status(signature_hash)
        if in_mem is not None:
            return {
                "signature_hash": signature_hash,
                "status": in_mem.status,
                "progress": in_mem.progress,
                "current_step": in_mem.current_step,
            }

        cdir = self.cache_dir(signature_hash)
        lock_path = cdir / "build.lock"
        meta_path = cdir / "meta.json"
        if lock_path.exists():
            return {
                "signature_hash": signature_hash,
                "status": "building",
                "progress": 1,
                "current_step": "build_lock_exists",
            }
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                status = str(meta.get("status", "ready"))
                progress = int(meta.get("progress", 100 if status == "ready" else 0))
                step = str(meta.get("current_step", "ready"))
                return {
                    "signature_hash": signature_hash,
                    "status": status,
                    "progress": progress,
                    "current_step": step,
                }
            except Exception:
                pass
        return {
            "signature_hash": signature_hash,
            "status": "not_found",
            "progress": 0,
            "current_step": "",
        }

    def start_build_async(
        self,
        signature_hash: str,
        img_waves: np.ndarray,
        img_fwhm: np.ndarray | None,
        ignore_water_bands: bool,
    ) -> None:
        current = self.status(signature_hash)
        if current["status"] in {"ready", "building"}:
            return
        t = Thread(
            target=self.build_sync,
            kwargs={
                "signature_hash": signature_hash,
                "img_waves": img_waves,
                "img_fwhm": img_fwhm,
                "ignore_water_bands": ignore_water_bands,
            },
            daemon=True,
        )
        t.start()

    def build_sync(
        self,
        signature_hash: str,
        img_waves: np.ndarray,
        img_fwhm: np.ndarray | None,
        ignore_water_bands: bool,
    ) -> Path:
        lock = self._get_lock(signature_hash)
        with lock:
            current = self.status(signature_hash)
            if current["status"] == "ready":
                return self.cache_dir(signature_hash)
            return self._build_locked(signature_hash, img_waves, img_fwhm, ignore_water_bands)

    def wait_until_ready(self, signature_hash: str, timeout_sec: int, poll_ms: int = 200) -> bool:
        timeout = max(0, int(timeout_sec))
        poll_interval = max(50, int(poll_ms)) / 1000.0
        deadline = time.monotonic() + timeout
        while time.monotonic() <= deadline:
            status = self.status(signature_hash)
            state = str(status.get("status", "not_found"))
            if state == "ready":
                return True
            if state in {"failed", "not_found"}:
                return False
            time.sleep(poll_interval)
        return str(self.status(signature_hash).get("status", "not_found")) == "ready"

    @staticmethod
    def _estimate_fwhm_from_waves(img_waves: np.ndarray) -> np.ndarray:
        waves = np.asarray(img_waves, dtype=np.float32)
        if waves.size <= 1:
            return np.ones_like(waves, dtype=np.float32)
        diffs = np.diff(waves)
        diffs = np.abs(diffs[np.isfinite(diffs)])
        if diffs.size == 0:
            return np.ones_like(waves, dtype=np.float32)
        median_step = float(np.median(diffs))
        if median_step <= 0:
            median_step = 1.0
        out = np.empty_like(waves, dtype=np.float32)
        out[0] = np.float32(abs(waves[1] - waves[0]))
        out[-1] = np.float32(abs(waves[-1] - waves[-2]))
        if waves.size > 2:
            out[1:-1] = np.abs((waves[2:] - waves[:-2]) * 0.5).astype(np.float32)
        out[~np.isfinite(out) | (out <= 0)] = np.float32(median_step)
        return out

    def _resolve_target_fwhm(self, img_waves: np.ndarray, img_fwhm: np.ndarray | None) -> np.ndarray:
        if img_fwhm is not None and len(img_fwhm) == len(img_waves):
            fwhm = np.asarray(img_fwhm, dtype=np.float32)
            bad = ~np.isfinite(fwhm) | (fwhm <= 0)
            if np.any(bad):
                estimated = self._estimate_fwhm_from_waves(img_waves)
                fwhm = fwhm.copy()
                fwhm[bad] = estimated[bad]
            return fwhm
        return self._estimate_fwhm_from_waves(img_waves)

    @staticmethod
    def _build_gaussian_response_weights(
        lib_waves: np.ndarray,
        img_waves: np.ndarray,
        img_fwhm: np.ndarray,
    ) -> np.ndarray:
        x = np.asarray(lib_waves, dtype=np.float32)
        centers = np.asarray(img_waves, dtype=np.float32)
        fwhm = np.asarray(img_fwhm, dtype=np.float32)

        if x.size == 0:
            return np.empty((centers.shape[0], 0), dtype=np.float32)

        sigma = fwhm / np.float32(2.355)
        bad_band = (~np.isfinite(centers)) | (~np.isfinite(sigma)) | (sigma <= 0)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            delta = (x.reshape(1, -1) - centers.reshape(-1, 1)) / sigma.reshape(-1, 1)
            weights = np.exp(-0.5 * delta * delta).astype(np.float32)

        weights[bad_band, :] = 0.0
        weights[:, ~np.isfinite(x)] = 0.0
        weights[~np.isfinite(weights)] = 0.0
        return weights

    @staticmethod
    def _resample_batch_with_response(spec_batch: np.ndarray, weights: np.ndarray) -> np.ndarray:
        if spec_batch.shape[0] == 0:
            return np.empty((0, weights.shape[0]), dtype=np.float32)
        if spec_batch.shape[1] == 0 or weights.shape[1] == 0:
            return np.full((spec_batch.shape[0], weights.shape[0]), np.nan, dtype=np.float32)
        valid = np.isfinite(spec_batch)
        spec_clean = np.nan_to_num(spec_batch, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        numer = spec_clean @ weights.T
        denom = valid.astype(np.float32) @ weights.T
        out = np.full((spec_batch.shape[0], weights.shape[0]), np.nan, dtype=np.float32)
        np.divide(numer, denom, out=out, where=denom > 1e-12)
        return out

    def load_active_signature(self, signature_hash: str) -> ActiveSignatureData:
        active = self.store.get_active_signature(signature_hash)
        if active is not None:
            return active

        cdir = self.cache_dir(signature_hash)
        meta_path = cdir / "meta.json"
        norm_path = cdir / "spectra_norm.f32"
        valid_mask_path = cdir / "valid_mask.npy"
        meta_ids_path = cdir / "meta_ids.npy"
        if not meta_path.exists() or not norm_path.exists() or not valid_mask_path.exists() or not meta_ids_path.exists():
            raise AppError(
                ErrorCode.CACHE_NOT_FOUND,
                f"signature cache not ready: {signature_hash}",
                status_code=404,
            )

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("status") != "ready":
            raise AppError(
                ErrorCode.CACHE_BUILDING,
                f"signature cache status is {meta.get('status')}",
                status_code=409,
            )
        total = int(meta["total_spectra"])
        bands = int(meta["bands"])

        valid_mask = np.load(valid_mask_path)
        meta_ids = np.load(meta_ids_path).astype(np.int32)
        spectra_norm = np.memmap(norm_path, mode="r", dtype=np.float32, shape=(total, bands))

        faiss_index = None
        faiss_path = cdir / "faiss.index"
        if faiss is not None and faiss_path.exists():
            faiss_index = faiss.read_index(str(faiss_path))

        active_data = ActiveSignatureData(
            signature_hash=signature_hash,
            cache_dir=cdir,
            valid_mask=valid_mask,
            meta_ids=meta_ids,
            spectra_norm=spectra_norm,
            faiss_index=faiss_index,
        )
        self.store.upsert_active_signature(active_data)
        return active_data

    def load_resampled_rows(self, signature_hash: str, row_indices: np.ndarray) -> np.ndarray:
        cdir = self.cache_dir(signature_hash)
        h5_path = cdir / "spectra_resampled.h5"
        if not h5_path.exists():
            return np.empty((0, 0), dtype=np.float32)
        idx = np.asarray(row_indices, dtype=np.int64)
        if idx.size == 0:
            return np.empty((0, 0), dtype=np.float32)
        order = np.argsort(idx)
        sorted_idx = idx[order]
        with h5py.File(h5_path, "r") as h5:
            ds = h5["/spectra/resampled"]
            sorted_rows = ds[sorted_idx].astype(np.float32)
        inverse = np.empty_like(order)
        inverse[order] = np.arange(order.size)
        return sorted_rows[inverse]

    def load_resampled_slice(self, signature_hash: str, start: int, end: int) -> np.ndarray:
        cdir = self.cache_dir(signature_hash)
        h5_path = cdir / "spectra_resampled.h5"
        if not h5_path.exists():
            return np.empty((0, 0), dtype=np.float32)
        a = int(max(0, start))
        b = int(max(a, end))
        if b <= a:
            return np.empty((0, 0), dtype=np.float32)
        with h5py.File(h5_path, "r") as h5:
            ds = h5["/spectra/resampled"]
            if a >= ds.shape[0]:
                return np.empty((0, 0), dtype=np.float32)
            b = min(b, int(ds.shape[0]))
            return ds[a:b].astype(np.float32)

    def _build_locked(
        self,
        signature_hash: str,
        img_waves: np.ndarray,
        img_fwhm: np.ndarray | None,
        ignore_water_bands: bool,
    ) -> Path:
        self.library_service.assert_ready()

        cdir = self.cache_dir(signature_hash)
        tmp_dir = cdir / "_tmp_build"
        lock_path = cdir / "build.lock"

        cdir.mkdir(parents=True, exist_ok=True)
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(datetime.now(tz=timezone.utc).isoformat(), encoding="utf-8")

        img_waves = np.asarray(img_waves, dtype=np.float32)
        valid_mask = build_valid_mask(img_waves, ignore_water_bands=ignore_water_bands)
        target_valid_bands = int(np.count_nonzero(valid_mask))
        if target_valid_bands < settings.min_valid_bands:
            raise AppError(
                ErrorCode.MATCH_FAILED,
                (
                    f"too few valid target bands after masking: "
                    f"{target_valid_bands} < {settings.min_valid_bands}"
                ),
                status_code=422,
            )
        target_fwhm = self._resolve_target_fwhm(img_waves, img_fwhm)
        use_response_resample = not settings.resample_algo_version.lower().startswith("linear")

        total_input = self.library_service.total_spectra()
        bands = img_waves.shape[0]
        min_coverage_ratio = max(0.0, min(float(settings.library_min_coverage_ratio), 1.0))

        resampled_tmp = tmp_dir / "spectra_resampled.f32"
        norm_tmp = tmp_dir / "spectra_norm.f32"
        meta_ids_tmp = tmp_dir / "meta_ids.i32"

        resampled_mm = np.memmap(resampled_tmp, mode="w+", dtype=np.float32, shape=(total_input, bands))
        norm_mm = np.memmap(norm_tmp, mode="w+", dtype=np.float32, shape=(total_input, bands))
        meta_ids_mm = np.memmap(meta_ids_tmp, mode="w+", dtype=np.int32, shape=(total_input,))

        offset = 0
        processed = 0
        fully_convertible = 0
        partially_convertible = 0
        zero_valid_bands = 0
        below_min_valid_bands = 0
        filtered_out = 0
        total_missing_valid_bands = 0
        max_missing_valid_bands = 0
        self.store.set_signature_status(signature_hash, "building", 1, "loading_library")
        try:
            grouped_mode_used = False
            if use_response_resample:
                with h5py.File(self.library_service.h5_path, "r") as h5:
                    if (
                        "/groups/start_idx" in h5
                        and "/groups/count" in h5
                        and "/meta/id" in h5
                        and "/spectra/raw_values" in h5
                        and "/spectra/raw_lengths" in h5
                        and "/spectra/raw_wave_start_idx" in h5
                        and "/waves/all_waves" in h5
                    ):
                        grouped_mode_used = True
                        raw_values_ds = h5["/spectra/raw_values"]
                        raw_lengths = h5["/spectra/raw_lengths"][:].astype(np.int32)
                        raw_wave_start_idx = h5["/spectra/raw_wave_start_idx"][:].astype(np.int64)
                        all_waves = h5["/waves/all_waves"][:].astype(np.float32)
                        all_ids = h5["/meta/id"][:].astype(np.int32)
                        group_starts = h5["/groups/start_idx"][:].astype(np.int64)
                        group_counts = h5["/groups/count"][:].astype(np.int64)
                        order = np.argsort(group_starts)

                        for gi in order.tolist():
                            g_start = int(group_starts[gi])
                            g_count = int(group_counts[gi])
                            if g_count <= 0:
                                continue
                            g_end = min(g_start + g_count, total_input)
                            if g_start >= g_end:
                                continue

                            length = int(raw_lengths[g_start])
                            wave_start = int(raw_wave_start_idx[g_start])
                            lib_waves = all_waves[wave_start : wave_start + length]
                            weights = self._build_gaussian_response_weights(lib_waves, img_waves, target_fwhm)

                            chunk_rows = 512
                            for row_start in range(g_start, g_end, chunk_rows):
                                row_end = min(row_start + chunk_rows, g_end)
                                spec_chunk = raw_values_ds[row_start:row_end, :length].astype(np.float32)
                                rs_batch = self._resample_batch_with_response(spec_chunk, weights)
                                rs_batch = sanitize_reflectance(
                                    rs_batch,
                                    clip_min=settings.clip_reflectance_min,
                                    clip_max=settings.clip_reflectance_max,
                                )

                                finite_valid_counts = np.count_nonzero(np.isfinite(rs_batch[:, valid_mask]), axis=1)
                                finite_valid_counts = finite_valid_counts.astype(np.int32, copy=False)
                                missing_valid = target_valid_bands - finite_valid_counts
                                total_missing_valid_bands += int(np.sum(missing_valid, dtype=np.int64))
                                if missing_valid.size:
                                    max_missing_valid_bands = max(max_missing_valid_bands, int(np.max(missing_valid)))

                                full_mask = finite_valid_counts == target_valid_bands
                                zero_mask = finite_valid_counts == 0
                                fully_convertible += int(np.count_nonzero(full_mask))
                                partially_convertible += int(finite_valid_counts.size - np.count_nonzero(full_mask))
                                zero_valid_bands += int(np.count_nonzero(zero_mask))
                                below_min_mask = finite_valid_counts < settings.min_valid_bands
                                below_min_valid_bands += int(np.count_nonzero(below_min_mask))

                                coverage_ratio = finite_valid_counts.astype(np.float32) / float(target_valid_bands)
                                keep_mask = (~below_min_mask) & (coverage_ratio >= min_coverage_ratio)
                                filtered_out += int(np.count_nonzero(~keep_mask))

                                if np.any(keep_mask):
                                    kept_batch = rs_batch[keep_mask]
                                    out_norm = normalize_rows(kept_batch, valid_mask)
                                    keep_ids = all_ids[row_start:row_end][keep_mask]
                                    keep_bsz = kept_batch.shape[0]
                                    end = offset + keep_bsz
                                    resampled_mm[offset:end] = kept_batch
                                    norm_mm[offset:end] = out_norm
                                    meta_ids_mm[offset:end] = keep_ids.astype(np.int32, copy=False)
                                    offset = end

                                processed += int(row_end - row_start)
                                progress = min(85, int((processed / max(total_input, 1)) * 85))
                                self.store.set_signature_status(signature_hash, "building", progress, "response_group_resampling")

            if not grouped_mode_used:
                for batch in self.library_service.iter_batches(batch_size=2048):
                    bsz = batch.ids.shape[0]
                    keep_specs: list[np.ndarray] = []
                    keep_ids: list[int] = []
                    for i in range(bsz):
                        length = int(batch.raw_lengths[i])
                        wave_start = int(batch.raw_wave_start_idx[i])
                        lib_spec = batch.raw_values[i, :length]
                        lib_waves = batch.all_waves[wave_start : wave_start + length]
                        if use_response_resample:
                            rs = resample_gaussian_fwhm(lib_waves, lib_spec, img_waves, target_fwhm)
                        else:
                            rs = resample_linear(lib_waves, lib_spec, img_waves)
                        rs = sanitize_reflectance(
                            rs,
                            clip_min=settings.clip_reflectance_min,
                            clip_max=settings.clip_reflectance_max,
                        )
                        finite_valid = int(np.count_nonzero(np.isfinite(rs[valid_mask])))
                        missing_valid = target_valid_bands - finite_valid
                        total_missing_valid_bands += max(0, missing_valid)
                        if missing_valid > max_missing_valid_bands:
                            max_missing_valid_bands = missing_valid

                        if finite_valid == target_valid_bands:
                            fully_convertible += 1
                        elif finite_valid == 0:
                            zero_valid_bands += 1
                            partially_convertible += 1
                        else:
                            partially_convertible += 1

                        if finite_valid < settings.min_valid_bands:
                            below_min_valid_bands += 1

                        coverage_ratio = float(finite_valid) / float(target_valid_bands)
                        if finite_valid < settings.min_valid_bands or coverage_ratio < min_coverage_ratio:
                            filtered_out += 1
                            continue

                        keep_specs.append(rs)
                        keep_ids.append(int(batch.ids[i]))

                    if keep_specs:
                        out_batch = np.stack(keep_specs, axis=0).astype(np.float32, copy=False)
                        out_norm = normalize_rows(out_batch, valid_mask)
                        keep_bsz = out_batch.shape[0]
                        end = offset + keep_bsz
                        resampled_mm[offset:end] = out_batch
                        norm_mm[offset:end] = out_norm
                        meta_ids_mm[offset:end] = np.asarray(keep_ids, dtype=np.int32)
                        offset = end

                    processed += bsz
                    progress = min(85, int((processed / max(total_input, 1)) * 85))
                    step = "response_resampling" if use_response_resample else "linear_resampling"
                    self.store.set_signature_status(signature_hash, "building", progress, step)

            total_kept = int(offset)
            if total_kept <= 0:
                raise RuntimeError(
                    (
                        "all spectra filtered out during resampling; "
                        "adjust SPECTRAL_MIN_VALID_BANDS or SPECTRAL_LIBRARY_MIN_COVERAGE_RATIO"
                    )
                )

            resampled_mm.flush()
            norm_mm.flush()
            meta_ids_mm.flush()
            del resampled_mm
            del norm_mm
            del meta_ids_mm

            self.store.set_signature_status(signature_hash, "building", 88, "writing_h5")
            resampled_read = np.memmap(resampled_tmp, mode="r", dtype=np.float32, shape=(total_kept, bands))
            with h5py.File(tmp_dir / "spectra_resampled.h5", "w") as h5:
                h5.create_dataset(
                    "/spectra/resampled",
                    data=resampled_read,
                    compression="gzip",
                    chunks=(min(2048, total_kept), bands),
                )
            del resampled_read

            np.save(tmp_dir / "valid_mask.npy", valid_mask.astype(bool))
            meta_ids_read = np.memmap(meta_ids_tmp, mode="r", dtype=np.int32, shape=(total_kept,))
            np.save(tmp_dir / "meta_ids.npy", meta_ids_read)
            del meta_ids_read

            final_norm_path = cdir / "spectra_norm.f32"
            if final_norm_path.exists():
                final_norm_path.unlink()
            shutil.move(str(norm_tmp), str(final_norm_path))

            self.store.set_signature_status(signature_hash, "building", 92, "building_index")
            faiss_enabled = False
            if faiss is not None:
                index = faiss.IndexFlatIP(bands)
                norm_read = np.memmap(final_norm_path, mode="r", dtype=np.float32, shape=(total_kept, bands))
                chunk = 8192
                for start in range(0, total_kept, chunk):
                    end = min(start + chunk, total_kept)
                    index.add(np.asarray(norm_read[start:end], dtype=np.float32))
                del norm_read
                faiss.write_index(index, str(tmp_dir / "faiss.index"))
                faiss_enabled = True

            for final_file in (
                cdir / "spectra_resampled.h5",
                cdir / "valid_mask.npy",
                cdir / "meta_ids.npy",
                cdir / "faiss.index",
            ):
                final_file.unlink(missing_ok=True)

            shutil.move(str(tmp_dir / "spectra_resampled.h5"), str(cdir / "spectra_resampled.h5"))
            shutil.move(str(tmp_dir / "valid_mask.npy"), str(cdir / "valid_mask.npy"))
            shutil.move(str(tmp_dir / "meta_ids.npy"), str(cdir / "meta_ids.npy"))
            if (tmp_dir / "faiss.index").exists():
                shutil.move(str(tmp_dir / "faiss.index"), str(cdir / "faiss.index"))

            meta = {
                "status": "ready",
                "progress": 100,
                "current_step": "faiss_index_built" if faiss_enabled else "ready",
                "signature_hash": signature_hash,
                "ignore_water_bands": ignore_water_bands,
                "resample_algo_version": settings.resample_algo_version,
                "clean_rules_version": settings.clean_rules_version,
                "created_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
                "total_spectra": total_kept,
                "total_input_spectra": total_input,
                "bands": bands,
                "valid_bands": int(np.count_nonzero(valid_mask)),
                "conversion_stats": {
                    "fully_convertible": fully_convertible,
                    "partially_convertible": partially_convertible,
                    "zero_valid_bands": zero_valid_bands,
                    "below_min_valid_bands": below_min_valid_bands,
                    "filtered_out": filtered_out,
                    "kept_for_matching": total_kept,
                    "target_valid_bands": target_valid_bands,
                    "total_missing_valid_band_values": total_missing_valid_bands,
                    "max_missing_valid_bands_in_one_spectrum": max_missing_valid_bands,
                    "min_coverage_ratio": min_coverage_ratio,
                    "resample_mode": "response_function_gaussian" if use_response_resample else "linear",
                },
                "faiss_enabled": faiss_enabled,
            }
            (cdir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            self.store.set_signature_status(signature_hash, "ready", 100, "ready")
        except Exception as exc:
            fail_meta = {
                "status": "failed",
                "progress": 0,
                "current_step": "failed",
                "signature_hash": signature_hash,
                "error": str(exc),
                "updated_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
            }
            (cdir / "meta.json").write_text(
                json.dumps(fail_meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self.store.set_signature_status(signature_hash, "failed", 0, "failed", error=str(exc))
            raise AppError(
                ErrorCode.CACHE_FAILED,
                f"failed to build cache for {signature_hash}: {exc}",
                status_code=500,
            ) from exc
        finally:
            lock_path.unlink(missing_ok=True)
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
            try:
                resampled_tmp.unlink(missing_ok=True)
            except OSError:
                pass
            try:
                meta_ids_tmp.unlink(missing_ok=True)
            except OSError:
                pass

        return cdir

    def _get_lock(self, signature_hash: str) -> Lock:
        with self._locks_lock:
            if signature_hash not in self._build_locks:
                self._build_locks[signature_hash] = Lock()
            return self._build_locks[signature_hash]
