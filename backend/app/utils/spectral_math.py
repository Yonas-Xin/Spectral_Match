from __future__ import annotations

import math

import numpy as np

WATER_BAND_RANGES = (
    (1340.0, 1455.0),
    (1790.0, 1960.0),
    (2480.0, 2500.0),
)


def merge_mask_ranges(
    ranges: list[tuple[float, float]] | tuple[tuple[float, float], ...],
) -> list[tuple[float, float]]:
    if not ranges:
        return []
    cleaned: list[tuple[float, float]] = []
    for start, end in ranges:
        a = float(start)
        b = float(end)
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        lo = min(a, b)
        hi = max(a, b)
        if hi - lo <= 0:
            continue
        cleaned.append((lo, hi))
    if not cleaned:
        return []
    cleaned.sort(key=lambda x: x[0])
    merged: list[tuple[float, float]] = [cleaned[0]]
    for lo, hi in cleaned[1:]:
        prev_lo, prev_hi = merged[-1]
        if lo <= prev_hi + 1e-6:
            merged[-1] = (prev_lo, max(prev_hi, hi))
        else:
            merged.append((lo, hi))
    return merged


def build_custom_range_mask(
    waves_nm: np.ndarray,
    masked_ranges: list[tuple[float, float]] | tuple[tuple[float, float], ...],
) -> np.ndarray:
    waves = np.asarray(waves_nm, dtype=np.float32)
    if waves.size == 0:
        return np.zeros((0,), dtype=bool)
    merged = merge_mask_ranges(masked_ranges)
    if not merged:
        return np.ones(waves.shape[0], dtype=bool)
    keep = np.ones(waves.shape[0], dtype=bool)
    for low, high in merged:
        keep &= ~((waves >= low) & (waves <= high))
    return keep


def build_water_mask(waves_nm: np.ndarray) -> np.ndarray:
    waves = np.asarray(waves_nm, dtype=np.float32)
    mask = np.ones(waves.shape[0], dtype=bool)
    for low, high in WATER_BAND_RANGES:
        mask &= ~((waves >= low) & (waves <= high))
    return mask


def build_valid_mask(waves_nm: np.ndarray, ignore_water_bands: bool) -> np.ndarray:
    waves = np.asarray(waves_nm, dtype=np.float32)
    valid = np.isfinite(waves) & (waves > 0)
    if ignore_water_bands:
        valid &= build_water_mask(waves)
    return valid


def resample_linear(
    lib_waves: np.ndarray,
    lib_spec: np.ndarray,
    img_waves: np.ndarray,
) -> np.ndarray:
    x = np.asarray(lib_waves, dtype=np.float32)
    y = np.asarray(lib_spec, dtype=np.float32)
    target = np.asarray(img_waves, dtype=np.float32)

    good = np.isfinite(x) & np.isfinite(y)
    x = x[good]
    y = y[good]
    if x.size < 2:
        return np.full(target.shape, np.nan, dtype=np.float32)

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    uniq_x, uniq_idx = np.unique(x, return_index=True)
    uniq_y = y[uniq_idx]
    if uniq_x.size < 2:
        return np.full(target.shape, np.nan, dtype=np.float32)

    out = np.interp(target, uniq_x, uniq_y, left=np.nan, right=np.nan).astype(np.float32)
    out[(target < uniq_x[0]) | (target > uniq_x[-1])] = np.nan
    return out


def resample_gaussian_fwhm(
    lib_waves: np.ndarray,
    lib_spec: np.ndarray,
    img_waves: np.ndarray,
    img_fwhm: np.ndarray,
) -> np.ndarray:
    waves = np.asarray(lib_waves, dtype=np.float32)
    spec = np.asarray(lib_spec, dtype=np.float32)
    centers = np.asarray(img_waves, dtype=np.float32)
    fwhm = np.asarray(img_fwhm, dtype=np.float32)

    good = np.isfinite(waves) & np.isfinite(spec)
    waves = waves[good]
    spec = spec[good]
    if waves.size < 2:
        return np.full(centers.shape, np.nan, dtype=np.float32)

    order = np.argsort(waves)
    waves = waves[order]
    spec = spec[order]

    out = np.empty_like(centers, dtype=np.float32)
    for i, (center, width) in enumerate(zip(centers, fwhm, strict=True)):
        if not np.isfinite(center) or not np.isfinite(width) or width <= 0:
            out[i] = np.nan
            continue
        sigma = width / 2.355
        if sigma <= 0:
            out[i] = np.nan
            continue
        weights = np.exp(-0.5 * ((waves - center) / sigma) ** 2, dtype=np.float32)
        total_w = float(np.sum(weights))
        if total_w <= 1e-10:
            out[i] = np.nan
            continue
        out[i] = float(np.sum(weights * spec) / total_w)
    return out


def sanitize_reflectance(
    spec: np.ndarray,
    clip_min: float = 0.0,
    clip_max: float | None = 1.0,
) -> np.ndarray:
    arr = np.asarray(spec, dtype=np.float32).copy()
    arr[~np.isfinite(arr)] = np.nan
    if clip_max is None:
        arr = np.maximum(arr, np.float32(clip_min), out=arr)
    else:
        arr = np.clip(arr, clip_min, clip_max, out=arr)
    return arr


def scale_to_unit_reflectance(
    spec: np.ndarray,
    scale_factor: float | None = None,
) -> np.ndarray:
    arr = np.asarray(spec, dtype=np.float32).copy()
    if scale_factor is not None and np.isfinite(scale_factor) and scale_factor > 1.0:
        return arr / np.float32(scale_factor)

    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return arr

    # Fallback for products stored as int reflectance*10000 without metadata scale factor.
    p99 = float(np.percentile(np.abs(finite), 99))
    if 2.0 < p99 <= 30000.0:
        return arr / np.float32(10000.0)
    return arr


def normalize_vector(vec: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    out = np.zeros(vec.shape, dtype=np.float32)
    valid_cols = np.asarray(valid_mask, dtype=bool)
    segment = np.asarray(vec[valid_cols], dtype=np.float32)
    segment = np.nan_to_num(segment, nan=0.0, posinf=0.0, neginf=0.0)
    norm = float(np.linalg.norm(segment))
    if norm > 1e-8:
        segment = segment / norm
    out[valid_cols] = segment
    return out


def normalize_rows(matrix: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    mat = np.asarray(matrix, dtype=np.float32)
    out = np.zeros_like(mat, dtype=np.float32)
    valid_cols = np.asarray(valid_mask, dtype=bool)
    seg = np.nan_to_num(mat[:, valid_cols], nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(seg, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    out[:, valid_cols] = seg / norms
    return out


def sam_angles(candidates_norm: np.ndarray, query_norm: np.ndarray) -> np.ndarray:
    score = np.asarray(candidates_norm, dtype=np.float32) @ np.asarray(query_norm, dtype=np.float32)
    score = np.clip(score, -1.0, 1.0)
    return np.arccos(score, dtype=np.float32)


def pearson_r(x: np.ndarray, y: np.ndarray, valid_mask: np.ndarray) -> float:
    mask = np.asarray(valid_mask, dtype=bool)
    xv = np.asarray(x, dtype=np.float32)[mask]
    yv = np.asarray(y, dtype=np.float32)[mask]
    good = np.isfinite(xv) & np.isfinite(yv)
    xv = xv[good]
    yv = yv[good]
    if xv.size < 2:
        return float("nan")
    x_mean = float(np.mean(xv))
    y_mean = float(np.mean(yv))
    num = float(np.sum((xv - x_mean) * (yv - y_mean)))
    den_x = math.sqrt(float(np.sum((xv - x_mean) ** 2)))
    den_y = math.sqrt(float(np.sum((yv - y_mean) ** 2)))
    den = den_x * den_y
    if den <= 1e-12:
        return float("nan")
    return num / den


def nearest_band_index(waves_nm: np.ndarray, target_nm: float) -> int:
    waves = np.asarray(waves_nm, dtype=np.float32)
    if waves.size == 0:
        return 0
    return int(np.nanargmin(np.abs(waves - np.float32(target_nm))))
