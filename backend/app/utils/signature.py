from __future__ import annotations

import hashlib

import numpy as np


def build_signature_hash(
    image_waves: np.ndarray,
    image_fwhm: np.ndarray | None,
    ignore_water_bands: bool,
    resample_algo_version: str,
    clean_rules_version: str,
) -> str:
    payload = {
        "waves": np.asarray(image_waves, dtype=np.float32).round(4).tolist(),
        "fwhm": (
            np.asarray(image_fwhm, dtype=np.float32).round(4).tolist()
            if image_fwhm is not None and len(image_fwhm) == len(image_waves)
            else None
        ),
        "ignore_water_bands": ignore_water_bands,
        "resample_algo_version": resample_algo_version,
        "clean_rules_version": clean_rules_version,
    }
    encoded = repr(payload).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()
