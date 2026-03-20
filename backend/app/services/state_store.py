from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np


@dataclass
class ImageContext:
    image_id: str
    image_path: Path
    lines: int
    samples: int
    bands: int
    interleave: str
    dtype: str
    wavelengths: np.ndarray
    wavelength_unit: str
    fwhm: np.ndarray | None
    reflectance_scale_factor: float | None
    data_ignore_value: float | None
    preview_path: Path
    rgb_bands: tuple[int, int, int]
    data_backend: str
    cube: np.ndarray | None


@dataclass
class SignatureStatus:
    signature_hash: str
    status: str
    progress: int
    current_step: str
    error: str | None = None
    updated_at: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
    )


@dataclass
class ActiveSignatureData:
    signature_hash: str
    cache_dir: Path
    valid_mask: np.ndarray
    meta_ids: np.ndarray
    spectra_norm: np.ndarray
    faiss_index: Any | None = None


class RuntimeStore:
    def __init__(self, max_active_signatures: int) -> None:
        self._images: dict[str, ImageContext] = {}
        self._signature_status: dict[str, SignatureStatus] = {}
        self._active_signatures: OrderedDict[str, ActiveSignatureData] = OrderedDict()
        self._image_mask_ranges: dict[str, list[tuple[float, float]]] = {}
        self._max_active_signatures = max_active_signatures
        self._lock = RLock()

    def upsert_image(self, image: ImageContext) -> None:
        with self._lock:
            self._images[image.image_id] = image
            self._image_mask_ranges[image.image_id] = []

    def get_image(self, image_id: str) -> ImageContext | None:
        with self._lock:
            return self._images.get(image_id)

    def set_signature_status(
        self,
        signature_hash: str,
        status: str,
        progress: int = 0,
        current_step: str = "",
        error: str | None = None,
    ) -> None:
        with self._lock:
            self._signature_status[signature_hash] = SignatureStatus(
                signature_hash=signature_hash,
                status=status,
                progress=progress,
                current_step=current_step,
                error=error,
            )

    def get_signature_status(self, signature_hash: str) -> SignatureStatus | None:
        with self._lock:
            return self._signature_status.get(signature_hash)

    def upsert_active_signature(self, data: ActiveSignatureData) -> None:
        with self._lock:
            self._active_signatures[data.signature_hash] = data
            self._active_signatures.move_to_end(data.signature_hash)
            while len(self._active_signatures) > self._max_active_signatures:
                self._active_signatures.popitem(last=False)

    def get_active_signature(self, signature_hash: str) -> ActiveSignatureData | None:
        with self._lock:
            data = self._active_signatures.get(signature_hash)
            if data is not None:
                self._active_signatures.move_to_end(signature_hash)
            return data

    def set_image_mask_ranges(self, image_id: str, ranges: list[tuple[float, float]]) -> None:
        with self._lock:
            self._image_mask_ranges[image_id] = [(float(a), float(b)) for a, b in ranges]

    def get_image_mask_ranges(self, image_id: str) -> list[tuple[float, float]]:
        with self._lock:
            ranges = self._image_mask_ranges.get(image_id, [])
            return [(float(a), float(b)) for a, b in ranges]
