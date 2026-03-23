from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import h5py
import numpy as np

from app.core.config import settings
from app.core.errors import AppError, ErrorCode
from app.utils.library_meta import normalize_display_name, parse_variant_mode
from app.utils.spectral_math import resample_gaussian_fwhm, resample_linear, sanitize_reflectance


@dataclass
class LibraryBatch:
    ids: np.ndarray
    raw_values: np.ndarray
    raw_lengths: np.ndarray
    raw_wave_start_idx: np.ndarray
    all_waves: np.ndarray


@dataclass
class SpectrumMeta:
    spectrum_id: int
    name: str
    class_name: str | None
    source: str | None
    raw_name: str | None = None
    instrument_variant: str | None = None
    measure_mode: str | None = None
    chapter: str | None = None


@dataclass
class RawSpectrum:
    spectrum_id: int
    wavelengths: np.ndarray
    values: np.ndarray


class LibraryService:
    def __init__(self) -> None:
        self.h5_path = settings.library_h5_path
        self.sqlite_path = settings.library_sqlite_path
        self._id_row_index: dict[int, int] | None = None

    def is_ready(self) -> bool:
        return self.h5_path.exists()

    def assert_ready(self) -> None:
        if not self.is_ready():
            raise AppError(
                ErrorCode.LIBRARY_NOT_READY,
                f"library not found: {self.h5_path}. Please compile the master library first.",
                status_code=500,
            )

    def total_spectra(self) -> int:
        self.assert_ready()
        with h5py.File(self.h5_path, "r") as h5:
            return int(h5["/spectra/raw_values"].shape[0])

    def iter_batches(self, batch_size: int = 4096) -> Generator[LibraryBatch, None, None]:
        self.assert_ready()
        with h5py.File(self.h5_path, "r") as h5:
            raw_values = h5["/spectra/raw_values"]
            raw_lengths = h5["/spectra/raw_lengths"]
            raw_start_idx = h5["/spectra/raw_wave_start_idx"]
            ids = h5["/meta/id"]
            all_waves = h5["/waves/all_waves"][:].astype(np.float32)

            total = int(raw_values.shape[0])
            for start in range(0, total, batch_size):
                end = min(start + batch_size, total)
                yield LibraryBatch(
                    ids=ids[start:end].astype(np.int32),
                    raw_values=raw_values[start:end].astype(np.float32),
                    raw_lengths=raw_lengths[start:end].astype(np.int32),
                    raw_wave_start_idx=raw_start_idx[start:end].astype(np.int64),
                    all_waves=all_waves,
                )

    def fetch_metadata(self, ids: list[int]) -> dict[int, SpectrumMeta]:
        if not ids:
            return {}
        if self.sqlite_path.exists():
            return self._fetch_metadata_from_sqlite(ids)
        if self.h5_path.exists():
            return self._fetch_metadata_from_h5(ids)
        return {}

    def search_metadata(self, query: str, limit: int = 20) -> tuple[int, list[SpectrumMeta]]:
        if self.sqlite_path.exists():
            return self._search_metadata_from_sqlite(query=query, limit=limit)
        if self.h5_path.exists():
            return self._search_metadata_from_h5(query=query, limit=limit)
        raise AppError(
            ErrorCode.LIBRARY_NOT_READY,
            f"library metadata not found: {self.sqlite_path}",
            status_code=500,
        )

    def load_raw_spectra(self, ids: list[int]) -> list[RawSpectrum]:
        wanted = [int(sid) for sid in ids if int(sid) > 0]
        if not wanted:
            return []

        id_row_index = self._load_id_row_index()
        ordered_rows: list[tuple[int, int]] = []
        for sid in wanted:
            row_idx = id_row_index.get(sid)
            if row_idx is not None:
                ordered_rows.append((sid, row_idx))

        if not ordered_rows:
            return []

        out: list[RawSpectrum] = []
        with h5py.File(self.h5_path, "r") as h5:
            raw_values = h5["/spectra/raw_values"]
            raw_lengths = h5["/spectra/raw_lengths"][:].astype(np.int32)
            raw_wave_start_idx = h5["/spectra/raw_wave_start_idx"][:].astype(np.int64)
            all_waves = h5["/waves/all_waves"][:].astype(np.float32)

            for sid, row_idx in ordered_rows:
                length = int(raw_lengths[row_idx])
                wave_start = int(raw_wave_start_idx[row_idx])
                values = raw_values[row_idx, :length].astype(np.float32)
                wavelengths = all_waves[wave_start : wave_start + length].astype(np.float32, copy=False)
                out.append(
                    RawSpectrum(
                        spectrum_id=sid,
                        wavelengths=wavelengths.copy(),
                        values=values,
                    )
                )

        return out

    def load_resampled_spectra(
        self,
        ids: list[int],
        target_waves: np.ndarray,
        target_fwhm: np.ndarray | None = None,
    ) -> dict[int, np.ndarray]:
        target = np.asarray(target_waves, dtype=np.float32)
        if target.size == 0:
            return {}

        use_gaussian = target_fwhm is not None
        fwhm = np.asarray(target_fwhm, dtype=np.float32) if target_fwhm is not None else None

        out: dict[int, np.ndarray] = {}
        for raw in self.load_raw_spectra(ids):
            if use_gaussian and fwhm is not None:
                resampled = resample_gaussian_fwhm(raw.wavelengths, raw.values, target, fwhm)
            else:
                resampled = resample_linear(raw.wavelengths, raw.values, target)
            out[raw.spectrum_id] = sanitize_reflectance(
                resampled,
                clip_min=settings.clip_reflectance_min,
                clip_max=settings.clip_reflectance_max,
            )
        return out

    def _fetch_metadata_from_sqlite(self, ids: list[int]) -> dict[int, SpectrumMeta]:
        with sqlite3.connect(self.sqlite_path) as conn:
            columns = {
                str(row[1]).lower()
                for row in conn.execute("PRAGMA table_info(spectra_meta)").fetchall()
            }

        select_cols = ["id", "name", "class", "source"]
        has_raw_name = "raw_name" in columns
        has_variant = "instrument_variant" in columns
        has_mode = "measure_mode" in columns
        has_chapter = "chapter" in columns
        if has_raw_name:
            select_cols.append("raw_name")
        if has_variant:
            select_cols.append("instrument_variant")
        if has_mode:
            select_cols.append("measure_mode")
        if has_chapter:
            select_cols.append("chapter")

        placeholders = ",".join("?" for _ in ids)
        sql = (
            f"SELECT {', '.join(select_cols)} "
            "FROM spectra_meta "
            f"WHERE id IN ({placeholders})"
        )
        out: dict[int, SpectrumMeta] = {}
        conn = sqlite3.connect(self.sqlite_path)
        try:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, ids).fetchall()
            for row in rows:
                sid = int(row["id"])
                out[sid] = SpectrumMeta(
                    spectrum_id=sid,
                    name=str(row["name"]),
                    class_name=str(row["class"]) if row["class"] is not None else None,
                    source=str(row["source"]) if row["source"] is not None else None,
                    raw_name=(str(row["raw_name"]) if has_raw_name and row["raw_name"] is not None else None),
                    instrument_variant=(
                        str(row["instrument_variant"]) if has_variant and row["instrument_variant"] is not None else None
                    ),
                    measure_mode=(
                        str(row["measure_mode"]) if has_mode and row["measure_mode"] is not None else None
                    ),
                    chapter=(
                        str(row["chapter"]) if has_chapter and row["chapter"] is not None else None
                    ),
                )
        finally:
            conn.close()
        return out

    def _fetch_metadata_from_h5(self, ids: list[int]) -> dict[int, SpectrumMeta]:
        wanted = set(ids)
        out: dict[int, SpectrumMeta] = {}
        with h5py.File(self.h5_path, "r") as h5:
            all_ids = h5["/meta/id"][:].astype(np.int32)
            if "/meta/name_ref" not in h5 or "/dict/names" not in h5:
                for sid in ids:
                    out[sid] = SpectrumMeta(
                        spectrum_id=sid,
                        name=f"Spectrum {sid}",
                        class_name=None,
                        source=None,
                        raw_name=None,
                        instrument_variant=None,
                        measure_mode=None,
                        chapter=None,
                    )
                return out

            name_ref = h5["/meta/name_ref"][:].astype(np.int32)
            class_ref = h5["/meta/class_ref"][:].astype(np.int32) if "/meta/class_ref" in h5 else None
            source_ref = h5["/meta/source_ref"][:].astype(np.int32) if "/meta/source_ref" in h5 else None

            names = h5["/dict/names"][:]
            classes = h5["/dict/classes"][:] if "/dict/classes" in h5 else []
            sources = h5["/dict/sources"][:] if "/dict/sources" in h5 else []

            id_to_idx = {int(v): i for i, v in enumerate(all_ids.tolist()) if int(v) in wanted}
            for sid in ids:
                idx = id_to_idx.get(int(sid))
                if idx is None:
                    out[int(sid)] = SpectrumMeta(
                        spectrum_id=int(sid),
                        name=f"Spectrum {sid}",
                        class_name=None,
                        source=None,
                        raw_name=None,
                        instrument_variant=None,
                        measure_mode=None,
                        chapter=None,
                    )
                    continue
                nref = int(name_ref[idx])
                cref = int(class_ref[idx]) if class_ref is not None else -1
                sref = int(source_ref[idx]) if source_ref is not None else -1
                name = self._decode_vlen(names[nref]) if 0 <= nref < len(names) else f"Spectrum {sid}"
                class_name = self._decode_vlen(classes[cref]) if 0 <= cref < len(classes) else None
                source = self._decode_vlen(sources[sref]) if 0 <= sref < len(sources) else None
                out[int(sid)] = SpectrumMeta(
                    spectrum_id=int(sid),
                    name=name,
                    class_name=class_name,
                    source=source,
                    raw_name=None,
                    instrument_variant=None,
                    measure_mode=None,
                    chapter=None,
                )
        return out

    def _search_metadata_from_sqlite(self, query: str, limit: int) -> tuple[int, list[SpectrumMeta]]:
        q = str(query or "").strip().lower()
        capped_limit = max(1, min(int(limit), 50))
        params: list[object] = []
        count_params: list[object] = []
        where_sql = ""

        if q:
            like = f"%{q}%"
            prefix = f"{q}%"
            where_sql = (
                "WHERE lower(name) LIKE ? "
                "OR lower(coalesce(raw_name, '')) LIKE ? "
                "OR lower(coalesce(class, '')) LIKE ? "
                "OR lower(coalesce(source, '')) LIKE ? "
                "OR lower(coalesce(chapter, '')) LIKE ? "
                "OR lower(coalesce(instrument_variant, '')) LIKE ? "
                "OR lower(coalesce(measure_mode, '')) LIKE ?"
            )
            count_params = [like, like, like, like, like, like, like]
            params.extend(count_params)
            params.extend([q, q, prefix, prefix, capped_limit])
        else:
            params.append(capped_limit)

        sql = (
            "SELECT id, name, raw_name, class, source, instrument_variant, measure_mode, chapter "
            "FROM spectra_meta "
            f"{where_sql} "
        )
        if q:
            sql += (
                "ORDER BY CASE "
                "WHEN lower(name) = ? THEN 0 "
                "WHEN lower(coalesce(raw_name, '')) = ? THEN 1 "
                "WHEN lower(name) LIKE ? THEN 2 "
                "WHEN lower(coalesce(class, '')) LIKE ? THEN 3 "
                "ELSE 4 END, "
                "name ASC "
                "LIMIT ?"
            )
        else:
            sql += "ORDER BY name ASC LIMIT ?"

        count_sql = f"SELECT COUNT(*) FROM spectra_meta {where_sql}"
        out: list[SpectrumMeta] = []
        conn = sqlite3.connect(self.sqlite_path)
        try:
            conn.row_factory = sqlite3.Row
            total = int(conn.execute(count_sql, count_params).fetchone()[0])
            rows = conn.execute(sql, params).fetchall()
            for row in rows:
                out.append(
                    SpectrumMeta(
                        spectrum_id=int(row["id"]),
                        name=str(row["name"]),
                        class_name=str(row["class"]) if row["class"] is not None else None,
                        source=str(row["source"]) if row["source"] is not None else None,
                        raw_name=str(row["raw_name"]) if row["raw_name"] is not None else None,
                        instrument_variant=(
                            str(row["instrument_variant"]) if row["instrument_variant"] is not None else None
                        ),
                        measure_mode=str(row["measure_mode"]) if row["measure_mode"] is not None else None,
                        chapter=str(row["chapter"]) if row["chapter"] is not None else None,
                    )
                )
        finally:
            conn.close()
        return total, out

    def _search_metadata_from_h5(self, query: str, limit: int) -> tuple[int, list[SpectrumMeta]]:
        q = str(query or "").strip().lower()
        matched: list[SpectrumMeta] = []
        meta_map = self._fetch_metadata_from_h5(
            self._list_all_ids()[:]
        )
        for sid in sorted(meta_map):
            meta = meta_map[sid]
            haystack = " ".join(
                part.lower()
                for part in (
                    meta.name,
                    meta.class_name or "",
                    meta.source or "",
                    meta.chapter or "",
                )
                if part
            )
            if q and q not in haystack:
                continue
            matched.append(meta)
        return len(matched), matched[: max(1, min(int(limit), 50))]

    def to_display_summary(self, meta: SpectrumMeta) -> SpectrumMeta:
        raw_name = meta.raw_name or meta.name
        sensor_type = meta.instrument_variant
        measure_mode = meta.measure_mode
        if sensor_type is None or measure_mode is None:
            parsed_sensor, parsed_mode = parse_variant_mode(raw_name)
            sensor_type = sensor_type or parsed_sensor
            measure_mode = measure_mode or parsed_mode
        return SpectrumMeta(
            spectrum_id=meta.spectrum_id,
            name=normalize_display_name(meta.name),
            class_name=meta.class_name,
            source=meta.source,
            raw_name=raw_name,
            instrument_variant=sensor_type,
            measure_mode=measure_mode,
            chapter=meta.chapter,
        )

    def _load_id_row_index(self) -> dict[int, int]:
        if self._id_row_index is not None:
            return self._id_row_index
        with h5py.File(self.h5_path, "r") as h5:
            ids = h5["/meta/id"][:].astype(np.int32)
        self._id_row_index = {int(sid): idx for idx, sid in enumerate(ids.tolist())}
        return self._id_row_index

    def _list_all_ids(self) -> list[int]:
        with h5py.File(self.h5_path, "r") as h5:
            return h5["/meta/id"][:].astype(np.int32).tolist()

    @staticmethod
    def _decode_vlen(value: bytes | str) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="ignore")
        return str(value)

    def write_metadata_index(
        self,
        rows: list[tuple[int, str, str | None, str | None, float, float, int, str]],
    ) -> Path:
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.sqlite_path)
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS spectra_meta (
                  id INTEGER PRIMARY KEY,
                  name TEXT NOT NULL,
                  class TEXT,
                  source TEXT,
                  wavelength_min REAL,
                  wavelength_max REAL,
                  points_count INTEGER,
                  hash TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_meta_name ON spectra_meta(name);
                CREATE INDEX IF NOT EXISTS idx_meta_class ON spectra_meta(class);
                CREATE INDEX IF NOT EXISTS idx_meta_hash ON spectra_meta(hash);
                """
            )
            conn.executemany(
                """
                INSERT OR REPLACE INTO spectra_meta (
                  id, name, class, source, wavelength_min, wavelength_max, points_count, hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()
        finally:
            conn.close()
        return self.sqlite_path
