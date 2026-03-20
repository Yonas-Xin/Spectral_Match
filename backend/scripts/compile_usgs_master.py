from __future__ import annotations

import argparse
import hashlib
import sqlite3
from pathlib import Path

import h5py
import numpy as np

try:
    from spectral.io import envi
except ImportError as exc:  # pragma: no cover
    raise SystemExit("spectral package is required: pip install spectral") from exc


def parse_header_array(value) -> np.ndarray:
    if value is None:
        return np.array([], dtype=np.float32)
    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value, dtype=np.float32)
    text = str(value)
    nums = []
    token = ""
    for ch in text:
        if ch in "0123456789+-.eE":
            token += ch
        elif token:
            nums.append(float(token))
            token = ""
    if token:
        nums.append(float(token))
    return np.asarray(nums, dtype=np.float32)


def derive_class(name: str) -> str:
    first = name.split()[0] if name else "unknown"
    return first.strip("_-")


def hash_spectrum(waves: np.ndarray, spec: np.ndarray) -> str:
    payload = np.concatenate([waves.astype(np.float32), spec.astype(np.float32)]).tobytes()
    return hashlib.sha1(payload).hexdigest()


def compile_library(hdr_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    h5_path = output_dir / "usgs_master.h5"
    sqlite_path = output_dir / "usgs_meta.db"

    lib = envi.open(str(hdr_path))
    spectra = np.asarray(lib.spectra, dtype=np.float32)
    if spectra.ndim != 2:
        raise RuntimeError(f"Unexpected spectra shape: {spectra.shape}")
    total, length = spectra.shape

    names = list(getattr(lib, "names", []))
    if len(names) != total:
        names = [f"Spectrum {i+1}" for i in range(total)]

    metadata = getattr(lib, "metadata", {}) or {}
    waves = parse_header_array(metadata.get("wavelength"))
    if waves.size != length:
        waves = np.arange(length, dtype=np.float32)

    ids = np.arange(1, total + 1, dtype=np.int32)
    raw_lengths = np.full(total, length, dtype=np.int32)
    raw_wave_start_idx = np.arange(0, total * length, length, dtype=np.int64)
    all_waves = np.tile(waves, total).astype(np.float32)

    classes = [derive_class(n) for n in names]
    sources = [hdr_path.name] * total

    unique_names = sorted(set(names))
    unique_classes = sorted(set(classes))
    unique_sources = sorted(set(sources))
    name_to_idx = {v: i for i, v in enumerate(unique_names)}
    class_to_idx = {v: i for i, v in enumerate(unique_classes)}
    source_to_idx = {v: i for i, v in enumerate(unique_sources)}

    name_ref = np.asarray([name_to_idx[n] for n in names], dtype=np.int32)
    class_ref = np.asarray([class_to_idx[c] for c in classes], dtype=np.int32)
    source_ref = np.asarray([source_to_idx[s] for s in sources], dtype=np.int32)

    with h5py.File(h5_path, "w") as h5:
        h5.create_dataset("/spectra/raw_values", data=spectra, compression="gzip", chunks=True)
        h5.create_dataset("/spectra/raw_lengths", data=raw_lengths)
        h5.create_dataset("/spectra/raw_wave_start_idx", data=raw_wave_start_idx)
        h5.create_dataset("/waves/all_waves", data=all_waves, compression="gzip", chunks=True)
        h5.create_dataset("/meta/id", data=ids)
        h5.create_dataset("/meta/name_ref", data=name_ref)
        h5.create_dataset("/meta/class_ref", data=class_ref)
        h5.create_dataset("/meta/source_ref", data=source_ref)
        h5.create_dataset("/dict/names", data=np.asarray(unique_names, dtype=h5py.string_dtype("utf-8")))
        h5.create_dataset("/dict/classes", data=np.asarray(unique_classes, dtype=h5py.string_dtype("utf-8")))
        h5.create_dataset("/dict/sources", data=np.asarray(unique_sources, dtype=h5py.string_dtype("utf-8")))

    rows = []
    for i in range(total):
        spec = spectra[i]
        sid = int(ids[i])
        rows.append(
            (
                sid,
                names[i],
                classes[i],
                sources[i],
                float(np.min(waves)),
                float(np.max(waves)),
                int(length),
                hash_spectrum(waves, spec),
            )
        )
    write_metadata_index(sqlite_path, rows)

    print(f"Compiled {total} spectra -> {h5_path}")
    print(f"SQLite index -> {sqlite_path}")


def write_metadata_index(
    sqlite_path: Path,
    rows: list[tuple[int, str, str | None, str | None, float, float, int, str]],
) -> None:
    conn = sqlite3.connect(sqlite_path)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile ENVI spectral library into HDF5 master store.")
    parser.add_argument("--library-hdr", required=True, help="Path to ENVI spectral library .hdr file")
    parser.add_argument("--output-dir", default="", help="Output library directory (default backend/data/library)")
    args = parser.parse_args()

    hdr_path = Path(args.library_hdr).expanduser().resolve()
    if not hdr_path.exists():
        raise SystemExit(f"File not found: {hdr_path}")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else Path(__file__).resolve().parents[1] / "data" / "library"
    )

    compile_library(hdr_path=hdr_path, output_dir=output_dir)


if __name__ == "__main__":
    main()
