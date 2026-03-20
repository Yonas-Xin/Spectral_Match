from __future__ import annotations

import argparse
import hashlib
import sqlite3
from pathlib import Path

import h5py
import numpy as np

DELETED_CHANNEL_THRESHOLD = -1.0e33

WAVE_FILES = {
    "ASD": "splib07a_Wavelengths_ASD_0.35-2.5_microns_2151_ch.txt",
    "AVIRIS": "splib07a_Wavelengths_AVIRIS_1996_0.37-2.5_microns.txt",
    "BECK": "splib07a_Wavelengths_BECK_Beckman_0.2-3.0_microns.txt",
    "NIC4": "splib07a_Wavelengths_NIC4_Nicolet_1.12-216microns.txt",
}

INSTRUMENT_ORDER = ("ASD", "AVIRIS", "BECK", "NIC4")


def parse_numeric_text(path: Path) -> np.ndarray:
    try:
        arr = np.loadtxt(path, dtype=np.float32, skiprows=1)
    except Exception:
        values: list[float] = []
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            token = line.strip()
            if not token:
                continue
            try:
                values.append(float(token))
            except ValueError:
                continue
        arr = np.asarray(values, dtype=np.float32)

    arr = np.atleast_1d(arr).astype(np.float32, copy=False)
    return arr


def normalize_wave_unit(waves: np.ndarray) -> np.ndarray:
    # splib07a wavelength files are in microns. Convert to nm for consistent matching with image wavelengths.
    if waves.size and float(np.nanmax(waves)) < 400.0:
        return waves * 1000.0
    return waves


def detect_instrument(file_name: str) -> str | None:
    upper_name = file_name.upper()
    if "NIC4" in upper_name:
        return "NIC4"
    if "AVIRIS" in upper_name:
        return "AVIRIS"
    if "BECK" in upper_name:
        return "BECK"
    if "ASDFR" in upper_name or "ASDHR" in upper_name or "ASDNG" in upper_name or "_ASD" in upper_name:
        return "ASD"
    return None


def chapter_to_class(chapter: str) -> str:
    if "_" not in chapter:
        return chapter
    return chapter.split("_", 1)[1]


def hash_spectrum(waves: np.ndarray, spec: np.ndarray) -> str:
    wave_part = np.asarray(waves, dtype=np.float32)
    spec_part = np.nan_to_num(np.asarray(spec, dtype=np.float32), nan=-9999.0, posinf=-9999.0, neginf=-9999.0)
    payload = np.concatenate([wave_part, spec_part]).tobytes()
    return hashlib.sha1(payload).hexdigest()


def collect_spectrum_files(source_dir: Path) -> list[tuple[Path, str]]:
    files: list[tuple[Path, str]] = []
    for chapter_dir in sorted(source_dir.glob("Chapter*")):
        if not chapter_dir.is_dir():
            continue
        for txt in sorted(chapter_dir.glob("splib07a_*.txt")):
            files.append((txt, chapter_dir.name))
    return files


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


def compile_splib07a_ascii(source_dir: Path, output_dir: Path, limit: int = 0) -> None:
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    h5_path = output_dir / "usgs_master.h5"
    sqlite_path = output_dir / "usgs_meta.db"

    for key, file_name in WAVE_FILES.items():
        if not (source_dir / file_name).exists():
            raise RuntimeError(f"Missing wavelength file for {key}: {source_dir / file_name}")

    wave_map = {
        key: normalize_wave_unit(parse_numeric_text(source_dir / file_name))
        for key, file_name in WAVE_FILES.items()
    }

    spectrum_files = collect_spectrum_files(source_dir)
    if limit > 0:
        spectrum_files = spectrum_files[:limit]
    if not spectrum_files:
        raise RuntimeError(f"No splib07a spectrum files found in: {source_dir}")

    spectra_list: list[np.ndarray] = []
    waves_list: list[np.ndarray] = []
    lengths: list[int] = []
    names: list[str] = []
    classes: list[str] = []
    sources: list[str] = []

    skipped = 0
    for idx, (spec_path, chapter_name) in enumerate(spectrum_files, start=1):
        instrument = detect_instrument(spec_path.name)
        if instrument is None:
            skipped += 1
            continue

        spec = parse_numeric_text(spec_path)
        spec = spec.astype(np.float32, copy=False)
        spec[spec <= DELETED_CHANNEL_THRESHOLD] = np.nan
        waves = wave_map[instrument]

        if spec.size != waves.size:
            # Strictly align lengths; if one side has extra trailing channels, truncate to min length.
            n = min(spec.size, waves.size)
            if n <= 0:
                skipped += 1
                continue
            spec = spec[:n]
            waves = waves[:n]

        spectra_list.append(spec)
        waves_list.append(waves)
        lengths.append(int(spec.size))
        names.append(spec_path.stem)
        classes.append(chapter_to_class(chapter_name))
        sources.append(f"splib07a_ascii_{instrument.lower()}")

        if idx % 300 == 0:
            print(f"Processed {idx}/{len(spectrum_files)} files...")

    if not spectra_list:
        raise RuntimeError("No valid spectra were parsed.")

    total = len(spectra_list)
    max_len = max(lengths)
    ids = np.arange(1, total + 1, dtype=np.int32)

    raw_values = np.full((total, max_len), np.nan, dtype=np.float32)
    raw_lengths = np.asarray(lengths, dtype=np.int32)
    raw_wave_start_idx = np.empty(total, dtype=np.int64)
    wave_segments: list[np.ndarray] = []

    offset = 0
    for i, (spec, waves) in enumerate(zip(spectra_list, waves_list, strict=True)):
        n = spec.size
        raw_values[i, :n] = spec
        raw_wave_start_idx[i] = offset
        offset += n
        wave_segments.append(waves.astype(np.float32, copy=False))

    all_waves = np.concatenate(wave_segments).astype(np.float32, copy=False)

    unique_names = sorted(set(names))
    unique_classes = sorted(set(classes))
    unique_sources = sorted(set(sources))
    name_to_idx = {v: i for i, v in enumerate(unique_names)}
    class_to_idx = {v: i for i, v in enumerate(unique_classes)}
    source_to_idx = {v: i for i, v in enumerate(unique_sources)}

    name_ref = np.asarray([name_to_idx[n] for n in names], dtype=np.int32)
    class_ref = np.asarray([class_to_idx[c] for c in classes], dtype=np.int32)
    source_ref = np.asarray([source_to_idx[s] for s in sources], dtype=np.int32)

    h5_path.unlink(missing_ok=True)
    sqlite_path.unlink(missing_ok=True)

    with h5py.File(h5_path, "w") as h5:
        h5.create_dataset("/spectra/raw_values", data=raw_values, compression="gzip", chunks=True)
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

    sqlite_rows = []
    for i in range(total):
        spec = raw_values[i, : raw_lengths[i]]
        waves = all_waves[raw_wave_start_idx[i] : raw_wave_start_idx[i] + raw_lengths[i]]
        valid_wave = waves[np.isfinite(waves)]
        if valid_wave.size == 0:
            w_min = float("nan")
            w_max = float("nan")
        else:
            w_min = float(np.min(valid_wave))
            w_max = float(np.max(valid_wave))
        sqlite_rows.append(
            (
                int(ids[i]),
                names[i],
                classes[i],
                sources[i],
                w_min,
                w_max,
                int(raw_lengths[i]),
                hash_spectrum(waves, spec),
            )
        )

    write_metadata_index(sqlite_path, sqlite_rows)

    print("Done.")
    print(f"Source dir: {source_dir}")
    print(f"Output HDF5: {h5_path}")
    print(f"Output SQLite: {sqlite_path}")
    print(f"Total parsed spectra: {total}")
    print(f"Skipped files: {skipped}")
    print(f"Max channels: {max_len}")
    print(f"all_waves size: {all_waves.size}")


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Compile USGS splib07a ASCII spectra into local HDF5+SQLite store.")
    parser.add_argument(
        "--source-dir",
        default=str(project_root / "ASCIIdata_splib07a"),
        help="Directory of ASCIIdata_splib07a",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory for usgs_master.h5 and usgs_meta.db (default backend/data/library).",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional limit for debugging.")
    args = parser.parse_args()

    source_dir = Path(args.source_dir).expanduser().resolve()
    if not source_dir.exists():
        raise SystemExit(f"Source directory not found: {source_dir}")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else Path(__file__).resolve().parents[1] / "data" / "library"
    )
    compile_splib07a_ascii(source_dir=source_dir, output_dir=output_dir, limit=args.limit)


if __name__ == "__main__":
    main()
