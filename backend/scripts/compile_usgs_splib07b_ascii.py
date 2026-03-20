from __future__ import annotations

import argparse
import hashlib
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

DELETED_CHANNEL_THRESHOLD = -1.0e33

WAVE_FILES = {
    "ASDFR": "splib07b_Wavelengths_ASDFR_0.35-2.5microns_2151ch.txt",
    "AVIRIS": "splib07b_Wavelengths_AVIRIS_1996_interp_to_2203ch.txt",
    "BECK": "splib07b_Wavelengths_BECK_Beckman_interp._3961_ch.txt",
    "NIC4": "splib07b_Wavelengths_NIC4_Nicolet_1.12-216microns.txt",
}

FWHM_FILES = {
    "ASDFR": "splib07b_Bandpass_(FWHM)_ASDFR_StandardResolution.txt",
    "ASDHR": "splib07b_Bandpass_(FWHM)_ASDHR_High-Resolution.txt",
    "ASDNG": "splib07b_Bandpass_(FWHM)_ASDNG_High-Res_NextGen.txt",
    "AVIRIS": "splib07b_Bandpass_(FWHM)_AVIRIS_1996_in_microns.txt",
    "BECK": "splib07b_Bandpass_(FWHM)_BECK_Beckman_in_microns.txt",
    "NIC4": "splib07b_Bandpass_(FWHM)_NIC4_Nicolet_in_microns.txt",
}

VARIANT_RE = re.compile(
    r"_(ASD(?:FR|HR|NG)[A-Za-z0-9]*|AVIRIS[A-Za-z0-9]*|BECK[A-Za-z0-9]*|NIC4[A-Za-z0-9]*)_(AREF|RREF|RTGC|TRAN)$",
    flags=re.IGNORECASE,
)


@dataclass
class SpectrumRecord:
    path: Path
    chapter: str
    class_name: str
    raw_name: str
    name: str
    instrument_variant: str
    wave_key: str
    fwhm_key: str
    measure_mode: str
    meta_group: str
    spectrum: np.ndarray
    waves: np.ndarray
    fwhm: np.ndarray
    source: str


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


def normalize_wave_unit(arr: np.ndarray) -> np.ndarray:
    if arr.size and float(np.nanmax(arr)) < 400.0:
        return arr * 1000.0
    return arr


def chapter_to_class(chapter: str) -> str:
    if "_" not in chapter:
        return chapter
    return chapter.split("_", 1)[1]


def hash_spectrum(waves: np.ndarray, spec: np.ndarray) -> str:
    wave_part = np.asarray(waves, dtype=np.float32)
    spec_part = np.nan_to_num(np.asarray(spec, dtype=np.float32), nan=-9999.0, posinf=-9999.0, neginf=-9999.0)
    payload = np.concatenate([wave_part, spec_part]).tobytes()
    return hashlib.sha1(payload).hexdigest()


def detect_variant_and_mode(stem: str) -> tuple[str, str] | None:
    m = VARIANT_RE.search(stem)
    if not m:
        return None
    variant = m.group(1).upper()
    mode = m.group(2).upper()
    return variant, mode


def normalize_display_name(stem: str) -> str:
    s = stem.strip()
    if s.lower().startswith("splib07b_"):
        s = s[9:]
    s = VARIANT_RE.sub("", s)
    s = s.replace("_", " ").strip()
    return s or stem


def map_keys(variant: str) -> tuple[str, str]:
    u = variant.upper()
    if u.startswith("ASDFR"):
        return "ASDFR", "ASDFR"
    if u.startswith("ASDHR"):
        return "ASDFR", "ASDHR"
    if u.startswith("ASDNG"):
        return "ASDFR", "ASDNG"
    if u.startswith("AVIRIS"):
        return "AVIRIS", "AVIRIS"
    if u.startswith("BECK"):
        return "BECK", "BECK"
    if u.startswith("NIC4"):
        return "NIC4", "NIC4"
    raise RuntimeError(f"Unsupported instrument variant: {variant}")


def collect_spectrum_files(source_dir: Path, include_errorbars: bool = False) -> list[tuple[Path, str]]:
    files: list[tuple[Path, str]] = []
    for chapter_dir in sorted(source_dir.glob("Chapter*")):
        if not chapter_dir.is_dir():
            continue
        for txt in sorted(chapter_dir.glob("splib07b_*.txt")):
            files.append((txt, chapter_dir.name))
    if include_errorbars:
        eb_dir = source_dir / "errorbars"
        if eb_dir.exists():
            for txt in sorted(eb_dir.glob("splib07b_*.txt")):
                files.append((txt, "errorbars"))
    return files


def ordered_group_keys(grouped: dict[str, list[SpectrumRecord]]) -> list[str]:
    wave_order = {"ASDFR": 0, "AVIRIS": 1, "BECK": 2, "NIC4": 3}
    fwhm_order = {"ASDFR": 0, "ASDHR": 1, "ASDNG": 2, "AVIRIS": 3, "BECK": 4, "NIC4": 5}

    def key_fn(meta_group: str) -> tuple[int, int, str, str]:
        wave_key, fwhm_key, mode = meta_group.split("|", 2)
        return (wave_order.get(wave_key, 99), fwhm_order.get(fwhm_key, 99), mode, meta_group)

    return sorted(grouped.keys(), key=key_fn)


def write_metadata_index(
    sqlite_path: Path,
    rows: list[tuple[int, str, str, str | None, str | None, float, float, int, str, str, str, str, str, str]],
    group_rows: list[tuple[int, str, str, str, str, int, int]],
) -> None:
    conn = sqlite3.connect(sqlite_path)
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS spectra_meta (
              id INTEGER PRIMARY KEY,
              name TEXT NOT NULL,
                            raw_name TEXT NOT NULL,
              class TEXT,
              source TEXT,
              wavelength_min REAL,
              wavelength_max REAL,
              points_count INTEGER,
              hash TEXT,
              chapter TEXT,
              instrument_variant TEXT,
              wave_key TEXT,
              fwhm_key TEXT,
              measure_mode TEXT,
              meta_group TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_meta_name ON spectra_meta(name);
            CREATE INDEX IF NOT EXISTS idx_meta_class ON spectra_meta(class);
            CREATE INDEX IF NOT EXISTS idx_meta_hash ON spectra_meta(hash);
            CREATE INDEX IF NOT EXISTS idx_meta_group ON spectra_meta(meta_group);
            CREATE INDEX IF NOT EXISTS idx_meta_wave ON spectra_meta(wave_key);
            CREATE INDEX IF NOT EXISTS idx_meta_variant ON spectra_meta(instrument_variant);

            CREATE TABLE IF NOT EXISTS spectra_groups (
              group_id INTEGER PRIMARY KEY,
              meta_group TEXT UNIQUE,
              wave_key TEXT,
              fwhm_key TEXT,
              measure_mode TEXT,
              start_idx INTEGER,
              points_count INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_groups_meta_group ON spectra_groups(meta_group);
            """
        )
        conn.executemany(
            """
            INSERT OR REPLACE INTO spectra_meta (
                            id, name, raw_name, class, source, wavelength_min, wavelength_max, points_count, hash,
              chapter, instrument_variant, wave_key, fwhm_key, measure_mode, meta_group
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.executemany(
            """
            INSERT OR REPLACE INTO spectra_groups (
              group_id, meta_group, wave_key, fwhm_key, measure_mode, start_idx, points_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            group_rows,
        )
        conn.commit()
    finally:
        conn.close()


def compile_splib07b_ascii(
    source_dir: Path,
    output_dir: Path,
    limit: int = 0,
    include_errorbars: bool = False,
) -> None:
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    h5_path = output_dir / "usgs_master.h5"
    sqlite_path = output_dir / "usgs_meta.db"

    for key, file_name in WAVE_FILES.items():
        if not (source_dir / file_name).exists():
            raise RuntimeError(f"Missing wavelength file for {key}: {source_dir / file_name}")
    for key, file_name in FWHM_FILES.items():
        if not (source_dir / file_name).exists():
            raise RuntimeError(f"Missing FWHM file for {key}: {source_dir / file_name}")

    wave_map = {k: normalize_wave_unit(parse_numeric_text(source_dir / v)) for k, v in WAVE_FILES.items()}
    fwhm_map = {k: normalize_wave_unit(parse_numeric_text(source_dir / v)) for k, v in FWHM_FILES.items()}

    spectrum_files = collect_spectrum_files(source_dir, include_errorbars=include_errorbars)
    if limit > 0:
        spectrum_files = spectrum_files[:limit]
    if not spectrum_files:
        raise RuntimeError(f"No splib07b spectrum files found in: {source_dir}")

    grouped: dict[str, list[SpectrumRecord]] = {}
    skipped = 0
    for idx, (spec_path, chapter_name) in enumerate(spectrum_files, start=1):
        parsed = detect_variant_and_mode(spec_path.stem)
        if parsed is None:
            skipped += 1
            continue
        instrument_variant, measure_mode = parsed
        wave_key, fwhm_key = map_keys(instrument_variant)

        spec = parse_numeric_text(spec_path).astype(np.float32, copy=False)
        spec[spec <= DELETED_CHANNEL_THRESHOLD] = np.nan
        waves = wave_map[wave_key]
        fwhm = fwhm_map[fwhm_key]

        if spec.size != waves.size:
            n = min(spec.size, waves.size)
            if n <= 0:
                skipped += 1
                continue
            spec = spec[:n]
            waves = waves[:n]
            if fwhm.size >= n:
                fwhm = fwhm[:n]
            else:
                fwhm = np.pad(fwhm, (0, n - fwhm.size), mode="edge")
        else:
            if fwhm.size >= spec.size:
                fwhm = fwhm[: spec.size]
            else:
                fwhm = np.pad(fwhm, (0, spec.size - fwhm.size), mode="edge")

        meta_group = f"{wave_key}|{fwhm_key}|{measure_mode}"
        rec = SpectrumRecord(
            path=spec_path,
            chapter=chapter_name,
            class_name=chapter_to_class(chapter_name),
            raw_name=spec_path.stem,
            name=normalize_display_name(spec_path.stem),
            instrument_variant=instrument_variant,
            wave_key=wave_key,
            fwhm_key=fwhm_key,
            measure_mode=measure_mode,
            meta_group=meta_group,
            spectrum=spec,
            waves=waves,
            fwhm=fwhm,
            source="splib07b_ascii",
        )
        grouped.setdefault(meta_group, []).append(rec)

        if idx % 400 == 0:
            print(f"Scanned {idx}/{len(spectrum_files)} files...")

    ordered_groups = ordered_group_keys(grouped)
    ordered_records: list[SpectrumRecord] = []
    group_ranges: list[tuple[str, int, int, str, str, str]] = []
    cursor = 0
    for gk in ordered_groups:
        records = sorted(grouped[gk], key=lambda r: r.name.lower())
        count = len(records)
        wave_key, fwhm_key, mode = gk.split("|", 2)
        group_ranges.append((gk, cursor, count, wave_key, fwhm_key, mode))
        ordered_records.extend(records)
        cursor += count

    if not ordered_records:
        raise RuntimeError("No valid splib07b spectra were parsed.")

    total = len(ordered_records)
    max_len = max(int(r.spectrum.size) for r in ordered_records)
    ids = np.arange(1, total + 1, dtype=np.int32)

    raw_values = np.full((total, max_len), np.nan, dtype=np.float32)
    raw_lengths = np.empty(total, dtype=np.int32)
    raw_wave_start_idx = np.empty(total, dtype=np.int64)
    raw_fwhm_start_idx = np.empty(total, dtype=np.int64)

    wave_segments: list[np.ndarray] = []
    fwhm_segments: list[np.ndarray] = []

    names: list[str] = []
    classes: list[str] = []
    sources: list[str] = []
    chapters: list[str] = []
    variants: list[str] = []
    wave_keys: list[str] = []
    fwhm_keys: list[str] = []
    modes: list[str] = []
    groups: list[str] = []

    wave_offset = 0
    fwhm_offset = 0
    for i, rec in enumerate(ordered_records):
        n = int(rec.spectrum.size)
        raw_values[i, :n] = rec.spectrum
        raw_lengths[i] = n
        raw_wave_start_idx[i] = wave_offset
        raw_fwhm_start_idx[i] = fwhm_offset
        wave_offset += n
        fwhm_offset += n
        wave_segments.append(rec.waves.astype(np.float32, copy=False))
        fwhm_segments.append(rec.fwhm.astype(np.float32, copy=False))

        names.append(rec.name)
        classes.append(rec.class_name)
        sources.append(rec.source)
        chapters.append(rec.chapter)
        variants.append(rec.instrument_variant)
        wave_keys.append(rec.wave_key)
        fwhm_keys.append(rec.fwhm_key)
        modes.append(rec.measure_mode)
        groups.append(rec.meta_group)

    all_waves = np.concatenate(wave_segments).astype(np.float32, copy=False)
    all_fwhm = np.concatenate(fwhm_segments).astype(np.float32, copy=False)

    unique_names = sorted(set(names))
    unique_classes = sorted(set(classes))
    unique_sources = sorted(set(sources))
    unique_chapters = sorted(set(chapters))
    unique_variants = sorted(set(variants))
    unique_wave_keys = sorted(set(wave_keys))
    unique_fwhm_keys = sorted(set(fwhm_keys))
    unique_modes = sorted(set(modes))
    unique_groups = ordered_groups

    def build_ref(values: list[str], uniq: list[str]) -> np.ndarray:
        mapper = {v: i for i, v in enumerate(uniq)}
        return np.asarray([mapper[v] for v in values], dtype=np.int32)

    name_ref = build_ref(names, unique_names)
    class_ref = build_ref(classes, unique_classes)
    source_ref = build_ref(sources, unique_sources)
    chapter_ref = build_ref(chapters, unique_chapters)
    variant_ref = build_ref(variants, unique_variants)
    wave_key_ref = build_ref(wave_keys, unique_wave_keys)
    fwhm_key_ref = build_ref(fwhm_keys, unique_fwhm_keys)
    mode_ref = build_ref(modes, unique_modes)
    group_ref = build_ref(groups, unique_groups)

    group_key_to_id = {gk: i for i, gk in enumerate(unique_groups)}
    group_start_idx = np.asarray([start for _, start, _, _, _, _ in group_ranges], dtype=np.int32)
    group_count = np.asarray([count for _, _, count, _, _, _ in group_ranges], dtype=np.int32)
    group_wave_key_ref = np.asarray(
        [unique_wave_keys.index(wk) for _, _, _, wk, _, _ in group_ranges],
        dtype=np.int32,
    )
    group_fwhm_key_ref = np.asarray(
        [unique_fwhm_keys.index(fk) for _, _, _, _, fk, _ in group_ranges],
        dtype=np.int32,
    )
    group_mode_ref = np.asarray(
        [unique_modes.index(md) for _, _, _, _, _, md in group_ranges],
        dtype=np.int32,
    )
    group_id_array = np.asarray([group_key_to_id[gk] for gk, _, _, _, _, _ in group_ranges], dtype=np.int32)

    h5_path.unlink(missing_ok=True)
    sqlite_path.unlink(missing_ok=True)

    with h5py.File(h5_path, "w") as h5:
        h5.create_dataset("/spectra/raw_values", data=raw_values, compression="gzip", chunks=True)
        h5.create_dataset("/spectra/raw_lengths", data=raw_lengths)
        h5.create_dataset("/spectra/raw_wave_start_idx", data=raw_wave_start_idx)
        h5.create_dataset("/spectra/raw_fwhm_start_idx", data=raw_fwhm_start_idx)
        h5.create_dataset("/waves/all_waves", data=all_waves, compression="gzip", chunks=True)
        h5.create_dataset("/fwhm/all_fwhm", data=all_fwhm, compression="gzip", chunks=True)

        h5.create_dataset("/meta/id", data=ids)
        h5.create_dataset("/meta/name_ref", data=name_ref)
        h5.create_dataset("/meta/class_ref", data=class_ref)
        h5.create_dataset("/meta/source_ref", data=source_ref)
        h5.create_dataset("/meta/chapter_ref", data=chapter_ref)
        h5.create_dataset("/meta/instrument_variant_ref", data=variant_ref)
        h5.create_dataset("/meta/wave_key_ref", data=wave_key_ref)
        h5.create_dataset("/meta/fwhm_key_ref", data=fwhm_key_ref)
        h5.create_dataset("/meta/measure_mode_ref", data=mode_ref)
        h5.create_dataset("/meta/group_ref", data=group_ref)

        h5.create_dataset("/dict/names", data=np.asarray(unique_names, dtype=h5py.string_dtype("utf-8")))
        h5.create_dataset("/dict/classes", data=np.asarray(unique_classes, dtype=h5py.string_dtype("utf-8")))
        h5.create_dataset("/dict/sources", data=np.asarray(unique_sources, dtype=h5py.string_dtype("utf-8")))
        h5.create_dataset("/dict/chapters", data=np.asarray(unique_chapters, dtype=h5py.string_dtype("utf-8")))
        h5.create_dataset(
            "/dict/instrument_variants",
            data=np.asarray(unique_variants, dtype=h5py.string_dtype("utf-8")),
        )
        h5.create_dataset("/dict/wave_keys", data=np.asarray(unique_wave_keys, dtype=h5py.string_dtype("utf-8")))
        h5.create_dataset("/dict/fwhm_keys", data=np.asarray(unique_fwhm_keys, dtype=h5py.string_dtype("utf-8")))
        h5.create_dataset("/dict/measure_modes", data=np.asarray(unique_modes, dtype=h5py.string_dtype("utf-8")))
        h5.create_dataset("/dict/meta_groups", data=np.asarray(unique_groups, dtype=h5py.string_dtype("utf-8")))

        h5.create_dataset("/groups/id", data=group_id_array)
        h5.create_dataset("/groups/start_idx", data=group_start_idx)
        h5.create_dataset("/groups/count", data=group_count)
        h5.create_dataset("/groups/wave_key_ref", data=group_wave_key_ref)
        h5.create_dataset("/groups/fwhm_key_ref", data=group_fwhm_key_ref)
        h5.create_dataset("/groups/measure_mode_ref", data=group_mode_ref)

    sqlite_rows: list[tuple[int, str, str, str | None, str | None, float, float, int, str, str, str, str, str, str]] = []
    for i in range(total):
        length = int(raw_lengths[i])
        waves = all_waves[raw_wave_start_idx[i] : raw_wave_start_idx[i] + length]
        spec = raw_values[i, :length]
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
                ordered_records[i].raw_name,
                classes[i],
                sources[i],
                w_min,
                w_max,
                length,
                hash_spectrum(waves, spec),
                chapters[i],
                variants[i],
                wave_keys[i],
                fwhm_keys[i],
                modes[i],
                groups[i],
            )
        )

    sqlite_group_rows: list[tuple[int, str, str, str, str, int, int]] = []
    for gid, (gk, start, count, wave_key, fwhm_key, mode) in enumerate(group_ranges):
        sqlite_group_rows.append((gid, gk, wave_key, fwhm_key, mode, start, count))

    write_metadata_index(sqlite_path, sqlite_rows, sqlite_group_rows)

    print("Done.")
    print(f"Source dir: {source_dir}")
    print(f"Output HDF5: {h5_path}")
    print(f"Output SQLite: {sqlite_path}")
    print(f"Total parsed spectra: {total}")
    print(f"Skipped files: {skipped}")
    print(f"Meta groups: {len(unique_groups)}")
    print("Top groups by size:")
    top = sorted(group_ranges, key=lambda x: x[2], reverse=True)[:10]
    for gk, _, count, _, _, _ in top:
        print(f"  {gk}: {count}")
    print(f"Max channels: {max_len}")
    print(f"all_waves size: {all_waves.size}")


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Compile USGS splib07b ASCII chapter spectra into local HDF5+SQLite store, "
            "grouped by shared metadata for batch resampling."
        )
    )
    parser.add_argument(
        "--source-dir",
        default=str(project_root / "ASCIIdata_splib07b"),
        help="Directory of ASCIIdata_splib07b",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory for usgs_master.h5 and usgs_meta.db (default backend/data/library).",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional limit for debugging.")
    parser.add_argument(
        "--include-errorbars",
        action="store_true",
        help="Also include spectra from errorbars folder (default false).",
    )
    args = parser.parse_args()

    source_dir = Path(args.source_dir).expanduser().resolve()
    if not source_dir.exists():
        raise SystemExit(f"Source directory not found: {source_dir}")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else Path(__file__).resolve().parents[1] / "data" / "library"
    )

    compile_splib07b_ascii(
        source_dir=source_dir,
        output_dir=output_dir,
        limit=args.limit,
        include_errorbars=args.include_errorbars,
    )


if __name__ == "__main__":
    main()
