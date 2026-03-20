from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import settings
from app.services.cache_service import SignatureCacheService
from app.services.library_service import LibraryService
from app.services.state_store import RuntimeStore
from app.utils.signature import build_signature_hash

FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


def parse_named_array(text: str, name: str) -> np.ndarray:
    match = re.search(rf"{name}\s*=\s*\{{(.*?)\}}", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        raise RuntimeError(f"Cannot find '{name} = {{...}}' block.")
    values = [float(token) for token in FLOAT_RE.findall(match.group(1))]
    if not values:
        raise RuntimeError(f"No numeric values found in '{name}' block.")
    return np.asarray(values, dtype=np.float32)


def load_sensor_definition(path: Path) -> tuple[np.ndarray, np.ndarray]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    waves = parse_named_array(text, "wavelength")
    fwhm = parse_named_array(text, "fwhm")
    if waves.shape[0] != fwhm.shape[0]:
        raise RuntimeError(
            f"Wavelength/FWHM length mismatch in {path}: {waves.shape[0]} vs {fwhm.shape[0]}"
        )
    return waves, fwhm


def build_sensor_cache(
    name: str,
    sensor_file: Path,
    cache_service: SignatureCacheService,
    ignore_water_bands: bool,
    force_rebuild: bool,
) -> dict[str, object]:
    waves, fwhm = load_sensor_definition(sensor_file)
    signature_hash = build_signature_hash(
        image_waves=waves,
        image_fwhm=fwhm,
        ignore_water_bands=ignore_water_bands,
        resample_algo_version=settings.resample_algo_version,
        clean_rules_version=settings.clean_rules_version,
    )
    cache_dir = cache_service.cache_dir(signature_hash)
    if force_rebuild and cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)

    started = time.perf_counter()
    cache_service.build_sync(
        signature_hash=signature_hash,
        img_waves=waves,
        img_fwhm=fwhm,
        ignore_water_bands=ignore_water_bands,
    )
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    meta_path = cache_dir / "meta.json"
    if not meta_path.exists():
        raise RuntimeError(f"Cache build finished but meta.json not found: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    conversion_stats = meta.get("conversion_stats", {})

    total_input = int(meta.get("total_input_spectra", meta.get("total_spectra", 0)))
    kept = int(meta.get("total_spectra", 0))
    filtered = int(conversion_stats.get("filtered_out", max(total_input - kept, 0)))
    full = int(conversion_stats.get("fully_convertible", 0))
    partial = int(conversion_stats.get("partially_convertible", 0))
    zero_valid = int(conversion_stats.get("zero_valid_bands", 0))
    below_min = int(conversion_stats.get("below_min_valid_bands", 0))

    print(
        (
            f"[{name}] bands={waves.shape[0]} "
            f"total={total_input} kept={kept} filtered={filtered} "
            f"full={full} partial={partial} zero_valid={zero_valid} below_min={below_min} "
            f"elapsed={elapsed_ms}ms"
        )
    )

    return {
        "sensor": name,
        "sensor_file": str(sensor_file),
        "signature_hash": signature_hash,
        "bands": int(waves.shape[0]),
        "elapsed_ms": elapsed_ms,
        "meta": meta,
    }


def main() -> None:
    project_root = BACKEND_ROOT.parent
    parser = argparse.ArgumentParser(
        description=(
            "Prebuild signature caches for GF5 and ZY-1-02D using provided wavelength/FWHM files. "
            "Outputs conversion statistics for incomplete conversions."
        )
    )
    parser.add_argument(
        "--gf5-file",
        default=str(project_root / "ASCIIdata_splib07b" / "gf5.txt"),
        help="Path to GF5 wavelength/fwhm definition file.",
    )
    parser.add_argument(
        "--zy102d-file",
        default=str(project_root / "ASCIIdata_splib07b" / "zy-1-02d.txt"),
        help="Path to ZY-1-02D wavelength/fwhm definition file.",
    )
    parser.add_argument(
        "--report-path",
        default="",
        help="Output JSON report path (default: backend/data/cache/sensor_prebuild_report.json).",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Delete existing sensor signature cache directories before build.",
    )
    parser.add_argument(
        "--ignore-water-bands",
        dest="ignore_water_bands",
        action="store_true",
        default=True,
        help="Ignore water absorption bands when building signature cache (default: true).",
    )
    parser.add_argument(
        "--keep-water-bands",
        dest="ignore_water_bands",
        action="store_false",
        help="Keep water absorption bands when building signature cache.",
    )
    args = parser.parse_args()

    gf5_file = Path(args.gf5_file).expanduser().resolve()
    zy102d_file = Path(args.zy102d_file).expanduser().resolve()
    if not gf5_file.exists():
        raise SystemExit(f"GF5 file not found: {gf5_file}")
    if not zy102d_file.exists():
        raise SystemExit(f"ZY-1-02D file not found: {zy102d_file}")

    report_path = (
        Path(args.report_path).expanduser().resolve()
        if args.report_path
        else (settings.cache_root / "sensor_prebuild_report.json")
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)

    store = RuntimeStore(max_active_signatures=settings.max_active_signatures)
    library_service = LibraryService()
    cache_service = SignatureCacheService(store=store, library_service=library_service)

    summaries = [
        build_sensor_cache(
            name="GF5",
            sensor_file=gf5_file,
            cache_service=cache_service,
            ignore_water_bands=args.ignore_water_bands,
            force_rebuild=args.force_rebuild,
        ),
        build_sensor_cache(
            name="ZY102D",
            sensor_file=zy102d_file,
            cache_service=cache_service,
            ignore_water_bands=args.ignore_water_bands,
            force_rebuild=args.force_rebuild,
        ),
    ]

    report = {
        "created_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        "ignore_water_bands": bool(args.ignore_water_bands),
        "resample_algo_version": settings.resample_algo_version,
        "clean_rules_version": settings.clean_rules_version,
        "library_min_coverage_ratio": float(settings.library_min_coverage_ratio),
        "min_valid_bands": int(settings.min_valid_bands),
        "sensors": summaries,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report written: {report_path}")


if __name__ == "__main__":
    main()
