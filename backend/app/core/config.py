from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    app_name: str = "Spectral Match Backend"
    api_prefix: str = "/api/v1"
    host: str = "127.0.0.1"
    port: int = 8000
    max_active_signatures: int = 2
    match_candidate_topk: int = 512
    min_valid_bands: int = 20
    cache_build_wait_timeout_sec: int = 20
    cache_build_wait_poll_ms: int = 200
    library_min_coverage_ratio: float = 0.0
    clip_reflectance_min: float = 0.0
    clip_reflectance_max: float = 1.0
    resample_algo_version: str = "rf_gaussian_v1"
    clean_rules_version: str = "clean_v2_unit01"
    cors_allow_origins: tuple[str, ...] = ("*",)
    cors_allow_methods: tuple[str, ...] = ("*",)
    cors_allow_headers: tuple[str, ...] = ("*",)
    cors_allow_credentials: bool = False

    data_dir: Path = Path(".")
    previews_dir: Path = Path(".")
    cache_root: Path = Path(".")
    signatures_dir: Path = Path(".")
    library_dir: Path = Path(".")
    library_h5_path: Path = Path(".")
    library_sqlite_path: Path = Path(".")


def _default_data_dir() -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / "data"


def get_settings() -> Settings:
    data_dir = Path(os.getenv("SPECTRAL_DATA_DIR", _default_data_dir())).resolve()
    previews_dir = data_dir / "previews"
    cache_root = data_dir / "cache"
    signatures_dir = cache_root / "signatures"
    library_dir = data_dir / "library"

    previews_dir.mkdir(parents=True, exist_ok=True)
    signatures_dir.mkdir(parents=True, exist_ok=True)
    library_dir.mkdir(parents=True, exist_ok=True)

    cors_origins = tuple(
        p.strip()
        for p in os.getenv("SPECTRAL_CORS_ALLOW_ORIGINS", "*").split(",")
        if p.strip()
    )
    cors_methods = tuple(
        p.strip()
        for p in os.getenv("SPECTRAL_CORS_ALLOW_METHODS", "*").split(",")
        if p.strip()
    )
    cors_headers = tuple(
        p.strip()
        for p in os.getenv("SPECTRAL_CORS_ALLOW_HEADERS", "*").split(",")
        if p.strip()
    )
    cors_allow_credentials = os.getenv("SPECTRAL_CORS_ALLOW_CREDENTIALS", "false").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    return Settings(
        max_active_signatures=int(os.getenv("SPECTRAL_MAX_ACTIVE_SIGNATURES", "2")),
        match_candidate_topk=int(os.getenv("SPECTRAL_MATCH_CANDIDATE_TOPK", "512")),
        min_valid_bands=int(os.getenv("SPECTRAL_MIN_VALID_BANDS", "20")),
        cache_build_wait_timeout_sec=int(os.getenv("SPECTRAL_CACHE_BUILD_WAIT_TIMEOUT_SEC", "20")),
        cache_build_wait_poll_ms=int(os.getenv("SPECTRAL_CACHE_BUILD_WAIT_POLL_MS", "200")),
        library_min_coverage_ratio=float(os.getenv("SPECTRAL_LIBRARY_MIN_COVERAGE_RATIO", "0.0")),
        cors_allow_origins=cors_origins if cors_origins else ("*",),
        cors_allow_methods=cors_methods if cors_methods else ("*",),
        cors_allow_headers=cors_headers if cors_headers else ("*",),
        cors_allow_credentials=cors_allow_credentials,
        data_dir=data_dir,
        previews_dir=previews_dir,
        cache_root=cache_root,
        signatures_dir=signatures_dir,
        library_dir=library_dir,
        library_h5_path=library_dir / "usgs_master.h5",
        library_sqlite_path=library_dir / "usgs_meta.db",
    )


settings = get_settings()
