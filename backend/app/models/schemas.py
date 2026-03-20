from __future__ import annotations

from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field


T = TypeVar("T")


class ApiResponse(BaseModel, Generic[T]):
    code: int = 0
    message: str = "ok"
    data: T | None = None


class SignatureInfo(BaseModel):
    hash: str
    ignore_water_bands: bool
    cache_exists: bool
    build_status: Literal["not_found", "building", "ready", "failed"]


class ImageLoadRequest(BaseModel):
    image_path: str
    display_mode: Literal["true_color", "false_color"] = "true_color"
    build_signature_cache: bool = True
    ignore_water_bands: bool = True


class ImageRGBBands(BaseModel):
    r: int
    g: int
    b: int


class ImageLoadData(BaseModel):
    image_id: str
    image_path: str
    samples: int
    lines: int
    bands: int
    interleave: str
    dtype: str
    wavelengths: list[float]
    preview_url: str
    rgb_bands: ImageRGBBands
    signature: SignatureInfo


class SignatureStatusData(BaseModel):
    signature_hash: str
    status: Literal["not_found", "building", "ready", "failed"]
    progress: int = Field(default=0, ge=0, le=100)
    current_step: str = ""


class RegionPoint(BaseModel):
    x: float
    y: float


class RegionSelection(BaseModel):
    mode: Literal["pixel", "box", "circle", "lasso"] = "pixel"
    x0: int | None = None
    y0: int | None = None
    x1: int | None = None
    y1: int | None = None
    cx: float | None = None
    cy: float | None = None
    radius: float | None = None
    points: list[RegionPoint] | None = None


class SpectralMaskRange(BaseModel):
    start: float
    end: float


class PixelMatchRequest(BaseModel):
    image_id: str
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    top_n: int = Field(default=10, ge=1, le=20)
    metric: Literal["sam"] = "sam"
    ignore_water_bands: bool = True
    min_valid_bands: int | None = Field(default=None, ge=1)
    return_candidate_curves: bool = True
    selection: RegionSelection | None = None
    custom_masked_ranges: list[SpectralMaskRange] = Field(default_factory=list)


class SpectrumExtractRequest(BaseModel):
    image_id: str
    x: int = Field(ge=0)
    y: int = Field(ge=0)


class ExportMatchResultRequest(BaseModel):
    image_id: str
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    top_n: int = Field(default=10, ge=1, le=20)
    format: Literal["csv", "txt"] = "csv"
    output_path: str
    include_query_spectrum: bool = True
    include_matched_curves: bool = True
    ignore_water_bands: bool = True
    min_valid_bands: int | None = Field(default=None, ge=1)
    selection: RegionSelection | None = None


class QuerySpectrumData(BaseModel):
    x: int
    y: int
    bands_total: int
    bands_used: int
    selection_mode: Literal["pixel", "box", "circle", "lasso"] = "pixel"
    selected_pixels: int = Field(default=1, ge=1)
    wavelengths: list[float]
    spectrum: list[float]


class MatchContextData(BaseModel):
    signature_hash: str
    metric: str
    ignore_water_bands: bool
    min_valid_bands: int
    custom_masked_ranges: list[SpectralMaskRange] = Field(default_factory=list)
    candidate_count: int
    elapsed_ms: int


class MatchResultItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    rank: int
    spectrum_id: int
    name: str
    class_name: str | None = Field(default=None, alias="class")
    sensor_type: str | None = None
    measure_mode: str | None = None
    source: str | None = None
    sam_score: float
    pearson_r: float | None = None
    curve: list[float] | None = None


class PixelMatchData(BaseModel):
    query: QuerySpectrumData
    match_context: MatchContextData
    results: list[MatchResultItem]


class SpectrumExtractData(BaseModel):
    x: int
    y: int
    wavelengths: list[float]
    spectrum: list[float]


class ExportData(BaseModel):
    output_path: str
    format: str
    file_size: int
    rows_written: int


class ImageMetaData(BaseModel):
    image_id: str
    path: str
    samples: int
    lines: int
    bands: int
    wavelength_unit: str
    wavelengths: list[float]
