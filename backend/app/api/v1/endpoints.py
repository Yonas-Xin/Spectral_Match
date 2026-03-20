from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import FileResponse

from app.core.config import settings
from app.core.errors import AppError, ErrorCode
from app.models.schemas import (
    ApiResponse,
    ExportData,
    ExportMatchResultRequest,
    ImageLoadData,
    ImageLoadRequest,
    ImageMetaData,
    PixelMatchData,
    PixelMatchRequest,
    SignatureStatusData,
    SpectrumExtractData,
    SpectrumExtractRequest,
)
from app.services.container import cache_service, export_service, image_service, match_service, store

router = APIRouter()


@router.post("/image/load", response_model=ApiResponse[ImageLoadData])
def load_image(req: ImageLoadRequest) -> ApiResponse[ImageLoadData]:
    ctx = image_service.load_image(image_path=req.image_path, display_mode=req.display_mode)
    prepared = match_service.prepare_signature(
        image_id=ctx.image_id,
        ignore_water_bands=req.ignore_water_bands,
        build_async=req.build_signature_cache,
    )
    data = ImageLoadData(
        image_id=ctx.image_id,
        image_path=str(ctx.image_path),
        samples=ctx.samples,
        lines=ctx.lines,
        bands=ctx.bands,
        interleave=ctx.interleave,
        dtype=ctx.dtype,
        wavelengths=ctx.wavelengths.astype(float).tolist(),
        preview_url=f"{settings.api_prefix}/image/preview/{ctx.image_id}.png",
        rgb_bands=image_service.rgb_model(*ctx.rgb_bands),
        signature=match_service.to_signature_info(prepared, req.ignore_water_bands),
    )
    return ApiResponse(code=0, message="ok", data=data)


@router.get("/image/preview/{image_id}.png")
def get_image_preview(image_id: str):
    ctx = store.get_image(image_id)
    if ctx is None:
        raise AppError(ErrorCode.IMAGE_CONTEXT_NOT_FOUND, f"image context not found: {image_id}", status_code=404)
    if not ctx.preview_path.exists():
        raise AppError(ErrorCode.IMAGE_PARSE_FAILED, f"preview not found: {ctx.preview_path}", status_code=404)
    return FileResponse(path=ctx.preview_path, media_type="image/png")


@router.get(
    "/cache/signature/{signature_hash}/status",
    response_model=ApiResponse[SignatureStatusData],
)
def get_signature_status(signature_hash: str) -> ApiResponse[SignatureStatusData]:
    status = cache_service.status(signature_hash)
    data = SignatureStatusData(
        signature_hash=signature_hash,
        status=status["status"],  # type: ignore[arg-type]
        progress=int(status["progress"]),
        current_step=str(status["current_step"]),
    )
    return ApiResponse(code=0, message="ok", data=data)


@router.post("/match/pixel", response_model=ApiResponse[PixelMatchData])
def match_pixel(req: PixelMatchRequest) -> ApiResponse[PixelMatchData]:
    data = match_service.match_pixel(
        image_id=req.image_id,
        x=req.x,
        y=req.y,
        top_n=req.top_n,
        ignore_water_bands=req.ignore_water_bands,
        min_valid_bands=req.min_valid_bands,
        return_candidate_curves=req.return_candidate_curves,
        selection=req.selection,
        custom_masked_ranges=req.custom_masked_ranges,
    )
    return ApiResponse(code=0, message="ok", data=data)


@router.post("/spectrum/extract", response_model=ApiResponse[SpectrumExtractData])
def extract_spectrum(req: SpectrumExtractRequest) -> ApiResponse[SpectrumExtractData]:
    waves, spec = match_service.extract_spectrum(req.image_id, req.x, req.y)
    data = SpectrumExtractData(
        x=req.x,
        y=req.y,
        wavelengths=waves.astype(float).tolist(),
        spectrum=spec.astype(float).tolist(),
    )
    return ApiResponse(code=0, message="ok", data=data)


@router.post("/export/match-result", response_model=ApiResponse[ExportData])
def export_match_result(req: ExportMatchResultRequest) -> ApiResponse[ExportData]:
    data = export_service.export_match_result(
        image_id=req.image_id,
        x=req.x,
        y=req.y,
        top_n=req.top_n,
        output_path=req.output_path,
        fmt=req.format,
        include_query_spectrum=req.include_query_spectrum,
        include_matched_curves=req.include_matched_curves,
        ignore_water_bands=req.ignore_water_bands,
        min_valid_bands=req.min_valid_bands,
        selection=req.selection,
    )
    return ApiResponse(code=0, message="ok", data=data)


@router.get("/image/{image_id}/meta", response_model=ApiResponse[ImageMetaData])
def get_image_meta(image_id: str) -> ApiResponse[ImageMetaData]:
    ctx = store.get_image(image_id)
    if ctx is None:
        raise AppError(ErrorCode.IMAGE_CONTEXT_NOT_FOUND, f"image context not found: {image_id}", status_code=404)

    data = ImageMetaData(
        image_id=ctx.image_id,
        path=str(ctx.image_path),
        samples=ctx.samples,
        lines=ctx.lines,
        bands=ctx.bands,
        wavelength_unit=ctx.wavelength_unit,
        wavelengths=ctx.wavelengths.astype(float).tolist(),
    )
    return ApiResponse(code=0, message="ok", data=data)
