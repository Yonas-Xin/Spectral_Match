from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.errors import AppError, ErrorCode
from app.models.schemas import ApiResponse

logger = logging.getLogger("spectral_match.backend")

app = FastAPI(title=settings.app_name, default_response_class=ORJSONResponse)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_allow_origins),
    allow_methods=list(settings.cors_allow_methods),
    allow_headers=list(settings.cors_allow_headers),
    allow_credentials=settings.cors_allow_credentials,
)
app.include_router(api_router)


@app.get("/healthz")
def healthz():
    return ApiResponse(code=0, message="ok", data={"status": "healthy"})


@app.exception_handler(AppError)
async def handle_app_error(_: Request, exc: AppError):
    logger.error("AppError code=%s status=%s message=%s", exc.code, exc.status_code, exc.message)
    payload = ApiResponse(code=exc.code, message=exc.message, data=None)
    return ORJSONResponse(status_code=exc.status_code, content=payload.model_dump())


@app.exception_handler(RequestValidationError)
async def handle_validation_error(_: Request, exc: RequestValidationError):
    logger.warning("Request validation error: %s", exc.errors())
    payload = ApiResponse(
        code=ErrorCode.INVALID_REQUEST,
        message="invalid request",
        data={"detail": exc.errors()},
    )
    return ORJSONResponse(status_code=422, content=payload.model_dump())


@app.exception_handler(Exception)
async def handle_unknown_error(_: Request, exc: Exception):
    logger.exception("Unhandled exception")
    payload = ApiResponse(
        code=5000,
        message=f"internal server error: {exc}",
        data=None,
    )
    return ORJSONResponse(status_code=500, content=payload.model_dump())
