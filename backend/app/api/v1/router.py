from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.endpoints import router as endpoint_router

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(endpoint_router)
