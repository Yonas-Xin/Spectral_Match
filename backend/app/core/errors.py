from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AppError(Exception):
    code: int
    message: str
    status_code: int = 400

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"


class ErrorCode:
    INVALID_REQUEST = 4000
    IMAGE_FILE_NOT_FOUND = 4001
    IMAGE_PARSE_FAILED = 4002
    IMAGE_CONTEXT_NOT_FOUND = 4003
    PIXEL_OUT_OF_RANGE = 4004
    CACHE_NOT_FOUND = 5001
    CACHE_BUILDING = 5002
    CACHE_FAILED = 5003
    LIBRARY_NOT_READY = 5004
    MATCH_FAILED = 5005
    EXPORT_FAILED = 5006
