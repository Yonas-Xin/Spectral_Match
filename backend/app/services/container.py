from __future__ import annotations

from app.core.config import settings
from app.services.cache_service import SignatureCacheService
from app.services.export_service import ExportService
from app.services.image_service import ImageService
from app.services.library_service import LibraryService
from app.services.match_service import MatchService
from app.services.state_store import RuntimeStore

store = RuntimeStore(max_active_signatures=settings.max_active_signatures)
library_service = LibraryService()
image_service = ImageService(store=store)
cache_service = SignatureCacheService(store=store, library_service=library_service)
match_service = MatchService(
    image_service=image_service,
    cache_service=cache_service,
    library_service=library_service,
)
export_service = ExportService(match_service=match_service)
