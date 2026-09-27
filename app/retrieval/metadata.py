from __future__ import annotations

from typing import Optional

from app.storage.sqlite import get_store


def metadata_retrieve(
    service: Optional[str] = None,
    document_type: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    version: Optional[str] = None,
    limit: int = 50,
    historical_year: Optional[int] = None,
) -> list[dict]:
    """
    Structured metadata search over SQLite. Returns full documents (not
    chunks) so callers get the entire content for a matching document.
    """
    store = get_store()
    return store.search_by_metadata(
        service=service,
        document_type=document_type,
        date_from=date_from,
        date_to=date_to,
        version=version,
        historical_year=historical_year,
        limit=limit,
    )
