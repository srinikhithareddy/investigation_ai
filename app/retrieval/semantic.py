from __future__ import annotations

from typing import Optional

from app.storage.chroma import semantic_search as _semantic_search


def distance_to_score(distance: Optional[float]) -> float:
    """Convert Chroma cosine distance to a bounded higher-is-better score."""
    if distance is None:
        return 0.0
    try:
        return max(0.0, min(1.0, 1.0 - float(distance)))
    except (TypeError, ValueError):
        return 0.0


def _normalize_service(service: Optional[str]) -> str:
    if not service:
        return ""
    return service.lower().strip().replace("_", "-").replace(" ", "-")


def semantic_retrieve(
    query: str,
    top_k: int = 10,
    service: Optional[str] = None
) -> list[dict]:
    """
    Semantic search over document chunks.

    Service is NOT used as a hard Chroma filter.
    Instead, semantic results are retrieved first and service
    relevance is applied afterward.
    """

    # Get semantic results without filtering by service
    results = _semantic_search(
        query,
        top_k=top_k,
        where=None
    )

    for result in results:
        result["score"] = distance_to_score(result.get("distance"))

    # No service specified → return normal semantic results
    if not service:
        return results

    requested_service = _normalize_service(service)

    results.sort(
        key=lambda result: (
            result["score"]
            + (0.2 if _normalize_service(result.get("service")) == requested_service else 0.0),
            result["score"],
        ),
        reverse=True,
    )

    return results