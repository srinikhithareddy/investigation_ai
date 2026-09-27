"""
Hybrid retrieval: combines semantic (Chroma) and metadata (SQLite) search,
deduplicates by document_id, and re-ranks using a scoring function that
considers semantic relevance, service match, date proximity, version match,
document type, and recency. This deliberately does NOT sort by raw vector
similarity alone.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from app.retrieval.metadata import metadata_retrieve
from app.retrieval.semantic import distance_to_score, semantic_retrieve
from app.storage.sqlite import get_store


def _parse_date(value: Optional[str]):
    if not value:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    # fall back to just the date portion (handles values with trailing "UTC", etc.)
    try:
        return datetime.strptime(value[:10], "%Y-%m-%d")
    except ValueError:
        return None


def _date_proximity_score(doc_date: Optional[str], anchor_date: Optional[str]) -> float:
    """Higher score for documents closer to the anchor date. 0..1 range."""
    if not doc_date or not anchor_date:
        return 0.0
    d1, d2 = _parse_date(doc_date), _parse_date(anchor_date)
    if not d1 or not d2:
        return 0.0
    delta_days = abs((d1 - d2).days)
    if delta_days == 0:
        return 1.0
    if delta_days <= 3:
        return 0.8
    if delta_days <= 14:
        return 0.5
    if delta_days <= 60:
        return 0.25
    return 0.05


def _recency_score(doc_date: Optional[str]) -> float:
    """Mild general recency boost independent of the anchor date."""
    d = _parse_date(doc_date)
    if not d:
        return 0.0
    days_old = max((datetime.utcnow() - d).days, 0)
    # decays slowly; documents within the last ~2 years get most of the boost
    return max(0.0, 1.0 - min(days_old, 730) / 730.0) * 0.3


def _semantic_score(distance: Optional[float]) -> float:
    return distance_to_score(distance)

def _normalize_service(service: Optional[str]) -> str:
    if not service:
        return ""

    return (
        service.lower()
        .strip()
        .replace("_", "-")
        .replace(" ", "-")
    )

def search_documents(
    query: str,
    service: Optional[str] = None,
    document_type: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    version: Optional[str] = None,
    anchor_date: Optional[str] = None,
    top_k: int = 10,
) -> list[dict]:
    """
    Hybrid retrieval combining semantic search and metadata filtering.

    Returns a ranked, deduplicated list of full documents (not chunks), each
    annotated with `_score` and `_score_breakdown` for transparency/debugging
    (these are internal fields, not part of the public evidence schema).
    """
    store = get_store()

    # 1. Semantic retrieval (over chunks)
    semantic_hits = semantic_retrieve(query, top_k=max(top_k * 3, 15), service=service) if query else []

    # 2. Metadata retrieval (over full documents)
    metadata_hits = metadata_retrieve(
        service=service,
        document_type=document_type,
        date_from=date_from,
        date_to=date_to,
        version=version,
        limit=max(top_k * 3, 20),
    )

    # 3. Merge candidates by document_id
    candidates: dict[str, dict] = {}
    semantic_best_distance: dict[str, float] = {}

    for hit in semantic_hits:
        doc_id = hit.get("document_id")
        if not doc_id:
            continue
        dist = hit.get("distance")
        if dist is not None:
            prev = semantic_best_distance.get(doc_id)
            if prev is None or dist < prev:
                semantic_best_distance[doc_id] = dist
        candidates.setdefault(doc_id, {"document_id": doc_id, "from_semantic": True, "from_metadata": False})

    for hit in metadata_hits:
        doc_id = hit.get("document_id")
        if not doc_id:
            continue
        entry = candidates.setdefault(doc_id, {"document_id": doc_id, "from_semantic": False, "from_metadata": True})
        entry["from_metadata"] = True

    if not candidates:
        return []

    # 4. Hydrate full documents from SQLite (single source of truth for content/metadata)
    full_docs = {d["document_id"]: d for d in store.get_documents(candidates.keys())}

    # 5. Score + rank
    ranked = []
    for doc_id, cand in candidates.items():
        doc = full_docs.get(doc_id)
        if not doc:
            continue

        distance = semantic_best_distance.get(doc_id)
        sem_score = _semantic_score(distance)
        normalized_query_service = _normalize_service(service)
        normalized_doc_service = _normalize_service(doc.get("service"))

        service_score = (
            1.0
            if normalized_query_service
            and normalized_doc_service
            and normalized_query_service == normalized_doc_service
            else 0.0
        )
        version_score = 1.0 if version and doc.get("version") == version else 0.0
        type_score = 1.0 if document_type and doc.get("type") and doc["type"].lower() == document_type.lower() else 0.0
        date_prox = _date_proximity_score(doc.get("date"), anchor_date or date_from or date_to)
        recency = _recency_score(doc.get("date"))
        metadata_match_bonus = 0.15 if cand.get("from_metadata") else 0.0

        total = (
            sem_score * 0.45
            + service_score * 0.15
            + version_score * 0.10
            + type_score * 0.05
            + date_prox * 0.15
            + recency * 0.05
            + metadata_match_bonus
        )

        ranked.append(
            {
                **doc,
                "distance": distance,
                "score": round(total, 4),
                "_score": round(total, 4),
                "_score_breakdown": {
                    "semantic": round(sem_score, 3),
                    "service_match": service_score,
                    "version_match": version_score,
                    "type_match": type_score,
                    "date_proximity": round(date_prox, 3),
                    "recency": round(recency, 3),
                    "metadata_match_bonus": metadata_match_bonus,
                },
            }
        )

    ranked.sort(key=lambda d: d["_score"], reverse=True)
    return ranked[:top_k]
