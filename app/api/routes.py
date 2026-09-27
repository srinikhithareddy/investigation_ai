from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Response

from app.config import settings
from app.graph.workflow import run_investigation
from app.models.request import InvestigateRequest
from app.models.response import ContradictionItem, EvidenceItem, HealthResponse, InvestigateResponse
from app.storage.chroma import get_collection, is_embedding_model_loaded
from app.storage.sqlite import get_store

logger = logging.getLogger("investigation")

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(response: Response) -> HealthResponse:
    document_count = None
    chunk_count = None

    try:
        document_count = get_store().count()
        database_status = "ok"
    except Exception:
        database_status = "unavailable"

    try:
        chunk_count = get_collection().count()
        vector_store_status = "ok"
    except Exception:
        vector_store_status = "unavailable"

    checks = {
        "database": database_status,
        "vector_store": vector_store_status,
        "embedding_model": "ready" if is_embedding_model_loaded() else "not_loaded",
        "gemini": "configured" if settings.gemini_api_key else "missing_api_key",
    }
    healthy = all(
        checks[name] == expected
        for name, expected in (
            ("database", "ok"),
            ("vector_store", "ok"),
            ("embedding_model", "ready"),
            ("gemini", "configured"),
        )
    )

    response.status_code = 200 if healthy else 503
    return HealthResponse(
        status="ok" if healthy else "degraded",
        checks=checks,
        document_count=document_count,
        chunk_count=chunk_count,
    )


@router.post("/investigate", response_model=InvestigateResponse)
def investigate(request: InvestigateRequest) -> InvestigateResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="`question` must be a non-empty string.")

    try:
        state = run_investigation(question)
    except Exception:
        logger.exception("Investigation failed for question: %s", question)
        raise HTTPException(status_code=500, detail="Investigation failed due to an internal error.")

    evidence = [
        EvidenceItem(
            document_id=e["document_id"],
            title=e.get("title") or "",
            type=e.get("type") or "",
            date=e.get("date"),
            document_date=e.get("document_date"),
            version=e.get("version"),
            status=e.get("status"),
            superseded_by=e.get("superseded_by"),
            valid_from=e.get("valid_from"),
            valid_until=e.get("valid_until"),
            content=e.get("content") or "",
        )
        for e in state.get("evidence", [])
    ]

    contradictions = [
        ContradictionItem(title=c["title"], description=c["description"])
        for c in state.get("contradictions", [])
    ]

    return InvestigateResponse(
        answer=state.get("final_answer") or "The investigation did not produce an answer.",
        confidence=state.get("confidence") or "insufficient",
        evidence=evidence,
        contradictions=contradictions,
        trace=state.get("investigation_steps", []),
    )
