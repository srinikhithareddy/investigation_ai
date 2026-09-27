"""
Final Answer Generation Agent.
"""
from __future__ import annotations

import json
import re
from datetime import date
from typing import Optional

from app.llm.client import LLMError, get_llm_client
from app.retrieval.temporal import document_topic_key, historical_year_from_text

ANSWER_SYSTEM = """You are writing the final answer for an internal engineering investigation \
assistant. You are given the original question and everything the investigation discovered: evidence \
documents, a timeline, contradictions, similar-incident comparisons, and any unresolved gaps.

Write an answer that:
  1. Directly answers the question.
  2. Explains the reasoning, citing document IDs naturally in prose (e.g. "INC-1042 reports...").
  3. Mentions relevant dates and versions when they matter to the reasoning.
  4. Explains any contradictions rather than silently picking one side.
  5. Explicitly states uncertainty or gaps where the evidence does not fully answer the question.
  6. NEVER invents documents, dates, versions, or facts not present in the provided evidence.
  7. If the evidence is insufficient to answer the question (e.g. no matching documents, or the \
documents describe different services/failure mechanisms than what's being asked about), say so \
plainly instead of guessing.
    8. Treat explicit status and validity metadata as authoritative for currentness. Never infer that a \
document is obsolete from age alone. Prefer a uniquely active version for current questions; use a \
superseded version when it matches the requested historical period. If lifecycle metadata does not \
establish which conflicting version is current, preserve the versions and explicitly state that uncertainty.
    9. Use accumulated KNOWN facts and hypotheses from all search rounds; do not answer from only the most \
recent search. Distinguish established evidence from unconfirmed hypotheses and unresolved gaps.

Also decide a confidence level for the answer, based on evidence quality (not how the answer "feels"):
  - "high": strong, direct, unambiguous evidence for the core claims
  - "medium": reasonable evidence but with some gaps, indirect inference, or minor ambiguity
  - "low": thin or tangential evidence; the answer is a best guess
  - "insufficient": the evidence does not meaningfully address the question (e.g. no relevant documents, \
or the retrieved documents describe unrelated services/mechanisms)

Respond with ONLY a JSON object:
{
  "answer": string,       // the full prose answer, several sentences to a short paragraph
  "confidence": "high" | "medium" | "low" | "insufficient"
}"""


def generate_answer(
    question: str,
    entities: dict,
    evidence: list[dict],
    contradictions: list[dict],
    timeline: list[dict],
    similar_incidents: list[dict],
    gaps: list[str],
    known_facts: Optional[list[str]] = None,
    hypotheses: Optional[list[str]] = None,
    search_history: Optional[list[dict]] = None,
) -> dict:
    client = get_llm_client()

    def _version_uncertainty_note() -> str:
        historical_question = historical_year_from_text(question) is not None or bool(
            re.search(r"\b(historical(?:ly)?|previously|used to|what was|in the past)\b", question, re.IGNORECASE)
        )
        explicitly_current = bool(
            re.search(r"\b(current|currently|latest|now|today)\b", question, re.IGNORECASE)
        )
        current_question = explicitly_current or not historical_question
        groups: dict[tuple[str, str, str], list[dict]] = {}
        for document in evidence:
            if document.get("version"):
                groups.setdefault(document_topic_key(document), []).append(document)

        for documents in groups.values():
            versions = sorted({str(document["version"]) for document in documents})
            if not current_question:
                continue
            active_versions = {
                str(document["version"])
                for document in documents
                if str(document.get("status") or "").strip().lower() == "active"
                and not document.get("superseded_by")
                and (not document.get("valid_from") or document["valid_from"] <= date.today().isoformat())
                and (not document.get("valid_until") or document["valid_until"] >= date.today().isoformat())
            }
            if len(active_versions) != 1:
                if len(versions) < 2:
                    return (
                        f"The available evidence includes version {versions[0]}, but its lifecycle "
                        "metadata does not establish whether it is current."
                    )
                return (
                    "The retrieved evidence includes versions "
                    f"{', '.join(versions)}, but its lifecycle metadata does not establish "
                    "which version is current."
                )
        return ""

    temporal_uncertainty = _version_uncertainty_note()

    def _with_temporal_uncertainty(text: str) -> str:
        if temporal_uncertainty and temporal_uncertainty.lower() not in text.lower():
            return f"{text.rstrip()} {temporal_uncertainty}"
        return text

    def _slim(d: dict) -> dict:
        return {
            "document_id": d.get("document_id"),
            "title": d.get("title"),
            "type": d.get("type"),
            "service": d.get("service"),
            "date": d.get("date"),
            "document_date": d.get("document_date"),
            "version": d.get("version"),
            "status": d.get("status"),
            "superseded_by": d.get("superseded_by"),
            "valid_from": d.get("valid_from"),
            "valid_until": d.get("valid_until"),
            "content": (d.get("content") or "")[:1200],
        }

    payload = {
        "question": question,
        "entities": entities,
        "evidence": [_slim(d) for d in evidence],
        "timeline": timeline,
        "contradictions": contradictions,
        "similar_incidents": similar_incidents,
        "unresolved_gaps": gaps,
        "known_facts": known_facts or [],
        "hypotheses": hypotheses or [],
        "search_history": search_history or [],
    }

    if not evidence:
        return {
            "answer": (
                "No relevant documents were found for this question. The available document set does "
                "not contain evidence that addresses it, so no evidence-backed answer can be given."
            ),
            "confidence": "insufficient",
        }

    try:
        result = client.complete_json(system=ANSWER_SYSTEM, user=json.dumps(payload), max_tokens=1500)
        answer = result.get("answer", "").strip()
        confidence = result.get("confidence", "").strip().lower()
        if confidence not in {"high", "medium", "low", "insufficient"}:
            confidence = "low" if answer else "insufficient"
        if not answer:
            answer = "The investigation could not produce a grounded answer from the available evidence."
            confidence = "insufficient"
        return {"answer": _with_temporal_uncertainty(answer), "confidence": confidence}
    except LLMError as exc:
        return {
            "answer": _with_temporal_uncertainty(
                "The investigation gathered evidence, but the final answer could not be generated due "
                f"to an LLM error ({exc}). Please review the evidence and trace below."
            ),
            "confidence": "insufficient",
        }
