"""
Evidence Analyzer Agent.

Three responsibilities, matching the architecture diagram's "Evidence
Analyzer" branch (contradiction detection lives in agents/contradiction.py):

  1. Timeline Analysis   - deterministic, built from document dates (no LLM
                            invention of times/causality).
  2. Similar Incident     - LLM classifies retrieved documents relative to the
     Analysis               current incident as: same / similar / related /
                            unrelated, with justification grounded in the
                            documents' actual content.
  3. Gap Analysis / Loop  - LLM identifies unanswered investigation targets
     Decision                and whether another search round is warranted.
"""
from __future__ import annotations

import json
from typing import Optional

from app.llm.client import LLMError, get_llm_client


def build_timeline(documents: list[dict]) -> list[dict]:
    """Deterministic timeline from documents that have a date. No invented times."""
    dated = [d for d in documents if d.get("date")]
    dated.sort(key=lambda d: d["date"])
    timeline = []
    for d in dated:
        timeline.append(
            {
                "date": d["date"],
                "document_id": d["document_id"],
                "title": d.get("title"),
                "type": d.get("type"),
                "version": d.get("version"),
                "summary": (d.get("content") or "")[:220].strip(),
            }
        )
    return timeline


SIMILAR_INCIDENT_SYSTEM = """You compare a set of candidate documents against the CURRENT incident \
being investigated, to determine how each candidate relates to it.

Classify each candidate document into exactly one category:
  - "same_incident": the document is clearly describing the exact same incident/event as the current one
  - "similar_incident": a different incident, but with meaningfully similar symptoms/service/failure mechanism
  - "related_but_different": touches the same service/area but a different failure mechanism or cause
  - "unrelated": not meaningfully connected

Base your classification ONLY on the actual content provided - service, symptoms, version, failure \
mechanism, affected component, root cause. Do NOT assume two incidents are the same just because they \
are semantically similar or involve the same service. Only classify as "same_incident" if the evidence \
clearly indicates they refer to the identical event (e.g. matching incident ID, matching exact date/time \
and description).

Respond with ONLY a JSON object:
{
  "comparisons": [
    {
      "document_id": string,
      "classification": "same_incident" | "similar_incident" | "related_but_different" | "unrelated",
      "reasoning": string  // 1-2 sentences, grounded in the actual document content and differences (service, date, version, symptoms, cause)
    }
  ]
}"""


def analyze_similar_incidents(question: str, current_evidence: list[dict], candidates: list[dict]) -> list[dict]:
    """
    Classify candidate documents relative to the "current" incident evidence.
    `current_evidence` is the evidence most closely tied to the question's
    stated date/service; `candidates` are other documents (e.g. postmortems)
    that might be historically similar.
    """
    if not candidates:
        return []

    client = get_llm_client()

    def _slim(d: dict) -> dict:
        return {
            "document_id": d.get("document_id"),
            "title": d.get("title"),
            "type": d.get("type"),
            "service": d.get("service"),
            "date": d.get("date"),
            "version": d.get("version"),
            "content": (d.get("content") or "")[:800],
        }

    payload = {
        "question": question,
        "current_incident_evidence": [_slim(d) for d in current_evidence],
        "candidate_documents": [_slim(d) for d in candidates],
    }

    try:
        result = client.complete_json(
            system=SIMILAR_INCIDENT_SYSTEM, user=json.dumps(payload), max_tokens=1200
        )
        comparisons = result.get("comparisons", [])
        valid = {"same_incident", "similar_incident", "related_but_different", "unrelated"}
        return [c for c in comparisons if c.get("classification") in valid and c.get("document_id")]
    except LLMError:
        return []


GAP_ANALYSIS_SYSTEM = """You are assessing the state of an ongoing document-based investigation.

Given the original question, the entities/investigation targets extracted from it, and a summary of \
evidence gathered so far, determine:
  1. Which investigation targets (things the user wants answered) remain unaddressed by the current evidence.
  2. Whether another round of searching is likely to find more relevant evidence (vs. the evidence being \
as complete as it will get, or the topic simply not being covered by the document set).

Respond with ONLY a JSON object:
{
    "known_facts": [string],      // established facts grounded in evidence IDs
    "hypotheses": [string],       // plausible explanations that remain unconfirmed
  "gaps": [string],              // specific unanswered questions/targets, empty list if none
  "needs_more_evidence": boolean,
  "reasoning": string             // brief explanation
}

Be conservative about requesting more searches: if the evidence already substantively addresses the \
question, or previous search rounds returned nothing new/relevant, set needs_more_evidence to false."""


def analyze_gaps(
    question: str,
    entities: dict,
    evidence: list[dict],
    iteration: int,
    max_iterations: int,
    last_round_found_new: bool,
    known_facts: Optional[list[str]] = None,
    hypotheses: Optional[list[str]] = None,
    search_history: Optional[list[dict]] = None,
    new_document_ids: Optional[list[str]] = None,
) -> dict:
    if iteration >= max_iterations:
        return {
            "known_facts": known_facts or [],
            "hypotheses": hypotheses or [],
            "gaps": [],
            "needs_more_evidence": False,
            "reasoning": "Maximum investigation iterations reached.",
        }

    client = get_llm_client()

    evidence_summary = [
        {
            "document_id": e.get("document_id"),
            "title": e.get("title"),
            "type": e.get("type"),
            "service": e.get("service"),
            "date": e.get("date"),
            "version": e.get("version"),
            "excerpt": (e.get("content") or "")[:300],
        }
        for e in evidence
    ]

    payload = {
        "question": question,
        "entities": entities,
        "investigation_targets": entities.get("investigation_targets", []),
        "previously_known_facts": known_facts or [],
        "previous_hypotheses": hypotheses or [],
        "previous_search_history": search_history or [],
        "newly_discovered_document_ids": new_document_ids or [],
        "evidence_gathered": evidence_summary,
        "iteration": iteration,
        "max_iterations": max_iterations,
        "last_round_found_new_documents": last_round_found_new,
    }

    try:
        result = client.complete_json(system=GAP_ANALYSIS_SYSTEM, user=json.dumps(payload), max_tokens=600)
        return {
            "known_facts": [
                item for item in result.get("known_facts", [])
                if isinstance(item, str) and item.strip()
            ],
            "hypotheses": [
                item for item in result.get("hypotheses", [])
                if isinstance(item, str) and item.strip()
            ],
            "gaps": result.get("gaps") or [],
            "needs_more_evidence": bool(result.get("needs_more_evidence")),
            "reasoning": result.get("reasoning", ""),
        }
    except LLMError:
        return {
            "known_facts": known_facts or [],
            "hypotheses": hypotheses or [],
            "gaps": [],
            "needs_more_evidence": False,
            "reasoning": "Gap analysis unavailable (LLM error).",
        }
