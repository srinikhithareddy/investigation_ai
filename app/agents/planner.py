"""
Investigation Planner Agent.

Responsible for:
  1. Analyzing the raw question into structured entities (question analysis node).
  2. Generating the initial set of search queries.
  3. Generating follow-up queries based on evidence gathered so far and
     identified gaps (query expansion node).
"""
from __future__ import annotations

import json
from typing import Optional

from app.llm.client import LLMError, get_llm_client

QUESTION_ANALYSIS_SYSTEM = """You are an incident-investigation assistant. Extract structured \
information from an engineer's natural language investigation question.

Respond with ONLY a JSON object (no markdown fences, no commentary) with this shape:
{
  "service": string or null,
  "date": string or null (ISO format YYYY-MM-DD if a specific date is mentioned),
  "date_range": {"from": string or null, "to": string or null},
  "versions": [string],
  "symptoms": [string],
  "suspected_causes": [string],
  "investigation_targets": [string]  // things the user wants checked, e.g. "deployment relationship", "previous similar incidents", "recommended remediation"
  "keywords": [string],  // important search keywords, entities, component names
  "requested_comparisons": [string]  // e.g. "compare to past incidents"
}

Only include information actually implied by the question. Use null / empty lists when something \
isn't mentioned. Do not guess a date or service if none is stated."""

INITIAL_QUERY_SYSTEM = """You generate search queries for a document retrieval system over internal \
engineering documents (incidents, deployments, postmortems, troubleshooting guides).

Given structured entities extracted from a user's question, produce 2-4 short, specific search \
queries that together would retrieve the most relevant evidence.

Respond with ONLY a JSON object: {"queries": [string, string, ...]}"""

FOLLOWUP_QUERY_SYSTEM = """You are directing an ongoing investigation over internal engineering \
documents. You will be given the original question, the entities already identified, a summary of \
evidence already found, and a list of gaps/unanswered questions.

Generate 1-3 NEW, more specific search queries that would help close those gaps. Do not repeat \
queries that have already been run or only returned already-known documents. Use known facts, \
hypotheses, remaining gaps, and per-query outcomes to make each query specific to missing information. \
Base new queries on discovered evidence (e.g. a version number, a root cause, a related component), \
not generic rephrasing.

Respond with ONLY a JSON object: {"queries": [string, ...]}

If no further searches would plausibly help (evidence is exhausted or sufficient), respond with \
{"queries": []}"""


def analyze_question(question: str) -> dict:
    """Extract structured entities from the raw question via the LLM."""
    client = get_llm_client()
    try:
        result = client.complete_json(
            system=QUESTION_ANALYSIS_SYSTEM,
            user=f"Question: {question}",
            max_tokens=800,
        )
    except LLMError:
        # Degrade gracefully: fall back to an empty-but-valid entity structure.
        result = {}

    return {
        "service": result.get("service"),
        "date": result.get("date"),
        "date_range": result.get("date_range") or {"from": None, "to": None},
        "versions": result.get("versions") or [],
        "symptoms": result.get("symptoms") or [],
        "suspected_causes": result.get("suspected_causes") or [],
        "investigation_targets": result.get("investigation_targets") or [],
        "keywords": result.get("keywords") or [],
        "requested_comparisons": result.get("requested_comparisons") or [],
    }


def generate_initial_queries(question: str, entities: dict) -> list[str]:
    client = get_llm_client()
    try:
        result = client.complete_json(
            system=INITIAL_QUERY_SYSTEM,
            user=json.dumps({"question": question, "entities": entities}),
            max_tokens=400,
        )
        queries = [q for q in result.get("queries", []) if isinstance(q, str) and q.strip()]
        if queries:
            return queries
    except LLMError:
        pass

    # Deterministic fallback so the system still works if the LLM call fails.
    fallback = [question]
    if entities.get("service"):
        parts = [entities["service"]]
        parts.extend(entities.get("symptoms", []))
        fallback.append(" ".join(parts))
    return fallback


def generate_followup_queries(
    question: str,
    entities: dict,
    evidence_summary: str,
    gaps: list[str],
    already_executed: list[str],
    known_facts: Optional[list[str]] = None,
    hypotheses: Optional[list[str]] = None,
    search_history: Optional[list[dict]] = None,
) -> list[str]:
    if not gaps:
        return []

    client = get_llm_client()
    try:
        result = client.complete_json(
            system=FOLLOWUP_QUERY_SYSTEM,
            user=json.dumps(
                {
                    "question": question,
                    "entities": entities,
                    "evidence_summary": evidence_summary,
                    "known_facts": known_facts or [],
                    "hypotheses": hypotheses or [],
                    "gaps": gaps,
                    "already_executed_queries": already_executed,
                    "search_history": search_history or [],
                }
            ),
            max_tokens=400,
        )
        queries = [q for q in result.get("queries", []) if isinstance(q, str) and q.strip()]
        # Filter out near-duplicates of already-executed queries
        normalize = lambda query: " ".join(query.casefold().strip(" .?!").split())
        seen = {normalize(query) for query in already_executed}
        seen.update(
            normalize(entry["query"])
            for entry in search_history or []
            if entry.get("query")
        )
        unique_queries = []
        for query in queries:
            normalized = normalize(query)
            if normalized and normalized not in seen:
                unique_queries.append(query)
                seen.add(normalized)
        return unique_queries
    except LLMError:
        return []
