"""
Explicit, serializable state for the investigation LangGraph.
"""
from __future__ import annotations

from typing import Any, Optional, TypedDict


class InvestigationState(TypedDict, total=False):
    # Input
    question: str
    original_question: str

    # Question analysis output
    entities: dict[str, Any]

    # Search state
    search_queries: list[str]
    current_queries: list[str]
    executed_queries: list[str]
    search_history: list[dict]
    discovered_document_ids: list[str]
    new_document_ids: list[str]
    retrieved_documents: list[dict]  # deduplicated, ranked docs seen so far (raw, with _score)
    evidence: list[dict]  # cleaned evidence objects matching the public schema

    # Accumulated investigation knowledge
    investigation_plan: dict[str, Any]
    known_facts: list[str]
    hypotheses: list[str]
    evidence_gaps: list[str]
    findings_history: list[dict]

    # Analysis outputs
    contradictions: list[dict]
    timeline: list[dict]
    similar_incidents: list[dict]
    gaps: list[str]

    # Control flow
    investigation_steps: list[str]  # human-readable trace
    iteration: int
    investigation_iteration: int
    max_iterations: int
    needs_more_evidence: bool
    last_round_found_new: bool

    # Output
    confidence: str
    final_answer: str
