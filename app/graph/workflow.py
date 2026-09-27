"""
LangGraph investigation workflow.

Graph shape (mirrors the architecture diagram):

    analyze_question
          |
          v
        search  <---------------------+
          |                           |
          v                           |
    analyze_evidence                  |
          |                           |
          v                           |
  detect_contradictions               |
          |                           |
          v                           |
    similar_incidents                 |
          |                           |
          v                           |
   [decision: need more evidence?] ---+ (expand_queries -> search)
          |
          v (no)
    generate_answer
          |
          v
         END
"""
from __future__ import annotations

from typing import Literal

from app.agents import analyzer, answer as answer_agent, contradiction, investigator, planner
from app.config import settings
from app.graph.state import InvestigationState
from app.retrieval.temporal import historical_year_from_text


def _append_trace(state: InvestigationState, message: str) -> list[str]:
    steps = list(state.get("investigation_steps", []))
    steps.append(message)
    return steps


def _merge_unique(existing: list[str], additions: list[str]) -> list[str]:
    merged = list(existing)
    seen = {item.casefold().strip() for item in existing}
    for item in additions:
        if not isinstance(item, str) or not item.strip():
            continue
        normalized = item.casefold().strip()
        if normalized not in seen:
            merged.append(item.strip())
            seen.add(normalized)
    return merged


def _docs_to_evidence(documents: dict[str, dict]) -> list[dict]:
    evidence = []
    for doc in documents.values():
        evidence.append(
            {
                "document_id": doc["document_id"],
                "title": doc.get("title") or "",
                "type": doc.get("type") or "",
                "service": doc.get("service"),
                "date": doc.get("date"),
                "version": doc.get("version"),
                "document_date": doc.get("document_date"),
                "status": doc.get("status"),
                "superseded_by": doc.get("superseded_by"),
                "valid_from": doc.get("valid_from"),
                "valid_until": doc.get("valid_until"),
                "content": doc.get("content") or "",
                "_score": doc.get("_score", 0.0),
            }
        )
    evidence.sort(key=lambda e: e.get("_score", 0.0), reverse=True)
    return evidence


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------


def analyze_question_node(state: InvestigationState) -> dict:
    question = state["question"]
    entities = planner.analyze_question(question)
    historical_year = historical_year_from_text(question)
    if historical_year is not None:
        entities["historical_year"] = historical_year
    queries = planner.generate_initial_queries(question, entities)

    summary_bits = []
    if entities.get("service"):
        summary_bits.append(entities["service"])
    if entities.get("date"):
        summary_bits.append(entities["date"])
    summary_bits.extend(entities.get("symptoms", []))
    summary = ", ".join(summary_bits) if summary_bits else "no specific entities detected"

    trace = _append_trace(state, f"Parsed question: {summary}.")
    trace.append(f"Planned initial search queries: {queries}.")

    return {
        "entities": entities,
        "original_question": question,
        "investigation_plan": {
            "objective": question,
            "targets": list(entities.get("investigation_targets", [])),
            "initial_queries": list(queries),
        },
        "search_queries": queries,
        "current_queries": list(queries),
        "executed_queries": [],
        "search_history": [],
        "discovered_document_ids": [],
        "retrieved_documents": [],
        "evidence": [],
        "known_facts": [],
        "hypotheses": list(entities.get("suspected_causes", [])),
        "evidence_gaps": [],
        "findings_history": [],
        "iteration": 0,
        "investigation_iteration": 0,
        "max_iterations": max(1, settings.max_investigation_iterations),
        "investigation_steps": trace,
    }


def search_node(state: InvestigationState) -> dict:
    queries = state.get("current_queries", state.get("search_queries", []))
    entities = state.get("entities", {})
    existing_raw = {d["document_id"]: d for d in state.get("retrieved_documents", [])}
    iteration = state.get("investigation_iteration", state.get("iteration", 0)) + 1
    search_history = list(state.get("search_history", []))

    updated_docs, search_trace = investigator.run_searches(
        queries,
        entities,
        existing_raw,
        search_history=search_history,
        iteration=iteration,
    )
    new_document_ids = [document_id for document_id in updated_docs if document_id not in existing_raw]
    discovered_document_ids = list(state.get("discovered_document_ids", []))
    discovered_document_ids = _merge_unique(discovered_document_ids, list(updated_docs))
    last_round_found_new = bool(new_document_ids)

    trace = list(state.get("investigation_steps", []))
    if iteration == 1:
        trace.append("Iteration 1: Initial searches from the original question.")
    else:
        trace.append(
            f"Iteration {iteration}: Targeted searches based on prior findings and evidence gaps."
        )
    trace.extend(search_trace)

    executed = list(state.get("executed_queries", []))
    executed = _merge_unique(executed, queries)

    evidence = _docs_to_evidence(updated_docs)

    return {
        "retrieved_documents": list(updated_docs.values()),
        "evidence": evidence,
        "executed_queries": executed,
        "search_queries": [],
        "current_queries": [],
        "search_history": search_history,
        "discovered_document_ids": discovered_document_ids,
        "new_document_ids": new_document_ids,
        "iteration": iteration,
        "investigation_iteration": iteration,
        "last_round_found_new": last_round_found_new,
        "investigation_steps": trace,
    }


def analyze_evidence_node(state: InvestigationState) -> dict:
    evidence = state.get("evidence", [])
    timeline = analyzer.build_timeline(evidence)

    trace = list(state.get("investigation_steps", []))
    if timeline:
        trace.append(f"Built timeline from {len(timeline)} dated document(s).")

    last_round_found_new = state.get("last_round_found_new", bool(evidence))

    iteration = state.get("investigation_iteration", state.get("iteration", 0))
    previous_known_facts = state.get("known_facts", [])
    previous_hypotheses = state.get("hypotheses", [])
    new_document_ids = state.get("new_document_ids", [])
    gap_result = analyzer.analyze_gaps(
        question=state.get("original_question", state["question"]),
        entities=state.get("entities", {}),
        evidence=evidence,
        iteration=iteration,
        max_iterations=state.get("max_iterations", settings.max_investigation_iterations),
        last_round_found_new=last_round_found_new,
        known_facts=previous_known_facts,
        hypotheses=previous_hypotheses,
        search_history=state.get("search_history", []),
        new_document_ids=new_document_ids,
    )

    gaps = gap_result.get("gaps", [])
    if gap_result.get("reasoning") == "Gap analysis unavailable (LLM error).":
        gaps = list(state.get("evidence_gaps", state.get("gaps", gaps)))
    needs_more = gap_result.get("needs_more_evidence", False)
    reported_facts = gap_result.get("known_facts", [])
    if not reported_facts:
        reported_facts = [
            f"{doc.get('document_id')}: {doc.get('title') or 'Untitled'}"
            + (f" ({doc.get('version')})" if doc.get("version") else "")
            + f" — {(doc.get('content') or '')[:180].strip()}"
            for doc in evidence
            if doc.get("document_id") in set(new_document_ids)
        ]
    known_facts = _merge_unique(previous_known_facts, reported_facts)
    hypotheses = _merge_unique(previous_hypotheses, gap_result.get("hypotheses", []))
    findings_history = list(state.get("findings_history", []))
    findings_history.append(
        {
            "iteration": iteration,
            "new_document_ids": list(new_document_ids),
            "known_facts": list(known_facts),
            "hypotheses": list(hypotheses),
            "evidence_gaps": list(gaps),
        }
    )
    investigation_plan = dict(state.get("investigation_plan", {}))
    investigation_plan["latest_evidence_gaps"] = list(gaps)
    investigation_plan["completed_iterations"] = iteration

    if gaps:
        trace.append(f"Iteration {iteration} UNKNOWN / EVIDENCE GAPS: {gaps}.")
    else:
        trace.append(f"Iteration {iteration} UNKNOWN / EVIDENCE GAPS: none remaining.")
    trace.append(
        f"Iteration {iteration} KNOWN: "
        + ("; ".join(known_facts) if known_facts else "no evidence-backed facts established yet")
        + "."
    )

    return {
        "timeline": timeline,
        "gaps": gaps,
        "evidence_gaps": list(gaps),
        "known_facts": known_facts,
        "hypotheses": hypotheses,
        "findings_history": findings_history,
        "investigation_plan": investigation_plan,
        "needs_more_evidence": needs_more,
        "investigation_steps": trace,
    }


def contradiction_node(state: InvestigationState) -> dict:
    evidence = state.get("evidence", [])
    contradictions = contradiction.detect_contradictions(evidence)

    trace = list(state.get("investigation_steps", []))
    if contradictions:
        titles = [c["title"] for c in contradictions]
        trace.append(f"Detected {len(contradictions)} contradiction(s): {titles}.")
    else:
        trace.append("No contradictions detected among current evidence.")

    return {"contradictions": contradictions, "investigation_steps": trace}


def similar_incidents_node(state: InvestigationState) -> dict:
    entities = state.get("entities", {})
    evidence = state.get("evidence", [])
    question = state["question"]

    wants_comparison = bool(entities.get("requested_comparisons")) or any(
        "previous" in t.lower() or "before" in t.lower() or "similar" in t.lower() or "seen this" in t.lower()
        for t in entities.get("investigation_targets", [])
    ) or "before" in question.lower() or "previous" in question.lower() or "seen this" in question.lower()

    trace = list(state.get("investigation_steps", []))

    if not wants_comparison or len(evidence) < 2:
        trace.append("Historical comparison not requested or insufficient evidence for comparison.")
        return {"similar_incidents": [], "investigation_steps": trace}

    # "Current" evidence = documents matching the question's date/service most closely;
    # candidates = the rest (e.g. postmortems from other dates).
    date = entities.get("date")
    service = entities.get("service")

    current = [
        d for d in evidence if (not date or d.get("date") == date) and (not service or d.get("service") == service)
    ]
    if not current:
        current = evidence[:1]

    current_ids = {d["document_id"] for d in current}
    candidates = [d for d in evidence if d["document_id"] not in current_ids]

    comparisons = analyzer.analyze_similar_incidents(question, current, candidates)

    # Attach classification back onto lightweight incident summaries
    by_id = {d["document_id"]: d for d in evidence}
    similar_incidents = []
    for c in comparisons:
        doc = by_id.get(c["document_id"])
        if not doc:
            continue
        similar_incidents.append(
            {
                "document_id": doc["document_id"],
                "title": doc.get("title"),
                "date": doc.get("date"),
                "version": doc.get("version"),
                "document_date": doc.get("document_date"),
                "status": doc.get("status"),
                "superseded_by": doc.get("superseded_by"),
                "valid_from": doc.get("valid_from"),
                "valid_until": doc.get("valid_until"),
                "classification": c["classification"],
                "reasoning": c.get("reasoning", ""),
            }
        )

    if similar_incidents:
        trace.append(
            "Compared current evidence against "
            f"{len(candidates)} other document(s) for historical similarity: "
            + ", ".join(f"{s['document_id']}={s['classification']}" for s in similar_incidents)
            + "."
        )
    else:
        trace.append("No prior incidents available for historical comparison.")

    return {"similar_incidents": similar_incidents, "investigation_steps": trace}


def expand_queries_node(state: InvestigationState) -> dict:
    question = state.get("original_question", state["question"])
    entities = state.get("entities", {})
    evidence = state.get("evidence", [])
    gaps = state.get("evidence_gaps", state.get("gaps", []))
    executed = state.get("executed_queries", [])

    evidence_summary = "; ".join(
        f"{e['document_id']} ({e.get('date') or 'no date'}, {e.get('version') or 'no version'}): "
        f"{(e.get('content') or '')[:150]}"
        for e in evidence[:8]
    )

    new_queries = planner.generate_followup_queries(
        question,
        entities,
        evidence_summary,
        gaps,
        executed,
        known_facts=state.get("known_facts", []),
        hypotheses=state.get("hypotheses", []),
        search_history=state.get("search_history", []),
    )

    trace = list(state.get("investigation_steps", []))
    if new_queries:
        iteration = state.get("investigation_iteration", state.get("iteration", 0)) + 1
        trace.append(
            f"Iteration {iteration}: Generated targeted searches from prior known facts and gaps: "
            f"{new_queries}."
        )
    else:
        trace.append("No productive follow-up searches could be generated; ending search phase.")

    return {
        "search_queries": new_queries,
        "current_queries": list(new_queries),
        "investigation_steps": trace,
    }


def generate_answer_node(state: InvestigationState) -> dict:
    result = answer_agent.generate_answer(
        question=state.get("original_question", state["question"]),
        entities=state.get("entities", {}),
        evidence=state.get("evidence", []),
        contradictions=state.get("contradictions", []),
        timeline=state.get("timeline", []),
        similar_incidents=state.get("similar_incidents", []),
        gaps=state.get("gaps", []),
        known_facts=state.get("known_facts", []),
        hypotheses=state.get("hypotheses", []),
        search_history=state.get("search_history", []),
    )

    trace = list(state.get("investigation_steps", []))
    trace.append(f"Generated final evidence-backed answer with confidence={result['confidence']}.")

    return {
        "final_answer": result["answer"],
        "confidence": result["confidence"],
        "investigation_steps": trace,
    }


def route_after_expansion(state: InvestigationState) -> Literal["search_again", "answer"]:
    return "search_again" if state.get("search_queries") else "answer"


# ---------------------------------------------------------------------------
# Conditional routing
# ---------------------------------------------------------------------------


def decide_next(state: InvestigationState) -> Literal["search_again", "answer"]:
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", settings.max_investigation_iterations)
    needs_more = state.get("needs_more_evidence", False)
    gaps = state.get("gaps", [])

    if iteration >= max_iterations:
        return "answer"
    if needs_more and gaps:
        return "search_again"
    return "answer"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

_compiled_graph = None


def build_graph():
    from langgraph.graph import END, StateGraph

    workflow = StateGraph(InvestigationState)

    workflow.add_node("analyze_question", analyze_question_node)
    workflow.add_node("search", search_node)
    workflow.add_node("analyze_evidence", analyze_evidence_node)
    workflow.add_node("detect_contradictions", contradiction_node)
    workflow.add_node("analyze_similar_incidents", similar_incidents_node)
    workflow.add_node("expand_queries", expand_queries_node)
    workflow.add_node("generate_answer", generate_answer_node)

    workflow.set_entry_point("analyze_question")
    workflow.add_edge("analyze_question", "search")
    workflow.add_edge("search", "analyze_evidence")
    workflow.add_edge("analyze_evidence", "detect_contradictions")
    workflow.add_edge("detect_contradictions", "analyze_similar_incidents")
    workflow.add_conditional_edges(
        "analyze_similar_incidents",
        decide_next,
        {"search_again": "expand_queries", "answer": "generate_answer"},
    )
    workflow.add_conditional_edges(
        "expand_queries",
        route_after_expansion,
        {"search_again": "search", "answer": "generate_answer"},
    )
    workflow.add_edge("generate_answer", END)

    return workflow.compile()


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_investigation(question: str) -> InvestigationState:
    graph = get_graph()
    initial_state: InvestigationState = {"question": question}
    final_state = graph.invoke(initial_state, config={"recursion_limit": 50})
    return final_state
