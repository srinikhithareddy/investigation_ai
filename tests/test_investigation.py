"""
Tests for graph node logic and decision routing. LLM-backed agent functions
are not invoked so these tests run deterministically without a live
GEMINI_API_KEY or network access.
"""
from types import SimpleNamespace

import pytest

from app.agents import analyzer
from app.graph.workflow import decide_next, _docs_to_evidence


def test_docs_to_evidence_sorts_by_score():
    docs = {
        "A": {"document_id": "A", "title": "a", "type": "t", "content": "c", "_score": 0.2},
        "B": {"document_id": "B", "title": "b", "type": "t", "content": "c", "_score": 0.9},
    }
    evidence = _docs_to_evidence(docs)
    assert [e["document_id"] for e in evidence] == ["B", "A"]


def test_decide_next_stops_at_max_iterations():
    state = {"iteration": 3, "max_iterations": 3, "needs_more_evidence": True, "gaps": ["x"]}
    assert decide_next(state) == "answer"


def test_decide_next_continues_when_gaps_and_budget_remain():
    state = {"iteration": 1, "max_iterations": 3, "needs_more_evidence": True, "gaps": ["x"]}
    assert decide_next(state) == "search_again"


def test_decide_next_stops_when_no_gaps():
    state = {"iteration": 1, "max_iterations": 3, "needs_more_evidence": False, "gaps": []}
    assert decide_next(state) == "answer"


def test_decide_next_stops_on_empty_gaps_even_if_flagged():
    state = {"iteration": 1, "max_iterations": 3, "needs_more_evidence": True, "gaps": []}
    assert decide_next(state) == "answer"


def test_build_timeline_sorts_chronologically_and_skips_undated():
    docs = [
        {"document_id": "B", "title": "b", "type": "t", "date": "2026-09-16", "version": None, "content": "c2"},
        {"document_id": "A", "title": "a", "type": "t", "date": "2026-09-15", "version": None, "content": "c1"},
        {"document_id": "C", "title": "c", "type": "t", "date": None, "version": None, "content": "c3"},
    ]
    timeline = analyzer.build_timeline(docs)
    assert [t["document_id"] for t in timeline] == ["A", "B"]


def test_similar_incidents_node_skips_when_not_requested(monkeypatch):
    from app.graph.workflow import similar_incidents_node

    state = {
        "question": "Why did the Order API become slow?",
        "entities": {"investigation_targets": [], "requested_comparisons": []},
        "evidence": [
            {"document_id": "A", "title": "a", "type": "t", "date": "2026-09-16", "content": "c"},
        ],
        "investigation_steps": [],
    }
    result = similar_incidents_node(state)
    assert result["similar_incidents"] == []


def test_similar_incidents_node_runs_when_comparison_requested(monkeypatch):
    from app.graph import workflow

    def fake_analyze_similar_incidents(question, current, candidates):
        return [
            {
                "document_id": candidates[0]["document_id"],
                "classification": "similar_incident",
                "reasoning": "same service, different root cause",
            }
        ]

    monkeypatch.setattr(analyzer, "analyze_similar_incidents", fake_analyze_similar_incidents)
    monkeypatch.setattr(workflow.analyzer, "analyze_similar_incidents", fake_analyze_similar_incidents)

    state = {
        "question": "Have we seen this before?",
        "entities": {
            "investigation_targets": ["previous similar incidents"],
            "requested_comparisons": [],
            "date": "2026-09-16",
            "service": "orders-api",
        },
        "evidence": [
            {
                "document_id": "INC-1042",
                "title": "current",
                "type": "incident_report",
                "date": "2026-09-16",
                "service": "orders-api",
                "content": "current incident",
            },
            {
                "document_id": "PM-211",
                "title": "past",
                "type": "postmortem",
                "date": "2026-05-03",
                "service": "orders-api",
                "content": "past incident, different cause",
            },
        ],
        "investigation_steps": [],
    }

    result = workflow.similar_incidents_node(state)
    assert len(result["similar_incidents"]) == 1
    assert result["similar_incidents"][0]["document_id"] == "PM-211"
    assert result["similar_incidents"][0]["classification"] == "similar_incident"


def _sample_documents():
    return {
        "INC-1042": {
            "document_id": "INC-1042",
            "title": "Order API latency spike",
            "type": "incident_report",
            "service": "orders-api",
            "date": "2026-09-16",
            "version": "v2.8.1",
            "content": "P95 latency increased after the deployment.",
            "_score": 0.9,
        },
        "PM-211": {
            "document_id": "PM-211",
            "title": "Previous latency incident",
            "type": "postmortem",
            "service": "orders-api",
            "date": "2026-05-03",
            "version": "v2.6.0",
            "content": "A migration exhausted the database connection pool.",
            "_score": 0.7,
        },
    }


def _configure_graph(monkeypatch, workflow, documents, calls):
    entities = {
        "service": "orders-api",
        "date": "2026-09-16",
        "investigation_targets": ["compare prior incidents"],
        "requested_comparisons": ["previous incidents"],
        "symptoms": ["high latency"],
    }
    monkeypatch.setattr(workflow.planner, "analyze_question", lambda question: entities)
    monkeypatch.setattr(
        workflow.planner,
        "generate_initial_queries",
        lambda question, parsed: ["orders api latency"],
    )

    def run_searches(queries, parsed, existing):
        calls.append("search")
        return {**existing, **documents}, [f"searched {queries}"]

    monkeypatch.setattr(workflow.investigator, "run_searches", run_searches)
    monkeypatch.setattr(workflow.analyzer, "build_timeline", lambda evidence: evidence)
    monkeypatch.setattr(workflow.contradiction, "detect_contradictions", lambda evidence: [])

    def compare(question, current, candidates):
        calls.append("similar_incidents")
        return [
            {
                "document_id": candidates[0]["document_id"],
                "classification": "similar_incident",
                "reasoning": "The earlier incident also involved connection-pool saturation.",
            }
        ]

    monkeypatch.setattr(workflow.analyzer, "analyze_similar_incidents", compare)
    monkeypatch.setattr(
        workflow.answer_agent,
        "generate_answer",
        lambda **kwargs: {"answer": "The deployment changed connection pooling.", "confidence": "medium"},
    )


def test_graph_runs_full_followup_cycle_and_tracks_new_evidence(monkeypatch):
    from app.graph import workflow

    calls = []
    documents = _sample_documents()
    _configure_graph(monkeypatch, workflow, documents, calls)
    gap_rounds = []

    def analyze_gaps(**kwargs):
        calls.append("analyze_evidence")
        gap_rounds.append(kwargs["last_round_found_new"])
        if kwargs["iteration"] == 1:
            return {"gaps": ["deployment configuration change"], "needs_more_evidence": True}
        return {"gaps": [], "needs_more_evidence": False}

    monkeypatch.setattr(workflow.analyzer, "analyze_gaps", analyze_gaps)

    def generate_followups(*args):
        calls.append("expand_queries")
        return ["connection pool configuration v2.8.1"]

    monkeypatch.setattr(workflow.planner, "generate_followup_queries", generate_followups)
    monkeypatch.setattr(
        workflow.contradiction,
        "detect_contradictions",
        lambda evidence: calls.append("contradictions") or [],
    )
    monkeypatch.setattr(workflow.answer_agent, "generate_answer", lambda **kwargs: (
        calls.append("generate_answer")
        or {"answer": "Answer grounded in evidence.", "confidence": "medium"}
    ))

    result = workflow.build_graph().invoke(
        {"question": "Have we seen orders-api latency like this before?"}
    )

    assert calls == [
        "search",
        "analyze_evidence",
        "contradictions",
        "similar_incidents",
        "expand_queries",
        "search",
        "analyze_evidence",
        "contradictions",
        "similar_incidents",
        "generate_answer",
    ]
    assert gap_rounds == [True, False]
    assert result["iteration"] == 2
    assert result["last_round_found_new"] is False
    assert result["similar_incidents"][0]["document_id"] == "PM-211"
    assert result["final_answer"] == "Answer grounded in evidence."


def test_graph_handles_empty_retrieval(monkeypatch):
    from app.graph import workflow

    calls = []
    monkeypatch.setattr(workflow.planner, "analyze_question", lambda question: {})
    monkeypatch.setattr(workflow.planner, "generate_initial_queries", lambda *args: ["unknown service"])
    monkeypatch.setattr(
        workflow.investigator,
        "run_searches",
        lambda *args: (calls.append("search") or ({}, ["no results"])),
    )
    monkeypatch.setattr(
        workflow.analyzer,
        "analyze_gaps",
        lambda **kwargs: {"gaps": [], "needs_more_evidence": False},
    )

    result = workflow.build_graph().invoke({"question": "Investigate an unknown service."})

    assert calls == ["search"]
    assert result["evidence"] == []
    assert result["confidence"] == "insufficient"
    assert "No relevant documents" in result["final_answer"]


def test_graph_ends_when_followup_expansion_is_empty(monkeypatch):
    from app.graph import workflow

    calls = []
    documents = _sample_documents()
    _configure_graph(monkeypatch, workflow, documents, calls)
    monkeypatch.setattr(
        workflow.analyzer,
        "analyze_gaps",
        lambda **kwargs: {"gaps": ["unanswered detail"], "needs_more_evidence": True},
    )
    monkeypatch.setattr(workflow.planner, "generate_followup_queries", lambda *args: [])

    result = workflow.build_graph().invoke({"question": "Have we seen this before?"})

    assert calls.count("search") == 1
    assert result["iteration"] == 1
    assert result["final_answer"]


def test_graph_enforces_maximum_search_iterations(monkeypatch):
    from app.graph import workflow

    calls = []
    documents = _sample_documents()
    _configure_graph(monkeypatch, workflow, documents, calls)
    monkeypatch.setattr(workflow, "settings", SimpleNamespace(max_investigation_iterations=2))
    monkeypatch.setattr(
        workflow.analyzer,
        "analyze_gaps",
        lambda **kwargs: {"gaps": ["more detail"], "needs_more_evidence": True},
    )
    monkeypatch.setattr(
        workflow.planner,
        "generate_followup_queries",
        lambda *args: ["more detail " + str(len(calls))],
    )

    result = workflow.build_graph().invoke({"question": "Investigate this service."})

    assert calls.count("search") == 2
    assert result["iteration"] == 2
    assert result["final_answer"]


def test_graph_degrades_gracefully_when_llm_calls_fail(monkeypatch):
    from app.agents import answer, contradiction, planner
    from app.graph import workflow
    from app.llm.client import LLMError

    class BrokenLLM:
        def complete_json(self, **kwargs):
            raise LLMError("simulated Gemini failure")

    calls = []
    documents = _sample_documents()
    for agent in (planner, analyzer, contradiction, answer):
        monkeypatch.setattr(agent, "get_llm_client", lambda: BrokenLLM())
    monkeypatch.setattr(
        workflow.investigator,
        "run_searches",
        lambda queries, entities, existing: (calls.append("search") or ({**existing, **documents}, [])),
    )

    result = workflow.build_graph().invoke(
        {"question": "Have we seen an orders-api incident like this before?"}
    )

    assert calls == ["search"]
    assert result["evidence"]
    assert result["confidence"] == "insufficient"
    assert "could not be generated" in result["final_answer"]
