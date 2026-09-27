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

    def run_searches(queries, parsed, existing, **kwargs):
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

    original_question = (
        "Why did Order API latency increase on 2026-09-16, and was it related to the latest deployment?"
    )
    initial_documents = {
        "INC-1042": {
            "document_id": "INC-1042", "title": "Order API latency spike",
            "type": "incident_report", "service": "orders-api", "date": "2026-09-16",
            "version": "v2.8.1", "content": "Latency increased after the v2.8.1 deployment.", "_score": 0.9,
        },
        "DEP-882": {
            "document_id": "DEP-882", "title": "Orders deployment",
            "type": "deployment_note", "service": "orders-api", "date": "2026-09-15",
            "version": "v2.8.1", "content": "The release changed pooling and caching.", "_score": 0.8,
        },
    }
    eng_document = {
        "document_id": "ENG-DISC-19", "title": "Connection pool investigation",
        "type": "engineering_discussion", "service": "orders-api", "date": "2026-09-16",
        "version": "v2.8.1", "content": "The maximum connection pool size was reduced.", "_score": 0.85,
    }
    prior_document = {
        "document_id": "PM-211", "title": "Previous latency incident",
        "type": "postmortem", "service": "orders-api", "date": "2026-05-03",
        "version": "v2.6.0", "content": "A migration exhausted the database connection pool.", "_score": 0.7,
    }
    entities = {
        "service": "orders-api", "date": "2026-09-16",
        "investigation_targets": ["deployment relationship", "prior incidents"],
        "requested_comparisons": [], "symptoms": ["high latency"],
    }
    monkeypatch.setattr(workflow.planner, "analyze_question", lambda question: entities)
    monkeypatch.setattr(
        workflow.planner,
        "generate_initial_queries",
        lambda question, parsed: ["orders-api latency 2026-09-16 deployment"],
    )
    search_calls = []

    def run_searches(queries, parsed, existing, search_history=None, iteration=1):
        existing_ids = set(existing)
        search_calls.append((iteration, list(queries), existing_ids, list(search_history or [])))
        if iteration == 1:
            found = initial_documents
        elif iteration == 2:
            assert {"INC-1042", "DEP-882"} <= existing_ids
            found = {"INC-1042": initial_documents["INC-1042"], "ENG-DISC-19": eng_document}
        else:
            assert {"INC-1042", "DEP-882", "ENG-DISC-19"} <= existing_ids
            found = {"INC-1042": initial_documents["INC-1042"], "PM-211": prior_document}

        new_ids = [document_id for document_id in found if document_id not in existing_ids]
        search_history.append({
            "iteration": iteration,
            "query": queries[0],
            "document_ids": list(found),
            "new_document_ids": new_ids,
            "produced_new_evidence": bool(new_ids),
            "skipped": False,
        })
        return {**existing, **found}, [f"searched {queries[0]}"]

    monkeypatch.setattr(workflow.investigator, "run_searches", run_searches)
    monkeypatch.setattr(workflow.analyzer, "build_timeline", lambda evidence: evidence)
    monkeypatch.setattr(workflow.contradiction, "detect_contradictions", lambda evidence: [])
    monkeypatch.setattr(workflow.analyzer, "analyze_similar_incidents", lambda *args: [])
    gap_rounds = []

    def analyze_gaps(**kwargs):
        gap_rounds.append(kwargs["last_round_found_new"])
        if kwargs["iteration"] == 1:
            assert {item["document_id"] for item in kwargs["evidence"]} == {"INC-1042", "DEP-882"}
            return {
                "known_facts": ["INC-1042 and DEP-882 link the latency increase to v2.8.1."],
                "hypotheses": ["A connection-pooling change may have caused the latency."],
                "gaps": ["Which v2.8.1 change increased connection wait time?"],
                "needs_more_evidence": True,
            }
        if kwargs["iteration"] == 2:
            assert {item["document_id"] for item in kwargs["evidence"]} == {
                "INC-1042", "DEP-882", "ENG-DISC-19"
            }
            assert "INC-1042 and DEP-882 link the latency increase to v2.8.1." in kwargs["known_facts"]
            assert kwargs["new_document_ids"] == ["ENG-DISC-19"]
            return {
                "known_facts": ["ENG-DISC-19 confirms the maximum connection pool size was reduced."],
                "hypotheses": ["Reduced pool capacity explains the elevated wait time."],
                "gaps": ["Was connection-pool exhaustion seen in a prior incident?"],
                "needs_more_evidence": True,
            }
        assert {item["document_id"] for item in kwargs["evidence"]} == {
            "INC-1042", "DEP-882", "ENG-DISC-19", "PM-211"
        }
        assert len(kwargs["search_history"]) == 3
        return {
            "known_facts": ["PM-211 documents prior connection-pool exhaustion during a migration."],
            "hypotheses": [],
            "gaps": [],
            "needs_more_evidence": False,
        }

    monkeypatch.setattr(workflow.analyzer, "analyze_gaps", analyze_gaps)

    followup_calls = []

    def generate_followups(question, parsed, evidence_summary, gaps, executed, **kwargs):
        followup_calls.append({
            "question": question,
            "evidence_summary": evidence_summary,
            "gaps": list(gaps),
            "executed": list(executed),
            **kwargs,
        })
        if len(followup_calls) == 1:
            assert kwargs["known_facts"] == ["INC-1042 and DEP-882 link the latency increase to v2.8.1."]
            assert gaps == ["Which v2.8.1 change increased connection wait time?"]
            return ["v2.8.1 connection pooling configuration orders-api"]
        assert "ENG-DISC-19 confirms the maximum connection pool size was reduced." in kwargs["known_facts"]
        assert gaps == ["Was connection-pool exhaustion seen in a prior incident?"]
        return ["orders-api previous connection pool exhaustion postmortem"]

    monkeypatch.setattr(workflow.planner, "generate_followup_queries", generate_followups)
    answer_payloads = []

    def generate_answer(**kwargs):
        answer_payloads.append(kwargs)
        return {"answer": "The deployment reduced pool size; prior evidence records similar exhaustion.", "confidence": "high"}

    monkeypatch.setattr(workflow.answer_agent, "generate_answer", generate_answer)

    result = workflow.build_graph().invoke(
        {"question": original_question}
    )

    assert [call[0] for call in search_calls] == [1, 2, 3]
    assert [len(call[3]) for call in search_calls] == [0, 1, 2]
    assert search_calls[1][1] == ["v2.8.1 connection pooling configuration orders-api"]
    assert search_calls[2][1] == ["orders-api previous connection pool exhaustion postmortem"]
    assert gap_rounds == [True, True, True]
    assert result["iteration"] == result["investigation_iteration"] == 3
    assert result["original_question"] == original_question
    assert result["investigation_plan"]["objective"] == original_question
    assert result["investigation_plan"]["completed_iterations"] == 3
    assert set(result["discovered_document_ids"]) == {"INC-1042", "DEP-882", "ENG-DISC-19", "PM-211"}
    assert len(result["evidence"]) == 4
    assert len({item["document_id"] for item in result["evidence"]}) == 4
    assert len(result["search_history"]) == 3
    assert len(result["findings_history"]) == 3
    assert result["evidence_gaps"] == []
    assert result["current_queries"] == []
    assert len(followup_calls) == 2
    assert len(answer_payloads) == 1
    assert {item["document_id"] for item in answer_payloads[0]["evidence"]} == {
        "INC-1042", "DEP-882", "ENG-DISC-19", "PM-211"
    }
    assert len(answer_payloads[0]["known_facts"]) == 3
    assert any("Iteration 2: Targeted searches" in line for line in result["investigation_steps"])
    assert any("Iteration 3: Targeted searches" in line for line in result["investigation_steps"])
    assert any("Iteration 1 KNOWN" in line for line in result["investigation_steps"])
    assert any("Iteration 1 UNKNOWN / EVIDENCE GAPS" in line for line in result["investigation_steps"])


def test_graph_handles_empty_retrieval(monkeypatch):
    from app.graph import workflow

    calls = []
    monkeypatch.setattr(workflow.planner, "analyze_question", lambda question: {})
    monkeypatch.setattr(workflow.planner, "generate_initial_queries", lambda *args: ["unknown service"])
    monkeypatch.setattr(
        workflow.investigator,
        "run_searches",
        lambda *args, **kwargs: (calls.append("search") or ({}, ["no results"])),
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
    monkeypatch.setattr(workflow.planner, "generate_followup_queries", lambda *args, **kwargs: [])

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
        lambda *args, **kwargs: ["more detail " + str(len(calls))],
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
        lambda queries, entities, existing, **kwargs: (calls.append("search") or ({**existing, **documents}, [])),
    )

    result = workflow.build_graph().invoke(
        {"question": "Have we seen an orders-api incident like this before?"}
    )

    assert calls == ["search"]
    assert result["evidence"]
    assert result["confidence"] == "insufficient"
    assert "could not be generated" in result["final_answer"]


def test_answer_surfaces_conflicting_versions_without_current_status(monkeypatch):
    import json

    from app.agents import answer

    class FakeLLM:
        def complete_json(self, **kwargs):
            payload = json.loads(kwargs["user"])
            assert {item["version"] for item in payload["evidence"]} == {"v1", "v2"}
            return {"answer": "The documents describe two architecture versions.", "confidence": "medium"}

    monkeypatch.setattr(answer, "get_llm_client", lambda: FakeLLM())
    evidence = [
        {
            "document_id": "ARCH-2018",
            "title": "Orders API architecture 2018",
            "type": "architecture",
            "service": "orders-api",
            "version": "v1",
            "document_date": "2018-03-01",
            "content": "Legacy architecture.",
        },
        {
            "document_id": "ARCH-2026",
            "title": "Orders API architecture 2026",
            "type": "architecture",
            "service": "orders-api",
            "version": "v2",
            "document_date": "2026-03-01",
            "content": "Newer architecture.",
        },
    ]

    result = answer.generate_answer(
        question="What is the current architecture of orders-api?",
        entities={},
        evidence=evidence,
        contradictions=[],
        timeline=[],
        similar_incidents=[],
        gaps=[],
    )

    assert "v1, v2" in result["answer"]
    assert "does not establish which version is current" in result["answer"]


def test_current_answer_marks_single_unclassified_version_uncertain(monkeypatch):
    from app.agents import answer

    class FakeLLM:
        def complete_json(self, **kwargs):
            return {"answer": "The architecture document describes a pooled database client.", "confidence": "medium"}

    monkeypatch.setattr(answer, "get_llm_client", lambda: FakeLLM())
    result = answer.generate_answer(
        question="What is the current architecture of orders-api?",
        entities={},
        evidence=[
            {
                "document_id": "ARCH-2018",
                "title": "Orders API architecture",
                "type": "architecture",
                "service": "orders-api",
                "version": "v1",
                "document_date": "2018-03-01",
                "content": "Uses a pooled database client.",
            }
        ],
        contradictions=[],
        timeline=[],
        similar_incidents=[],
        gaps=[],
    )

    assert "does not establish whether it is current" in result["answer"]


def test_searches_propagate_historical_year_and_single_version(monkeypatch):
    from app.agents import investigator

    calls = []
    monkeypatch.setattr(
        investigator,
        "search_documents_tool",
        lambda **kwargs: calls.append(kwargs) or [],
    )

    investigator.run_searches(
        ["orders architecture"],
        {"service": "orders-api", "historical_year": 2016, "versions": ["v1"]},
        {},
    )

    assert calls[0]["historical_year"] == 2016
    assert calls[0]["version"] == "v1"


def test_search_history_skips_repeated_query_and_marks_duplicate_results(monkeypatch):
    from app.agents import investigator

    document = {"document_id": "INC-1", "title": "Latency", "_score": 0.8}
    search_calls = []
    monkeypatch.setattr(
        investigator,
        "search_documents_tool",
        lambda **kwargs: search_calls.append(kwargs["query"]) or [document],
    )
    search_history = []

    first, _ = investigator.run_searches(
        ["orders api latency"], {}, {}, search_history=search_history, iteration=1
    )
    second, _ = investigator.run_searches(
        [" Orders   API Latency "],
        {},
        first,
        search_history=search_history,
        iteration=2,
    )
    third, _ = investigator.run_searches(
        ["orders api connection issue"],
        {},
        second,
        search_history=search_history,
        iteration=3,
    )

    assert search_calls == ["orders api latency", "orders api connection issue"]
    assert list(third) == ["INC-1"]
    assert search_history[1]["skipped"] is True
    assert search_history[2]["new_document_ids"] == []
    assert search_history[2]["produced_new_evidence"] is False


def test_followup_planner_uses_findings_and_filters_repeated_queries(monkeypatch):
    import json

    from app.agents import planner

    captured = {}

    class FakeLLM:
        def complete_json(self, **kwargs):
            captured.update(json.loads(kwargs["user"]))
            return {
                "queries": [
                    "pool config v2.8.1",
                    "connection-pool remediation",
                    " CONNECTION-POOL remediation! ",
                ]
            }

    monkeypatch.setattr(planner, "get_llm_client", lambda: FakeLLM())
    queries = planner.generate_followup_queries(
        question="Why did orders-api latency increase?",
        entities={"service": "orders-api"},
        evidence_summary="INC-1042 linked the spike to v2.8.1.",
        gaps=["Which pooling setting changed?"],
        already_executed=["pool config v2.8.1"],
        known_facts=["INC-1042 followed the v2.8.1 deployment."],
        hypotheses=["Reduced pool size may explain the wait time."],
        search_history=[{"query": "pool config v2.8.1", "produced_new_evidence": False}],
    )

    assert queries == ["connection-pool remediation"]
    assert captured["known_facts"] == ["INC-1042 followed the v2.8.1 deployment."]
    assert captured["hypotheses"] == ["Reduced pool size may explain the wait time."]
    assert captured["gaps"] == ["Which pooling setting changed?"]
    assert captured["search_history"][0]["produced_new_evidence"] is False
