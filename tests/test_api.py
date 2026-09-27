import pytest
from types import SimpleNamespace
from fastapi.testclient import TestClient

from app.main import app
import app.api.routes as routes


@pytest.fixture()
def client():
    return TestClient(app)


def configure_healthy_backend(monkeypatch):
    monkeypatch.setattr(routes, "get_store", lambda: SimpleNamespace(count=lambda: 10))
    monkeypatch.setattr(routes, "get_collection", lambda: SimpleNamespace(count=lambda: 10))
    monkeypatch.setattr(routes, "is_embedding_model_loaded", lambda: True)
    monkeypatch.setattr(routes, "settings", SimpleNamespace(gemini_api_key="test-key"))


def test_health_reports_backend_checks(client, monkeypatch):
    configure_healthy_backend(monkeypatch)

    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {
        "status": "ok",
        "checks": {
            "database": "ok",
            "vector_store": "ok",
            "embedding_model": "ready",
            "gemini": "configured",
        },
        "document_count": 10,
        "chunk_count": 10,
    }


def test_health_returns_503_when_a_backend_component_is_unavailable(client, monkeypatch):
    configure_healthy_backend(monkeypatch)
    monkeypatch.setattr(routes, "get_collection", lambda: (_ for _ in ()).throw(RuntimeError()))

    resp = client.get("/health")

    assert resp.status_code == 503
    assert resp.json()["status"] == "degraded"
    assert resp.json()["checks"]["vector_store"] == "unavailable"
    assert resp.json()["document_count"] == 10
    assert resp.json()["chunk_count"] is None


def test_health_returns_503_when_gemini_key_is_missing(client, monkeypatch):
    configure_healthy_backend(monkeypatch)
    monkeypatch.setattr(routes, "settings", SimpleNamespace(gemini_api_key=""))

    resp = client.get("/health")

    assert resp.status_code == 503
    assert resp.json()["checks"]["gemini"] == "missing_api_key"


def test_investigate_rejects_empty_question(client):
    resp = client.post("/investigate", json={"question": "   "})
    assert resp.status_code == 400


def test_investigate_rejects_missing_question(client):
    resp = client.post("/investigate", json={})
    assert resp.status_code == 400


def test_investigate_rejects_non_string_question(client):
    resp = client.post("/investigate", json={"question": 42})
    assert resp.status_code == 400


def test_investigate_happy_path(client, monkeypatch):
    configure_healthy_backend(monkeypatch)
    fake_state = {
        "final_answer": "The latency spike was associated with deployment v2.8.1 (INC-1042, DEP-882).",
        "confidence": "medium",
        "evidence": [
            {
                "document_id": "INC-1042",
                "title": "Order API latency spike",
                "type": "incident_report",
                "date": "2026-09-16",
                "document_date": "2026-09-16",
                "version": "v2.8.1",
                "status": "active",
                "valid_from": "2026-09-16",
                "content": "P95 latency increased...",
            }
        ],
        "contradictions": [],
        "investigation_steps": ["Parsed question.", "Searched documents.", "Generated final answer."],
    }

    called_with = []
    monkeypatch.setattr(
        routes,
        "run_investigation",
        lambda question: called_with.append(question) or fake_state,
    )

    resp = client.post(
        "/investigate",
        json={"question": "  Why did the Order API become slow?  "},
    )
    assert resp.status_code == 200
    assert called_with == ["Why did the Order API become slow?"]
    body = resp.json()
    assert set(body) == {"answer", "confidence", "evidence", "contradictions", "trace"}
    assert body["answer"] == fake_state["final_answer"]
    assert body["confidence"] == "medium"
    assert len(body["evidence"]) == 1
    assert body["evidence"][0]["document_id"] == "INC-1042"
    assert set(body["evidence"][0]) == {
        "document_id", "title", "type", "date", "document_date", "version",
        "status", "superseded_by", "valid_from", "valid_until", "content"
    }
    assert body["evidence"][0]["status"] == "active"
    assert body["evidence"][0]["valid_from"] == "2026-09-16"
    assert body["trace"] == fake_state["investigation_steps"]


def test_investigate_returns_500_on_internal_error(client, monkeypatch):
    def boom(question):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(routes, "run_investigation", boom)

    resp = client.post("/investigate", json={"question": "anything"})
    assert resp.status_code == 500
    assert "internal error" in resp.json()["detail"].lower()
