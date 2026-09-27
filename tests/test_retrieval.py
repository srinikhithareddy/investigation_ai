import os
import tempfile

import pytest

from app.ingestion.documents import build_chunks, chunk_text, validate_document
from app.storage.sqlite import DocumentRecord, SQLiteStore


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def test_chunk_text_short_text_returns_single_chunk():
    text = "This is a short document."
    chunks = chunk_text(text, chunk_size=1200, overlap=150)
    assert chunks == [text]


def test_chunk_text_long_text_splits_into_multiple_chunks():
    paragraph = "Sentence about latency and errors. " * 40  # ~1440 chars
    chunks = chunk_text(paragraph, chunk_size=300, overlap=50)
    assert len(chunks) > 1
    # No chunk should wildly exceed the target size
    assert all(len(c) <= 400 for c in chunks)


def test_chunk_text_rejects_invalid_overlap():
    with pytest.raises(ValueError, match="overlap"):
        chunk_text("content", chunk_size=100, overlap=100)


def test_build_chunks_preserves_parent_document_identity():
    doc = {
        "document_id": "INC-9999",
        "title": "Test incident",
        "type": "incident_report",
        "service": "orders-api",
        "date": "2026-09-16",
        "version": "v2.8.1",
        "content": "Latency spike investigation. " * 100,
    }
    chunks = build_chunks(doc)
    assert len(chunks) >= 1
    for c in chunks:
        assert c["document_id"] == "INC-9999"
        assert c["service"] == "orders-api"
        assert c["chunk_id"].startswith("INC-9999::chunk-")


def test_ingest_collapses_duplicate_document_ids_to_latest_record(
    tmp_path, tmp_store, monkeypatch
):
    from app.ingestion import documents

    source = tmp_path / "documents.json"
    source.write_text(
        '[{"document_id":"DOC-1","title":"old","content":"old"},'
        '{"document_id":" DOC-1 ","title":"new","content":"new"}]',
        encoding="utf-8",
    )
    indexed_chunks = []
    monkeypatch.setattr(documents, "upsert_chunks", indexed_chunks.extend)

    result = documents.ingest(path=str(source), store=tmp_store)

    assert result["ingested_documents"] == 1
    assert result["total_documents_in_store"] == 1
    assert tmp_store.get_document("DOC-1")["title"] == "new"
    assert [chunk["content"] for chunk in indexed_chunks] == ["new"]


def test_validate_document_flags_missing_fields():
    errors = validate_document({"document_id": "X"})
    assert errors  # missing title/content

    errors_ok = validate_document(
        {"document_id": "X", "title": "T", "content": "C"}
    )
    assert errors_ok == []


# ---------------------------------------------------------------------------
# SQLite store
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_store():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test.db")
        yield SQLiteStore(db_path=db_path)


def test_sqlite_upsert_and_get(tmp_store):
    record = DocumentRecord(
        document_id="INC-1",
        title="Test",
        type="incident_report",
        service="orders-api",
        date="2026-09-16",
        version="v2.8.1",
        content="content here",
    )
    tmp_store.upsert_document(record)

    fetched = tmp_store.get_document("INC-1")
    assert fetched is not None
    assert fetched["title"] == "Test"
    assert tmp_store.count() == 1

    # Upserting again with the same id should not create a duplicate
    tmp_store.upsert_document(record)
    assert tmp_store.count() == 1


def test_sqlite_metadata_filtering(tmp_store):
    tmp_store.upsert_document(
        DocumentRecord("A", "A title", "incident_report", "orders-api", "2026-09-16", "v2.8.1", "c")
    )
    tmp_store.upsert_document(
        DocumentRecord("B", "B title", "postmortem", "catalog-api", "2026-05-01", "v1.0.0", "c")
    )

    results = tmp_store.search_by_metadata(service="orders-api")
    assert len(results) == 1
    assert results[0]["document_id"] == "A"

    results_type = tmp_store.search_by_metadata(document_type="postmortem")
    assert len(results_type) == 1
    assert results_type[0]["document_id"] == "B"


# ---------------------------------------------------------------------------
# Hybrid ranking helper functions (pure logic, no external services)
# ---------------------------------------------------------------------------


def test_date_proximity_score_exact_and_far():
    from app.retrieval.hybrid import _date_proximity_score

    assert _date_proximity_score("2026-09-16", "2026-09-16") == 1.0
    assert _date_proximity_score("2020-01-01", "2026-09-16") < 0.1
    assert _date_proximity_score(None, "2026-09-16") == 0.0


def test_semantic_score_converts_distance():
    from app.retrieval.hybrid import _semantic_score

    assert _semantic_score(0.0) == 1.0
    assert _semantic_score(None) == 0.0
    assert 0.0 <= _semantic_score(0.5) <= 1.0


def test_semantic_retrieve_reads_flat_chroma_fields_and_scores(monkeypatch):
    from app.retrieval import semantic

    hits = [
        {
            "chunk_id": "OTHER::chunk-0",
            "document_id": "OTHER",
            "title": "Other service",
            "type": "incident_report",
            "service": "catalog-api",
            "date": "2026-09-16",
            "version": "v1",
            "content": "other content",
            "distance": 0.1,
        },
        {
            "chunk_id": "INC-1::chunk-0",
            "document_id": "INC-1",
            "title": "Matching service",
            "type": "incident_report",
            "service": "orders_api",
            "date": "2026-09-16",
            "version": "v2",
            "content": "matching content",
            "distance": 0.25,
        },
    ]
    monkeypatch.setattr(semantic, "_semantic_search", lambda *args, **kwargs: hits)

    results = semantic.semantic_retrieve("latency", service="orders-api")

    assert results[0]["document_id"] == "INC-1"
    assert results[0]["score"] == 0.75
    assert results[0]["distance"] == 0.25
    assert results[0]["title"] == "Matching service"
    assert results[0]["type"] == "incident_report"
    assert results[0]["service"] == "orders_api"
    assert results[0]["date"] == "2026-09-16"
    assert results[0]["version"] == "v2"
    assert results[0]["content"] == "matching content"
    assert results[1]["score"] == 0.9


def test_semantic_retrieve_always_returns_score(monkeypatch):
    from app.retrieval import semantic

    hit = {"document_id": "INC-1", "distance": 0.4}
    monkeypatch.setattr(semantic, "_semantic_search", lambda *args, **kwargs: [hit])

    results = semantic.semantic_retrieve("latency")

    assert results[0]["score"] == 0.6


def test_chroma_upsert_removes_stale_chunks_and_is_idempotent(monkeypatch):
    from app.storage import chroma

    class InMemoryCollection:
        def __init__(self):
            self.items = {}

        def upsert(self, ids, documents, metadatas, embeddings):
            for chunk_id, document, metadata, embedding in zip(
                ids, documents, metadatas, embeddings
            ):
                self.items[chunk_id] = {
                    "document": document,
                    "metadata": metadata,
                    "embedding": embedding,
                }

        def get(self, where, include):
            document_id = where["document_id"]
            return {
                "ids": [
                    chunk_id
                    for chunk_id, item in self.items.items()
                    if item["metadata"]["document_id"] == document_id
                ]
            }

        def delete(self, ids):
            for chunk_id in ids:
                self.items.pop(chunk_id, None)

    collection = InMemoryCollection()
    monkeypatch.setattr(chroma, "get_collection", lambda: collection)
    monkeypatch.setattr(chroma, "embed_texts", lambda texts: [[0.1] for _ in texts])

    def chunk(chunk_id, document_id, content):
        return {
            "chunk_id": chunk_id,
            "document_id": document_id,
            "content": content,
        }

    chroma.upsert_chunks(
        [
            chunk("INC-1::chunk-0", "INC-1", "old first chunk"),
            chunk("INC-1::chunk-1", "INC-1", "old second chunk"),
            chunk("INC-2::chunk-0", "INC-2", "unrelated document"),
        ]
    )
    replacement = [chunk("INC-1::chunk-0", "INC-1", "short replacement")]
    chroma.upsert_chunks(replacement)
    chroma.upsert_chunks(replacement)

    assert set(collection.items) == {"INC-1::chunk-0", "INC-2::chunk-0"}
    assert collection.items["INC-1::chunk-0"]["document"] == "short replacement"
