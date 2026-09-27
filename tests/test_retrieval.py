import os
import sqlite3
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


def test_build_chunks_preserves_optional_lifecycle_metadata():
    from app.ingestion.documents import build_chunks

    doc = {
        "document_id": "ARCH-1",
        "title": "Orders API architecture",
        "content": "Architecture notes.",
        "version": "v2",
        "document_date": "2026-04-01",
        "status": "active",
        "valid_from": "2026-04-01",
    }

    chunk = build_chunks(doc)[0]

    assert chunk["document_date"] == "2026-04-01"
    assert chunk["status"] == "active"
    assert chunk["valid_from"] == "2026-04-01"
    assert chunk["superseded_by"] is None


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


def test_ingest_persists_optional_lifecycle_metadata(tmp_path, tmp_store, monkeypatch):
    from app.ingestion import documents

    source = tmp_path / "documents.json"
    source.write_text(
        '[{"document_id":"ARCH-NEW","title":"Orders API architecture",'
        '"content":"Current design.","version":"v2","document_date":"2026-04-01",'
        '"status":"ACTIVE","superseded_by":null,"valid_from":"2026-04-01",'
        '"valid_until":null}]',
        encoding="utf-8",
    )
    indexed_chunks = []
    monkeypatch.setattr(documents, "upsert_chunks", indexed_chunks.extend)

    result = documents.ingest(path=str(source), store=tmp_store)

    stored = tmp_store.get_document("ARCH-NEW")
    assert result["ingested_documents"] == 1
    assert stored["version"] == "v2"
    assert stored["document_date"] == "2026-04-01"
    assert stored["status"] == "active"
    assert stored["valid_from"] == "2026-04-01"
    assert indexed_chunks[0]["status"] == "active"


def test_ingest_preserves_stored_lifecycle_when_legacy_source_omits_fields(
    tmp_path, tmp_store, monkeypatch
):
    from app.ingestion import documents

    tmp_store.upsert_document(
        DocumentRecord(
            document_id="ARCH-1",
            title="Old title",
            type="architecture",
            service="orders-api",
            date="2018-01-01",
            version="v1",
            content="Old architecture.",
            document_date="2018-01-01",
            status="superseded",
            superseded_by="ARCH-2",
            valid_from="2018-01-01",
            valid_until="2020-01-01",
        )
    )
    source = tmp_path / "legacy-documents.json"
    source.write_text(
        '[{"document_id":"ARCH-1","title":"Updated title","content":"Updated content."}]',
        encoding="utf-8",
    )
    indexed_chunks = []
    monkeypatch.setattr(documents, "upsert_chunks", indexed_chunks.extend)

    documents.ingest(path=str(source), store=tmp_store)

    stored = tmp_store.get_document("ARCH-1")
    assert stored["status"] == "superseded"
    assert stored["superseded_by"] == "ARCH-2"
    assert indexed_chunks[0]["status"] == "superseded"
    assert indexed_chunks[0]["superseded_by"] == "ARCH-2"


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


def test_sqlite_migrates_legacy_rows_without_lifecycle_metadata(tmp_path):
    db_path = tmp_path / "legacy.db"
    connection = sqlite3.connect(db_path)
    connection.execute(
        """CREATE TABLE documents (
            document_id TEXT PRIMARY KEY, title TEXT NOT NULL, type TEXT,
            service TEXT, date TEXT, version TEXT, content TEXT NOT NULL, source TEXT
        )"""
    )
    connection.execute(
        "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("OLD-1", "Legacy architecture", "architecture", "orders-api", "2016-01-01", "v1", "old", None),
    )
    connection.commit()
    connection.close()

    store = SQLiteStore(db_path=str(db_path))
    migrated = store.get_document("OLD-1")

    assert migrated["document_date"] is None
    assert migrated["status"] is None
    assert migrated["superseded_by"] is None
    assert migrated["valid_from"] is None
    assert migrated["valid_until"] is None
    assert migrated["content"] == "old"


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


def _temporal_record(
    document_id,
    version,
    document_date,
    status=None,
    superseded_by=None,
    valid_from=None,
    valid_until=None,
    legacy_date=None,
):
    return DocumentRecord(
        document_id=document_id,
        title=f"Orders API architecture {version}",
        type="architecture",
        service="orders-api",
        date=legacy_date,
        version=version,
        content=f"Architecture described by {version}.",
        document_date=document_date,
        status=status,
        superseded_by=superseded_by,
        valid_from=valid_from,
        valid_until=valid_until,
    )


def _temporal_search(store, monkeypatch, query):
    from app.retrieval import hybrid

    monkeypatch.setattr(hybrid, "get_store", lambda: store)
    monkeypatch.setattr(hybrid, "semantic_retrieve", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        hybrid,
        "metadata_retrieve",
        lambda **kwargs: store.search_by_metadata(**kwargs),
    )
    return hybrid.search_documents(
        query,
        service="orders-api",
        document_type="architecture",
        top_k=10,
    )


def test_current_query_prefers_latest_active_document(tmp_store, monkeypatch):
    tmp_store.upsert_documents(
        [
            _temporal_record("ARCH-2018", "v1", "2018-03-01", status="active"),
            _temporal_record("ARCH-2024", "v2", "2024-02-01", status="active"),
            _temporal_record(
                "ARCH-2026-OLD", "v3", "2026-01-01", status="superseded", superseded_by="ARCH-2026"
            ),
            _temporal_record("ARCH-2026", "v4", "2026-06-01", status="active"),
        ]
    )

    results = _temporal_search(
        tmp_store, monkeypatch, "What is the current architecture of orders-api?"
    )

    assert results[0]["document_id"] == "ARCH-2026"
    assert next(item for item in results if item["document_id"] == "ARCH-2026-OLD")["_score"] < results[0]["_score"]


def test_historical_query_retrieves_matching_old_document(tmp_store, monkeypatch):
    tmp_store.upsert_documents(
        [
            _temporal_record(
                "ARCH-2016", "v1", "2016-04-12", status="superseded",
                superseded_by="ARCH-2026", valid_from="2016-04-12", valid_until="2020-01-01",
            ),
            _temporal_record("ARCH-2026", "v2", "2026-02-01", status="active", valid_from="2026-02-01"),
        ]
    )

    results = _temporal_search(
        tmp_store, monkeypatch, "What was the architecture of orders-api in 2016?"
    )

    assert [item["document_id"] for item in results] == ["ARCH-2016"]
    assert results[0]["status"] == "superseded"


def test_historical_query_respects_explicit_validity_over_publication_date(tmp_store, monkeypatch):
    tmp_store.upsert_document(
        _temporal_record(
            "ARCH-REVISED",
            "v2",
            "2016-04-12",
            status="active",
            valid_from="2026-01-01",
        )
    )

    results = _temporal_search(
        tmp_store, monkeypatch, "What was the architecture of orders-api in 2016?"
    )

    assert results == []


def test_superseded_record_is_downgraded_for_current_query(tmp_store, monkeypatch):
    tmp_store.upsert_documents(
        [
            _temporal_record("ARCH-OLD", "v1", "2024-01-01", status="superseded", superseded_by="ARCH-NEW"),
            _temporal_record("ARCH-NEW", "v2", "2025-01-01", status="active"),
        ]
    )

    results = _temporal_search(
        tmp_store, monkeypatch, "What is the current architecture of orders-api?"
    )

    assert results[0]["document_id"] == "ARCH-NEW"
    assert next(item for item in results if item["document_id"] == "ARCH-OLD")["_score"] < results[0]["_score"]


def test_old_document_without_lifecycle_metadata_remains_usable(tmp_store, monkeypatch):
    tmp_store.upsert_document(
        _temporal_record("ARCH-LEGACY", "v1", None, legacy_date="2016-06-30")
    )

    results = _temporal_search(
        tmp_store, monkeypatch, "What was the architecture of orders-api in 2016?"
    )

    assert len(results) == 1
    assert results[0]["document_id"] == "ARCH-LEGACY"
    assert results[0]["status"] is None

    current_results = _temporal_search(
        tmp_store, monkeypatch, "What is the current architecture of orders-api?"
    )
    assert any(item["document_id"] == "ARCH-LEGACY" for item in current_results)


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

    def chunk(chunk_id, document_id, content, **metadata):
        return {
            "chunk_id": chunk_id,
            "document_id": document_id,
            "content": content,
            **metadata,
        }

    chroma.upsert_chunks(
        [
            chunk("INC-1::chunk-0", "INC-1", "old first chunk"),
            chunk("INC-1::chunk-1", "INC-1", "old second chunk"),
            chunk("INC-2::chunk-0", "INC-2", "unrelated document"),
        ]
    )
    replacement = [
        chunk(
            "INC-1::chunk-0",
            "INC-1",
            "short replacement",
            version="v2",
            document_date="2026-01-01",
            status="active",
            valid_from="2026-01-01",
        )
    ]
    chroma.upsert_chunks(replacement)
    chroma.upsert_chunks(replacement)

    assert set(collection.items) == {"INC-1::chunk-0", "INC-2::chunk-0"}
    assert collection.items["INC-1::chunk-0"]["document"] == "short replacement"
    assert collection.items["INC-1::chunk-0"]["metadata"]["status"] == "active"


def test_chroma_semantic_search_returns_lifecycle_metadata(monkeypatch):
    from app.storage import chroma

    class Collection:
        def count(self):
            return 1

        def query(self, **kwargs):
            return {
                "ids": [["ARCH-1::chunk-0"]],
                "documents": [["Architecture notes."]],
                "metadatas": [[
                    {
                        "document_id": "ARCH-1",
                        "title": "Orders API architecture",
                        "type": "architecture",
                        "service": "orders-api",
                        "date": "2026-04-01",
                        "version": "v2",
                        "document_date": "2026-04-01",
                        "status": "active",
                        "superseded_by": "",
                        "valid_from": "2026-04-01",
                        "valid_until": "",
                    }
                ]],
                "distances": [[0.2]],
            }

    monkeypatch.setattr(chroma, "get_collection", lambda: Collection())
    monkeypatch.setattr(chroma, "embed_texts", lambda texts: [[0.1, 0.2]])

    result = chroma.semantic_search("architecture")[0]

    assert result["document_id"] == "ARCH-1"
    assert result["document_date"] == "2026-04-01"
    assert result["status"] == "active"
    assert result["valid_from"] == "2026-04-01"
    assert result["superseded_by"] is None
    assert result["distance"] == 0.2


def test_multi_year_query_is_not_arbitrarily_filtered():
    from app.retrieval.temporal import historical_year_from_text

    assert historical_year_from_text("Compare the architecture in 2018 and 2026") is None
    assert historical_year_from_text("What was the architecture in 2016?") == 2016
