"""
SQLite storage for document metadata + raw content.

This is the source of truth for structured metadata (service, date, version,
type) and is used for metadata-filtered retrieval. Chroma is used only for
semantic search over chunk embeddings; the full document record always lives
here.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

from app.config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    type TEXT,
    service TEXT,
    date TEXT,
    version TEXT,
    document_date TEXT,
    status TEXT,
    superseded_by TEXT,
    valid_from TEXT,
    valid_until TEXT,
    content TEXT NOT NULL,
    source TEXT
);

CREATE INDEX IF NOT EXISTS idx_documents_service ON documents(service);
CREATE INDEX IF NOT EXISTS idx_documents_type ON documents(type);
CREATE INDEX IF NOT EXISTS idx_documents_date ON documents(date);
CREATE INDEX IF NOT EXISTS idx_documents_version ON documents(version);
"""

LIFECYCLE_COLUMNS = {
    "document_date": "TEXT",
    "status": "TEXT",
    "superseded_by": "TEXT",
    "valid_from": "TEXT",
    "valid_until": "TEXT",
}


@dataclass
class DocumentRecord:
    document_id: str
    title: str
    type: Optional[str]
    service: Optional[str]
    date: Optional[str]
    version: Optional[str]
    content: str
    source: Optional[str] = None
    document_date: Optional[str] = None
    status: Optional[str] = None
    superseded_by: Optional[str] = None
    valid_from: Optional[str] = None
    valid_until: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "document_id": self.document_id,
            "title": self.title,
            "type": self.type,
            "service": self.service,
            "date": self.date,
            "version": self.version,
            "document_date": self.document_date,
            "status": self.status,
            "superseded_by": self.superseded_by,
            "valid_from": self.valid_from,
            "valid_until": self.valid_until,
            "content": self.content,
            "source": self.source,
        }


class SQLiteStore:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or settings.database_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA)
            existing_columns = {
                row["name"] for row in conn.execute("PRAGMA table_info(documents)")
            }
            for column, column_type in LIFECYCLE_COLUMNS.items():
                if column not in existing_columns:
                    conn.execute(
                        f"ALTER TABLE documents ADD COLUMN {column} {column_type}"
                    )
            conn.executescript(
                """
                CREATE INDEX IF NOT EXISTS idx_documents_document_date ON documents(document_date);
                CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
                CREATE INDEX IF NOT EXISTS idx_documents_validity ON documents(valid_from, valid_until);
                """
            )

    def upsert_document(self, doc: DocumentRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO documents (
                    document_id, title, type, service, date, version,
                    document_date, status, superseded_by, valid_from, valid_until,
                    content, source
                )
                VALUES (
                    :document_id, :title, :type, :service, :date, :version,
                    :document_date, :status, :superseded_by, :valid_from, :valid_until,
                    :content, :source
                )
                ON CONFLICT(document_id) DO UPDATE SET
                    title=excluded.title,
                    type=excluded.type,
                    service=excluded.service,
                    date=excluded.date,
                    version=excluded.version,
                    document_date=excluded.document_date,
                    status=excluded.status,
                    superseded_by=excluded.superseded_by,
                    valid_from=excluded.valid_from,
                    valid_until=excluded.valid_until,
                    content=excluded.content,
                    source=excluded.source
                """,
                doc.to_dict(),
            )

    def upsert_documents(self, docs: Iterable[DocumentRecord]) -> int:
        count = 0
        for doc in docs:
            self.upsert_document(doc)
            count += 1
        return count

    def get_document(self, document_id: str) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE document_id = ?", (document_id,)
            ).fetchone()
            return dict(row) if row else None

    def get_documents(self, document_ids: Iterable[str]) -> list[dict]:
        ids = list(document_ids)
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM documents WHERE document_id IN ({placeholders})", ids
            ).fetchall()
            return [dict(r) for r in rows]

    def all_documents(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM documents").fetchall()
            return [dict(r) for r in rows]

    def search_by_metadata(
        self,
        service: Optional[str] = None,
        document_type: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        version: Optional[str] = None,
        limit: int = 50,
        historical_year: Optional[int] = None,
    ) -> list[dict]:
        clauses = []
        params: list = []

        if service:
            clauses.append("LOWER(service) = LOWER(?)")
            params.append(service)
        if document_type:
            clauses.append("LOWER(type) = LOWER(?)")
            params.append(document_type)
        if version:
            clauses.append("version = ?")
            params.append(version)
        if date_from and not historical_year:
            clauses.append("COALESCE(document_date, date) >= ?")
            params.append(date_from)
        if date_to and not historical_year:
            clauses.append("COALESCE(document_date, date) <= ?")
            params.append(date_to)

        if historical_year:
            year_start = f"{historical_year:04d}-01-01"
            year_end = f"{historical_year:04d}-12-31"
            clauses.append(
                "((valid_from IS NULL AND valid_until IS NULL "
                "AND COALESCE(document_date, date) BETWEEN ? AND ?) "
                "OR ((valid_from IS NOT NULL OR valid_until IS NOT NULL) "
                "AND (valid_from IS NULL OR valid_from <= ?) "
                "AND (valid_until IS NULL OR valid_until >= ?)))"
            )
            params.extend([year_start, year_end, year_end, year_start])

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = (
            f"SELECT * FROM documents {where} "
            "ORDER BY COALESCE(document_date, date) DESC LIMIT ?"
        )
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    def count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) as c FROM documents").fetchone()
            return row["c"] if row else 0


_store: Optional[SQLiteStore] = None


def get_store() -> SQLiteStore:
    global _store
    if _store is None:
        _store = SQLiteStore()
    return _store
