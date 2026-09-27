"""
Centralized configuration for the Investigation backend.

All values are read from environment variables (optionally loaded from a
.env file via python-dotenv). Nothing here should hold a real secret -
only defaults and env var lookups.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - dotenv is optional at import time
    pass


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str = field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY", "").strip()
    )
    gemini_model: str = field(
        default_factory=lambda: os.getenv("GEMINI_MODEL", "").strip()
        or "gemini-2.5-flash"
    )

    database_path: str = field(default_factory=lambda: os.getenv("DATABASE_PATH", "./data/app.db"))
    chroma_path: str = field(default_factory=lambda: os.getenv("CHROMA_PATH", "./data/chroma"))
    documents_path: str = field(
        default_factory=lambda: os.getenv("DOCUMENTS_PATH", "./data/documents.json")
    )

    max_investigation_iterations: int = field(
        default_factory=lambda: _get_int("MAX_INVESTIGATION_ITERATIONS", 3)
    )

    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    )

    chroma_collection_name: str = field(
        default_factory=lambda: os.getenv("CHROMA_COLLECTION", "investigation_documents")
    )

    # Chunking parameters (characters, not tokens - kept simple for the hackathon scope)
    chunk_size: int = field(default_factory=lambda: _get_int("CHUNK_SIZE", 1200))
    chunk_overlap: int = field(default_factory=lambda: _get_int("CHUNK_OVERLAP", 150))


settings = Settings()
