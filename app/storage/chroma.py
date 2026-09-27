"""
ChromaDB-backed vector store for semantic search over document chunks.

Embeddings are generated locally with sentence-transformers
(BAAI/bge-small-en-v1.5 by default) - no external embedding API is used.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from app.config import settings

_embedder = None
_chroma_client = None
_collection = None


def _new_chroma_client():
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    Path(settings.chroma_path).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=settings.chroma_path,
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def get_embedder():
    """Load the sentence-transformers embedding model once per process."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer(settings.embedding_model)
    return _embedder


def is_embedding_model_loaded() -> bool:
    return _embedder is not None


def warm_up() -> None:
    """Load the local embedding model during API startup."""
    get_embedder()


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_embedder()
    vectors = model.encode(list(texts), normalize_embeddings=True)
    return [v.tolist() if hasattr(v, "tolist") else list(v) for v in vectors]


def get_collection():
    """Lazily initialize the persistent Chroma client/collection."""
    global _chroma_client, _collection
    if _collection is None:
        _chroma_client = _new_chroma_client()
        _collection = _chroma_client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def reset_collection():
    """Used by tests / re-ingestion to start from a clean collection."""
    global _chroma_client, _collection
    _chroma_client = _new_chroma_client()
    try:
        _chroma_client.delete_collection(settings.chroma_collection_name)
    except Exception:
        pass
    _collection = _chroma_client.get_or_create_collection(
        name=settings.chroma_collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    return _collection


def upsert_chunks(chunks: list[dict]) -> None:
    """
    chunks: list of dicts with keys:
        chunk_id, document_id, title, type, service, date, version,
        document_date, status, superseded_by, valid_from, valid_until, content
    """
    if not chunks:
        return

    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    chunks = list(chunks_by_id.values())
    collection = get_collection()
    ids = [c["chunk_id"] for c in chunks]
    active_ids = set(ids)
    documents = [c["content"] for c in chunks]
    metadatas = [
        {
            "document_id": c["document_id"],
            "title": c.get("title") or "",
            "type": c.get("type") or "",
            "service": c.get("service") or "",
            "date": c.get("date") or "",
            "version": c.get("version") or "",
            "document_date": c.get("document_date") or "",
            "status": c.get("status") or "",
            "superseded_by": c.get("superseded_by") or "",
            "valid_from": c.get("valid_from") or "",
            "valid_until": c.get("valid_until") or "",
        }
        for c in chunks
    ]
    embeddings = embed_texts(documents)
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)

    stale_ids = []
    for document_id in {chunk["document_id"] for chunk in chunks}:
        existing = collection.get(where={"document_id": document_id}, include=[])
        stale_ids.extend(
            chunk_id
            for chunk_id in existing.get("ids", [])
            if chunk_id not in active_ids
        )
    if stale_ids:
        collection.delete(ids=stale_ids)


def semantic_search(query: str, top_k: int = 10, where: Optional[dict] = None) -> list[dict]:
    """
    Returns a list of dicts: {chunk_id, document_id, title, type, service,
    date, version, content, distance}
    """
    collection = get_collection()
    if collection.count() == 0:
        return []

    query_embedding = embed_texts([query])[0]
    kwargs = dict(
        query_embeddings=[query_embedding],
        n_results=min(top_k, max(collection.count(), 1)),
    )
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)

    out = []
    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0] if results.get("distances") else [None] * len(ids)

    for i, chunk_id in enumerate(ids):
        meta = metas[i] or {}
        out.append(
            {
                "chunk_id": chunk_id,
                "document_id": meta.get("document_id"),
                "title": meta.get("title"),
                "type": meta.get("type"),
                "service": meta.get("service") or None,
                "date": meta.get("date") or None,
                "version": meta.get("version") or None,
                "document_date": meta.get("document_date") or None,
                "status": meta.get("status") or None,
                "superseded_by": meta.get("superseded_by") or None,
                "valid_from": meta.get("valid_from") or None,
                "valid_until": meta.get("valid_until") or None,
                "content": docs[i],
                "distance": dists[i],
            }
        )
    return out
