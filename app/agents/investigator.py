"""
Investigator Agent.

Exposes retrieval as controlled "tools" (plain Python functions - the LLM
never gets direct DB/SQL access) and executes searches on behalf of the
investigation graph: semantic search, metadata search, and hybrid combined
search. All results are merged into the running set of retrieved documents.
"""
from __future__ import annotations

from typing import Optional

from app.retrieval.hybrid import search_documents as _hybrid_search
from app.retrieval.metadata import metadata_retrieve as _metadata_search
from app.retrieval.temporal import historical_year_from_text


def search_documents_tool(
    query: str,
    service: Optional[str] = None,
    document_type: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    version: Optional[str] = None,
    anchor_date: Optional[str] = None,
    top_k: int = 8,
    historical_year: Optional[int] = None,
) -> list[dict]:
    """Tool: hybrid semantic + metadata search. This is the primary retrieval tool."""
    return _hybrid_search(
        query=query,
        service=service,
        document_type=document_type,
        date_from=date_from,
        date_to=date_to,
        version=version,
        anchor_date=anchor_date,
        historical_year=historical_year,
        top_k=top_k,
    )


def search_by_metadata_tool(
    service: Optional[str] = None,
    document_type: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    version: Optional[str] = None,
    limit: int = 20,
) -> list[dict]:
    """Tool: pure metadata/structured search, no query text required."""
    return _metadata_search(
        service=service,
        document_type=document_type,
        date_from=date_from,
        date_to=date_to,
        version=version,
        limit=limit,
    )


def find_similar_incidents_tool(service: Optional[str], symptoms: list[str], exclude_ids: list[str]) -> list[dict]:
    """Tool: search for historically similar incidents/postmortems for a service."""
    query = " ".join([service or "", *symptoms]).strip() or "similar incident"
    results = _hybrid_search(
        query=query,
        service=service,
        document_type=None,
        top_k=10,
    )
    return [r for r in results if r.get("document_id") not in set(exclude_ids)]


def run_searches(
    queries: list[str],
    entities: dict,
    existing_documents: dict[str, dict],
    search_history: Optional[list[dict]] = None,
    iteration: int = 1,
) -> tuple[dict[str, dict], list[str]]:
    """
    Execute a batch of search queries with entity-derived filters, merge
    results into `existing_documents` (keyed by document_id, keeping the
    highest score seen), and return (updated_documents, trace_lines).
    """
    trace: list[str] = []
    documents = dict(existing_documents)
    search_history = search_history if search_history is not None else []
    previously_searched = {
        " ".join(str(entry.get("query", "")).casefold().split())
        for entry in search_history
        if entry.get("query")
    }

    service = entities.get("service")
    date = entities.get("date")
    date_range = entities.get("date_range") or {}
    historical_year = entities.get("historical_year")
    versions = entities.get("versions") or []
    version = entities.get("version") or (versions[0] if len(versions) == 1 else None)
    date_from = date_range.get("from") or date
    date_to = date_range.get("to") or date

    for query in queries:
        normalized_query = " ".join(query.casefold().split())
        if normalized_query in previously_searched:
            search_history.append(
                {
                    "iteration": iteration,
                    "query": query,
                    "document_ids": [],
                    "new_document_ids": [],
                    "produced_new_evidence": False,
                    "skipped": True,
                }
            )
            trace.append(f"Skipped previously executed query \"{query}\".")
            continue

        previously_searched.add(normalized_query)
        results = search_documents_tool(
            query=query,
            service=service,
            version=version,
            date_from=date_from,
            date_to=date_to,
            historical_year=historical_year or historical_year_from_text(query),
            anchor_date=date,
            top_k=8,
        )
        new_ids = []
        result_ids = []
        for doc in results:
            doc_id = doc["document_id"]
            if doc_id not in result_ids:
                result_ids.append(doc_id)
            prior = documents.get(doc_id)
            if prior is None or doc.get("_score", 0) > prior.get("_score", 0):
                documents[doc_id] = doc
            if prior is None:
                new_ids.append(doc_id)

        search_history.append(
            {
                "iteration": iteration,
                "query": query,
                "document_ids": result_ids,
                "new_document_ids": new_ids,
                "produced_new_evidence": bool(new_ids),
                "skipped": False,
            }
        )

        if results:
            if new_ids:
                trace.append(
                    f"Searched \"{query}\" -> found {len(results)} result(s), "
                    f"{len(new_ids)} new: {', '.join(new_ids)}."
                )
            else:
                trace.append(f"Searched \"{query}\" -> found {len(results)} result(s), all already known.")
        else:
            trace.append(f"Searched \"{query}\" -> no matching documents found.")

    return documents, trace
