# The Investigation Nobody Could Answer — Backend

An agentic incident-investigation backend. Given a natural-language question
about an internal engineering incident, it runs a multi-step LangGraph
investigation (analyze → search → analyze evidence → detect contradictions →
compare against history → decide if more evidence is needed → answer) over a
document corpus, and returns an evidence-backed answer with a trace of what
it did.

This is a hackathon-scoped implementation: one working investigation graph,
real hybrid retrieval, real contradiction/timeline/similar-incident analysis,
and a clean `/investigate` API — not a sprawling framework.

## Architecture

```
FastAPI  POST /investigate
        │
        ▼
 analyze_question  (LLM extracts service/date/version/symptoms/targets)
        │
        ▼
     search  <───────────────────────────┐   hybrid retrieval:
        │                                │     - semantic (Chroma + sentence-transformers)
        ▼                                │     - metadata (SQLite filters)
 analyze_evidence (timeline + gap/LLM)   │     - merged, deduped, re-ranked
        │                                │
        ▼                                │
 detect_contradictions (LLM)             │
        │                                │
        ▼                                │
 similar_incidents (LLM classification:  │
   same / similar / related / unrelated) │
        │                                │
        ▼                                │
 need more evidence? ──yes──▶ expand_queries (LLM query expansion) ──▶ search
        │no
        ▼
 generate_answer (LLM: answer + confidence, grounded in evidence)
        │
        ▼
     JSON response
```

The loop is capped at `MAX_INVESTIGATION_ITERATIONS` (default 3) to prevent
runaway searching. Every LLM call is isolated behind `app/llm/client.py`, so
the model can be swapped without touching agent logic, and every retrieval
call goes through controlled Python functions in `app/retrieval/` — the LLM
never touches SQL or the vector store directly.

### Project layout

```
investigation_ai/
├── app/
│   ├── main.py              FastAPI app + error handlers
│   ├── api/routes.py        POST /investigate, GET /health
│   ├── agents/
│   │   ├── planner.py       question analysis + query generation/expansion
│   │   ├── investigator.py  retrieval "tools" + search execution
│   │   ├── analyzer.py      timeline, gap analysis, similar-incident classification
│   │   ├── contradiction.py contradiction detection
│   │   └── answer.py        final answer + confidence generation
│   ├── graph/
│   │   ├── state.py         InvestigationState (TypedDict)
│   │   └── workflow.py      LangGraph StateGraph wiring the nodes above
│   ├── retrieval/
│   │   ├── semantic.py      Chroma semantic search
│   │   ├── metadata.py      SQLite metadata search
│   │   └── hybrid.py        merges + re-ranks both (not pure vector similarity)
│   ├── storage/
│   │   ├── sqlite.py        document metadata/content store
│   │   └── chroma.py        embeddings + vector store (sentence-transformers + Chroma)
│   ├── ingestion/documents.py  loads documents.json, chunks, embeds, indexes
│   ├── llm/client.py        Google Gemini wrapper (LLMClient abstraction)
│   ├── models/               Pydantic request/response schemas
│   └── config.py             environment-driven settings
├── data/
│   ├── documents.json        example corpus (covers all 3 test scenarios)
│   ├── app.db                 created by ingestion
│   └── chroma/                 created by ingestion
├── tests/
│   ├── test_retrieval.py     chunking, SQLite, ranking helpers
│   ├── test_investigation.py graph node/decision logic (LLM calls stubbed)
│   ├── test_api.py           FastAPI endpoint tests (graph stubbed)
│   └── test_llm_client.py    Gemini SDK request/configuration tests
├── requirements.txt
├── .env.example
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Installation and Configuration

Requires Python 3.11+, network access to install dependencies, download the
Sentence Transformers model on first use, and reach the Gemini API.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

On macOS/Linux, use `python3.11 -m venv .venv`,
`source .venv/bin/activate`, and `cp .env.example .env`.

Set `GEMINI_API_KEY` in `.env` to a Google AI Studio API key. `GEMINI_MODEL`
defaults to `gemini-2.5-flash`; set it to another model available to your key
to change models. All supported variables and defaults are listed in
`.env.example`. Do not commit `.env`; it is ignored by Git and Docker builds.

### Ingest the example documents

```bash
python -m app.ingestion.documents
```

This loads `data/documents.json`, validates each record, stores metadata +
content in SQLite (`data/app.db`), chunks each document, generates local
embeddings (`BAAI/bge-small-en-v1.5`, via sentence-transformers — no external
embedding API), and indexes the chunks in Chroma (`data/chroma/`). SQLite
upserts by `document_id`; Chroma uses deterministic chunk IDs and prunes stale
chunks when a document changes. Re-running ingestion is idempotent. Use
`--reset` only when you intentionally want to discard and rebuild the Chroma
collection; it does not clear SQLite.

### Start the Server

```bash
uvicorn app.main:app --reload --port 8000
```

Open the frontend at `http://localhost:8000/ui`. FastAPI also serves
interactive API documentation at `http://localhost:8000/docs`.

```bash
curl http://localhost:8000/health
# health status and component checks
```

### Run tests

```bash
python -m pytest
```

Tests stub the LangGraph/LLM boundaries and Chroma embedding calls; they do not
require a Gemini API key, network access, or a Sentence Transformers model
download. Coverage includes health and request validation, response mapping,
retrieval, graph routing and iteration limits, and Gemini client error cases.

### Docker

```bash
docker build -t incident-agent .
docker run --env-file .env -p 8000:8000 incident-agent
```

The image installs `requirements.txt`, excludes `.env` and local database/vector
artifacts from its build context, ingests on startup, and only starts Uvicorn
if ingestion succeeds. Credentials are runtime configuration, never build
arguments. Compose reads the ignored local `.env` through `env_file` and
persists data under `./data`:

```bash
docker compose up --build
```

## API

### `GET /health`

Reports database, vector-store, embedding-model, and Gemini-key configuration
status. It returns HTTP 200 with `status: "ok"` when ready, or HTTP 503 with
`status: "degraded"` when a required component is unavailable. It does not
make a Gemini API request.

```json
{
  "status": "ok",
  "checks": {
    "database": "ok",
    "vector_store": "ok",
    "embedding_model": "ready",
    "gemini": "configured"
  },
  "document_count": 10,
  "chunk_count": 10
}
```

### `POST /investigate`

Request:

```json
{"question": "Why did the Order API become slow on September 16? Check whether the deployment was related and whether we have seen this before."}
```

The response contains `answer`, `confidence`, `evidence`, `contradictions`,
and `trace`. Evidence fields are `document_id`, `title`, `type`, `date`,
`version`, and `content`. Empty/whitespace or invalid requests return HTTP 400;
unexpected investigation failures return HTTP 500.

Example `curl`:

```bash
curl -X POST http://localhost:8000/investigate \
  -H "Content-Type: application/json" \
  -d '{"question": "Why did the Order API become slow on September 16? Check whether the deployment was related and whether we have seen this before."}'
```

Example response (shape; exact wording depends on the live LLM):

```json
{
  "answer": "The evidence indicates that the September 16 latency spike on orders-api was temporally associated with deployment v2.8.1. INC-1042 reports the P95 latency increase beginning shortly after the release, and DEP-882 records that v2.8.1 (which changed connection pooling configuration) was deployed at 18:10 UTC the previous day; ENG-DISC-19 notes that connection wait-time metrics correlated with the spike. A previous latency incident is documented in PM-211 (2026-05-03, v2.6.0), but it was caused by database connection saturation during a schema migration - a different mechanism. The documents support that a similar latency incident occurred before, but do not establish that it shared the same root cause. Confidence: medium.",
  "confidence": "medium",
  "evidence": [
    {
      "document_id": "INC-1042",
      "title": "Order API latency spike",
      "type": "incident_report",
      "date": "2026-09-16",
      "version": "v2.8.1",
      "content": "P95 latency increased significantly on the orders-api service starting on 2026-09-16..."
    }
  ],
  "contradictions": [],
  "trace": [
    "Parsed question: orders-api, 2026-09-16, slow, latency.",
    "Planned initial search queries: ['orders-api latency September 16 deployment', 'orders-api previous latency incidents'].",
    "Searched \"orders-api latency September 16 deployment\" -> found 5 result(s), 5 new: INC-1042, DEP-882, ENG-DISC-19, CUST-77, ARCH-05.",
    "Built timeline from 4 dated document(s).",
    "No significant evidence gaps identified.",
    "No contradictions detected among current evidence.",
    "Compared current evidence against 1 other document(s) for historical similarity: PM-211=similar_incident.",
    "Generated final evidence-backed answer with confidence=medium."
  ]
}
```

## Frontend and Troubleshooting

Open `http://localhost:8000/ui`. The status indicator polls `/health`; the
question form posts to `/investigate` on the same origin and renders the
answer, confidence, evidence, contradictions, and investigation trace. It
distinguishes backend errors, unavailable connections, invalid/empty responses,
and requests that exceed its 120-second timeout. POST requests are not
automatically retried, avoiding duplicate investigations.

- **Health is degraded: `gemini: missing_api_key`**: set `GEMINI_API_KEY` in
  `.env` and restart the server or container.
- **The embedding model is not ready**: wait for startup model loading; first
  use requires network access to download the configured model.
- **Database or vector store is unavailable**: check `DATABASE_PATH`,
  `CHROMA_PATH`, filesystem permissions, and the Compose `./data` mount.
- **The frontend cannot reach the backend**: confirm the server is running and
  open `/ui` from that same server origin. Check the browser network panel for
  `/health` and `/investigate` responses.
- **An investigation times out**: inspect server logs and Gemini availability
  before retrying; the browser stops waiting after 120 seconds.
- **Compose cannot find `.env`**: copy `.env.example` to `.env` first. The
  file is supplied at runtime and is excluded from the image build context.

## The three required test scenarios

The bundled `data/documents.json` contains documents for all three scenarios
from the spec:

- **A — deployment-related incident**: `INC-1042` / `DEP-882` / `PM-211`
  (plus supporting `ENG-DISC-19`, `CUST-77`, `ARCH-05`). Ask: *"Why did the
  Order API become slow on September 16? Check whether the deployment was
  related and whether we have seen this before."*
- **B — contradictory guidance**: `GUIDE-12` (2024, restart) vs. `GUIDE-41`
  (2026, do not restart during dependency failures). Ask: *"The service is
  failing after a deployment. What should the on-call engineer do first?"*
- **C — insufficient evidence**: `INC-300` (catalog-api, database
  saturation) vs. `INC-301` (orders-api, expired certificate) — different
  services and failure mechanisms. Ask: *"Did this exact failure happen
  before?"* — the system should explicitly decline to claim a match rather
  than force one.

After ingesting, exercise these with `curl` against `POST /investigate` (or
`/docs` for interactive Swagger UI) to verify:

```bash
curl -X POST http://localhost:8000/investigate -H "Content-Type: application/json" \
  -d '{"question": "The service is failing after a deployment. What should the on-call engineer do first?"}'

curl -X POST http://localhost:8000/investigate -H "Content-Type: application/json" \
  -d '{"question": "Did this exact failure happen before?"}'
```

## Design notes

- **Hybrid retrieval, not pure vector similarity** (`app/retrieval/hybrid.py`):
  merges semantic hits (Chroma) with metadata hits (SQLite), deduplicates by
  `document_id`, and re-ranks with a weighted score over semantic relevance,
  service match, version match, type match, date proximity to the question's
  anchor date, and general recency.
- **Query expansion is a real graph node** (`expand_queries_node`): after the
  first search + evidence analysis pass, an LLM call looks at what was
  actually discovered (e.g. a version number, a root cause keyword) and gaps
  identified by the gap-analysis step, and proposes new, more specific
  queries — this is what makes the system re-search based on findings rather
  than just running one retrieval pass.
- **Contradiction detection is evidence-grounded**: the LLM is explicitly
  instructed not to assume "newer supersedes older" unless the documents
  themselves indicate that relationship, and to output structured
  `{title, description}` pairs matching the API contract.
- **Similar-incident classification is a 4-way classification**
  (`same_incident` / `similar_incident` / `related_but_different` /
  `unrelated`), not a binary "is this the same" — matching the requirement
  that semantic similarity alone is never treated as proof of sameness.
- **Confidence is evidence-driven**: the final-answer prompt is explicit that
  `confidence` must reflect evidence quality (direct vs. indirect vs. no
  matching documents), and the code falls back to `"insufficient"` whenever
  no evidence was retrieved at all, regardless of what the model returns.
- **Everything the model can see is grounded**: evidence objects are always
  hydrated from SQLite (the source of truth for content/metadata), so the
  answer/contradiction/similar-incident prompts only ever receive real
  document content — nothing is synthesized before it reaches the LLM.
