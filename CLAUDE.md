# CLAUDE.md — SOLVRO MCP

## Project Overview

**SOLVRO MCP** is a Knowledge Graph RAG system for Wrocław University of Science and Technology (ToPWR). It answers natural-language questions (in Polish) about university entities — courses, professors, departments, articles — by generating Cypher queries against a Neo4j graph database.

**Architecture:** Four loosely-coupled services + a data pipeline:
1. **Frontend** — React 18 + TypeScript chatbot UI served by Nginx (port 80); proxies `/api/*` to ToPWR API
2. **MCP Server** — FastMCP server exposing a `knowledge_graph_tool` (port 8005)
3. **ToPWR API** — FastAPI HTTP backend, session management, user-facing chat endpoint (port 8000)
4. **Data Pipeline** — Prefect ETL: Azure Blob → PDF extraction → LLM Cypher generation → Neo4j
5. **MCP Client** — CLI for direct graph queries

**Core tech stack:**
| Layer | Technology |
|---|---|
| Frontend | React 18, TypeScript, Vite, TailwindCSS v3 |
| Frontend serving | Nginx (SPA fallback + API proxy) |
| LLM orchestration | LangChain, LangGraph (state machines) |
| MCP protocol | FastMCP >=2.12.4 |
| Graph database | Neo4j (async driver via langchain-neo4j) |
| API framework | FastAPI + Uvicorn |
| Data pipeline | Prefect >=3.6.7 |
| Cloud storage | Azure Blob Storage |
| Observability | Langfuse (optional) |
| Config validation | Pydantic v2 |
| Package manager | uv (NOT pip); npm for frontend |
| Linter/formatter | Ruff (Python); TypeScript strict mode |
| Python version | >=3.11 (Docker images use 3.12) |

---

## Development Setup

### Prerequisites
- Python >=3.11
- Node.js >=20 + npm (for frontend development)
- [uv](https://docs.astral.sh/uv/) package manager
- Docker + Docker Compose (for full stack)
- Neo4j instance
- At least one LLM API key (OpenAI preferred; DeepSeek, Google, CLARIN as alternatives)

### Install Dependencies
```bash
uv sync
# or for initial setup (also installs frontend npm deps):
just setup   # runs uv sync + generates Pydantic models + npm install
```

### Environment Variables
Copy `.env.example` to `.env` and fill in values:

```bash
cp .env.example .env
```

**Required (at minimum):**
```
# LLM (one of these)
OPENAI_API_KEY=...
DEEPSEEK_API_KEY=...
GOOGLE_API_KEY=...
CLARIN_API_KEY=...

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=...

# Langfuse (optional — omit to disable tracing)
LANGFUSE_HOST=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_PUBLIC_KEY=...
```

**Data pipeline extras:**
```
AZURE_STORAGE_CONNECTION_STRING=...
AZURE_CONTAINER_NAME=...

# Concurrent pipeline controls
DATA_PIPELINE_MAX_CONCURRENCY=4
DATA_PIPELINE_CLAIM_STALE_MINUTES=30

DATA_PIPELINE_STAGING_DIR=data/staging

# Cap on extra extraction passes for pages that dropped list/table rows (0 = unlimited)
DATA_PIPELINE_MAX_MISSED_ROW_PASSES=0

# Source acquisition / refresh (source_refresh.py)
DATA_PIPELINE_SOURCE_URLS=          # comma-separated seed URLs (required for refresh)
DATA_PIPELINE_CRAWL_DEPTH=1         # link hops from each seed (0 = seeds only)
DATA_PIPELINE_REQUEST_DELAY=1.0     # seconds between requests (too fast trips bot protection)
DATA_PIPELINE_EXCLUDE_PATTERNS=     # comma-separated URL substrings to skip (e.g. /addtrack/)
DATA_PIPELINE_MAX_DOCUMENTS=0       # cap documents per run (0 = unlimited)
DATA_PIPELINE_REFRESH_CRON=0 3 * * *

# OCR fallback for scanned/image PDFs
OCR_MIN_TEXT_CHARS=50
OCR_PDF_RENDER_SCALE=2.0
OCR_LANG=pol+eng
TESSERACT_CMD=/opt/homebrew/bin/tesseract
```

**Service ports (have defaults):**
```
MCP_BIND_HOST=0.0.0.0
MCP_HOST=localhost
MCP_PORT=8005
TOPWR_API_HOST=0.0.0.0
TOPWR_API_PORT=8000
```

### Run Locally

```bash
# MCP server
just mcp-server
# or: uv run server

# FastAPI backend
just api
# or: uv run topwr-api

# Frontend dev server (requires running API on :8000)
just frontend-dev      # → http://localhost:3000

# Query CLI (requires running MCP server)
just kg "Kto wykłada analizę matematyczną?"
# or: uv run kg "<question>"

# Full stack via Docker (includes frontend at http://localhost)
just up
just down
```

### Run Tests
```bash
just test            # pytest with coverage
just test-verbose    # verbose + HTML coverage report
just ci              # lint + test (full CI pipeline)
```

---

## Project Structure

```
ml-mcp/
├── src/
│   ├── config/
│   │   ├── config.py            # Singleton config loader (loads graph_config.yaml)
│   │   └── config_models.py     # AUTO-GENERATED Pydantic models — do not edit manually
│   ├── mcp_server/
│   │   ├── server.py            # FastMCP app, /health route, tool registration, lifespan
│   │   └── tools/knowledge_graph/
│   │       ├── rag.py           # LangGraph state machine (guardrails → cypher → retrieve → grade)
│   │       ├── state.py         # GraphState TypedDict definition
│   │       ├── cypher_guardrails.py # Read-only validation, LIMIT enforcement
│   │       ├── question_analysis.py # Polish question-literal detection, search phrases
│   │       └── graph_visualizer.py  # Mermaid diagram generator
│   ├── topwr_api/
│   │   ├── server.py            # FastAPI app, endpoints, MCP client integration
│   │   ├── models.py            # Pydantic models (ChatRequest, ChatResponse, Session)
│   │   └── session_manager.py   # Thread-safe in-memory session store
│   ├── mcp_client/
│   │   └── client.py            # CLI client for knowledge graph queries
│   ├── data_pipeline/
│   │   ├── pipeline.py          # Top-level Prefect @flow orchestrator
│   │   ├── staging.py           # Staging dir, manifest, atomic writes, source_id mapping
│   │   ├── completeness.py      # List/table row coverage check for extracted pages
│   │   ├── label_vocabulary.py  # Closed node-label set and the rewrite that enforces it
│   │   ├── canonical_nodes.py   # Canonical merge keys (one node per real entity)
│   │   └── flows/
│   │       ├── source_refresh.py        # Scheduled discovery + fetch of source docs (web connector)
│   │       ├── data_acquisition.py      # Staging dir scan → document references
│   │       ├── ocr_extraction.py        # PDF/TXT/DOCX → text (OCR fallback)
│   │       ├── llm_cypher_generation.py # LLM → Cypher INSERT statements
│   │       ├── graph_dedup.py           # Post-ingest relabel, key backfill, duplicate merge
│   │       └── graph_populating.py      # Execute Cypher against Neo4j
│   └── scripts/
│       ├── api_smoke.py         # Manual smoke check against a live API (uv run api-smoke)
│       ├── populate_graph.py    # One-off graph population script
│       └── config/
│           └── generate_models.py   # Runs datamodel-codegen to regenerate config_models.py
├── tests/
│   ├── conftest.py                             # Puts the repo root on sys.path
│   ├── test_cypher_guardrails.py               # Read-only Cypher validation, in isolation
│   ├── test_llm_fallback_guardrails.py         # Provider selection and fallback chain
│   ├── test_rag_retrieve_path.py               # retrieve(): blocks mutations, enforces LIMIT
│   ├── test_rag_generation_guardrails_path.py  # generate_cypher / guardrails_system nodes
│   ├── test_rag_graph_end_to_end.py            # Full graph run, both branches, sync + async
│   ├── test_rag_empty_retrieval_escalation.py  # Empty-result retries and the abstention answer
│   ├── test_rag_fulltext_fallback.py           # Scored index-backed rescue, procedure allowlist
│   ├── test_rag_context_grader.py              # Grading rows before they can become an answer
│   ├── test_question_analysis.py               # Question-literal detection, phrase extraction
│   ├── test_llm_determinism_config.py          # Both models pinned to temperature 0
│   ├── test_graph_schema_config.py             # Closed label set stays internally consistent
│   ├── test_mcp_health_signal.py               # /health, ToolError on failure, driver shutdown
│   ├── test_kg_cli_failure_reporting.py        # kg reports a failed call instead of a traceback
│   ├── test_rag_graph_connection_lifecycle.py  # ping_database / close on the graph driver
│   └── data_pipeline/                          # Concurrency, acquisition, OCR, extraction quality
├── docker/
│   ├── compose.stack.yml        # Full stack (neo4j, postgres, mcp, api, prefect)
│   ├── compose.prefect.yml      # Data pipeline only
│   ├── Dockerfile.mcp           # MCP server image
│   ├── Dockerfile.api           # FastAPI image
│   └── Dockerfile.prefect       # Data pipeline image
├── graph_config.yaml            # Master config: LLM settings, graph schema, prompts
├── pyproject.toml               # Dependencies, ruff config, entry points
├── justfile                     # All dev commands
└── .env.example                 # Environment variable template
```

---

## Common Commands

```bash
# Development
just setup              # First-time: uv sync + generate models
just generate-models    # Regenerate src/config/config_models.py from graph_config.yaml
just lint               # Ruff format + check
just test               # pytest --cov=src --cov-report=term tests/
just ci                 # lint + test

# Running services locally
just mcp-server         # Start MCP server
just api                # Start FastAPI
just kg "<question>"    # Query the knowledge graph

# Docker
just up                 # docker compose -f docker/compose.stack.yml up -d
just down               # stop stack
just restart            # restart stack
just ps                 # container status
just logs               # all logs
just logs-mcp           # MCP server logs
just logs-api           # API logs
just nuke               # remove containers + volumes

# Data pipeline (Docker)
just prefect-up         # start Prefect stack
just prefect-down       # stop Prefect stack
just prefect-logs       # Prefect logs
just pipeline           # run pipeline locally (uv run prefect_pipeline)
uv run dedup-graph      # one-off full duplicate repair over an existing graph

# Build
just build              # uv build
just clean              # remove dist/, .cache/, __pycache__
```

---

## Development Conventions

### Code Style (enforced by Ruff)
- **Line length:** 100 characters
- **Quotes:** double quotes
- **Indent:** 4 spaces
- **Import order:** stdlib → third-party → local (isort-compatible)
- **Target:** Python 3.13

### Type Hints
Always use type hints for all function parameters and return values.

### Docstrings
Google-style docstrings for all public functions:
```python
async def query_graph(user_input: str, session_id: str = "default") -> Dict[str, Any]:
    """
    Query the knowledge graph with natural language.

    Args:
        user_input: User's natural language question
        session_id: Session identifier for grouping queries

    Returns:
        Dictionary containing answer and metadata

    Raises:
        Neo4jQueryError: If database query fails
    """
```

### Async
Use `async`/`await` for all I/O (Neo4j queries, LLM calls, HTTP requests).

### Error Handling
Wrap external calls in `try-except`; use custom exceptions (`KnowledgeGraphError` hierarchy).

### Naming
- Functions/variables: `snake_case`
- Classes: `PascalCase`
- MCP tools: `snake_case` (becomes tool name visible to AI clients)
- Prefect flows/tasks: decorated with `@flow` / `@task`

---

## Important Patterns

### LangGraph State Machine
The RAG pipeline is a LangGraph `StateGraph` with typed state:
```python
class State(MessagesState):
    user_question: str
    context: Optional[List[Document]]
    answer: Optional[str]
    next_node: str
    generated_cypher: Optional[str]
    guardrail_decision: Optional[str]
    trace_id: Optional[str]
```
Nodes: `guardrails_system` → conditional → `generate_cypher` → `retrieve` → `return_none`

### Config System
`graph_config.yaml` is the single source of truth. Loaded as a validated Pydantic singleton:
```python
from src.config.config import get_config
config = get_config()   # cached singleton
```
**Never edit `src/config/config_models.py` by hand** — run `just generate-models` after changing `graph_config.yaml`.

### Langfuse Observability
Optional; enabled when `LANGFUSE_SECRET_KEY` is set. Use `@observe` decorator + `CallbackHandler` for LangChain:
```python
from langfuse import observe
from langfuse.langchain import CallbackHandler

@observe(name="Cypher Generation")
async def generate_cypher(self, user_input: str, trace_id: str = None, **langfuse_kwargs) -> str:
    handler = CallbackHandler(trace_id=trace_id, session_id=langfuse_kwargs.get("session_id"))
    response = await self.llm.ainvoke(prompt, config={"callbacks": [handler]})
    return response.content
```

### Neo4j Operations
Always use async Neo4j sessions; enforce LIMIT on all queries:
```python
async with self.driver.session() as session:
    result = await session.run(cypher_query)  # must include LIMIT
    return await result.data()
```

### Cypher Generation (Data Pipeline)
LLM generates Cypher INSERT statements separated by `|` (pipe character). Strict rules are
enforced via prompt and string values are folded deterministically before execution:
- Unique variable names per statement
- Polish characters normalized (ó→o, ę→e, etc.)
- Token limit: 65536 to avoid DeepSeek API errors

The pipe-separated shape is a hard contract: `graph_populating.populate_graph` splits on `|` and
runs the parts as **one** query, because relationship clauses reference variables bound by
earlier node clauses.

### Ingestion Extraction Quality

Three deterministic passes run over the model's output in `llm_cypher_generation`, in this
order. Each exists because the prompt alone only gets it right most of the time, and "most of
the time" is what silently corrupts a knowledge graph.

**1. Completeness (`completeness.py`).** The prompt requires every list/table row to become its
own node. Rows are also counted from the source page and checked against the generated values: a
row counts as covered when most of its wording appears somewhere in the output, so rephrasing is
fine but a row the model never read is caught. Missing rows go into one extra extraction pass
(`prompts.cypher_insert_missing_rows`) whose statements are appended. The pass only runs when
something is missing, and a failure there leaves the first pass output intact.

That extra pass is a second model call for every page that lost rows.
`DATA_PIPELINE_MAX_MISSED_ROW_PASSES` caps how many a run may spend (0, the default, means
unlimited); past the cap the miss is still logged with the rows that stay absent. Every run
reports `missed_row_passes` in its summary, so the cost is visible before anyone bounds it.

This exists because the academic-calendar page kept the days off with a proper name and dropped
`2 XI 2026 r. — dzień wolny od zajęć`. That page is the regression case in
`tests/data_pipeline/test_completeness.py`.

**2. Closed label set (`label_vocabulary.py`).** `graph_schema.node_labels` in
`graph_config.yaml` is the only vocabulary; `graph_schema.label_aliases` maps known drift
(`Program`→`StudyProgram`, `CriterionItem`→`Criterion`, `Holiday`→`DayOff`, …) and anything still
unrecognised becomes `graph_schema.fallback_label`. Matching ignores case and diacritics. Only
node labels are rewritten — relationship types and quoted values are left verbatim, since a
wrong relationship type is visible in the traversal while a wrong label silently hides the node.

**3. Canonical merge keys (`canonical_nodes.py`).** A node MERGE keys on `key`, a normalized form
of the title (case folded, diacritics folded, punctuation dropped), never on `title + context`.

A trailing bracket is dropped from the key **only when it abbreviates the title** — `(CBE)`,
`(PWr)`, a faculty code like `(W4)`: one short alphanumeric token carrying capitals. Everything
else is a qualifier that is the only thing telling two entities apart, and it stays:
`Informatyka (studia I stopnia)` and `Informatyka (studia II stopnia)` are two nodes, as are
`(stacjonarne)`/`(niestacjonarne)`, a Roman numeral, and a year. Fusing two entities into one is
worse than splitting one into two — a split leaves both halves visible, a fusion loses an entity
with nothing in the graph showing it happened. `looks_like_abbreviation` reads the title *before*
case folding, since folding destroys the capitalisation the decision rests on. Properties are applied through
`ON CREATE SET` / `ON MATCH SET`, so a second mention enriches the existing node: the fuller
title wins and a new context is appended (capped at 2000 chars) rather than replacing what was
there. Relationship MERGEs and combined patterns are left alone — only a lone node MERGE can take
`ON CREATE` / `ON MATCH` without changing the pattern.

`data_pipeline_flow` creates a `key` index per configured label before extracting, because MERGE
on an unindexed property scans the whole label.

### Post-Ingest Deduplication

`graph_dedup.deduplicate_graph` repairs entities split across several nodes. It has two modes,
sharing one query — a null key list means "everything" — so they cannot drift apart.

**Per run (automatic).** `populate_graph` returns the canonical keys it wrote, the flow collects
them across pages, and only those groups are examined. The cost tracks what changed rather than
how large the graph has grown, and the `key` index carries the lookup. A run with no changed
documents returns before the repair is reached, so it never touches the graph.

**Full walk (explicit): `uv run dedup-graph`.** Run this once after upgrading an existing
database. It additionally:

1. moves nodes under an off-vocabulary label to their canonical label;
2. backfills `key` on titled nodes that predate it — computed in Python, so a backfilled node
   and a freshly extracted one can never disagree about the key for a title;
3. merges every group sharing a label and a key across the whole graph.

Steps 1 and 2 are deliberately absent from the per-run path: they repair nodes written before
the rules existed, and no later run can reintroduce either, so paying for them every time buys
nothing.

Merging keeps the fullest title, every distinct context, and the relationships of the nodes it
absorbs. `ProcessedDocument` and `PipelineRun` are excluded by label: relabelling
`ProcessedDocument` would replay every page that has already been ingested. Merging needs APOC;
without it the pass logs and changes nothing, because a half-finished merge is worse than a
duplicate. The pass is idempotent — a second run is a no-op.

### Text2Cypher Search Normalization

- The Cypher prompt receives both the original Polish question and a lowercase,
  diacritic-folded search form.
- Generated Cypher string literals are diacritic-folded again before validation and execution.
  Case-insensitive human-readable matching uses `toLower(...)` on both sides, while stable IDs
  retain their original case.
- Labels, relationship types, property keys, and Cypher clauses are never normalized; they must
  be copied exactly from the live Neo4j schema.
- Read-only guardrails remain authoritative after normalization. `CALL` is blocked for generated
  Cypher; the only reviewed exception is the full-text lookup described below, which passes its
  one procedure through `validate_read_only(..., allowed_procedures=...)`.

### Text2Cypher Determinism and Empty-Result Escalation

Both pipeline models run at `temperature: 0` (`llm.fast_model`, `llm.accurate_model`), so the
same question routes and generates the same Cypher on every run. Do not reintroduce sampling:
`tests/test_llm_determinism_config.py` fails if either temperature moves off zero.

A generated query that executes but matches nothing is retried rather than reported as missing
data, because the two most common Text2Cypher mistakes both surface as zero rows. `retrieve()`
escalates in `src/mcp_server/tools/knowledge_graph/rag.py`:

1. **primary** — the model's query, after diacritic folding and `toLower` enforcement.
2. **repaired_literals** — fuzzy predicates whose literal is copied question text (detected by
   `question_analysis.is_question_like_literal`) are replaced with `true`, keeping the traversal
   the model wrote but dropping a filter that could never match a stored title.
3. **label_agnostic_phrases** — `FALLBACK_SEARCH_CYPHER` queries the `entity_search` full-text
   index for noun phrases extracted from the question
   (`question_analysis.extract_search_phrases` → `build_lucene_query`), which recovers an answer
   stored under a label the model did not pick. Longer phrases are boosted, and hits scoring
   below `rag.fallback_min_score` never leave the database. Gated by
   `rag.enable_fallback_search`.

The chosen step is reported as `metadata.retrieval_strategy` and logged by the MCP server.
Escalation only follows a *successful* execution — a blocked or failing query is still reported
as blocked or failed, never retried.

Phrase extraction never starts or ends a phrase with a Polish question word or function word, so
`"Co obejmuje udział w konferencjach?"` yields `"udzial w konferencjach"` (the stored title) and
never the truncated `"udzial w"`.

### The `entity_search` Full-Text Index

`RAG.ensure_fulltext_index` creates it at startup over `title` and `context`. A full-text index
must name its labels, so it is built from the labels the database currently holds — minus
`ProcessedDocument` and `PipelineRun` — and dropped and recreated when that set changes. An empty
fallback result re-checks the index once before being believed, so a label that ingestion added
since startup does not stay invisible until the next deploy.

Reading it needs `CALL`, which `validate_read_only` blocks. Rather than loosening the guardrail,
it takes an explicit `allowed_procedures` allowlist: only the internally authored fallback passes
one (`ALLOWED_RETRIEVAL_PROCEDURES`), and generated Cypher is still validated with an empty set,
so the model cannot call anything. `CALL` subqueries are rejected outright — they name no
procedure to vet.

**Inflected Polish.** The index uses Lucene's standard analyzer and no Polish analyzer ships
with this Neo4j build, so nothing stems: a question in an oblique case would never reach a
nominative title. `build_lucene_query` closes that query-side. Tokens of 5 characters or more
also get a prefix clause (`semestrze` → `semestr*`) and an edit-distance clause
(`semestrze~1`), both boosted below 1 so an exact phrase still outranks them. Short tokens stay
exact — a 3-character prefix matches half the graph.

This widens recall deliberately; the score floor and the grader are what keep precision, which
is the point of having both. If a Polish analyzer ever ships in the deployed build, configuring
it at index creation is the better fix and this expansion can go.

### Abstention Is a Retrieval Decision

Whether to answer is settled before an answer is written, not by the answering model:

1. **Score threshold.** Fallback hits below `rag.fallback_min_score` are filtered in the
   database. Lucene scores are corpus-relative, so this is a junk floor, not a precision
   guarantee. Keep it low: an inflected hit legitimately scores around 1.4 where the nominative
   form of the same question scores 10, so a floor chosen from nominative scores would drop the
   matches the inflection expansion exists to recover.
2. **`grade_context` node** (between `retrieve` and the end of the graph). One cheap fast-model
   call is shown the question and the retrieved rows and returns which of them actually answer
   it. Rejecting all of them sets `retrieval_strategy = graded_out` and empties the context, so
   the caller gets `NO_GRAPH_DATA_MESSAGE` without the answering model being consulted.
   Rows from a `primary` query skip grading — that query expressed the question's own structure.
   The grader **fails open**: a failed call or an unreadable reply keeps the rows, because a
   provider outage must not be indistinguishable from an empty graph.
3. **The answer payload carries its provenance.** `RAG._format_result` returns
   `{"retrieval_strategy", "context_graded", "rows"}`, and `prompts.final_answer` treats
   `primary` rows as the answer and rescue rows as candidates.

There is no "answer from general knowledge" escape hatch anywhere in `prompts.final_answer`. A
confidently wrong date in a student-facing chatbot is worse than an admission of ignorance —
keep every layer of this in place.

Two distinct sentinels live in `src/config/messages.py`:
- `OFF_TOPIC_MESSAGE` — the guardrail routed the question away from retrieval.
- `NO_GRAPH_DATA_MESSAGE` — retrieval ran and found nothing worth answering with.

### Guardrail Breadth

`prompts.guardrails` is deliberately permissive and defaults to `generate` when in doubt: a
wrongly rejected question costs the user an answer, while a wrongly accepted one just produces an
empty retrieval that the escalation and abstention paths already handle. Malformed guardrail
output is a separate matter and still fails closed to `end` (`_parse_guardrail_output`).

### Concurrent Pipeline Idempotency
The data pipeline processes pages in batches with configurable concurrency and hash-based idempotency:
- `DATA_PIPELINE_MAX_CONCURRENCY` controls max parallel pages per batch.
- `ProcessedDocument {hash}` nodes prevent duplicate processing for repeated pages.
- Failed or stale `processing` claims can be retried/reclaimed based on `DATA_PIPELINE_CLAIM_STALE_MINUTES`.
- `source_id` is stable per file/page (`file://relative/path#page=N`) so unchanged staged docs are skipped.
- Pages whose extraction raises are skipped and left out of the run's source hashes, so the next run retries them; pages that extract successfully but yield little text are recorded as-is.

### Source Refresh (Acquisition)
`refresh_sources_flow` (scheduled via `prefect-refresh`, cron in `DATA_PIPELINE_REFRESH_CRON`)
crawls `DATA_PIPELINE_SOURCE_URLS`, stages `.pdf/.txt/.md` (HTML saved as stripped-text `.md`)
into `DATA_PIPELINE_STAGING_DIR` with atomic `*.part` → rename writes, and tracks state in
`manifest.json` (`source_id = file://{relative_path}` → etag/last-modified/sha256).
Only changed docs trigger the downstream pipeline; failed fetches are retried next run.
Discovery prefers the host's `sitemap.xml` (declared in `robots.txt`, else probed at the root) and
falls back to breadth-first crawling only when no sitemap is available — on wit.pwr.edu.pl that is
1770 documents in 5 requests instead of 184 documents in 184. `robots.txt` `Disallow` rules and
`DATA_PIPELINE_EXCLUDE_PATTERNS` filter both paths; `DATA_PIPELINE_MAX_DOCUMENTS` caps a run.
Requests are paced by `DATA_PIPELINE_REQUEST_DELAY` and identify the crawler via User-Agent;
a document that collapses to under 30% of its known size (bot check, outage) is treated as a
failed fetch so good staged content is never overwritten.
`acquire_data()` scans the staging dir and feeds document references to `ocr_extraction` →
`pipeline.py`. Run once with `just refresh`; serve the schedule with `just refresh-serve`.

### OCR Runtime Requirements

- OCR fallback is handled by `src/data_pipeline/flows/ocr_extraction.py` with `pytesseract` + `PyMuPDF`.
- Install Tesseract OCR on the host/container (`tesseract --version` should work).
- Optionally set `TESSERACT_CMD` when the binary is outside `PATH`.
- `.docx` files are parsed with `python-docx` (paragraphs and tables), not OCR.

### The Health Signal Means "I Can Serve"

`GET /health` on the MCP server (a `@mcp.custom_route`, so it sits next to `/mcp` on port 8005)
runs a trivial query against Neo4j and answers `200 {"status": "healthy"}` or `503` with a
`reason`: `rag_not_initialized`, `neo4j_unreachable` (carrying the driver's own message), or
`neo4j_timeout`. `docker/compose.stack.yml` curls it.

It replaced a socket probe that went green the moment uvicorn bound the port, which meant
`depends_on: service_healthy` could hand a caller a server whose graph was gone. A bound socket
is not an interface anyone can depend on.

The ping runs in a worker thread under `HEALTH_PING_TIMEOUT_SECONDS` (5s) so a stalled graph
returns a 503 naming the reason instead of hanging until docker kills the probe; the compose
`timeout` is deliberately larger (10s) so ours is the one that fires. A hung Neo4j leaves the
worker thread behind until the driver's own timeout trips it, which is why the ping is a bare
`RETURN 1` and not something that can queue.

**Failure is not content.** `knowledge_graph_tool` raises `ToolError` when the graph cannot be
consulted at all — no RAG, or the pipeline timed out. `fastmcp.Client.call_tool` raises on
`isError` by default, so `topwr_api` lands in its `except` branch and tags the turn
`source="error"` rather than feeding "Error: RAG not initialized" to the answering model as if
it were graph data. `OFF_TOPIC_MESSAGE` and `NO_GRAPH_DATA_MESSAGE` are *answers* — retrieval
ran and found nothing — and keep coming back as ordinary results.

Both consumers of the tool had to learn the difference. `topwr_api` already caught the
exception. The `kg` CLI did not, and a raised `ToolError` would have surfaced as a traceback, so
it now prints the failure to **stderr** and exits non-zero — the answer owns stdout, and anything
piping `kg` must not read an error as one.

**The driver is closed on shutdown.** RAG opens `Neo4jGraph` in its constructor and holds it for
the process lifetime; `RAG.close()` releases it and the FastMCP `lifespan` calls it, so a restart
loop no longer leaks one per cycle. `close_rag()` is idempotent and swallows a failing close,
because a shutdown derailed by its own cleanup is worse than a leaked socket.

### Session Management
`SessionManager` is thread-safe in-memory storage (dict + `threading.Lock`). Not persisted across restarts. Suitable for single-instance deployments only.

### Multi-LLM Fallback
The system tries LLM providers in order: OpenAI → DeepSeek → Google Gemini. Configured in `graph_config.yaml` under `llm.fast_model` and `llm.accurate_model`.

---

## Gotchas & Notes

1. **`config_models.py` is auto-generated** — never edit it directly. Run `just generate-models` after changing `graph_config.yaml`.

2. **Langfuse is optional** — if `LANGFUSE_SECRET_KEY` is not set, traces are silently skipped. The code checks for the env var before initializing.

3. **Session storage is in-memory** — restarting the API loses all sessions. No database persistence layer for sessions.

4. **`topwr_api`'s `/health` is still shallow** — it reports the session store and never checks
   that the MCP server is reachable, so it can read healthy while the graph behind it is not.
   The MCP side was fixed in #64; this half was left alone deliberately, since failing the API's
   probe on a dependency hiccup would restart a container that is itself fine.

5. **Cypher LIMIT enforcement** — the RAG pipeline strips and re-adds `LIMIT` to all generated Cypher queries. Do not rely on LLM to add it.

6. **Pipeline Cypher delimiter** — the data pipeline LLM generates statements joined by `|`. Splitting logic lives in `llm_cypher_generation.py`.

7. **Polish language** — prompts are in Polish; guardrails check if a query is university-related in Polish context; CLARIN model is used as alternative for Polish-specific tasks.

8. **uv, not pip** — this project uses `uv` for dependency management. Do not use `pip install`. Lockfile: `uv.lock`.

9. **Docker multi-stage builds** — MCP and API Dockerfiles use `ghcr.io/astral-sh/uv:python3.12` as builder then copy to `python:3.12-slim`. This keeps images small.

10. **The containerised pipeline does not run the schedule** — `Dockerfile.prefect` installs from `uv.lock` (Prefect 3.6.11), so the version mismatch this used to warn about is gone. What is missing is the deployment: the image's `CMD` is only `prefect server start`, nothing invokes `serve_refresh` (`uv run prefect-refresh`), so the cron added in #51 exists on a developer machine and not in Docker. `compose.prefect.yml` is also its own stack with no `neo4j` service and no link to `mcp_network`, so the pipeline has no route to the graph. See #54.

11. **Graph schema** — `graph_schema` in `graph_config.yaml` enumerates 27 node labels and 32 relationship types. The label set is closed and enforced at ingestion (see *Ingestion Extraction Quality*); adding a label means editing the config and running `just generate-models`. Retrieval still reads the live schema from Neo4j, which may also contain labels written before the set was enforced until the dedup pass relabels them.
