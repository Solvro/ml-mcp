# CLAUDE.md — SOLVRO MCP

## Project Overview

**SOLVRO MCP** is a Knowledge Graph RAG system for Wrocław University of Science and Technology (ToPWR). It answers natural-language questions (in Polish) about university entities — courses, professors, departments, articles — by generating Cypher queries against a Neo4j graph database.

**Architecture:** Three loosely-coupled services + a data pipeline:
1. **MCP Server** — FastMCP server exposing a `knowledge_graph_tool` (port 8005)
2. **ToPWR API** — FastAPI HTTP backend, session management, user-facing chat endpoint (port 8000)
3. **Data Pipeline** — Prefect ETL: Azure Blob → PDF extraction → LLM Cypher generation → Neo4j
4. **MCP Client** — CLI for direct graph queries

**Core tech stack:**
| Layer | Technology |
|---|---|
| LLM orchestration | LangChain, LangGraph (state machines) |
| MCP protocol | FastMCP >=2.12.4 |
| Graph database | Neo4j (async driver via langchain-neo4j) |
| API framework | FastAPI + Uvicorn |
| Data pipeline | Prefect >=3.6.7 |
| Cloud storage | Azure Blob Storage |
| Observability | Langfuse (optional) |
| Config validation | Pydantic v2 |
| Package manager | uv (NOT pip) |
| Linter/formatter | Ruff |
| Python version | >=3.11 (Docker images use 3.12) |

---

## Development Setup

### Prerequisites
- Python >=3.11
- [uv](https://docs.astral.sh/uv/) package manager
- Docker + Docker Compose (for full stack)
- Neo4j instance
- At least one LLM API key (OpenAI preferred; DeepSeek, Google, CLARIN as alternatives)

### Install Dependencies
```bash
uv sync
# or for initial setup:
just setup   # runs uv sync + generates Pydantic models
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

# Query CLI (requires running MCP server)
just kg "Kto wykłada analizę matematyczną?"
# or: uv run kg "<question>"

# Full stack via Docker
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
│   │   ├── server.py            # FastMCP app, tool registration, startup
│   │   └── tools/knowledge_graph/
│   │       ├── rag.py           # LangGraph state machine (guardrails → cypher → retrieve)
│   │       ├── state.py         # GraphState TypedDict definition
│   │       └── graph_visualizer.py  # Mermaid diagram generator
│   ├── topwr_api/
│   │   ├── server.py            # FastAPI app, endpoints, MCP client integration
│   │   ├── models.py            # Pydantic models (ChatRequest, ChatResponse, Session)
│   │   ├── session_manager.py   # Thread-safe in-memory session store
│   │   └── test_api.py          # Integration test script (uv run test-topwr-api)
│   ├── mcp_client/
│   │   └── client.py            # CLI client for knowledge graph queries
│   ├── data_pipeline/
│   │   ├── pipeline.py          # Top-level Prefect @flow orchestrator
│   │   └── flows/
│   │       ├── data_acquisition.py      # Azure Blob download
│   │       ├── text_extraction.py       # PDF/TXT → text
│   │       ├── llm_cypher_generation.py # LLM → Cypher INSERT statements
│   │       └── graph_populating.py      # Execute Cypher against Neo4j
│   └── scripts/config/
│       └── generate_models.py   # Runs datamodel-codegen to regenerate config_models.py
├── docker/
│   ├── compose.stack.yml        # Full stack (neo4j, postgres, mcp, api, prefect)
│   ├── compose.prefect.yml      # Data pipeline only
│   ├── Dockerfile.mcp           # MCP server image
│   ├── Dockerfile.api           # FastAPI image
│   ├── Dockerfile.prefect       # Data pipeline image
│   └── prefect-entrypoint.sh   # Starts Prefect server then runs pipeline
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
LLM generates Cypher INSERT statements separated by `|` (pipe character). Strict rules enforced via prompt:
- Unique variable names per statement
- Polish characters normalized (ó→o, ę→e, etc.)
- Token limit: 65536 to avoid DeepSeek API errors

### Session Management
`SessionManager` is thread-safe in-memory storage (dict + `threading.Lock`). Not persisted across restarts. Suitable for single-instance deployments only.

### Multi-LLM Fallback
The system tries LLM providers in order: OpenAI → DeepSeek → Google Gemini. Configured in `graph_config.yaml` under `llm.fast_model` and `llm.accurate_model`.

---

## Gotchas & Notes

1. **`config_models.py` is auto-generated** — never edit it directly. Run `just generate-models` after changing `graph_config.yaml`.

2. **Langfuse is optional** — if `LANGFUSE_SECRET_KEY` is not set, traces are silently skipped. The code checks for the env var before initializing.

3. **Session storage is in-memory** — restarting the API loses all sessions. No database persistence layer for sessions.

4. **Cypher LIMIT enforcement** — the RAG pipeline strips and re-adds `LIMIT` to all generated Cypher queries. Do not rely on LLM to add it.

5. **Pipeline Cypher delimiter** — the data pipeline LLM generates statements joined by `|`. Splitting logic lives in `llm_cypher_generation.py`.

6. **Polish language** — prompts are in Polish; guardrails check if a query is university-related in Polish context; CLARIN model is used as alternative for Polish-specific tasks.

7. **uv, not pip** — this project uses `uv` for dependency management. Do not use `pip install`. Lockfile: `uv.lock`.

8. **Docker multi-stage builds** — MCP and API Dockerfiles use `ghcr.io/astral-sh/uv:python3.12` as builder then copy to `python:3.12-slim`. This keeps images small.

9. **Prefect version mismatch** — `Dockerfile.prefect` pins `prefect==2.*` while `pyproject.toml` requires `prefect>=3.6.7`. Verify which version is actually running before modifying pipeline code. (Flagged as inconsistency.)

10. **Graph schema** — 27 node types and 32 relation types defined in `graph_config.yaml`. The RAG pipeline fetches this schema dynamically from Neo4j and falls back to config if the DB is empty.
