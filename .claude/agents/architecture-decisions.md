# Architecture Decisions — SOLVRO MCP

## Core Architecture: Four-Component System

```
[Azure Blob] → [Prefect Pipeline] → [Neo4j Graph DB] ← [MCP Server] ← [ToPWR API] ← [Users]
                                                              ↑
                                                        [LangGraph RAG]
```

1. **MCP Server** — single source of graph intelligence; stateless; exposed as FastMCP tool
2. **ToPWR API** — user-facing HTTP API; owns sessions and conversation history; delegates intelligence to MCP
3. **Data Pipeline** — one-way ETL; documents become graph nodes/relations via LLM-generated Cypher
4. **MCP Client** — CLI; same protocol as API; not used in production path

## Key Patterns to Preserve

### Singleton Config
`src/config/config.py` loads `graph_config.yaml` once and caches it. All components call `get_config()`. **Do not load config elsewhere.** Changes to `graph_config.yaml` require: `just generate-models` to regenerate `config_models.py`, then restart services.

### LangGraph State Machine (RAG)
The RAG pipeline is a typed state machine — not a chain. Each node is a pure function: `State → State`. Routing decisions (guardrails → cypher vs. end) happen via conditional edges. When adding pipeline steps:
1. Add field to `State` TypedDict in `state.py`
2. Add node method in `rag.py`
3. Wire into graph with `add_node()` + `add_edge()` or `add_conditional_edges()`

**Do not bypass LangGraph** by calling nodes directly — the graph handles routing, observability, and error recovery.

### Optional Langfuse
Langfuse is never a hard dependency. All observability code must guard on `LANGFUSE_SECRET_KEY` being set. The pattern is: check env var → conditionally create `CallbackHandler` → pass to LLM invoke config. **Never make Langfuse required.**

### Cypher Generation Rules
The LLM Cypher generation prompts (in `graph_config.yaml` under `prompts.cypher_insert` and `prompts.cypher_search`) encode critical constraints:
- LIMIT on all MATCH queries
- Unique variable names per statement
- Polish character normalization
- Pipe (`|`) delimiter for pipeline output

**Do not change these prompts casually** — they directly affect Neo4j safety and data quality.

## What NOT to Change

| Item | Why |
|---|---|
| `src/config/config_models.py` | Auto-generated — edit `graph_config.yaml` instead |
| State machine edges in `rag.py` | Core routing logic — test thoroughly before changing |
| Cypher LIMIT enforcement | Safety guard — Neo4j can return unbounded results |
| Session in-memory storage | Intentional simplicity — if persistence is needed, add a proper DB layer |
| Non-root Docker users (`mcpuser`, `apiuser`) | Security requirement |

## Extension Points

### Add a new MCP tool
→ Create `src/mcp_server/tools/<tool_name>/` package with `__init__.py` and main module. Register with `@mcp.tool()` in `server.py`.

### Add a new RAG pipeline node
→ Add field to `State`, add node method to `RAG` class in `rag.py`, wire into `StateGraph`.

### Add a new API endpoint
→ Add route to `src/topwr_api/server.py`, add Pydantic models to `models.py` if needed.

### Add a new data source (not Azure)
→ Create new flow in `src/data_pipeline/flows/data_acquisition.py` or add a new flow file. Update `pipeline.py` orchestrator to call it.

### Add a new graph node/relation type
→ Add to `graph_config.yaml` under `graph.nodes` / `graph.relations`. Run `just generate-models`. Update Cypher generation prompts if needed.

### Add a new LLM provider
→ Add config under `llm` in `graph_config.yaml`. Add detection logic in `rag.py` where LLM is instantiated (currently checks for OpenAI/DeepSeek → Google fallback).

## Intentional Simplifications

- **In-memory sessions:** Single-instance deployment. If horizontal scaling is needed, sessions must move to Redis or a DB.
- **Synchronous Prefect pipeline:** Pipeline runs sequentially (acquire → extract → generate → populate). Parallelism is possible with Prefect `.map()` but not currently used.
- **LLM-only Cypher generation:** No pre-built graph schema enforcement beyond the prompt. Generated Cypher is executed as-is (with LIMIT injection).

## Known Inconsistencies

1. **Prefect version mismatch:** `Dockerfile.prefect` installs `prefect==2.*` but `pyproject.toml` requires `prefect>=3.6.7`. Resolve before extending the pipeline.
2. **graph_visualizer.py** appears unused in the production path — exists for debugging/docs generation.
3. **`scripts/data_pipeline/`** was deleted; ensure no references remain.
