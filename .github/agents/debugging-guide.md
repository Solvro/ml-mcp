# Debugging Guide — SOLVRO MCP

## Logging

The project uses Python's standard `logging` module with structured extras:

```python
import logging
logger = logging.getLogger(__name__)

logger.info("Processing query", extra={"session_id": session_id, "trace_id": trace_id})
logger.error(f"Query failed: {e}", extra={"trace_id": trace_id})
```

Log level is configured at startup. Set `DEBUG` for verbose output.

## Langfuse Tracing (Primary Observability)

When `LANGFUSE_SECRET_KEY` is set, all LLM calls are traced:

1. Each `@observe`-decorated function appears as a span
2. `CallbackHandler` traces LangChain chain calls
3. Traces are grouped by `session_id` and identified by `trace_id`
4. Tags like `["cypher_generation", "rag"]` allow filtering in the Langfuse UI

**Debug LLM calls:** Check Langfuse UI → filter by `session_id` or `trace_id`.

**If Langfuse is not configured:** Traces are silently skipped. Check `LANGFUSE_SECRET_KEY` env var.

## Debug Mode (RAG Pipeline)

Enable `enable_debug: true` in `graph_config.yaml` under `rag:` to activate the `debug_print` node in the LangGraph pipeline. This prints intermediate state to stdout.

## Docker Log Commands

```bash
just logs           # tail all services
just logs-mcp       # MCP server only
just logs-api       # FastAPI only
just logs-neo4j     # Neo4j only
just prefect-logs   # Prefect pipeline
```

Or directly:
```bash
docker compose -f docker/compose.stack.yml logs -f mcp-server
docker compose -f docker/compose.stack.yml logs -f topwr-api
```

## Health Checks

```bash
# API health
curl http://localhost:8000/health

# Stats endpoint
curl http://localhost:8000/api/stats

# Neo4j browser
open http://localhost:7474

# Prefect UI
open http://localhost:4200
```

## Common Failure Modes

### Neo4j connection fails
- Check `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` in `.env`
- Verify Neo4j container is healthy: `docker compose ps`
- Check Neo4j logs: `just logs-neo4j`
- RAG pipeline falls back to config schema if DB is empty (not an error)

### LLM API errors
- `OPENAI_API_KEY` not set → check which model is configured in `graph_config.yaml`
- DeepSeek: token limit errors → enforced at 65536 in pipeline config
- Check `llm.fast_model.model` and `llm.accurate_model.model` in `graph_config.yaml`

### Cypher generation fails
- Enable `enable_debug: true` in `graph_config.yaml`
- Check the generated Cypher in Langfuse traces
- Common issues: Polish characters not normalized, variable name collisions
- Pipeline: verify `|` delimiter splitting in `llm_cypher_generation.py`

### MCP server not reachable from API
- Check `MCP_HOST` and `MCP_PORT` match the running server
- In Docker: services communicate via service names (`mcp-server:8005`)
- Health check: `docker compose ps` → mcp-server should show `healthy`

### Session not found (API)
- Sessions are **in-memory only** — lost on API restart
- Use `/api/sessions/{session_id}` to verify session exists before sending messages

### Prefect pipeline stuck
- Check `prefect-entrypoint.sh` — it starts Prefect server first, waits for health, then runs pipeline
- Prefect UI at port 4200 shows flow run status
- Check env: `AZURE_STORAGE_CONNECTION_STRING` must be set for data acquisition

## Useful Debug Commands

```bash
# Test knowledge graph query directly (requires running MCP server)
uv run kg "Kto wykłada analizę matematyczną?"

# Run pipeline locally without Docker
uv run prefect_pipeline

# Run API integration tests against live server
uv run test-topwr-api

# Check graph schema cached in config
python -c "from src.config.config import get_config; c = get_config(); print(c.graph.nodes)"

# Verify Ruff is happy
just lint
```

## Graph Visualizer

Generate a Mermaid diagram of the RAG state machine:
```python
from src.mcp_server.tools.knowledge_graph.graph_visualizer import visualize_graph
# produces Mermaid JS markup for the LangGraph pipeline
```

Paste output into mermaid.live or any Mermaid renderer to visualize the pipeline.
