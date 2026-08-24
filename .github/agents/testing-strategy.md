# Testing Strategy — SOLVRO MCP

## Framework & Tools

- **Framework:** `pytest` with `pytest-asyncio` for async tests
- **Coverage:** `pytest-cov` (`--cov=src`)
- **Mocking:** `unittest.mock` (`AsyncMock`, `MagicMock`, `patch`)
- **Commands:**
  ```bash
  just test           # pytest --cov=src --cov-report=term tests/
  just test-verbose   # verbose + HTML coverage report
  ```

## Test Location

```
tests/
├── conftest.py         # Puts the repo root on sys.path
├── test_*.py           # Knowledge-graph tests: guardrails, RAG nodes, full graph runs
└── data_pipeline/      # Pipeline flow tests with their own conftest
```

Note: `src/scripts/api_smoke.py` is a manual smoke script, not a test. It hits a live API at the configured host/port and prints what it gets back — it asserts nothing and pytest does not collect it. Run it by hand via `uv run api-smoke`.

## What to Test

### MCP Server / RAG Pipeline
- Each LangGraph node in isolation (guardrails, generate_cypher, retrieve)
- State transitions and conditional routing
- Schema fallback behavior (empty Neo4j → config fallback)

### FastAPI Backend
- All endpoints: `/api/chat`, `/api/sessions/*`, `/health`
- Session creation/retrieval/deletion
- Chat continuation (existing session reuse)
- Error responses for invalid session IDs

### Session Manager
- Thread-safety (inferred from Lock usage — worth testing with concurrent writes)
- `get_active_session()` returns most recent active session
- `deactivate_session()` marks correctly

### Data Pipeline
- Text extraction from PDF and TXT inputs
- Cypher statement splitting on `|` delimiter
- Error handling when Azure Blob is unavailable

## Mocking Patterns

### Mock Neo4j
```python
from unittest.mock import AsyncMock, patch

async def test_retrieve_node():
    mock_session = AsyncMock()
    mock_session.run.return_value.data.return_value = [{"name": "Jan Kowalski"}]

    with patch("src.mcp_server.tools.knowledge_graph.rag.neo4j_graph") as mock_db:
        mock_db.session.return_value.__aenter__.return_value = mock_session
        # test retrieve node
```

### Mock LLM
```python
mock_llm = AsyncMock()
mock_llm.ainvoke.return_value.content = "MATCH (p:Professor) RETURN p LIMIT 5"

with patch("src.mcp_server.tools.knowledge_graph.rag.RAG._llm_accurate", mock_llm):
    result = await rag.generate_cypher_node(state)
```

### Mock Langfuse
Langfuse is optional and keyed off env vars. In tests, either:
- Unset `LANGFUSE_SECRET_KEY` (tracing auto-disabled), or
- Mock `langfuse.observe` decorator

### Mock Azure Blob
```python
from unittest.mock import patch, MagicMock

with patch("src.data_pipeline.flows.data_acquisition.BlobServiceClient") as mock_client:
    mock_client.from_connection_string.return_value.get_container_client.return_value = ...
```

## Test Patterns

### Async Tests
```python
import pytest

@pytest.mark.asyncio
async def test_knowledge_graph_tool():
    mock_rag = AsyncMock()
    mock_rag.ainvoke.return_value = {"answer": "Prof. Kowalski"}

    with patch("src.mcp_server.server.rag", mock_rag):
        result = await knowledge_graph_tool("Kto wykłada analizę?")

    assert "Kowalski" in result
    mock_rag.ainvoke.assert_called_once()
```

### Guardrails Node Test
```python
@pytest.mark.asyncio
async def test_guardrails_routes_relevant_query():
    state = State(user_question="Kto wykłada matematykę?", next_node="", ...)
    result = await rag.guardrails_node(state)
    assert result["guardrail_decision"] == "yes"
    assert result["next_node"] == "generate_cypher"
```

### Session Manager Tests
```python
def test_session_creation():
    manager = SessionManager()
    session = manager.create_session(user_id="user1")
    assert session.user_id == "user1"
    assert session.is_active

def test_get_nonexistent_session_returns_none():
    manager = SessionManager()
    assert manager.get_session("nonexistent") is None
```

## Coverage Expectations

- Aim for high coverage on core business logic: `rag.py`, `session_manager.py`, `pipeline.py`
- Lower priority for: auto-generated `config_models.py`, Docker entrypoint scripts
- Manual scripts under `src/scripts/` are not pytest — they are excluded from coverage via `omit` in `[tool.coverage.run]`

## Current Test Gap

The suite lives in `tests/` and runs via `just test` (pytest + coverage). It runs on every pull request through the `test` job in `.github/workflows/main.yaml`.

- `tests/test_cypher_guardrails.py` — read-only Cypher validation in isolation
- `tests/test_llm_fallback_guardrails.py` — provider selection and the fallback chain
- `tests/test_rag_retrieve_path.py` — `retrieve()`: mutating queries never reach the driver, LIMIT is enforced
- `tests/test_rag_generation_guardrails_path.py` — the `generate_cypher` and `guardrails_system` nodes, driven with stubbed models
- `tests/test_rag_graph_end_to_end.py` — full graph runs, both guardrail branches, `invoke` and `ainvoke`
- `tests/data_pipeline/` — pipeline concurrency, data acquisition, OCR extraction

Path-level tests exist because unit tests are structurally blind to wiring bugs: during #45 the Cypher validator was unplugged from `retrieve()` and the whole suite stayed green, because every test verified the validator in isolation and nothing verified it was still being called.
