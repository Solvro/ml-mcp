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
tests/                  # All tests live here (inferred — no tests found in repo yet)
├── unit/
│   ├── mcp_server/     # RAG pipeline node tests
│   ├── topwr_api/      # API endpoint + session manager tests
│   └── data_pipeline/  # Flow tests
└── integration/        # Full-stack integration tests
```

Note: `src/topwr_api/test_api.py` is an integration test script (not pytest), run via `uv run test-topwr-api`. It hits a live API at configured host/port.

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
- Integration scripts (`test_api.py`) are not pytest — don't include in coverage

## Current Test Gap

As of analysis: no `tests/` directory was found in the repository. The only test infrastructure is `src/topwr_api/test_api.py` (integration script). Unit tests need to be created.
