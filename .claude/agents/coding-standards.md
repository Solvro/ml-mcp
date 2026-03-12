# Coding Standards — SOLVRO MCP

## File Structure Conventions

```
src/
├── config/           # Config loading only — no business logic
├── mcp_server/       # FastMCP server + RAG tools
│   └── tools/<name>/ # Each tool in its own package: __init__.py, main module, state.py
├── topwr_api/        # FastAPI app — server.py, models.py, session_manager.py
├── mcp_client/       # CLI clients
├── data_pipeline/    # Prefect flows
│   ├── pipeline.py   # Orchestrating @flow only — no logic
│   └── flows/        # Individual @task/@flow modules
└── scripts/          # One-off scripts (codegen, etc.)
```

New tool packages go in `src/mcp_server/tools/<tool_name>/`. New pipeline stages go in `src/data_pipeline/flows/`.

## Import Order (enforced by Ruff isort)

```python
# 1. Standard library
import asyncio
import os
from typing import Dict, Any, Optional, List

# 2. Third-party
from langchain_openai import ChatOpenAI
from langfuse import observe
from pydantic import BaseModel

# 3. Local (absolute paths from src/)
from src.config.config import get_config
from src.mcp_server.tools.knowledge_graph.state import State
```

## Type Hints

Always annotate all function parameters and return types. For Python <3.10 compatibility, use `Optional[X]` instead of `X | None`.

```python
# Correct
async def process(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    ...

# Wrong — missing annotations
async def process(query, limit=10):
    ...
```

## Docstrings — Google Style

Required for all public functions and classes:

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

## Async Patterns

- Use `async`/`await` for ALL I/O (Neo4j, LLM calls, HTTP)
- Prefer `asyncio.gather()` for concurrent calls over sequential `await`
- Use async context managers for Neo4j sessions

```python
# Good
results = await asyncio.gather(llm1.ainvoke(p1), llm2.ainvoke(p2), return_exceptions=True)

# Avoid
r1 = await llm1.ainvoke(p1)
r2 = await llm2.ainvoke(p2)
```

## Error Handling

Wrap all external I/O in try-except. Raise custom exceptions from the `KnowledgeGraphError` hierarchy. Log before re-raising:

```python
try:
    result = await session.run(cypher_query)
except Exception as e:
    logger.error(f"Neo4j query failed: {e}", extra={"query": cypher_query})
    raise Neo4jQueryError(f"Query execution failed: {e}") from e
```

Custom exception hierarchy lives in the relevant tool package. Do not use bare `Exception` in raises.

## Pydantic Models

- Use `pydantic.BaseModel` for all external data shapes (API request/response, config)
- Config models: use `graph_config.yaml` as source of truth; run `just generate-models` to regenerate `config_models.py`
- **Never edit `src/config/config_models.py` by hand**

## LangGraph State

- State is a `TypedDict` (or `MessagesState` subclass)
- All state fields must be explicitly typed
- Optional fields use `Optional[T] = None`
- Each node takes `State` and returns updated `State`

```python
class State(MessagesState):
    user_question: str
    generated_cypher: Optional[str] = None
    next_node: str
```

## FastMCP Tools

```python
@mcp.tool()
async def tool_name(param1: str, param2: int = 10) -> str:
    """
    One-line description (shown to AI clients in tool catalog).

    Args:
        param1: Description
        param2: Description (default: 10)

    Returns:
        JSON-serializable string result
    """
```

- snake_case tool names
- Clear docstrings (visible to AI clients)
- Return JSON-serializable types

## Prefect Pipeline

- Top-level `@flow` in `pipeline.py` only orchestrates — no logic
- Logic lives in `@flow` or `@task` functions in `flows/` submodules
- Use `log_prints=True` on flows for Prefect UI visibility

## Formatting (Ruff — do not override)

- Line length: 100
- Quotes: double (`"`)
- Indent: 4 spaces
- Target: Python 3.13

Run `just lint` before committing.
