# AI Agent Instructions for SOLVRO MCP Codebase

This document provides guidelines for AI coding assistants (GitHub Copilot, Claude, GPT, etc.) when working on the SOLVRO MCP project.

## Project Context

SOLVRO MCP is a Knowledge Graph RAG system using:
- **FastMCP** for Model Context Protocol server
- **LangGraph** for state machine orchestration
- **Neo4j** for graph database
- **LangChain** for LLM orchestration
- **Langfuse** for observability

## Code Style Guidelines

### General Principles

1. **Type Hints**: Always use type hints for function parameters and return values
2. **Async/Await**: Use async functions for I/O operations (Neo4j, LLM calls)
3. **Docstrings**: Use Google-style docstrings for all public functions
4. **Error Handling**: Wrap external calls in try-except blocks
5. **Logging**: Use structured logging with context

### Python Standards

```python
# Good Example
async def query_graph(
    user_input: str,
    session_id: str = "default",
    trace_id: str = "default",
) -> Dict[str, Any]:
    """
    Query the knowledge graph with natural language.
    
    Args:
        user_input: User's natural language question
        session_id: Session identifier for grouping queries
        trace_id: Trace identifier for observability
        
    Returns:
        Dictionary containing answer and metadata
        
    Raises:
        Neo4jConnectionError: If database connection fails
        LLMAPIError: If LLM API call fails
    """
    try:
        # Implementation
        pass
    except Exception as e:
        logger.error(f"Query failed: {e}", extra={"trace_id": trace_id})
        raise
```

### Formatting Rules

- **Line Length**: Maximum 100 characters (configured in Ruff)
- **Imports**: Group in order: standard library, third-party, local
- **Quotes**: Use double quotes for strings
- **Indentation**: 4 spaces (no tabs)

```python
# Import Order
import asyncio
import os
from typing import Dict, Any

from langchain_openai import ChatOpenAI
from langfuse import observe

from src.mcp_server.tools.knowledge_graph.state import GraphState
```

## Architecture Patterns

### 1. LangGraph State Machines

When modifying RAG pipeline:

```python
from langgraph.graph import StateGraph, END
from src.mcp_server.tools.knowledge_graph.state import GraphState

# Define nodes as methods
def guardrails_node(self, state: GraphState) -> GraphState:
    """Determine if query is relevant."""
    # Implementation
    return state

# Build graph
workflow = StateGraph(GraphState)
workflow.add_node("guardrails", self.guardrails_node)
workflow.add_edge("guardrails", "cypher_generation")
```

**Rules:**
- Always type state parameter as `GraphState`
- Return updated state from each node
- Use conditional edges for routing decisions
- Add END edge for terminal states

### 2. Langfuse Observability

Always wrap LLM calls with observability:

```python
from langfuse import observe
from langfuse.langchain import CallbackHandler

@observe(name="Cypher Generation")
async def generate_cypher(
    self,
    user_input: str,
    trace_id: str = None,
    **langfuse_kwargs,  # Accept session_id and other Langfuse params
) -> str:
    """Generate Cypher query from natural language."""
    
    handler = CallbackHandler(
        trace_id=trace_id,
        session_id=langfuse_kwargs.get("session_id"),
        tags=["cypher_generation", "rag"],
    )
    
    response = await self.llm.ainvoke(
        prompt,
        config={"callbacks": [handler]},
    )
    
    return response.content
```

**Rules:**
- Use `@observe` decorator for traced functions
- Accept `**langfuse_kwargs` to capture session_id
- Pass `session_id` via function parameters, not imports
- Use `CallbackHandler` for LangChain integration
- Tag traces with component identifiers

### 3. Neo4j Database Operations

Always use async Neo4j driver:

```python
from neo4j import AsyncGraphDatabase

async def query_database(self, cypher_query: str) -> List[Dict]:
    """Execute Cypher query against Neo4j."""
    
    async with self.driver.session() as session:
        try:
            result = await session.run(cypher_query)
            records = await result.data()
            return records
        except Exception as e:
            logger.error(f"Neo4j query failed: {e}")
            raise Neo4jQueryError(f"Query execution failed: {e}")
```

**Rules:**
- Use async context managers for sessions
- Always limit query results (add LIMIT clause)
- Handle `Neo4jError` exceptions
- Log failed queries for debugging
- Close driver on application shutdown

### 4. FastMCP Tools

When adding new MCP tools:

```python
from fastmcp import FastMCP

mcp = FastMCP("SOLVRO MCP")

@mcp.tool()
async def new_tool_name(
    param1: str,
    param2: int = 10,
) -> str:
    """
    Brief tool description (shown to AI clients).
    
    Args:
        param1: Description of first parameter
        param2: Description of second parameter (default: 10)
        
    Returns:
        Description of return value
    """
    # Implementation
    return result
```

**Rules:**
- Use descriptive tool names (snake_case)
- Provide clear docstrings (visible to AI clients)
- Use type hints for all parameters
- Return JSON-serializable data
- Handle errors gracefully

## Common Patterns

### Pattern 1: Adding a New RAG Node

```python
# 1. Update state in state.py
class GraphState(TypedDict):
    message: str
    new_field: str  # Add new field

# 2. Add node method in rag.py
def new_node(self, state: GraphState) -> GraphState:
    """Process new step in pipeline."""
    
    # Extract state
    message = state["message"]
    
    # Process
    result = self._process_logic(message)
    
    # Update state
    state["new_field"] = result
    return state

# 3. Add to workflow
workflow.add_node("new_step", self.new_node)
workflow.add_edge("previous_step", "new_step")
```

### Pattern 2: Adding Langfuse Traces

```python
# For standalone functions
@observe(name="Component Name")
async def process_data(
    input_data: str,
    trace_id: str = None,
    **langfuse_kwargs,
) -> str:
    """Process data with observability."""
    # Implementation
    return result

# For LangChain LLM calls
handler = CallbackHandler(
    trace_id=trace_id,
    session_id=session_id,
    tags=["component_tag"],
    metadata={"custom": "metadata"},
)

response = await llm.ainvoke(
    prompt,
    config={
        "callbacks": [handler],
        "metadata": {
            "run_name": "Descriptive Name",
        },
    },
)
```

### Pattern 3: Multi-threaded Data Processing

```python
from concurrent.futures import ThreadPoolExecutor
from typing import List

def process_documents(
    documents: List[str],
    num_threads: int = 4,
) -> None:
    """Process documents in parallel."""
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(self._process_single, doc)
            for doc in documents
        ]
        
        for future in futures:
            try:
                future.result()
            except Exception as e:
                logger.error(f"Document processing failed: {e}")
```

## Testing Guidelines

### Unit Tests

```python
import pytest
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_knowledge_graph_tool():
    """Test knowledge graph tool with valid input."""
    
    # Arrange
    mock_rag = AsyncMock()
    mock_rag.ainvoke.return_value = {"answer": "test"}
    
    # Act
    with patch("src.mcp_server.server.rag", mock_rag):
        result = await knowledge_graph_tool("test query")
    
    # Assert
    assert result == "test"
    mock_rag.ainvoke.assert_called_once()
```

**Rules:**
- Use `pytest` framework
- Mock external dependencies (Neo4j, LLM APIs)
- Test both success and failure cases
- Use descriptive test names
- Group related tests in classes

## Environment Variables

Always check for required environment variables:

```python
import os
from dotenv import load_dotenv

load_dotenv()

REQUIRED_VARS = [
    "NEO4J_URI",
    "NEO4J_USER",
    "NEO4J_PASSWORD",
    "OPENAI_API_KEY",
    "LANGFUSE_SECRET_KEY",
]

missing = [var for var in REQUIRED_VARS if not os.getenv(var)]
if missing:
    raise ValueError(f"Missing environment variables: {missing}")
```

## Error Handling

### Custom Exceptions

```python
class KnowledgeGraphError(Exception):
    """Base exception for knowledge graph errors."""
    pass

class Neo4jConnectionError(KnowledgeGraphError):
    """Raised when Neo4j connection fails."""
    pass

class CypherGenerationError(KnowledgeGraphError):
    """Raised when Cypher query generation fails."""
    pass
```

### Error Recovery

```python
async def robust_query(self, query: str) -> Dict:
    """Query with retry logic."""
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return await self._execute_query(query)
        except Neo4jConnectionError as e:
            if attempt == max_retries - 1:
                raise
            logger.warning(f"Retry {attempt + 1}/{max_retries}: {e}")
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
```

## File Organization

### Adding New Modules

```
src/mcp_server/tools/
├── knowledge_graph/
│   ├── __init__.py         # Export public API
│   ├── rag.py              # Main RAG logic
│   ├── state.py            # State definitions
│   ├── graph_visualizer.py # Utilities
│   └── new_module.py       # New functionality
```

### Import Structure

```python
# In __init__.py
from .rag import RAG
from .state import GraphState

__all__ = ["RAG", "GraphState"]

# In other files
from src.mcp_server.tools.knowledge_graph import RAG, GraphState
```

## Performance Considerations

### Caching

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def get_graph_schema() -> Dict:
    """Get cached graph schema."""
    with open("graph_config.json") as f:
        return json.load(f)
```

### Async Concurrency

```python
# Good: Concurrent API calls
results = await asyncio.gather(
    llm1.ainvoke(prompt1),
    llm2.ainvoke(prompt2),
    return_exceptions=True,
)

# Bad: Sequential calls
result1 = await llm1.ainvoke(prompt1)
result2 = await llm2.ainvoke(prompt2)
```

## Documentation

### Function Documentation

```python
def complex_function(
    param1: str,
    param2: List[int],
    param3: Optional[Dict] = None,
) -> Tuple[str, int]:
    """
    One-line summary of function purpose.
    
    Detailed explanation of what the function does,
    including any important behavioral notes.
    
    Args:
        param1: Description of first parameter
        param2: Description of second parameter
        param3: Optional parameter description. Defaults to None.
        
    Returns:
        Tuple containing:
            - str: Description of first return value
            - int: Description of second return value
            
    Raises:
        ValueError: When param1 is empty
        TypeError: When param2 contains non-integers
        
    Example:
        >>> result = complex_function("test", [1, 2, 3])
        >>> print(result)
        ('processed', 3)
    """
```

## Git Commit Messages

Follow conventional commits:

```
feat: Add new RAG node for entity extraction
fix: Resolve Neo4j connection timeout issue
docs: Update API reference for knowledge_graph_tool
refactor: Simplify Cypher generation logic
test: Add unit tests for guardrails node
chore: Update dependencies in pyproject.toml
```

## Questions to Ask

When modifying code, consider:

1. **Does this need observability?** → Add Langfuse traces
2. **Is this I/O bound?** → Use async/await
3. **Will this be called frequently?** → Consider caching
4. **Does this modify state?** → Update GraphState TypedDict
5. **Is this user-facing?** → Add clear error messages
6. **Can this fail?** → Add error handling and logging

## Important Notes
- Always prioritize code readability and maintainability
- Update Readme and documentation for significant changes

## Resources

- [FastMCP Documentation](https://github.com/jlowin/fastmcp)
- [LangGraph Guide](https://langchain-ai.github.io/langgraph/)
- [Neo4j Python Driver](https://neo4j.com/docs/python-manual/current/)
- [Langfuse Python SDK](https://langfuse.com/docs/sdk/python)
- [Project README](../README.md)