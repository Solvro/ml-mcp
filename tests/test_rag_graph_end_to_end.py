import asyncio
from typing import Any, NamedTuple

import pytest
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

from src.config.messages import NO_GRAPH_DATA_MESSAGE, OFF_TOPIC_MESSAGE
from src.mcp_server.tools.knowledge_graph.graph_visualizer import GraphVisualizer
from src.mcp_server.tools.knowledge_graph.rag import RAG

QUESTION = "test question"
SCHEMA_TEXT = "Node properties: X\nRelationship properties: Y\nThe relationships: Z"
SAFE_CYPHER = "MATCH (n) RETURN n"
EXECUTED_CYPHER = f"{SAFE_CYPHER} LIMIT 5"
EMPTY_ANSWER = OFF_TOPIC_MESSAGE
TIMEOUT_MESSAGE = "exceeded the maximum allowed wait time"


class FakeDatabase:
    """Minimal Neo4j stand-in for graph end-to-end tests."""

    def __init__(self, schema: str, rows: list[dict[str, Any]]) -> None:
        self.get_schema = schema
        self.rows = rows
        self.queries: list[str] = []

    def query(self, cypher_query: str) -> list[dict[str, Any]]:
        self.queries.append(cypher_query)
        return self.rows


class GraphStub(NamedTuple):
    """A compiled RAG graph plus everything its stubbed models recorded."""

    rag: RAG
    database: FakeDatabase
    guardrails_prompts: list[str]
    cypher_prompts: list[str]
    invoke_configs: list[dict[str, Any]]


def _recording_runnable(
    reply: str, prompt_log: list[str], config_log: list[dict[str, Any]]
) -> RunnableLambda:
    def _invoke(prompt_value: Any, config: dict[str, Any] | None = None) -> str:
        prompt_log.append(prompt_value.to_string())
        config_log.append(config or {})
        return reply

    return RunnableLambda(_invoke)


def _build_rag_graph_stub(
    *,
    guardrails_reply: str,
    cypher_reply: str,
    db_rows: list[dict[str, Any]],
    max_results: int = 5,
    graph_timeout_sec: float = 5.0,
) -> GraphStub:
    """Create a runnable RAG graph without running RAG.__init__."""
    rag = object.__new__(RAG)
    rag.enable_debug = False
    rag.max_results = max_results
    rag.enable_fallback_search = False
    rag.graph_timeout_sec = graph_timeout_sec
    rag._cached_schema = None
    rag.visualizer = GraphVisualizer()

    database = FakeDatabase(schema=SCHEMA_TEXT, rows=db_rows)
    rag.database = database

    rag.generate_cypher_template = PromptTemplate(
        input_variables=["user_question", "schema"],
        template="Q: {user_question}\nSchema:\n{schema}",
    )
    rag.guard_rails_template = PromptTemplate(
        input_variables=["user_question"],
        template="Question: {user_question}",
    )

    guardrails_prompts: list[str] = []
    cypher_prompts: list[str] = []
    invoke_configs: list[dict[str, Any]] = []

    rag.fast_llm = _recording_runnable(guardrails_reply, guardrails_prompts, invoke_configs)
    rag.cypher_llm = _recording_runnable(cypher_reply, cypher_prompts, invoke_configs)

    rag.graph = rag._build_processing_graph()
    return GraphStub(rag, database, guardrails_prompts, cypher_prompts, invoke_configs)


def test_graph_run_generate_branch_executes_retrieve() -> None:
    stub = _build_rag_graph_stub(
        guardrails_reply='{"decision":"generate"}',
        cypher_reply=SAFE_CYPHER,
        db_rows=[{"value": 1}],
        max_results=5,
    )

    result = stub.rag.invoke(QUESTION)

    assert result["metadata"]["guardrail_decision"] == "generate_cypher"
    assert result["metadata"]["context"] == [{"value": 1}]
    assert result["metadata"]["cypher_query"] == EXECUTED_CYPHER
    assert "value" in result["answer"]

    assert len(stub.database.queries) == 1
    assert stub.database.queries[0].endswith("LIMIT 5")
    assert len(stub.guardrails_prompts) == 1
    assert len(stub.cypher_prompts) == 1
    assert QUESTION in stub.guardrails_prompts[0]
    assert QUESTION in stub.cypher_prompts[0]


def test_graph_run_end_branch_skips_retrieve() -> None:
    stub = _build_rag_graph_stub(
        guardrails_reply='{"decision":"end"}',
        cypher_reply=SAFE_CYPHER,
        db_rows=[{"unused": 1}],
    )

    result = stub.rag.invoke(QUESTION)

    assert result["answer"] == EMPTY_ANSWER
    assert result["metadata"]["guardrail_decision"] == "end"
    assert result["metadata"]["cypher_query"] is None
    assert result["metadata"]["context"] == []

    assert stub.database.queries == []
    assert len(stub.guardrails_prompts) == 1
    assert stub.cypher_prompts == []


def test_async_graph_run_generate_branch_matches_sync_path() -> None:
    """ainvoke is the entry point the MCP server actually calls."""
    stub = _build_rag_graph_stub(
        guardrails_reply='{"decision":"generate"}',
        cypher_reply=SAFE_CYPHER,
        db_rows=[{"value": 1}],
    )

    result = asyncio.run(stub.rag.ainvoke(QUESTION, session_id="s-1", trace_id="tr-1"))

    assert result["metadata"]["guardrail_decision"] == "generate_cypher"
    assert result["metadata"]["cypher_query"] == EXECUTED_CYPHER
    assert result["metadata"]["context"] == [{"value": 1}]
    assert stub.database.queries[0].endswith("LIMIT 5")


def test_async_graph_run_end_branch_skips_retrieve() -> None:
    stub = _build_rag_graph_stub(
        guardrails_reply='{"decision":"end"}',
        cypher_reply=SAFE_CYPHER,
        db_rows=[{"unused": 1}],
    )

    result = asyncio.run(stub.rag.ainvoke(QUESTION))

    assert result["answer"] == EMPTY_ANSWER
    assert result["metadata"]["cypher_query"] is None
    assert stub.database.queries == []
    assert stub.cypher_prompts == []


def test_graph_run_reports_no_data_when_retrieval_finds_nothing() -> None:
    """An empty retrieval must reach the user as an explicit abstention, not as "[]"."""
    stub = _build_rag_graph_stub(
        guardrails_reply='{"decision":"generate"}',
        cypher_reply=SAFE_CYPHER,
        db_rows=[],
    )

    result = stub.rag.invoke(QUESTION)

    assert result["answer"] == NO_GRAPH_DATA_MESSAGE
    assert result["metadata"]["retrieval_strategy"] == "empty"
    assert result["metadata"]["cypher_query"] == EXECUTED_CYPHER
    assert result["metadata"]["context"] == []


def test_graph_run_reports_the_strategy_that_produced_the_context() -> None:
    stub = _build_rag_graph_stub(
        guardrails_reply='{"decision":"generate"}',
        cypher_reply=SAFE_CYPHER,
        db_rows=[{"value": 1}],
    )

    result = stub.rag.invoke(QUESTION)

    assert result["metadata"]["retrieval_strategy"] == "primary"


def test_async_graph_run_propagates_session_id_to_every_node() -> None:
    """Only ainvoke puts tracing context into graph state - invoke passes the question alone."""
    stub = _build_rag_graph_stub(
        guardrails_reply='{"decision":"generate"}',
        cypher_reply=SAFE_CYPHER,
        db_rows=[{"value": 1}],
    )

    asyncio.run(stub.rag.ainvoke(QUESTION, session_id="s-1", trace_id="tr-1"))

    session_ids = [config["metadata"]["langfuse_session_id"] for config in stub.invoke_configs]
    assert session_ids == ["s-1", "s-1"]


def test_async_graph_run_enforces_timeout() -> None:
    stub = _build_rag_graph_stub(
        guardrails_reply='{"decision":"end"}',
        cypher_reply=SAFE_CYPHER,
        db_rows=[],
        graph_timeout_sec=0.0,
    )

    with pytest.raises(TimeoutError, match=TIMEOUT_MESSAGE):
        asyncio.run(stub.rag.ainvoke(QUESTION))
