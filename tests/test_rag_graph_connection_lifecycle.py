"""RAG owns a Neo4j driver from construction to close.

Issue #64: nothing ever closed it. One long-lived process never noticed, but every restart in a
loop leaked a driver, and there was no way to ask the instance whether its connection still
worked - which is what the health route needs.
"""

from __future__ import annotations

from typing import Any

import pytest

import src.mcp_server.tools.knowledge_graph.rag as rag_module
from src.mcp_server.tools.knowledge_graph.rag import RAG


class FakeDatabase:
    """Neo4jGraph stand-in that records what was asked of it."""

    def __init__(self, *, error: Exception | None = None):
        self.error = error
        self.queries: list[str] = []
        self.close_calls = 0

    def query(self, query: str, params: dict | None = None):
        self.queries.append(query)
        if self.error:
            raise self.error
        return [{"ok": 1}]

    def close(self) -> None:
        self.close_calls += 1


def _rag_with(database) -> RAG:
    """A RAG holding just a database - the constructor would want a live graph and an LLM."""
    rag = RAG.__new__(RAG)
    rag.database = database
    return rag


def test_ping_runs_a_query_against_the_graph() -> None:
    database = FakeDatabase()

    _rag_with(database).ping_database()

    assert len(database.queries) == 1, "a health ping that queries nothing proves nothing"


def test_ping_propagates_the_driver_error() -> None:
    """The caller reports why the graph is unreachable, so the reason must not be swallowed."""
    database = FakeDatabase(error=RuntimeError("Unable to connect to bolt://neo4j:7687"))

    with pytest.raises(RuntimeError, match="Unable to connect"):
        _rag_with(database).ping_database()


def test_close_releases_the_driver() -> None:
    database = FakeDatabase()

    _rag_with(database).close()

    assert database.close_calls == 1


def test_close_can_be_called_twice() -> None:
    """Shutdown paths overlap; closing twice must not be an error."""
    database = FakeDatabase()
    rag = _rag_with(database)

    rag.close()
    rag.close()

    assert database.close_calls == 2, "Neo4jGraph.close is itself idempotent"


def test_close_without_a_database_is_not_an_error() -> None:
    """A RAG that failed partway through construction still has to be closeable."""
    rag = RAG.__new__(RAG)

    rag.close()


class CapturingNeo4jGraph:
    def __init__(self, sink: dict[str, Any], **kwargs: Any) -> None:
        sink.update(kwargs)


@pytest.fixture
def capture_neo4j_graph_init(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        rag_module,
        "Neo4jGraph",
        lambda **kwargs: CapturingNeo4jGraph(captured, **kwargs),
    )
    monkeypatch.setattr(
        rag_module.RAG, "_build_llm_with_fallback", lambda self, use_accurate=False: object()
    )
    monkeypatch.setattr(rag_module.RAG, "_initialize_prompt_templates", lambda self: None)
    monkeypatch.setattr(rag_module.RAG, "_build_processing_graph", lambda self: object())
    monkeypatch.setattr(rag_module.RAG, "ensure_fulltext_index", lambda self: True)

    return captured


def _build_rag(**kwargs: Any) -> RAG:
    return RAG(
        api_key="test-key",
        neo4j_url="bolt://neo4j:7687",
        neo4j_username="neo4j",
        neo4j_password="secret",
        **kwargs,
    )


def test_rag_passes_neo4j_runtime_limits_to_driver(
    capture_neo4j_graph_init: dict[str, Any],
) -> None:
    _build_rag(
        llm_timeout_sec=30,
        graph_timeout_sec=20,
        neo4j_query_timeout_sec=7,
        neo4j_connection_timeout_sec=4,
        neo4j_max_transaction_retry_sec=3,
    )

    assert capture_neo4j_graph_init["timeout"] == 7
    assert capture_neo4j_graph_init["driver_config"] == {
        "connection_timeout": 4,
        "max_transaction_retry_time": 3,
    }


def test_rag_caps_query_and_retry_timeout_to_graph_budget(
    capture_neo4j_graph_init: dict[str, Any],
) -> None:
    """Connection and retry are spent in sequence, so the graph budget has to cover both."""
    rag = _build_rag(
        llm_timeout_sec=30,
        graph_timeout_sec=5,
        neo4j_query_timeout_sec=12,
        neo4j_connection_timeout_sec=4,
        neo4j_max_transaction_retry_sec=9,
    )

    assert rag.neo4j_query_timeout_sec == 5
    assert rag.neo4j_max_transaction_retry_sec == 1, "4s connecting leaves 1s of the 5s budget"
    assert capture_neo4j_graph_init["timeout"] == 5
    assert capture_neo4j_graph_init["driver_config"]["max_transaction_retry_time"] == 1


def test_rag_drops_retries_when_connecting_alone_fills_the_budget(
    capture_neo4j_graph_init: dict[str, Any],
) -> None:
    """A retry that could not start before the request expires is worse than no retry."""
    rag = _build_rag(
        llm_timeout_sec=30,
        graph_timeout_sec=3,
        neo4j_query_timeout_sec=3,
        neo4j_connection_timeout_sec=8,
        neo4j_max_transaction_retry_sec=9,
    )

    assert rag.neo4j_max_transaction_retry_sec == 0
    assert capture_neo4j_graph_init["driver_config"]["max_transaction_retry_time"] == 0
