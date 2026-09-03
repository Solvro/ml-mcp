"""RAG owns a Neo4j driver from construction to close.

Issue #64: nothing ever closed it. One long-lived process never noticed, but every restart in a
loop leaked a driver, and there was no way to ask the instance whether its connection still
worked - which is what the health route needs.
"""

import pytest

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
