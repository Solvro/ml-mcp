from typing import Any

import pytest

from src.mcp_server.tools.knowledge_graph.rag import RAG


class FakeDatabase:
    def __init__(
        self, response: list[dict[str, Any]] | None = None, error: Exception | None = None
    ) -> None:
        self.response = [] if response is None else response
        self.error = error
        self.calls: list[str] = []

    def query(self, cypher_query: str):
        self.calls.append(cypher_query)
        if self.error is not None:
            raise self.error
        return self.response


def _build_rag_for_test(
    max_results: int = 5,
    db_response: list[dict[str, Any]] | None = None,
    db_error: Exception | None = None,
) -> tuple[RAG, FakeDatabase]:
    fake_db = FakeDatabase(response=db_response, error=db_error)

    rag = RAG.__new__(RAG)
    rag.database = fake_db
    rag.max_results = max_results
    rag.enable_debug = False

    return rag, fake_db


@pytest.mark.parametrize(
    "query",
    [
        "MATCH (n) DETACH DELETE n RETURN n",
        "MATCH (n:Node) SET n.value = 1 RETURN n",
        "CREATE (n:Node) RETURN n",
    ],
)
def test_retrieve_blocks_mutating_query_before_db_call(query):
    rag, fake_db = _build_rag_for_test()

    result = rag.retrieve({"generated_cypher": query})

    assert fake_db.calls == []
    assert result["context"] == []
    assert "Blocked unsafe Cypher" in result["generated_cypher"]


def test_retrieve_reports_database_failure():
    rag, fake_db = _build_rag_for_test(db_error=RuntimeError("neo4j unavailable"))

    result = rag.retrieve({"generated_cypher": "MATCH (n:Node) RETURN n"})

    assert len(fake_db.calls) == 1
    assert result["context"] == []
    assert result["generated_cypher"] == "Query failed: neo4j unavailable"


def test_retrieve_blocks_missing_cypher():
    rag, fake_db = _build_rag_for_test()

    result = rag.retrieve({})

    assert fake_db.calls == []
    assert "Blocked unsafe Cypher" in result["generated_cypher"]


def test_retrieve_executes_safe_query_with_enforced_limit():
    rag, fake_db = _build_rag_for_test(max_results=5, db_response=[{"id": 1}])

    state = {"generated_cypher": "MATCH (n:Node) RETURN n"}
    result = rag.retrieve(state)

    assert len(fake_db.calls) == 1
    assert fake_db.calls[0].strip().endswith("LIMIT 5")
    assert result["context"] == [{"id": 1}]


def test_retrieve_preserves_existing_limit():
    rag, fake_db = _build_rag_for_test(max_results=5, db_response=[{"id": 1}])

    state = {"generated_cypher": "MATCH (n:Node) RETURN n LIMIT 2"}
    result = rag.retrieve(state)

    assert len(fake_db.calls) == 1
    assert fake_db.calls[0].strip().endswith("LIMIT 2")
    assert result["context"] == [{"id": 1}]


def test_retrieve_strips_code_fences_before_query():
    rag, fake_db = _build_rag_for_test(max_results=3, db_response=[{"name": "x"}])

    state = {"generated_cypher": "```cypher\nMATCH (n) RETURN n\n```"}
    result = rag.retrieve(state)

    assert len(fake_db.calls) == 1
    assert "```" not in fake_db.calls[0]
    assert fake_db.calls[0].strip().endswith("LIMIT 3")
    assert result["context"] == [{"name": "x"}]
