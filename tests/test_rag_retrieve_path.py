from typing import Any

import pytest
from neo4j.exceptions import ClientError, ServiceUnavailable

from src.mcp_server.tools.knowledge_graph.rag import (
    RAG,
    KnowledgeGraphQueryError,
    KnowledgeGraphUnavailableError,
)

READ_QUERY = "MATCH (n:Node) RETURN n"
READ_QUERY_WITH_LIMIT = f"{READ_QUERY} LIMIT 2"
FENCED_QUERY = "```cypher\nMATCH (n) RETURN n\n```"

MUTATING_QUERIES = [
    "MATCH (n) DETACH DELETE n RETURN n",
    "MATCH (n:Node) SET n.value = 1 RETURN n",
    "CREATE (n:Node) RETURN n",
]


class FakeDatabase:
    def __init__(
        self, response: list[dict[str, Any]] | None = None, error: Exception | None = None
    ) -> None:
        self.response = [] if response is None else response
        self.error = error
        self.calls: list[str] = []

    def query(self, cypher_query: str) -> list[dict[str, Any]]:
        self.calls.append(cypher_query)
        if self.error is not None:
            raise self.error
        return self.response


def _statement_error(message: str = "bad query") -> ClientError:
    return ClientError._hydrate_neo4j(
        code="Neo.ClientError.Statement.SyntaxError",
        message=message,
    )


def _build_rag_for_test(
    max_results: int = 5,
    db_response: list[dict[str, Any]] | None = None,
    db_error: Exception | None = None,
) -> tuple[RAG, FakeDatabase]:
    fake_db = FakeDatabase(response=db_response, error=db_error)

    rag = RAG.__new__(RAG)
    rag.database = fake_db
    rag.max_results = max_results
    rag.enable_fallback_search = False

    return rag, fake_db


@pytest.mark.parametrize("query", MUTATING_QUERIES)
def test_retrieve_blocks_mutating_query_before_db_call(query):
    rag, fake_db = _build_rag_for_test()

    with pytest.raises(KnowledgeGraphQueryError, match="blocked") as raised:
        rag.retrieve({"generated_cypher": query})

    assert fake_db.calls == []
    assert raised.value.cypher == query


def test_retrieve_raises_when_neo4j_is_unreachable():
    rag, fake_db = _build_rag_for_test(db_error=ServiceUnavailable("neo4j unavailable"))

    with pytest.raises(KnowledgeGraphUnavailableError, match="neo4j unavailable"):
        rag.retrieve({"generated_cypher": READ_QUERY})

    assert len(fake_db.calls) == 1


def test_retrieve_raises_a_query_error_for_a_statement_neo4j_rejects():
    """This stub has the fallback search disabled, so the rejected statement stays a failure;
    with it enabled it is escalated to the label-agnostic search instead (issue #3, see
    test_rag_empty_retrieval_escalation)."""
    rag, fake_db = _build_rag_for_test(db_error=_statement_error("invalid input"))

    with pytest.raises(KnowledgeGraphQueryError, match="invalid input") as raised:
        rag.retrieve({"generated_cypher": READ_QUERY})

    assert len(fake_db.calls) == 1
    assert raised.value.cypher == fake_db.calls[0], "the operator sees what actually ran"
    assert not isinstance(raised.value, KnowledgeGraphUnavailableError), (
        "a rejected statement is not an outage; the caller's messages differ"
    )


def test_retrieve_raises_when_the_query_outlives_its_timeout():
    timed_out = ClientError._hydrate_neo4j(
        code="Neo.ClientError.Transaction.TransactionTimedOutClientConfiguration",
        message="The transaction has been terminated.",
    )
    rag, fake_db = _build_rag_for_test(db_error=timed_out)

    with pytest.raises(KnowledgeGraphUnavailableError, match="terminated"):
        rag.retrieve({"generated_cypher": READ_QUERY})

    assert len(fake_db.calls) == 1


def test_retrieve_reports_non_neo4j_runtime_failure():
    rag, fake_db = _build_rag_for_test(db_error=RuntimeError("unexpected failure"))

    with pytest.raises(KnowledgeGraphQueryError, match="unexpected failure"):
        rag.retrieve({"generated_cypher": READ_QUERY})

    assert len(fake_db.calls) == 1


def test_retrieve_blocks_missing_cypher():
    rag, fake_db = _build_rag_for_test()

    with pytest.raises(KnowledgeGraphQueryError, match="blocked"):
        rag.retrieve({})

    assert fake_db.calls == []


def test_retrieve_executes_safe_query_with_enforced_limit():
    rag, fake_db = _build_rag_for_test(max_results=5, db_response=[{"id": 1}])

    result = rag.retrieve({"generated_cypher": READ_QUERY})

    assert len(fake_db.calls) == 1
    assert fake_db.calls[0].strip().endswith("LIMIT 5")
    assert result["context"] == [{"id": 1}]


def test_retrieve_preserves_existing_limit():
    rag, fake_db = _build_rag_for_test(max_results=5, db_response=[{"id": 1}])

    result = rag.retrieve({"generated_cypher": READ_QUERY_WITH_LIMIT})

    assert len(fake_db.calls) == 1
    assert fake_db.calls[0].strip().endswith("LIMIT 2")
    assert result["context"] == [{"id": 1}]


def test_retrieve_strips_code_fences_before_query():
    rag, fake_db = _build_rag_for_test(max_results=3, db_response=[{"name": "x"}])

    result = rag.retrieve({"generated_cypher": FENCED_QUERY})

    assert len(fake_db.calls) == 1
    assert "```" not in fake_db.calls[0]
    assert fake_db.calls[0].strip().endswith("LIMIT 3")
    assert result["context"] == [{"name": "x"}]
