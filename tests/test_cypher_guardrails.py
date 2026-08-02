import pytest

from src.mcp_server.tools.knowledge_graph.cypher_guardrails import (
    UnsafeCypherQueryError,
    ensure_limit,
    strip_code_fences,
    validate_read_only,
)

READ_QUERY = "MATCH (n:Node) RETURN n.value"
READ_QUERY_WITH_LIMIT = f"{READ_QUERY} LIMIT 10"

BLOCKED_KEYWORDS = [
    "ALTER",
    "CREATE",
    "DELETE",
    "DENY",
    "DETACH",
    "DROP",
    "FOREACH",
    "GRANT",
    "LOAD",
    "MERGE",
    "REMOVE",
    "REVOKE",
    "SET",
]


@pytest.mark.parametrize(
    "query",
    [
        READ_QUERY_WITH_LIMIT,
        "OPTIONAL MATCH (a:Node)-[:REL]->(b:Node) RETURN a.value, b.value LIMIT 10",
        "WITH [1, 2] AS values UNWIND values AS v "
        "MATCH (n:Node) WHERE n.value = v RETURN n.value LIMIT 10",
    ],
)
def test_accepts_read_only_query(query):
    validate_read_only(query)


@pytest.mark.parametrize("keyword", BLOCKED_KEYWORDS)
def test_ignores_write_keyword_inside_string_literal(keyword):
    validate_read_only(f"MATCH (n:Node) WHERE n.value CONTAINS '{keyword}' RETURN n.value LIMIT 10")


@pytest.mark.parametrize("keyword", BLOCKED_KEYWORDS)
def test_ignores_write_keyword_inside_comment(keyword):
    validate_read_only(f"// {keyword}\n{READ_QUERY_WITH_LIMIT}")


def test_accepts_fenced_query():
    validate_read_only(f"```cypher\n{READ_QUERY_WITH_LIMIT}\n```")


def test_strip_code_fences_removes_wrapper():
    assert strip_code_fences(f"```cypher\n{READ_QUERY}\n```") == READ_QUERY


@pytest.mark.parametrize(
    "query",
    [
        "MATCH (n:Node) DETACH DELETE n RETURN n",
        "CREATE (n:Node {value: 1}) RETURN n",
        "MERGE (n:Node {value: 1}) RETURN n",
        "MATCH (n:Node) SET n.value = 1 RETURN n",
        "MATCH (n:Node) REMOVE n.value RETURN n",
        "DROP INDEX node_value IF EXISTS RETURN 1",
        "MATCH (n:Node) FOREACH (x IN [1] | SET n.value = x) RETURN n",
        "LOAD CSV FROM 'file:///tmp/example.csv' AS row RETURN row",
        "ALTER DATABASE neo4j SET ACCESS READ WRITE RETURN 1",
        "GRANT TRAVERSE ON GRAPH * NODES * TO public RETURN 1",
        "DENY WRITE ON GRAPH neo4j TO public RETURN 1",
        "REVOKE MATCH {*} ON GRAPH neo4j FROM public RETURN 1",
    ],
)
def test_rejects_write_or_admin_operations(query):
    with pytest.raises(UnsafeCypherQueryError):
        validate_read_only(query)


@pytest.mark.parametrize(
    "query",
    [
        "CALL db.procedure() YIELD value RETURN value",
        "SHOW DATABASES",
    ],
)
def test_rejects_disallowed_start_clause(query):
    with pytest.raises(UnsafeCypherQueryError, match="read-only clause"):
        validate_read_only(query)


def test_rejects_multiple_statements():
    with pytest.raises(UnsafeCypherQueryError, match="multiple"):
        validate_read_only(f"{READ_QUERY}; {READ_QUERY}")


def test_rejects_query_without_return():
    with pytest.raises(UnsafeCypherQueryError, match="return"):
        validate_read_only("MATCH (n:Node)")


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_rejects_empty_query(query):
    with pytest.raises(UnsafeCypherQueryError, match="empty"):
        validate_read_only(query)


def test_ensure_limit_appends_when_missing():
    assert ensure_limit(READ_QUERY, max_results=10) == READ_QUERY_WITH_LIMIT


def test_ensure_limit_preserves_existing_clause():
    assert ensure_limit(READ_QUERY_WITH_LIMIT, max_results=5) == READ_QUERY_WITH_LIMIT


def test_ensure_limit_ignores_limit_inside_string_literal():
    query = "MATCH (n:Node) WHERE n.value CONTAINS 'LIMIT' RETURN n.value"
    assert ensure_limit(query, max_results=10).endswith("LIMIT 10")


def test_ensure_limit_rejects_non_positive_max_results():
    with pytest.raises(ValueError):
        ensure_limit(READ_QUERY, max_results=0)
