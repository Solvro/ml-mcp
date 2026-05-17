import pytest

from src.mcp_server.tools.knowledge_graph.rag import (
    UnsafeCypherQueryError,
    validate_read_only_cypher,
)


def test_read_only_cypher_accepts_match_return_limit():
    validate_read_only_cypher(
        "MATCH (p:Person) WHERE toLower(p.title) CONTAINS toLower('kowalski') "
        "RETURN p.title LIMIT 5"
    )


@pytest.mark.parametrize(
    "cypher_query",
    [
        "MATCH (n) DETACH DELETE n RETURN n",
        "CREATE (n:Person {title: 'x'}) RETURN n",
        "MERGE (n:Person {title: 'x'}) RETURN n",
        "MATCH (n) SET n.title = 'x' RETURN n",
        "DROP INDEX course_title IF EXISTS",
    ],
)
def test_read_only_cypher_rejects_mutating_keywords(cypher_query):
    with pytest.raises(UnsafeCypherQueryError):
        validate_read_only_cypher(cypher_query)


def test_read_only_cypher_ignores_keywords_inside_strings_and_comments():
    validate_read_only_cypher(
        """
        // DELETE should not count inside a comment
        MATCH (c:Course)
        WHERE c.title CONTAINS 'CREATE'
        RETURN c.title
        """
    )


def test_read_only_cypher_rejects_multiple_statements():
    with pytest.raises(UnsafeCypherQueryError):
        validate_read_only_cypher("MATCH (n) RETURN n; MATCH (m) RETURN m")


def test_read_only_cypher_requires_return_data():
    with pytest.raises(UnsafeCypherQueryError):
        validate_read_only_cypher("MATCH (n)")


def test_read_only_cypher_requires_read_only_start_clause():
    with pytest.raises(UnsafeCypherQueryError):
        validate_read_only_cypher("SHOW USERS RETURN user")
