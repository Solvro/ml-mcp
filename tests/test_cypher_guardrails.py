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


def test_rejects_disallowed_start_clause():
    with pytest.raises(UnsafeCypherQueryError, match="read-only clause"):
        validate_read_only("SHOW DATABASES")


@pytest.mark.parametrize(
    "query",
    [
        "CALL db.procedure() YIELD value RETURN value",
        "CALL apoc.cypher.runWrite('CREATE (n)', {}) YIELD value RETURN value",
        "MATCH (n) CALL db.index.fulltext.queryNodes('i', 'x') YIELD node RETURN node",
        # Cypher allows a no-argument procedure to be called without parentheses. Requiring them
        # in the detector meant no name was captured, so the allowlist vetted nothing and these
        # passed. Both were reported in review on PR #57.
        "CALL db.labels YIELD label RETURN label",
        "CALL apoc.periodic.iterate YIELD x RETURN x",
        "MATCH (n) CALL db.labels YIELD label RETURN label",
        "CALL  dbms.components  YIELD name RETURN name",
    ],
)
def test_rejects_procedure_calls_without_an_allowlist(query):
    """Generated Cypher is validated with an empty allowlist, so no CALL is reachable from it."""
    with pytest.raises(UnsafeCypherQueryError, match="blocked Cypher procedure call"):
        validate_read_only(query)


@pytest.mark.parametrize(
    "query",
    [
        "CALL db.labels YIELD label RETURN label",
        "CALL db.labels() YIELD label RETURN label",
    ],
)
def test_an_allowlist_does_not_cover_a_procedure_it_does_not_name(query):
    """Whether the call has parentheses must not change which procedures are permitted."""
    with pytest.raises(UnsafeCypherQueryError, match="db.labels"):
        validate_read_only(query, allowed_procedures=frozenset({"db.index.fulltext.queryNodes"}))


@pytest.mark.parametrize(
    "query",
    [
        "CALL db.labels YIELD label RETURN label",
        "CALL db.labels() YIELD label RETURN label",
    ],
)
def test_an_allowlisted_procedure_is_permitted_with_or_without_parentheses(query):
    validate_read_only(query, allowed_procedures=frozenset({"db.labels"}))


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
    assert ensure_limit(READ_QUERY, max_results=10) == f"{READ_QUERY}\nLIMIT 10"


def test_ensure_limit_clamps_a_larger_limit_to_the_configured_cap():
    # Issue #85: any LIMIT at all used to satisfy the check, so `LIMIT 999999999` passed and
    # rag.max_results decided nothing.
    assert ensure_limit(f"{READ_QUERY} LIMIT 999999999", max_results=5) == f"{READ_QUERY} LIMIT 5"


def test_ensure_limit_clamps_the_limit_the_prompt_asks_for():
    # The two agree in graph_config.yaml (test_llm_determinism_config pins that), so this is the
    # case where an edit moved the config and the model is still writing the old number.
    assert ensure_limit(READ_QUERY_WITH_LIMIT, max_results=5) == f"{READ_QUERY} LIMIT 5"


def test_ensure_limit_leaves_a_smaller_limit_alone():
    # A model narrowing its own result set is not what the cap is defending against.
    narrowed = f"{READ_QUERY} LIMIT 3"

    assert ensure_limit(narrowed, max_results=5) == narrowed


def test_ensure_limit_leaves_an_intermediate_limit_alone():
    # `WITH ... LIMIT 100` shapes an intermediate result; rewriting it changes what the query
    # means, while what reaches the answering model is decided by the trailing clause.
    query = "MATCH (n:Node) WITH n ORDER BY n.rank DESC LIMIT 100 RETURN n.value"

    assert ensure_limit(query, max_results=5) == f"{query}\nLIMIT 5"


def test_ensure_limit_caps_a_query_whose_only_limit_is_intermediate():
    query = "MATCH (n:Node) WITH n LIMIT 1000 RETURN n.value"

    capped = ensure_limit(query, max_results=5)

    assert capped.endswith("LIMIT 5")
    assert "LIMIT 1000" in capped


def test_ensure_limit_ignores_limit_inside_string_literal():
    query = "MATCH (n:Node) WHERE n.value CONTAINS 'LIMIT 900' RETURN n.value"

    assert ensure_limit(query, max_results=10) == f"{query}\nLIMIT 10"


def test_ensure_limit_survives_a_trailing_comment():
    query = f"{READ_QUERY} // zwroc wszystko"

    capped = ensure_limit(query, max_results=5)

    assert capped.endswith("LIMIT 5")
    assert capped.splitlines()[0] == query


def test_ensure_limit_clamps_past_a_trailing_comment():
    capped = ensure_limit(f"{READ_QUERY} LIMIT 900 // zwroc wszystko", max_results=5)

    assert capped == f"{READ_QUERY} LIMIT 5"


def test_ensure_limit_drops_a_trailing_semicolon():
    assert ensure_limit(f"{READ_QUERY} LIMIT 900 ;", max_results=5) == f"{READ_QUERY} LIMIT 5"


def test_ensure_limit_rejects_non_positive_max_results():
    with pytest.raises(ValueError):
        ensure_limit(READ_QUERY, max_results=0)


def test_ensure_limit_caps_only_the_last_branch_of_a_union():
    """Known and accepted, review of PR #90: capping the whole thing needs a CALL subquery.

    Pinned so the behaviour is a decision someone can find rather than a surprise. It is not a
    regression - before ensure_limit clamped anything, this case was uncapped entirely.
    """
    union = "MATCH (n:A) RETURN n LIMIT 100 UNION MATCH (n:B) RETURN n LIMIT 100"

    capped = ensure_limit(union, max_results=5)

    assert capped == "MATCH (n:A) RETURN n LIMIT 100 UNION MATCH (n:B) RETURN n LIMIT 5"
    validate_read_only(capped)
