import pytest

from src.config.system_labels import SYSTEM_LABELS
from src.data_pipeline.canonical_nodes import rewrite_merge_to_canonical_key
from src.data_pipeline.ingestion_guardrails import (
    StrandedStatementError,
    UnsafeIngestionStatementError,
    refuse_unsafe_statements,
    validate_ingestion_statement,
)

NOTHING_BOUND: frozenset[str] = frozenset()


def _refusal(statement: str, bound: frozenset[str] = NOTHING_BOUND) -> str:
    with pytest.raises(UnsafeIngestionStatementError) as error:
        validate_ingestion_statement(statement, bound)
    return error.value.reason


def test_accepts_what_the_canonical_key_rewrite_produces():
    statement = rewrite_merge_to_canonical_key(
        r"MERGE (node1:Person {title: 'Jan Kowalski', context: 'it\'s at https://pwr.edu.pl',"
        " year: 2020})"
    )

    assert validate_ingestion_statement(statement, NOTHING_BOUND) == {"node1"}


@pytest.mark.parametrize(
    "statement",
    [
        "MERGE (a)-[:BELONGS_TO]->(b)",
        "MERGE (b)<-[:BELONGS_TO]-(a)",
        "MERGE (a)-[:HAS {since: date('2020-01-01')}]->(b)",
        "MERGE (a)-[:PART_OF]->(:Semester {title: 'Semestr zimowy'})",
        "MERGE (a)-[r:PART_OF]->(b) ON CREATE SET r.note = toLower(a.title) + ' / ' + b.title",
    ],
)
def test_accepts_relationships_between_the_pages_own_nodes(statement):
    validate_ingestion_statement(statement, frozenset({"a", "b"}))


def test_accepts_a_pattern_that_merges_its_nodes_and_relationship_at_once():
    statement = "MERGE (a:Course {title: 'X'})-[:PART_OF]->(b:Semester {title: 'Y'})"

    assert validate_ingestion_statement(statement, NOTHING_BOUND) == {"a", "b"}


def test_accepts_values_built_from_literals_and_the_pages_own_properties():
    statement = (
        'MERGE (`node 1`:Topic {key: "x"}) '
        "SET `node 1`.n = -1.5e3, `node 1`.tags = ['a', null, true], "
        "`node 1`.flag = `node 1`.x IS NOT NULL AND NOT `node 1`.y STARTS WITH 'a'"
    )

    validate_ingestion_statement(statement, NOTHING_BOUND)


def test_a_clause_inside_a_string_literal_stays_text():
    """Read like Neo4j reads it: `\\'` does not close the literal, so this is one string."""
    statement = r"MERGE (a:Topic {key: 'x'}) SET a.t = 'x\' DETACH DELETE a //'"

    validate_ingestion_statement(statement, NOTHING_BOUND)


@pytest.mark.parametrize(
    ("statement", "reason"),
    [
        ("MATCH (n) DETACH DELETE n", "must start with MERGE, not `MATCH`"),
        (
            "LOAD CSV FROM 'http://127.0.0.1:9/never.csv' AS row RETURN count(row)",
            "must start with MERGE, not `LOAD`",
        ),
        ("CALL apoc.refactor.mergeNodes([], {})", "must start with MERGE, not `CALL`"),
        ("DROP INDEX entity_key_topic", "must start with MERGE, not `DROP`"),
        ("MERGE (a:Topic {key: 'x'}) DETACH DELETE a", "`DETACH` is not allowed"),
        # A URL literal must not swallow what follows it on the line.
        ("MERGE (a:Topic {key: 'x'}) SET a.url = 'http://x' DELETE a", "`DELETE` is not allowed"),
        ("MERGE (a:Topic {key: 'x'}) REMOVE a.key", "`REMOVE` is not allowed"),
        ("MERGE (a:Topic {key: 'x'}) WITH a MATCH (m) DELETE m", "`WITH` is not allowed"),
        ("MERGE (a:Topic {key: 'x'}) CREATE INDEX i FOR (n:Topic) ON (n.x)", "`CREATE`"),
        ("MERGE (a:Topic {key: 'x'}) FOREACH (x IN [1] SET a.y = x)", "`FOREACH`"),
        ("MERGE (a:Topic {key: 'x'}) // note\nDETACH DELETE a", "comments are not allowed"),
        ("MERGE (a:Topic {key: 'x'});MATCH (n) DELETE n", "unexpected character ';'"),
        ("MERGE (a:Topic {key: $key})", "unexpected character '$'"),
        ("MERGE (a:Topic {key: 'x'}) SET a.t = 'open", "unterminated '"),
        ("MERGE (a:Topic {key: apoc.text.join(['a'], '')})", "`apoc.text.join` is not allowed"),
        ("MERGE (a:Topic {key: 'x'}) SET a.n = COUNT { MATCH (m) RETURN m }", "`COUNT`"),
        ("MERGE (a:Topic {key: 'x'}) SET a.n = (a)--(a)", "not the node itself"),
        ("MERGE (a:Topic {key: 'x'}) SET a:Course", "not labels or a whole map"),
        ("MERGE (a:Topic {key: 'x'}) SET a += {title: 'y'}", "not labels or a whole map"),
        ("MERGE (a:Topic) SET a.title = 'y'", "needs both a label and properties"),
        # No label, so the properties could match a bookkeeping node.
        ("MERGE (d {hash: 'abc'}) SET d.status = 'processed'", "needs both a label"),
    ],
)
def test_refuses_anything_but_merging_the_pages_own_entities(statement, reason):
    assert reason in _refusal(statement)


@pytest.mark.parametrize("label", sorted(SYSTEM_LABELS))
def test_refuses_a_bookkeeping_label_in_any_spelling_or_position(label):
    for statement in (
        f"MERGE (n:{label} {{key: 'x'}})",
        f"MERGE (n:Topic:`{label}` {{key: 'x'}})",
        f"MERGE (n:{label.lower()} {{key: 'x'}})",
        f"MERGE (a)-[:FROM_SOURCE]->(:{label} {{source_id: 'x'}})",
    ):
        assert "bookkeeping" in _refusal(statement, frozenset({"a"}))


def test_a_bare_node_the_page_never_bound_is_refused_and_named():
    """`MERGE (node13)` binds nothing of its own: it matches every node in the graph."""
    with pytest.raises(UnsafeIngestionStatementError) as error:
        validate_ingestion_statement("MERGE (node13)-[:R]->(b)", frozenset({"b"}))

    assert error.value.unbound_variable == "node13"


def test_set_on_a_variable_the_page_never_bound_is_refused_and_named():
    with pytest.raises(UnsafeIngestionStatementError) as error:
        validate_ingestion_statement("MERGE (a:Topic {key: 'x'}) SET b.title = 'y'", NOTHING_BOUND)

    assert error.value.unbound_variable == "b"


def test_deep_nesting_is_refused_rather_than_crashing_the_page():
    statement = "MERGE (a:Topic {key: " + "(" * 5000 + "1" + ")" * 5000 + "})"

    assert "nested too deeply" in _refusal(statement)


def test_refuse_unsafe_statements_drops_the_statement_and_keeps_the_page():
    statements = [
        "MERGE (a:Topic {key: 'a'})",
        "MATCH (n) DETACH DELETE n",
        "MERGE (b:Topic {key: 'b'})",
        "MERGE (a)-[:RELATED_TO]->(b)",
    ]

    kept, refused = refuse_unsafe_statements(statements)

    assert kept == [statements[0], statements[2], statements[3]]
    assert [item.statement for item in refused] == ["MATCH (n) DETACH DELETE n"]
    assert "MATCH" in refused[0].reason


def test_refuse_unsafe_statements_fails_the_page_when_a_refused_statement_bound_a_used_variable():
    statements = [
        "MERGE (a:Topic {key: 'a'}) WITH a MATCH (m) DETACH DELETE m",
        "MERGE (b:Topic {key: 'b'})",
        "MERGE (a)-[:RELATED_TO]->(b)",
    ]

    with pytest.raises(StrandedStatementError) as error:
        refuse_unsafe_statements(statements)

    assert error.value.unbound_variable == "a"


def test_refuse_unsafe_statements_reads_variables_in_page_order():
    """A relationship written before its nodes would match any node, so it goes on its own."""
    statements = [
        "MERGE (a)-[:RELATED_TO]->(b)",
        "MERGE (a:Topic {key: 'a'})",
        "MERGE (b:Topic {key: 'b'})",
    ]

    kept, refused = refuse_unsafe_statements(statements)

    assert kept == statements[1:]
    assert [item.statement for item in refused] == [statements[0]]
