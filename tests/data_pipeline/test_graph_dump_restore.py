"""Restore reads the cypher-shell dump itself and sends its statements over the driver (#21).

The image's APOC plugin has no ``apoc.cypher.runFile`` — it is an APOC Extended procedure —
so ``restore-graph`` and the pipeline's bootstrap-from-dump path could never load what
``dump-graph`` wrote. The parser below is what stands in for it; the shapes are the ones
``apoc.export.cypher.all(..., {format: 'cypher-shell'})`` actually writes.
"""

import pytest

from src.data_pipeline import graph_dump

REAL_SHAPE = "\n".join(
    [
        ":begin",
        "CREATE FULLTEXT INDEX entity_search FOR (n:Course|Person) ON EACH [n.title, n.context];",
        "CREATE RANGE INDEX FOR (n:Course) ON (n.key);",
        "CREATE CONSTRAINT UNIQUE_IMPORT_NAME FOR (node:`UNIQUE IMPORT LABEL`) REQUIRE (node.`UNIQUE IMPORT ID`) IS UNIQUE;",  # noqa: E501
        ":commit",
        "CALL db.awaitIndexes(300);",
        ":begin",
        'UNWIND [{_id:0, properties:{title:"Analiza; czesc 1", context:"a"}}, {_id:1, properties:{title:"B"}}] AS row',  # noqa: E501
        "CREATE (n:`UNIQUE IMPORT LABEL`{`UNIQUE IMPORT ID`: row._id}) SET n += row.properties SET n:Course;",  # noqa: E501
        ":commit",
        ":begin",
        "UNWIND [{start: {_id:0}, end: {_id:1}, properties:{}}] AS row",
        "MATCH (start:`UNIQUE IMPORT LABEL`{`UNIQUE IMPORT ID`: row.start._id})",
        "MATCH (end:`UNIQUE IMPORT LABEL`{`UNIQUE IMPORT ID`: row.end._id})",
        "CREATE (start)-[r:RELATED_TO]->(end) SET r += row.properties;",
        ":commit",
        ":begin",
        "MATCH (n:`UNIQUE IMPORT LABEL`)  WITH n LIMIT 20000 REMOVE n:`UNIQUE IMPORT LABEL` REMOVE n.`UNIQUE IMPORT ID`;",  # noqa: E501
        ":commit",
        ":begin",
        "DROP CONSTRAINT UNIQUE_IMPORT_NAME;",
        ":commit",
        "",
    ]
)


def test_transactions_follow_begin_commit_and_loose_statements_stand_alone() -> None:
    batches = graph_dump.parse_cypher_shell_dump(REAL_SHAPE)

    assert [len(b) for b in batches] == [3, 1, 1, 1, 1, 1]
    assert batches[1] == ["CALL db.awaitIndexes(300)"]
    assert batches[4] == [
        "MATCH (n:`UNIQUE IMPORT LABEL`)  WITH n LIMIT 20000 "
        "REMOVE n:`UNIQUE IMPORT LABEL` REMOVE n.`UNIQUE IMPORT ID`"
    ]
    assert batches[5] == ["DROP CONSTRAINT UNIQUE_IMPORT_NAME"]


def test_a_statement_spanning_lines_is_joined_and_a_semicolon_in_a_value_does_not_end_it():
    batches = graph_dump.parse_cypher_shell_dump(REAL_SHAPE)

    (node_statement,) = batches[2]
    assert node_statement.startswith('UNWIND [{_id:0, properties:{title:"Analiza; czesc 1"')
    assert node_statement.endswith("SET n:Course")
    assert node_statement.count("\n") == 1

    (rel_statement,) = batches[3]
    assert rel_statement.count("\n") == 3
    assert rel_statement.endswith("SET r += row.properties")


def test_a_semicolon_ending_a_line_inside_a_string_does_not_end_the_statement() -> None:
    text = 'CREATE (n:Note {context:"first line;\nsecond line"});\nRETURN 1;\n'

    batches = graph_dump.parse_cypher_shell_dump(text)

    assert batches == [['CREATE (n:Note {context:"first line;\nsecond line"})'], ["RETURN 1"]]


def test_an_escaped_quote_does_not_open_a_string() -> None:
    text = 'CREATE (n:Note {title:"say \\"hi\\"; now"});\nRETURN 1;\n'

    batches = graph_dump.parse_cypher_shell_dump(text)

    assert batches == [['CREATE (n:Note {title:"say \\"hi\\"; now"})'], ["RETURN 1"]]


def test_schema_creates_become_idempotent_and_are_not_doubled() -> None:
    batches = graph_dump.parse_cypher_shell_dump(REAL_SHAPE)

    assert batches[0] == [
        "CREATE FULLTEXT INDEX entity_search IF NOT EXISTS FOR (n:Course|Person) "
        "ON EACH [n.title, n.context]",
        "CREATE RANGE INDEX IF NOT EXISTS FOR (n:Course) ON (n.key)",
        "CREATE CONSTRAINT UNIQUE_IMPORT_NAME IF NOT EXISTS FOR (node:`UNIQUE IMPORT LABEL`) "
        "REQUIRE (node.`UNIQUE IMPORT ID`) IS UNIQUE",
    ]

    already = "CREATE INDEX entity_key_course IF NOT EXISTS FOR (n:Course) ON (n.key);\n"
    assert graph_dump.parse_cypher_shell_dump(already) == [[already.strip().rstrip(";")]]


def test_a_data_create_is_left_alone() -> None:
    text = 'CREATE (n:Course {title:"FOR loops"});\n'

    assert graph_dump.parse_cypher_shell_dump(text) == [['CREATE (n:Course {title:"FOR loops"})']]


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (":param x => 1\nRETURN 1;\n", "unsupported cypher-shell command"),
        (":begin\nRETURN 1;\n", "':begin' without ':commit'"),
        (":begin\n:begin\n", "inside an open transaction"),
        ("RETURN 1;\n:commit\n", "':commit' without ':begin'"),
        ("RETURN 1\n", "middle of a statement"),
        ('CREATE (n {t:"open;\n', "middle of a statement"),
    ],
)
def test_a_file_that_is_not_a_cypher_shell_dump_is_refused(text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        graph_dump.parse_cypher_shell_dump(text)


class _Result:
    def consume(self) -> None:
        return None


class _Tx:
    def __init__(self, log: list[tuple[str, ...]]) -> None:
        self.log = log

    def __enter__(self) -> "_Tx":
        self.log.append(("begin",))
        return self

    def __exit__(self, *_exc: object) -> None:
        self.log.append(("close",))

    def run(self, statement: str) -> _Result:
        if "BOOM" in statement:
            raise RuntimeError("Neo4j said no")
        self.log.append(("run", statement))
        return _Result()

    def commit(self) -> None:
        self.log.append(("commit",))


class _Session:
    def __init__(self, log: list[tuple[str, ...]]) -> None:
        self.log = log

    def __enter__(self) -> "_Session":
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def begin_transaction(self) -> _Tx:
        return _Tx(self.log)


class _Driver:
    def __init__(self, log: list[tuple[str, ...]]) -> None:
        self.log = log

    def __enter__(self) -> "_Driver":
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def session(self) -> _Session:
        return _Session(self.log)


@pytest.fixture
def driver_log(monkeypatch, tmp_path):
    log: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        graph_dump.GraphDatabase, "driver", staticmethod(lambda _uri, auth: _Driver(log))
    )
    monkeypatch.setenv("NEO4J_URI", "bolt://example.invalid:7687")
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "pw")
    monkeypatch.setattr(graph_dump, "host_dump_path", lambda: tmp_path / "graph_export.cypher")
    return log


def test_import_runs_each_group_in_its_own_transaction(driver_log, tmp_path) -> None:
    (tmp_path / "graph_export.cypher").write_text(
        ":begin\nCREATE (a:X);\nCREATE (b:Y);\n:commit\nCALL db.awaitIndexes(300);\n",
        encoding="utf-8",
    )

    graph_dump.import_graph_from_cypher_dump()

    assert driver_log == [
        ("begin",),
        ("run", "CREATE (a:X)"),
        ("run", "CREATE (b:Y)"),
        ("commit",),
        ("close",),
        ("begin",),
        ("run", "CALL db.awaitIndexes(300)"),
        ("commit",),
        ("close",),
    ]


def test_import_needs_no_apoc_procedure(driver_log, tmp_path) -> None:
    (tmp_path / "graph_export.cypher").write_text(":begin\nCREATE (a:X);\n:commit\n")

    graph_dump.import_graph_from_cypher_dump()

    assert not any("apoc." in entry[1] for entry in driver_log if entry[0] == "run")


def test_a_rejected_statement_stops_the_restore_without_committing_its_group(
    driver_log, tmp_path
) -> None:
    (tmp_path / "graph_export.cypher").write_text(
        ":begin\nCREATE (a:X);\n:commit\n:begin\nCREATE (b:Y);\nBOOM;\n:commit\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="Neo4j said no"):
        graph_dump.import_graph_from_cypher_dump()

    assert driver_log == [
        ("begin",),
        ("run", "CREATE (a:X)"),
        ("commit",),
        ("close",),
        ("begin",),
        ("run", "CREATE (b:Y)"),
        ("close",),
    ]


def test_import_reads_the_dump_as_utf8_whatever_the_locale(driver_log, tmp_path) -> None:
    (tmp_path / "graph_export.cypher").write_bytes(
        'CREATE (n:Course {title:"Analiza matematyczna – ćwiczenia"});\n'.encode()
    )

    graph_dump.import_graph_from_cypher_dump()

    assert driver_log[1] == ("run", 'CREATE (n:Course {title:"Analiza matematyczna – ćwiczenia"})')
