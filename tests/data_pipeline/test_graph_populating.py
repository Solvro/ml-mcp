import pytest

from src.data_pipeline.flows import graph_populating


class FakePopulator:
    def __init__(self):
        self.executed: list[str] = []
        self.processed: list[str] = []
        self.failed: list[tuple[str, str]] = []
        self.fail_on: str | None = None

    def execute_cypher(self, query: str) -> None:
        if self.fail_on and self.fail_on in query:
            raise RuntimeError("boom")
        self.executed.append(query)

    def mark_document_processed(self, doc_hash: str) -> None:
        self.processed.append(doc_hash)

    def mark_document_failed(self, doc_hash: str, error_message: str) -> None:
        self.failed.append((doc_hash, error_message))


@pytest.fixture
def fake_populator(monkeypatch):
    fake = FakePopulator()
    monkeypatch.setattr(graph_populating, "GraphPopulator", lambda: fake)
    return fake


def test_populate_graph_runs_pipe_joined_clauses_as_one_query(fake_populator):
    """Relationship clauses reference variables from node clauses, so all
    clauses of a document must execute in a single query context."""
    graph_populating.populate_graph.fn(
        "MERGE (a:X {t: 'x'})|MERGE (b:Y {t: 'y'})| MERGE (a)-[:REL]->(b) ",
        "hash1",
        "file://docs/a.pdf#page=1",
    )
    assert fake_populator.executed == [
        "MERGE (a:X {t: 'x'})\nMERGE (b:Y {t: 'y'})\nMERGE (a)-[:REL]->(b)"
    ]
    assert fake_populator.processed == ["hash1"]
    assert fake_populator.failed == []


def test_populate_graph_single_statement_unchanged(fake_populator):
    graph_populating.populate_graph.fn("MERGE (a:X)", "hash2", "file://docs/a.pdf#page=1")
    assert fake_populator.executed == ["MERGE (a:X)"]
    assert fake_populator.processed == ["hash2"]
    assert fake_populator.failed == []


def test_populate_graph_failure_marks_failed_and_raises(fake_populator):
    fake_populator.fail_on = "b:Y"
    with pytest.raises(RuntimeError):
        graph_populating.populate_graph.fn(
            "MERGE (a:X)|MERGE (b:Y)",
            "hash3",
            "file://docs/a.pdf#page=1",
        )
    assert fake_populator.executed == []
    assert fake_populator.processed == []
    assert fake_populator.failed and fake_populator.failed[0][0] == "hash3"
