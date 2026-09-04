import pytest

from src.data_pipeline.flows import graph_populating


class FakePopulator:
    def __init__(self):
        self.executed: list[tuple[str, dict[str, str]]] = []
        self.processed: list[str] = []
        self.failed: list[tuple[str, str]] = []
        self.fail_on: str | None = None
        self.linked: list[tuple[str, str]] = []

    def execute_cypher(self, query: str, params: dict[str, str] | None = None) -> None:
        if self.fail_on and self.fail_on in query:
            raise RuntimeError("boom")
        self.executed.append((query, params or {}))

    def mark_document_processed(self, doc_hash: str) -> None:
        self.processed.append(doc_hash)

    def mark_document_failed(self, doc_hash: str, error_message: str) -> None:
        self.failed.append((doc_hash, error_message))

    def link_processed_document_to_source(self, doc_hash: str, source_id: str) -> None:
        self.linked.append((doc_hash, source_id))


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
        (
            "MERGE (a:X {t: 'x'})\n"
            "MERGE (b:Y {t: 'y'})\n"
            "MERGE (a)-[:REL]->(b)\n"
            "WITH a, b\n"
            "MERGE (prov_source:Source {source_id: $source_id})\n"
            "MERGE (a)-[:FROM_SOURCE]->(prov_source)\n"
            "MERGE (b)-[:FROM_SOURCE]->(prov_source)",
            {"source_id": "file://docs/a.pdf#page=1"},
        )
    ]
    assert fake_populator.processed == ["hash1"]
    assert fake_populator.failed == []
    assert fake_populator.linked == [("hash1", "file://docs/a.pdf#page=1")]


def test_populate_graph_appends_provenance_to_single_statement(fake_populator):
    graph_populating.populate_graph.fn("MERGE (a:X)", "hash2", "file://docs/a.pdf#page=1")
    assert fake_populator.executed == [
        (
            "MERGE (a:X)\n"
            "WITH a\n"
            "MERGE (prov_source:Source {source_id: $source_id})\n"
            "MERGE (a)-[:FROM_SOURCE]->(prov_source)",
            {"source_id": "file://docs/a.pdf#page=1"},
        )
    ]
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
    assert fake_populator.linked == [("hash3", "file://docs/a.pdf#page=1")]


def test_populate_graph_without_merge_vars_runs_without_provenance(fake_populator, caplog):
    graph_populating.populate_graph.fn("MATCH (a) RETURN a", "hash4", "file://docs/a.pdf#page=1")

    assert fake_populator.executed == [("MATCH (a) RETURN a", {})]
    assert fake_populator.processed == ["hash4"]
    assert any("without provenance" in rec.message for rec in caplog.records)


def test_populate_graph_avoids_colliding_with_a_generated_source_variable(fake_populator):
    graph_populating.populate_graph.fn(
        "MERGE (prov_source:X {t: 'x'})",
        "hash5",
        "file://docs/a.pdf#page=1",
    )

    query, params = fake_populator.executed[0]
    assert "MERGE (prov_source_:Source {source_id: $source_id})" in query
    assert "MERGE (prov_source)-[:FROM_SOURCE]->(prov_source_)" in query
    assert params == {"source_id": "file://docs/a.pdf#page=1"}


def test_delete_sources_for_documents_confirms_every_completed_id():
    class FakeGraphDb:
        def __init__(self):
            self.calls = []

        def query(self, query, params=None):
            self.calls.append((query, params))
            # b.pdf contributed nothing unique, so it leaves no orphans behind —
            # that is a completed deletion, not a failed one.
            if params["doc_source_id"] == "file://docs/a.pdf":
                return [{"sources_deleted": 2, "orphans_removed": 3}]
            return [{"sources_deleted": 1, "orphans_removed": 0}]

    pop = graph_populating.GraphPopulator.__new__(graph_populating.GraphPopulator)
    pop.graph_db = FakeGraphDb()

    deleted = pop.delete_sources_for_documents(
        ["file://docs/a.pdf", "file://docs/b.pdf", "file://docs/a.pdf"]
    )

    assert deleted == {"file://docs/a.pdf", "file://docs/b.pdf"}
    assert len(pop.graph_db.calls) == 2
    assert pop.graph_db.calls[0][1] == {
        "doc_source_id": "file://docs/a.pdf",
        "doc_prefix": "file://docs/a.pdf#",
    }


def test_delete_sources_for_documents_skips_ids_whose_query_failed():
    class FailingGraphDb:
        def query(self, query, params=None):
            if params["doc_source_id"] == "file://docs/b.pdf":
                raise RuntimeError("neo4j down")
            return [{"sources_deleted": 1, "orphans_removed": 1}]

    pop = graph_populating.GraphPopulator.__new__(graph_populating.GraphPopulator)
    pop.graph_db = FailingGraphDb()

    deleted = pop.delete_sources_for_documents(["file://docs/a.pdf", "file://docs/b.pdf"])

    assert deleted == {"file://docs/a.pdf"}


def test_delete_sources_for_documents_warns_when_source_nodes_are_missing(caplog):
    class EmptyMatchGraphDb:
        def query(self, query, params=None):
            return [{"sources_deleted": 0, "orphans_removed": 0}]

    pop = graph_populating.GraphPopulator.__new__(graph_populating.GraphPopulator)
    pop.graph_db = EmptyMatchGraphDb()

    deleted = pop.delete_sources_for_documents(["file://docs/a.pdf"])

    assert deleted == {"file://docs/a.pdf"}
    assert any(
        "matched no Source nodes" in rec.message and rec.levelname == "WARNING"
        for rec in caplog.records
    )


def test_mirror_entity_provenance_for_duplicate_hash_links_existing_entities():
    class FakeGraphDb:
        def __init__(self):
            self.calls = []

        def query(self, query, params=None):
            self.calls.append((query, params))
            return [{"entities_linked": 4}]

    pop = graph_populating.GraphPopulator.__new__(graph_populating.GraphPopulator)
    pop.graph_db = FakeGraphDb()

    linked = pop.mirror_entity_provenance_for_duplicate_hash("hash-a", "file://docs/b.pdf#page=1")

    assert linked == 4
    assert len(pop.graph_db.calls) == 1
    _, params = pop.graph_db.calls[0]
    assert params["doc_hash"] == "hash-a"
    assert params["source_id"] == "file://docs/b.pdf#page=1"
    assert params["system_labels"] == ["PipelineRun", "ProcessedDocument", "Source"]


def test_mirror_entity_provenance_for_duplicate_hash_skips_empty_inputs():
    class NeverCalledGraphDb:
        def query(self, query, params=None):
            raise AssertionError("query should not be called")

    pop = graph_populating.GraphPopulator.__new__(graph_populating.GraphPopulator)
    pop.graph_db = NeverCalledGraphDb()

    assert pop.mirror_entity_provenance_for_duplicate_hash("", "file://docs/b.pdf#page=1") == 0
    assert pop.mirror_entity_provenance_for_duplicate_hash("hash-a", "") == 0
