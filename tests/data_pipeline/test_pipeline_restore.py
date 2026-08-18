from src.data_pipeline import pipeline as pipeline_module
from tests.data_pipeline.test_pipeline_concurrency import ImmediateFuture, SubmitStub


def _stub_extraction(monkeypatch, populate_calls):
    monkeypatch.setattr(
        pipeline_module, "acquire_data", lambda: [{"source_id": "file://a.md", "path": "/x/a.md"}]
    )
    monkeypatch.setattr(pipeline_module, "ocr_extraction", lambda _a: [("file://a.md", "content")])
    monkeypatch.setattr(pipeline_module, "reflect_on_schema", lambda: "schema")
    claim_stub = SubmitStub(lambda _h: ImmediateFuture(True))
    monkeypatch.setattr(pipeline_module, "claim_document_for_processing", claim_stub)
    generate_stub = SubmitStub(lambda _c, _s: ImmediateFuture("MERGE (n:X)"))
    monkeypatch.setattr(pipeline_module, "generate_cypher_queries", generate_stub)

    def populate_submit(_cypher_future, doc_hash):
        populate_calls.append(doc_hash)
        return ImmediateFuture(None)

    populate_stub = SubmitStub(populate_submit)
    monkeypatch.setattr(pipeline_module, "populate_graph", populate_stub)


def _dump_file(monkeypatch, tmp_path):
    dump = tmp_path / "graph_export.cypher"
    dump.write_text("MERGE (n:X)", encoding="utf-8")
    monkeypatch.setattr(pipeline_module, "host_dump_path", lambda: dump)


def test_dump_restore_skipped_when_graph_has_data(monkeypatch, tmp_path):
    _dump_file(monkeypatch, tmp_path)
    monkeypatch.setattr(pipeline_module.GraphPopulator, "graph_has_data", lambda self: True)
    restore_calls: list[int] = []
    monkeypatch.setattr(
        pipeline_module, "import_graph_from_cypher_dump", lambda: restore_calls.append(1)
    )

    populate_calls: list[str] = []
    _stub_extraction(monkeypatch, populate_calls)

    pipeline_module.data_pipeline_flow()
    assert restore_calls == []
    assert len(populate_calls) == 1


def test_dump_restore_runs_on_empty_graph(monkeypatch, tmp_path):
    _dump_file(monkeypatch, tmp_path)
    monkeypatch.setattr(pipeline_module.GraphPopulator, "graph_has_data", lambda self: False)
    restore_calls: list[int] = []
    monkeypatch.setattr(
        pipeline_module, "import_graph_from_cypher_dump", lambda: restore_calls.append(1)
    )

    populate_calls: list[str] = []
    _stub_extraction(monkeypatch, populate_calls)

    pipeline_module.data_pipeline_flow()
    assert restore_calls == [1]
    assert populate_calls == []


def test_dump_restore_failure_falls_back_to_extraction(monkeypatch, tmp_path):
    _dump_file(monkeypatch, tmp_path)
    monkeypatch.setattr(pipeline_module.GraphPopulator, "graph_has_data", lambda self: False)

    def failing_restore():
        raise RuntimeError("no apoc")

    monkeypatch.setattr(pipeline_module, "import_graph_from_cypher_dump", failing_restore)

    populate_calls: list[str] = []
    _stub_extraction(monkeypatch, populate_calls)

    pipeline_module.data_pipeline_flow()
    assert len(populate_calls) == 1
