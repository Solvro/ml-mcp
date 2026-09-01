from collections.abc import Callable

from src.data_pipeline import pipeline as pipeline_module


class ImmediateFuture:
    def __init__(
        self,
        value=None,
        *,
        error: Exception | None = None,
        on_result: Callable[[], None] | None = None,
    ):
        self._value = value
        self._error = error
        self._on_result = on_result

    def result(self):
        if self._on_result:
            self._on_result()
        if self._error is not None:
            raise self._error
        return self._value


class SubmitStub:
    def __init__(self, submit_impl: Callable[..., ImmediateFuture]):
        self.submit_impl = submit_impl
        self.calls: list[tuple[tuple, dict]] = []

    def submit(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.submit_impl(*args, **kwargs)


def _acquired_stub():
    return [{"source_id": "file://docs/a.pdf", "path": "/tmp/docs/a.pdf"}]


def _to_extracted_pages(contents: list[str]) -> list[tuple[str, str]]:
    return [(f"file://docs/a.pdf#page={i + 1}", c) for i, c in enumerate(contents)]


def test_pipeline_uses_batched_parallel_processing(monkeypatch):
    pages = [f"page-{i}" for i in range(10)]
    extracted_pages = _to_extracted_pages(pages)
    reflection_calls: list[int] = []
    active_populates = 0
    max_active_populates = 0

    monkeypatch.setattr(pipeline_module, "acquire_data", _acquired_stub)
    monkeypatch.setattr(pipeline_module, "ocr_extraction", lambda _acquired: extracted_pages)
    monkeypatch.setenv("DATA_PIPELINE_MAX_CONCURRENCY", "3")
    monkeypatch.setattr(
        pipeline_module.GraphPopulator,
        "get_latest_pipeline_source_hashes",
        lambda self: {},
    )
    monkeypatch.setattr(
        pipeline_module.GraphPopulator,
        "record_pipeline_run",
        lambda self, *args, **kwargs: None,
    )

    def fake_reflect():
        reflection_calls.append(1)
        return "schema-summary"

    monkeypatch.setattr(pipeline_module, "reflect_on_schema", fake_reflect)

    claim_stub = SubmitStub(lambda _doc_hash: ImmediateFuture(True))
    monkeypatch.setattr(pipeline_module, "claim_document_for_processing", claim_stub)

    generate_stub = SubmitStub(
        lambda _page_content, _schema_context: ImmediateFuture("MERGE (n:Entity {title: 'x'})")
    )
    monkeypatch.setattr(pipeline_module, "generate_cypher_queries", generate_stub)

    def populate_submit(_cypher_future, _doc_hash, _source_id):
        nonlocal active_populates
        nonlocal max_active_populates

        active_populates += 1
        max_active_populates = max(max_active_populates, active_populates)

        def release_slot():
            nonlocal active_populates
            active_populates -= 1

        return ImmediateFuture(None, on_result=release_slot)

    populate_stub = SubmitStub(populate_submit)
    monkeypatch.setattr(pipeline_module, "populate_graph", populate_stub)

    pipeline_module.data_pipeline_flow()

    assert len(claim_stub.calls) == 10
    assert len(generate_stub.calls) == 10
    assert len(populate_stub.calls) == 10
    assert max_active_populates <= 3
    assert max_active_populates == 3
    assert len(reflection_calls) == 2


def test_pipeline_skips_duplicate_hashes(monkeypatch):
    pages = ["same", "same", "unique"]
    extracted_pages = _to_extracted_pages(pages)
    linked_calls: list[tuple[str, str]] = []

    def link_stub(self, doc_hash: str, source_id: str):
        linked_calls.append((doc_hash, source_id))

    monkeypatch.setattr(
        pipeline_module.GraphPopulator,
        "link_processed_document_to_source",
        link_stub,
    )

    monkeypatch.setenv("DATA_PIPELINE_MAX_CONCURRENCY", "4")
    monkeypatch.setattr(pipeline_module, "acquire_data", _acquired_stub)
    monkeypatch.setattr(pipeline_module, "ocr_extraction", lambda _acquired: extracted_pages)
    monkeypatch.setattr(
        pipeline_module.GraphPopulator,
        "get_latest_pipeline_source_hashes",
        lambda self: {},
    )
    monkeypatch.setattr(
        pipeline_module.GraphPopulator,
        "record_pipeline_run",
        lambda self, *args, **kwargs: None,
    )
    monkeypatch.setattr(pipeline_module, "reflect_on_schema", lambda: "schema-summary")

    claim_results = iter([True, False, True])
    claim_stub = SubmitStub(lambda _doc_hash: ImmediateFuture(next(claim_results)))
    monkeypatch.setattr(pipeline_module, "claim_document_for_processing", claim_stub)

    generate_stub = SubmitStub(
        lambda _page_content, _schema_context: ImmediateFuture("MERGE (n:Entity {title: 'x'})")
    )
    monkeypatch.setattr(pipeline_module, "generate_cypher_queries", generate_stub)

    populate_stub = SubmitStub(lambda _cypher_future, _doc_hash, _source_id: ImmediateFuture(None))
    monkeypatch.setattr(pipeline_module, "populate_graph", populate_stub)

    outcome = pipeline_module.data_pipeline_flow()

    assert len(claim_stub.calls) == 3
    assert len(generate_stub.calls) == 2
    assert len(populate_stub.calls) == 2

    first_hash = claim_stub.calls[0][0][0]
    second_hash = claim_stub.calls[1][0][0]
    third_hash = claim_stub.calls[2][0][0]
    assert first_hash == second_hash
    assert first_hash != third_hash
    # A duplicate hash means the page is already in the graph, so the document
    # still counts as confirmed.
    assert outcome.processed == {"file://docs/a.pdf"}
    assert outcome.deleted == set()
    assert linked_calls == [(second_hash, "file://docs/a.pdf#page=2")]


def test_pipeline_continues_after_page_failure(monkeypatch):
    pages = ["p0", "p1", "p2"]
    extracted_pages = _to_extracted_pages(pages)
    reflection_calls: list[int] = []

    monkeypatch.setenv("DATA_PIPELINE_MAX_CONCURRENCY", "3")
    monkeypatch.setattr(pipeline_module, "acquire_data", _acquired_stub)
    monkeypatch.setattr(pipeline_module, "ocr_extraction", lambda _acquired: extracted_pages)
    monkeypatch.setattr(
        pipeline_module.GraphPopulator,
        "get_latest_pipeline_source_hashes",
        lambda self: {},
    )
    monkeypatch.setattr(
        pipeline_module.GraphPopulator,
        "record_pipeline_run",
        lambda self, *args, **kwargs: None,
    )

    def fake_reflect():
        reflection_calls.append(1)
        return "schema-summary"

    monkeypatch.setattr(pipeline_module, "reflect_on_schema", fake_reflect)
    monkeypatch.setattr(
        pipeline_module,
        "claim_document_for_processing",
        SubmitStub(lambda _doc_hash: ImmediateFuture(True)),
    )
    monkeypatch.setattr(
        pipeline_module,
        "generate_cypher_queries",
        SubmitStub(
            lambda _page_content, _schema_context: ImmediateFuture("MERGE (n:Entity {x: 1})")
        ),
    )

    failure_index = {"value": 0}

    def populate_submit(_cypher_future, _doc_hash, _source_id):
        idx = failure_index["value"]
        failure_index["value"] += 1
        if idx == 1:
            return ImmediateFuture(error=RuntimeError("boom"))
        return ImmediateFuture(None)

    populate_stub = SubmitStub(populate_submit)
    monkeypatch.setattr(pipeline_module, "populate_graph", populate_stub)

    pipeline_module.data_pipeline_flow()

    assert len(populate_stub.calls) == 3
    assert len(reflection_calls) == 2


def test_pipeline_incremental_skips_when_source_hashes_unchanged(monkeypatch):
    pages = ["page-0", "page-1"]
    extracted_pages = _to_extracted_pages(pages)
    last = {sid: pipeline_module._compute_page_hash(text) for sid, text in extracted_pages}
    monkeypatch.setattr(pipeline_module, "acquire_data", _acquired_stub)
    monkeypatch.setattr(pipeline_module, "ocr_extraction", lambda _acquired: extracted_pages)
    monkeypatch.setattr(
        pipeline_module.GraphPopulator,
        "get_latest_pipeline_source_hashes",
        lambda self: last,
    )
    monkeypatch.setattr(
        pipeline_module.GraphPopulator,
        "record_pipeline_run",
        lambda self, *args, **kwargs: None,
    )
    reflect_calls: list[int] = []

    def fake_reflect():
        reflect_calls.append(1)
        return ""

    monkeypatch.setattr(pipeline_module, "reflect_on_schema", fake_reflect)

    pipeline_module.data_pipeline_flow()

    assert reflect_calls == []


def test_pipeline_second_run_is_noop_for_unchanged_extracted_pages(monkeypatch):
    extracted_pages = [
        ("file://docs/a.pdf#page=1", "page-one"),
        ("file://docs/a.pdf#page=2", "page-two"),
    ]

    state = {"last_source_hashes": {}, "record_calls": 0}

    monkeypatch.setattr(
        pipeline_module,
        "acquire_data",
        lambda: [{"source_id": "file://docs/a.pdf", "path": "/tmp/docs/a.pdf"}],
    )
    monkeypatch.setattr(pipeline_module, "ocr_extraction", lambda _acquired: extracted_pages)

    monkeypatch.setattr(
        pipeline_module.GraphPopulator,
        "get_latest_pipeline_source_hashes",
        lambda self: state["last_source_hashes"],
    )

    def record_pipeline_run(self, full_source_hashes, mode="full"):
        state["last_source_hashes"] = dict(full_source_hashes)
        state["record_calls"] += 1

    monkeypatch.setattr(
        pipeline_module.GraphPopulator,
        "record_pipeline_run",
        record_pipeline_run,
    )

    reflect_calls: list[int] = []

    def fake_reflect():
        reflect_calls.append(1)
        return "schema-summary"

    monkeypatch.setattr(pipeline_module, "reflect_on_schema", fake_reflect)

    claim_stub = SubmitStub(lambda _doc_hash: ImmediateFuture(True))
    generate_stub = SubmitStub(
        lambda _page_content, _schema_context: ImmediateFuture("MERGE (n:Entity {title: 'x'})")
    )
    populate_stub = SubmitStub(lambda _cypher_future, _doc_hash, _source_id: ImmediateFuture(None))

    monkeypatch.setattr(pipeline_module, "claim_document_for_processing", claim_stub)
    monkeypatch.setattr(pipeline_module, "generate_cypher_queries", generate_stub)
    monkeypatch.setattr(pipeline_module, "populate_graph", populate_stub)

    # First run: should process 2 pages.
    pipeline_module.data_pipeline_flow()
    # Second run with unchanged pages: should be no-op.
    pipeline_module.data_pipeline_flow()

    assert len(claim_stub.calls) == 2
    assert len(generate_stub.calls) == 2
    assert len(populate_stub.calls) == 2
    assert state["record_calls"] == 1
    assert len(reflect_calls) == 2


def test_hash_function_is_stable():
    left = pipeline_module._compute_page_hash("some content")
    right = pipeline_module._compute_page_hash("some content")
    different = pipeline_module._compute_page_hash("other content")

    assert left == right
    assert left != different
    assert len(left) == 64


def test_pipeline_filters_extraction_to_changed_paths(monkeypatch):
    acquired_docs = [
        {"source_id": "file://docs/a.pdf", "path": "/tmp/docs/a.pdf"},
        {"source_id": "file://docs/b.pdf", "path": "/tmp/docs/b.pdf"},
    ]
    monkeypatch.setattr(pipeline_module, "acquire_data", lambda: acquired_docs)

    ocr_calls: list[list[dict[str, str]]] = []

    def ocr_stub(acquired):
        ocr_calls.append(acquired)
        out: list[tuple[str, str]] = []
        for item in acquired:
            out.append((f"{item['source_id']}#page=1", "page"))
        return out

    monkeypatch.setattr(pipeline_module, "ocr_extraction", ocr_stub)
    monkeypatch.setattr(
        pipeline_module.GraphPopulator,
        "get_latest_pipeline_source_hashes",
        lambda self: {},
    )
    monkeypatch.setattr(
        pipeline_module.GraphPopulator,
        "record_pipeline_run",
        lambda self, *args, **kwargs: None,
    )
    monkeypatch.setattr(pipeline_module, "reflect_on_schema", lambda: "schema-summary")

    deletion_calls: list[list[str]] = []

    def delete_stub(self, document_source_ids):
        deletion_calls.append(list(document_source_ids))
        return set(document_source_ids)

    monkeypatch.setattr(
        pipeline_module.GraphPopulator,
        "delete_sources_for_documents",
        delete_stub,
    )

    claim_stub = SubmitStub(lambda _doc_hash: ImmediateFuture(True))
    generate_stub = SubmitStub(
        lambda _page_content, _schema_context: ImmediateFuture("MERGE (n:Entity {title: 'x'})")
    )
    populate_stub = SubmitStub(lambda _cypher_future, _doc_hash, _source_id: ImmediateFuture(None))

    monkeypatch.setattr(pipeline_module, "claim_document_for_processing", claim_stub)
    monkeypatch.setattr(pipeline_module, "generate_cypher_queries", generate_stub)
    monkeypatch.setattr(pipeline_module, "populate_graph", populate_stub)

    outcome = pipeline_module.data_pipeline_flow(
        changed=["docs/a.pdf", "docs/missing.pdf"],
        deleted=["docs/removed.pdf"],
    )

    assert len(ocr_calls) == 1
    assert [item["source_id"] for item in ocr_calls[0]] == ["file://docs/a.pdf"]
    assert len(claim_stub.calls) == 1
    assert len(generate_stub.calls) == 1
    assert len(populate_stub.calls) == 1
    assert outcome.processed == {"file://docs/a.pdf"}
    assert deletion_calls == [["file://docs/removed.pdf"]]
    assert outcome.deleted == {"file://docs/removed.pdf"}


def test_pipeline_returns_documents_confirmed_in_graph(monkeypatch):
    extracted_pages = _to_extracted_pages(["p0", "p1"])

    monkeypatch.setattr(pipeline_module, "acquire_data", _acquired_stub)
    monkeypatch.setattr(pipeline_module, "ocr_extraction", lambda _acquired: extracted_pages)
    monkeypatch.setattr(
        pipeline_module.GraphPopulator,
        "get_latest_pipeline_source_hashes",
        lambda self: {},
    )
    monkeypatch.setattr(
        pipeline_module.GraphPopulator,
        "record_pipeline_run",
        lambda self, *args, **kwargs: None,
    )
    monkeypatch.setattr(pipeline_module, "reflect_on_schema", lambda: "schema-summary")
    monkeypatch.setattr(
        pipeline_module,
        "claim_document_for_processing",
        SubmitStub(lambda _doc_hash: ImmediateFuture(True)),
    )
    monkeypatch.setattr(
        pipeline_module,
        "generate_cypher_queries",
        SubmitStub(lambda _page, _schema: ImmediateFuture("MERGE (n:Entity {x: 1})")),
    )
    monkeypatch.setattr(
        pipeline_module,
        "populate_graph",
        SubmitStub(lambda _cypher, _hash, _source_id: ImmediateFuture(None)),
    )

    outcome = pipeline_module.data_pipeline_flow()

    assert outcome.processed == {"file://docs/a.pdf"}
    assert outcome.deleted == set()


def test_pipeline_returns_empty_when_extraction_yields_nothing(monkeypatch):
    monkeypatch.setattr(pipeline_module, "acquire_data", _acquired_stub)
    monkeypatch.setattr(pipeline_module, "ocr_extraction", lambda _acquired: [])

    outcome = pipeline_module.data_pipeline_flow()

    assert outcome.processed == set()


def test_pipeline_skips_extraction_when_changed_set_is_empty(monkeypatch):
    monkeypatch.setattr(
        pipeline_module,
        "acquire_data",
        lambda: [{"source_id": "file://docs/a.pdf", "path": "/tmp/docs/a.pdf"}],
    )

    ocr_calls: list[object] = []

    def ocr_stub(_acquired):
        ocr_calls.append(1)
        return []

    monkeypatch.setattr(pipeline_module, "ocr_extraction", ocr_stub)

    outcome = pipeline_module.data_pipeline_flow(changed=[])

    assert ocr_calls == []
    assert outcome.processed == set()
    assert outcome.deleted == set()


def test_full_scan_after_incremental_does_not_reclaim_untouched_documents(monkeypatch):
    acquired = [
        {"source_id": "file://docs/a.pdf", "path": "/tmp/docs/a.pdf"},
        {"source_id": "file://docs/b.pdf", "path": "/tmp/docs/b.pdf"},
    ]
    texts = {"file://docs/a.pdf": "a-one", "file://docs/b.pdf": "b-one"}
    recorded: dict[str, str] = {}

    def fake_record(self, hashes, mode="full"):
        recorded.clear()
        recorded.update(hashes)

    monkeypatch.setattr(pipeline_module, "acquire_data", lambda: list(acquired))
    monkeypatch.setattr(
        pipeline_module,
        "ocr_extraction",
        lambda items: [(f"{item['source_id']}#page=1", texts[item["source_id"]]) for item in items],
    )
    monkeypatch.setattr(
        pipeline_module.GraphPopulator,
        "get_latest_pipeline_source_hashes",
        lambda self: dict(recorded),
    )
    monkeypatch.setattr(pipeline_module.GraphPopulator, "record_pipeline_run", fake_record)
    monkeypatch.setattr(pipeline_module, "reflect_on_schema", lambda: "schema-summary")

    claim_stub = SubmitStub(lambda _doc_hash: ImmediateFuture(True))
    monkeypatch.setattr(pipeline_module, "claim_document_for_processing", claim_stub)
    monkeypatch.setattr(
        pipeline_module,
        "generate_cypher_queries",
        SubmitStub(lambda _page, _schema: ImmediateFuture("MERGE (n:Entity {x: 1})")),
    )
    monkeypatch.setattr(
        pipeline_module,
        "populate_graph",
        SubmitStub(lambda _cypher, _hash, _source_id: ImmediateFuture(None)),
    )

    pipeline_module.data_pipeline_flow()  # full scan: a and b recorded

    texts["file://docs/a.pdf"] = "a-two"
    pipeline_module.data_pipeline_flow(changed=["docs/a.pdf"])

    assert "file://docs/b.pdf#page=1" in recorded

    claim_stub.calls.clear()
    pipeline_module.data_pipeline_flow()

    assert claim_stub.calls == []


def test_incremental_merge_drops_stale_hash_for_attempted_but_missing_extraction(monkeypatch):
    acquired = [
        {"source_id": "file://docs/a.pdf", "path": "/tmp/docs/a.pdf"},
        {"source_id": "file://docs/b.pdf", "path": "/tmp/docs/b.pdf"},
    ]

    old_a = pipeline_module._compute_page_hash("a-old")
    old_b = pipeline_module._compute_page_hash("b-old")
    recorded: dict[str, str] = {}

    def fake_record(self, hashes, mode="full"):
        recorded.clear()
        recorded.update(hashes)

    monkeypatch.setattr(pipeline_module, "acquire_data", lambda: list(acquired))
    monkeypatch.setattr(
        pipeline_module,
        "ocr_extraction",
        lambda _items: [("file://docs/b.pdf#page=1", "b-new")],
    )
    monkeypatch.setattr(
        pipeline_module.GraphPopulator,
        "get_latest_pipeline_source_hashes",
        lambda self: {
            "file://docs/a.pdf#page=1": old_a,
            "file://docs/b.pdf#page=1": old_b,
        },
    )
    monkeypatch.setattr(pipeline_module.GraphPopulator, "record_pipeline_run", fake_record)
    monkeypatch.setattr(pipeline_module, "reflect_on_schema", lambda: "schema-summary")
    monkeypatch.setattr(
        pipeline_module,
        "claim_document_for_processing",
        SubmitStub(lambda _doc_hash: ImmediateFuture(True)),
    )
    monkeypatch.setattr(
        pipeline_module,
        "generate_cypher_queries",
        SubmitStub(lambda _page, _schema: ImmediateFuture("MERGE (n:Entity {x: 1})")),
    )
    monkeypatch.setattr(
        pipeline_module,
        "populate_graph",
        SubmitStub(lambda _cypher, _hash, _source_id: ImmediateFuture(None)),
    )

    pipeline_module.data_pipeline_flow(changed=["docs/a.pdf", "docs/b.pdf"])

    assert "file://docs/a.pdf#page=1" not in recorded
    assert recorded["file://docs/b.pdf#page=1"] == pipeline_module._compute_page_hash("b-new")
