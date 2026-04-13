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


def test_pipeline_uses_batched_parallel_processing(monkeypatch):
    pages = [f"page-{i}" for i in range(10)]
    reflection_calls: list[int] = []
    active_populates = 0
    max_active_populates = 0

    monkeypatch.setenv("DATA_PIPELINE_MAX_CONCURRENCY", "3")
    monkeypatch.setattr(pipeline_module, "acquire_data", lambda: pages)
    monkeypatch.setattr(pipeline_module, "_get_latest_pipeline_source_hashes", lambda: {})
    monkeypatch.setattr(pipeline_module, "_record_pipeline_run", lambda _d: None)

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

    def populate_submit(_cypher_future, _doc_hash):
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

    monkeypatch.setenv("DATA_PIPELINE_MAX_CONCURRENCY", "4")
    monkeypatch.setattr(pipeline_module, "acquire_data", lambda: pages)
    monkeypatch.setattr(pipeline_module, "_get_latest_pipeline_source_hashes", lambda: {})
    monkeypatch.setattr(pipeline_module, "_record_pipeline_run", lambda _d: None)
    monkeypatch.setattr(pipeline_module, "reflect_on_schema", lambda: "schema-summary")

    claim_results = iter([True, False, True])
    claim_stub = SubmitStub(lambda _doc_hash: ImmediateFuture(next(claim_results)))
    monkeypatch.setattr(pipeline_module, "claim_document_for_processing", claim_stub)

    generate_stub = SubmitStub(
        lambda _page_content, _schema_context: ImmediateFuture("MERGE (n:Entity {title: 'x'})")
    )
    monkeypatch.setattr(pipeline_module, "generate_cypher_queries", generate_stub)

    populate_stub = SubmitStub(lambda _cypher_future, _doc_hash: ImmediateFuture(None))
    monkeypatch.setattr(pipeline_module, "populate_graph", populate_stub)

    pipeline_module.data_pipeline_flow()

    assert len(claim_stub.calls) == 3
    assert len(generate_stub.calls) == 2
    assert len(populate_stub.calls) == 2

    first_hash = claim_stub.calls[0][0][0]
    second_hash = claim_stub.calls[1][0][0]
    third_hash = claim_stub.calls[2][0][0]
    assert first_hash == second_hash
    assert first_hash != third_hash


def test_pipeline_continues_after_page_failure(monkeypatch):
    pages = ["p0", "p1", "p2"]
    reflection_calls: list[int] = []

    monkeypatch.setenv("DATA_PIPELINE_MAX_CONCURRENCY", "3")
    monkeypatch.setattr(pipeline_module, "acquire_data", lambda: pages)
    monkeypatch.setattr(pipeline_module, "_get_latest_pipeline_source_hashes", lambda: {})
    monkeypatch.setattr(pipeline_module, "_record_pipeline_run", lambda _d: None)

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

    def populate_submit(_cypher_future, _doc_hash):
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
    last = {f"legacy://{i}": pipeline_module._compute_page_hash(p) for i, p in enumerate(pages)}
    monkeypatch.setattr(pipeline_module, "acquire_data", lambda: pages)
    monkeypatch.setattr(pipeline_module, "_get_latest_pipeline_source_hashes", lambda: last)
    monkeypatch.setattr(pipeline_module, "_record_pipeline_run", lambda _d: None)
    reflect_calls: list[int] = []

    def fake_reflect():
        reflect_calls.append(1)
        return ""

    monkeypatch.setattr(pipeline_module, "reflect_on_schema", fake_reflect)

    pipeline_module.data_pipeline_flow()

    assert reflect_calls == []


def test_hash_function_is_stable():
    left = pipeline_module._compute_page_hash("some content")
    right = pipeline_module._compute_page_hash("some content")
    different = pipeline_module._compute_page_hash("other content")

    assert left == right
    assert left != different
    assert len(left) == 64
