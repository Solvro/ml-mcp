from src.data_pipeline import tracing


def test_config_without_langfuse_only_names_the_run(monkeypatch):
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    tracing._langfuse_enabled.cache_clear()

    assert tracing.llm_run_config("pipeline.x") == {"run_name": "pipeline.x"}


def test_each_call_gets_its_own_trace_in_the_run_session(monkeypatch):
    monkeypatch.setattr(tracing, "_langfuse_enabled", lambda: True)
    monkeypatch.setattr(tracing.flow_run, "id", "run-123", raising=False)

    first = tracing.llm_run_config("pipeline.x")
    second = tracing.llm_run_config("pipeline.x")

    assert first["metadata"] == {
        "langfuse_tags": [tracing.PIPELINE_TAG],
        "langfuse_user_id": tracing.PIPELINE_USER_ID,
        "langfuse_session_id": "run-123",
    }
    first_trace = first["callbacks"][0].trace_context["trace_id"]
    second_trace = second["callbacks"][0].trace_context["trace_id"]
    assert first_trace != second_trace


def test_tracing_failure_never_fails_the_call(monkeypatch):
    monkeypatch.setattr(tracing, "_langfuse_enabled", lambda: True)

    def broken(*args, **kwargs):
        raise RuntimeError("langfuse down")

    monkeypatch.setattr("langfuse.langchain.CallbackHandler", broken)

    assert tracing.llm_run_config("pipeline.x") == {"run_name": "pipeline.x"}
