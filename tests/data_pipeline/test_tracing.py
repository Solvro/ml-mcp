from src.data_pipeline import tracing


def test_config_without_langfuse_only_names_the_run(monkeypatch):
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    tracing._langfuse_handler.cache_clear()

    assert tracing.llm_run_config("pipeline.x") == {"run_name": "pipeline.x"}


def test_config_with_langfuse_attaches_handler_tag_and_session(monkeypatch):
    handler = object()
    monkeypatch.setattr(tracing, "_langfuse_handler", lambda: handler)
    monkeypatch.setattr(tracing.flow_run, "root_flow_run_id", "run-123", raising=False)

    config = tracing.llm_run_config("pipeline.x")

    assert config["callbacks"] == [handler]
    assert config["metadata"] == {
        "langfuse_tags": [tracing.PIPELINE_TAG],
        "langfuse_session_id": "run-123",
    }
