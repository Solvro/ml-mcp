from pathlib import Path

import pytest

from src.data_pipeline import pipeline as pipeline_module

NONEXISTENT_PATH = Path("/__pipeline_tests__/nonexistent/graph_export.cypher")


@pytest.fixture(autouse=True)
def _pipeline_flow_no_external_services(monkeypatch):
    """Isolate pipeline tests from external services (Neo4j, filesystem side effects)."""

    # 1. DATABASE MOCKING (GraphPopulator)
    # Instead of patching init internals, replace methods to no-op or return empty data.
    monkeypatch.setattr(pipeline_module.GraphPopulator, "__init__", lambda self: None)
    monkeypatch.setattr(
        pipeline_module.GraphPopulator, "get_latest_pipeline_source_hashes", lambda *a, **k: {}
    )
    monkeypatch.setattr(pipeline_module.GraphPopulator, "record_pipeline_run", lambda *a, **k: None)
    monkeypatch.setattr(pipeline_module.GraphPopulator, "record_restore_run", lambda *a, **k: None)

    # 2. FILESYSTEM AND EXPORT MOCKING
    monkeypatch.setattr(pipeline_module, "host_dump_path", lambda *a, **k: NONEXISTENT_PATH)
    monkeypatch.setattr(pipeline_module, "ensure_host_dump_dir", lambda *a, **k: Path("/tmp"))
    monkeypatch.setattr(pipeline_module, "export_graph_to_cypher", lambda *a, **k: None)

    # 3. FUTURES ITERATION MOCKING
    monkeypatch.setattr(pipeline_module, "as_completed", lambda futures: futures)
