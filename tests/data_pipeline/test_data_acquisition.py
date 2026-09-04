from pathlib import Path

from src.data_pipeline.flows.data_acquisition import acquire_data


def test_acquire_data_reads_staging_documents(monkeypatch, tmp_path: Path):
    (tmp_path / "a.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "b.txt").write_text("test", encoding="utf-8")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "c.md").write_text("markdown", encoding="utf-8")
    (tmp_path / "ignored.bin").write_bytes(b"\x00")
    monkeypatch.setenv("DATA_PIPELINE_STAGING_DIR", str(tmp_path))
    result = acquire_data.fn()
    source_ids = [item["source_id"] for item in result]
    assert source_ids == [
        "file://a.pdf",
        "file://b.txt",
        "file://nested/c.md",
    ]
    assert all("path" in item for item in result)


def test_env_not_set_logs_warning(monkeypatch, caplog):
    monkeypatch.delenv("DATA_PIPELINE_STAGING_DIR", raising=False)
    caplog.set_level("WARNING")
    result = acquire_data.fn()
    assert result == []
    assert "DATA_PIPELINE_STAGING_DIR is not set" in caplog.text


def test_acquire_data_skips_archive_dir(monkeypatch, tmp_path: Path):
    (tmp_path / ".archive" / "old.pdf").parent.mkdir(parents=True)
    (tmp_path / ".archive" / "old.pdf").write_bytes(b"%PDF")
    (tmp_path / "live.pdf").write_bytes(b"%PDF")
    monkeypatch.setenv("DATA_PIPELINE_STAGING_DIR", str(tmp_path))

    result = acquire_data.fn()
    source_ids = [item["source_id"] for item in result]

    assert source_ids == ["file://live.pdf"]
