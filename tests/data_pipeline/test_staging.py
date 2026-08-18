from pathlib import Path

from src.data_pipeline import staging


def test_get_staging_dir_default(monkeypatch):
    monkeypatch.delenv("DATA_PIPELINE_STAGING_DIR", raising=False)
    assert staging.get_staging_dir() == Path("data/staging")


def test_get_staging_dir_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_PIPELINE_STAGING_DIR", str(tmp_path / "stage"))
    assert staging.get_staging_dir() == tmp_path / "stage"


def test_url_to_relative_path_is_deterministic():
    url = "https://pwr.edu.pl/studenci/aktualnosci"
    first = staging.url_to_relative_path(url, is_html=True)
    second = staging.url_to_relative_path(url, is_html=True)
    assert first == second
    assert first == "pwr.edu.pl/studenci/aktualnosci.md"


def test_url_to_relative_path_keeps_pdf_extension():
    url = "https://pwr.edu.pl/files/regulamin.pdf"
    assert staging.url_to_relative_path(url, is_html=False) == "pwr.edu.pl/files/regulamin.pdf"


def test_url_to_relative_path_sanitizes_and_handles_query():
    url = "https://pwr.edu.pl/szukaj?q=ana liza"
    path_a = staging.url_to_relative_path(url, is_html=True)
    path_b = staging.url_to_relative_path("https://pwr.edu.pl/szukaj?q=inne", is_html=True)
    assert path_a != path_b
    assert " " not in path_a
    assert path_a.startswith("pwr.edu.pl/szukaj_")
    assert path_a.endswith(".md")


def test_url_to_relative_path_root_url_maps_to_index():
    assert (
        staging.url_to_relative_path("https://pwr.edu.pl/", is_html=True) == "pwr.edu.pl/index.md"
    )


def test_source_id_for():
    assert staging.source_id_for("pwr.edu.pl/a.md") == "file://pwr.edu.pl/a.md"


def test_atomic_write_bytes_creates_parents_and_no_part_left(tmp_path):
    target = tmp_path / "sub" / "doc.pdf"
    staging.atomic_write_bytes(target, b"content")
    assert target.read_bytes() == b"content"
    assert list(tmp_path.rglob("*.part")) == []


def test_atomic_write_bytes_overwrites(tmp_path):
    target = tmp_path / "doc.txt"
    staging.atomic_write_bytes(target, b"old")
    staging.atomic_write_bytes(target, b"new")
    assert target.read_bytes() == b"new"


def test_manifest_roundtrip(tmp_path):
    manifest = {"file://a.md": {"origin": "https://x/a", "sha256": "abc"}}
    staging.save_manifest(tmp_path, manifest)
    assert staging.load_manifest(tmp_path) == manifest


def test_load_manifest_missing_or_corrupt_returns_empty(tmp_path):
    assert staging.load_manifest(tmp_path) == {}
    (tmp_path / staging.MANIFEST_FILENAME).write_text("{not json", encoding="utf-8")
    assert staging.load_manifest(tmp_path) == {}


def test_url_to_relative_path_truncates_overlong_segments():
    long_token = "A" * 400
    url = f"https://wit.pwr.edu.pl/addtrack/{long_token}"
    path = staging.url_to_relative_path(url, is_html=True)
    assert all(len(seg.encode("utf-8")) <= 255 for seg in path.split("/"))
    assert path == staging.url_to_relative_path(url, is_html=True)
    other = staging.url_to_relative_path(
        f"https://wit.pwr.edu.pl/addtrack/{'B' * 400}", is_html=True
    )
    assert path != other
    assert path.endswith(".md")


def test_url_to_relative_path_truncation_keeps_pdf_extension():
    url = f"https://pwr.edu.pl/files/{'x' * 300}.pdf"
    path = staging.url_to_relative_path(url, is_html=False)
    assert path.endswith(".pdf")
    assert all(len(seg.encode("utf-8")) <= 255 for seg in path.split("/"))
