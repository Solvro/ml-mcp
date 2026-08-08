from pathlib import Path

from src.data_pipeline.flows import ocr_extraction as ocr_module


def test_ocr_extraction_docx(monkeypatch, tmp_path: Path):
    class FakeParagraph:
        def __init__(self, text: str):
            self.text = text

    class FakeCell:
        def __init__(self, text: str):
            self.text = text

    class FakeRow:
        def __init__(self, cells):
            self.cells = cells

    class FakeTable:
        def __init__(self, rows):
            self.rows = rows

    class FakeDoc:
        paragraphs = [FakeParagraph("Ala"), FakeParagraph("ma kota")]
        tables = [FakeTable([FakeRow([FakeCell("kol1"), FakeCell("kol2")])])]

    monkeypatch.setattr(ocr_module, "Document", lambda _path: FakeDoc())

    path = tmp_path / "doc.docx"
    path.write_bytes(b"stub")

    result = ocr_module.ocr_extraction.fn([{"source_id": "file://doc.docx", "path": str(path)}])

    assert result == [("file://doc.docx", "Ala ma kota kol1 | kol2")]
