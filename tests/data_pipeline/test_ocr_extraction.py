from pathlib import Path

import pymupdf

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
        paragraphs = [FakeParagraph("Ola"), FakeParagraph("ma kota")]
        tables = [FakeTable([FakeRow([FakeCell("kol1"), FakeCell("kol2")])])]

    monkeypatch.setattr(ocr_module, "Document", lambda _path: FakeDoc())

    path = tmp_path / "doc.docx"
    path.write_bytes(b"stub")

    result = ocr_module.ocr_extraction.fn([{"source_id": "file://doc.docx", "path": str(path)}])

    assert result == [("file://doc.docx", "Ola ma kota kol1 | kol2")]


def test_pdf_text_layer_extraction(monkeypatch, tmp_path: Path):
    pdf_path = tmp_path / "text-layer.pdf"

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Politechnika Wroclawska test")
    doc.save(str(pdf_path))
    doc.close()

    monkeypatch.setenv("OCR_MIN_TEXT_CHARS", "5")

    result = ocr_module.ocr_extraction.fn(
        [{"source_id": "file://text-layer.pdf", "path": str(pdf_path)}]
    )

    assert len(result) == 1
    assert result[0][0] == "file://text-layer.pdf#page=1"
    assert "Politechnika" in result[0][1]

def test_pdf_falls_back_to_ocr_when_text_below_threshold(monkeypatch, tmp_path: Path):
    pdf_path = tmp_path / "short-text.pdf"

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "abc")
    doc.save(str(pdf_path))
    doc.close()

    monkeypatch.setenv("OCR_MIN_TEXT_CHARS", "50")

    called = {"count": 0}

    def fake_ocr_pdf_page(_doc, _page_index, _scale, _lang):
        called["count"] += 1
        return "OCR fallback content"

    monkeypatch.setattr(ocr_module, "_ocr_pdf_page", fake_ocr_pdf_page)

    result = ocr_module.ocr_extraction.fn(
        [{"source_id": "file://short-text.pdf", "path": str(pdf_path)}]
    )

    assert called["count"] == 1
    assert result == [("file://short-text.pdf#page=1", "OCR fallback content")]
