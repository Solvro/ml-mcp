from src.data_pipeline.section_parser import extract_heading_row_sections
from tests.data_pipeline.test_completeness import COMPETENCY_PAGE

HEADING_ROWS_PAGE = """Kompetencje pozadane dla naukowca R2:
1) Publikacje o zasiegu miedzynarodowym
2) Wspolpraca z zespolami badawczymi

Kompetencje wymagane dla naukowca R2:
a) Samodzielne prowadzenie badan
b) Pozyskiwanie finansowania projektow
"""


def test_heading_sections_keep_heading_and_row_order() -> None:
    sections = extract_heading_row_sections(HEADING_ROWS_PAGE)

    assert [section.heading for section in sections] == [
        "Kompetencje pozadane dla naukowca R2",
        "Kompetencje wymagane dla naukowca R2",
    ]
    assert list(sections[0].rows) == [
        "Publikacje o zasiegu miedzynarodowym",
        "Wspolpraca z zespolami badawczymi",
    ]
    assert list(sections[1].rows) == [
        "Samodzielne prowadzenie badan",
        "Pozyskiwanie finansowania projektow",
    ]


def test_rows_without_a_heading_do_not_form_a_section() -> None:
    sections = extract_heading_row_sections("1) Publikacja\n2) Grant\n")

    assert sections == []


def test_competency_page_section_keeps_inline_colon_row_under_r1() -> None:
    sections = extract_heading_row_sections(COMPETENCY_PAGE)

    assert len(sections) == 1
    assert sections[0].heading == "R1 - Naukowiec początkujący"
    assert "publikuje wyniki swoich badań w czasopismach naukowych:" in sections[0].rows
    assert sections[0].rows[-1].startswith("W grupie pracowników dydaktycznych")


def test_section_parser_splits_next_heading_when_blank_line_is_missing() -> None:
    page = """Kompetencje pozadane dla naukowca R2:
1) Publikacje o zasiegu miedzynarodowym
2) Pozyskiwanie finansowania Kompetencje niezbedne dla naukowca R2:
a) Samodzielne prowadzenie badan
b) Mentoring doktorantów
"""

    sections = extract_heading_row_sections(page)

    assert [section.heading for section in sections] == [
        "Kompetencje pozadane dla naukowca R2",
        "Kompetencje niezbedne dla naukowca R2",
    ]
    assert sections[0].rows == (
        "Publikacje o zasiegu miedzynarodowym",
        "Pozyskiwanie finansowania",
    )
    assert sections[1].rows == (
        "Samodzielne prowadzenie badan",
        "Mentoring doktorantów",
    )


def test_stage_context_keeps_page_qualifier_for_r4() -> None:
    page = """Polska Rama Kompetencji Naukowca
Kompetencje pozadane dla naukowcow na kolejnych etapach kariery (R1-R4)
R1 - Naukowiec poczatkujacy
1) R1 kompetencja
R2 - Naukowiec uznany
1) R2 kompetencja
R3 - Doswiadczony naukowiec
1) R3 kompetencja
R4 - Wiodacy naukowiec
1) R4 kompetencja
"""

    sections = extract_heading_row_sections(page)

    assert [section.heading for section in sections] == [
        "R1 - Naukowiec poczatkujacy",
        "R2 - Naukowiec uznany",
        "R3 - Doswiadczony naukowiec",
        "R4 - Wiodacy naukowiec",
    ]
    assert (
        "Kompetencje pozadane dla naukowcow na kolejnych etapach kariery (R1-R4)"
        in sections[3].context_headings
    )
    assert not any(
        sibling in sections[3].context_headings
        for sibling in (
            "R1 - Naukowiec poczatkujacy",
            "R2 - Naukowiec uznany",
            "R3 - Doswiadczony naukowiec",
        )
    )


def test_stage_specific_qualifier_does_not_leak_to_sibling_stage() -> None:
    page = """Kompetencje pozadane dla naukowcow na kolejnych etapach kariery (R1-R4)
R1 - Naukowiec poczatkujacy
Kompetencje niezbedne dla naukowca R1:
1) R1 wymaganie
R2 - Naukowiec uznany
1) R2 kompetencja
"""

    sections = extract_heading_row_sections(page)

    assert [section.heading for section in sections] == [
        "Kompetencje niezbedne dla naukowca R1",
        "R2 - Naukowiec uznany",
    ]
    assert "Kompetencje niezbedne dla naukowca R1" not in sections[1].context_headings


def test_paragraphs_between_stages_do_not_replace_page_qualifier_context() -> None:
    page = """Kompetencje pozadane dla naukowcow na kolejnych etapach kariery (R1-R3)
R1 - Naukowiec poczatkujacy
1) R1 kompetencja

Annual review happens once a year.

R2 - Naukowiec uznany
1) R2 kompetencja

Annual review happens once a year.

R3 - Doswiadczony naukowiec
1) R3 kompetencja
"""

    sections = extract_heading_row_sections(page)

    assert [section.heading for section in sections] == [
        "R1 - Naukowiec poczatkujacy",
        "R2 - Naukowiec uznany",
        "R3 - Doswiadczony naukowiec",
    ]
    expected_parent = "Kompetencje pozadane dla naukowcow na kolejnych etapach kariery (R1-R3)"
    for section in sections:
        assert expected_parent in section.context_headings
        assert "Annual review happens once a year." not in section.context_headings
