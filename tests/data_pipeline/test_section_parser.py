import pytest

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
    assert extract_heading_row_sections("1) Publikacja\n2) Grant\n") == []


def test_competency_page_section_keeps_inline_colon_row_under_r1() -> None:
    sections = extract_heading_row_sections(COMPETENCY_PAGE)

    assert len(sections) == 1
    assert sections[0].heading == "R1 - Naukowiec początkujący"
    assert "publikuje wyniki swoich badań w czasopismach naukowych:" in sections[0].rows
    assert sections[0].rows[-1].startswith("W grupie pracowników dydaktycznych")


def test_embedded_heading_without_blank_line_is_split_into_two_sections() -> None:
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


@pytest.mark.parametrize(
    ("page", "index", "required_context", "forbidden_context"),
    [
        (
            """Polska Rama Kompetencji Naukowca
Kompetencje pozadane dla naukowcow na kolejnych etapach kariery (R1-R4)
R1 - Naukowiec poczatkujacy
1) R1 kompetencja
R2 - Naukowiec uznany
1) R2 kompetencja
R3 - Doswiadczony naukowiec
1) R3 kompetencja
R4 - Wiodacy naukowiec
1) R4 kompetencja
""",
            3,
            "Kompetencje pozadane dla naukowcow na kolejnych etapach kariery (R1-R4)",
            ("R1 - Naukowiec poczatkujacy", "R2 - Naukowiec uznany", "R3 - Doswiadczony naukowiec"),
        ),
        (
            """Kompetencje pozadane dla naukowcow na kolejnych etapach kariery (R1-R4)
R1 - Naukowiec poczatkujacy
Kompetencje niezbedne dla naukowca R1:
1) R1 wymaganie
R2 - Naukowiec uznany
1) R2 kompetencja
""",
            1,
            "Kompetencje pozadane dla naukowcow na kolejnych etapach kariery (R1-R4)",
            ("Kompetencje niezbedne dla naukowca R1",),
        ),
        (
            """Kompetencje pozadane dla naukowcow na kolejnych etapach kariery (R1-R3)
R1 - Naukowiec poczatkujacy
1) R1 kompetencja

Ocena kompetencji odbywa się raz w roku.

R2 - Naukowiec uznany
1) R2 kompetencja

Ocena kompetencji odbywa się raz w roku.

R3 - Doswiadczony naukowiec
1) R3 kompetencja
""",
            2,
            "Kompetencje pozadane dla naukowcow na kolejnych etapach kariery (R1-R3)",
            ("Ocena kompetencji odbywa się raz w roku.",),
        ),
    ],
    ids=[
        "r4_keeps_the_page_title",
        "sibling_qualifier_does_not_leak",
        "paragraph_does_not_reset_context",
    ],
)
def test_context_inheritance_for_stage_sections(
    page: str,
    index: int,
    required_context: str,
    forbidden_context: tuple[str, ...],
) -> None:
    sections = extract_heading_row_sections(page)
    context = sections[index].context_headings
    assert required_context in context
    assert not any(value in context for value in forbidden_context)
