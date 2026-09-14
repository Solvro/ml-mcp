"""The two pages whose rows went missing, and the check that now reports them.

Issue #53, the academic calendar: the page lists five days off. The model kept the ones with a
proper name and dropped "2 XI 2026 r. - dzien wolny od zajec", which is described only
generically, so the answer looked complete while a date was missing.

Issue #78, the R1-R4 competency page: a PDF text layer had put every bullet on a line of its own,
so the check saw no rows at all, and the rows it did see counted as covered because their wording
sat inside one category node's context. R4's competencies became nodes, R1-R3's did not, and
"Jakie sa kompetencje pozadane dla naukowca R2?" answered "Nie wiem".

These tests cover the check that makes such a miss visible and the second pass that collects it.
"""

from unittest.mock import MagicMock

import pytest

from src.data_pipeline.completeness import extract_list_rows, rows_missing_from_cypher
from src.data_pipeline.flows import llm_cypher_generation as cypher_module

CALENDAR_PAGE = """Dni wolne od zajec w semestrze zimowym 2026/2027

- 1 XI 2026 r. - Wszystkich Swietych
- 2 XI 2026 r. - dzien wolny od zajec
- 11 XI 2026 r. - Swieto Niepodleglosci
- 16 XI 2026 r. - Obchody Swieta PWr
- 24 XII 2026 r. - Wigilia

Zajecia odbywaja sie zgodnie z planem.
"""

# What the model actually produced: the generically described row is absent in any spelling.
INCOMPLETE_EXTRACTION = [
    "MERGE (n1:DayOff {title: '1 XI 2026 r.', context: 'Wszystkich Swietych'})",
    "MERGE (n2:DayOff {title: '11 XI 2026 r.', context: 'Swieto Niepodleglosci'})",
    "MERGE (n3:DayOff {title: '16 XI 2026 r.', context: 'Obchody Swieta PWr'})",
    "MERGE (n4:DayOff {title: '24 XII 2026 r.', context: 'Wigilia'})",
]
COMPLETE_EXTRACTION = INCOMPLETE_EXTRACTION + [
    "MERGE (n5:DayOff {title: '2 XI 2026 r.', context: 'dzien wolny od zajec'})",
]

MISSED_ROW = "2 XI 2026 r. - dzien wolny od zajec"


def test_every_bullet_row_is_counted() -> None:
    rows = extract_list_rows(CALENDAR_PAGE)

    assert len(rows) == 5
    assert MISSED_ROW in rows


def test_prose_lines_are_not_counted_as_rows() -> None:
    rows = extract_list_rows(CALENDAR_PAGE)

    assert not any("Zajecia odbywaja sie" in row for row in rows)
    assert not any("Dni wolne od zajec w semestrze" in row for row in rows)


def test_numbered_and_table_rows_are_counted() -> None:
    page = "1. Pierwszy punkt regulaminu\n2) Drugi punkt regulaminu\n| Kurs | 5 ECTS |\n"

    rows = extract_list_rows(page)

    assert "Pierwszy punkt regulaminu" in rows
    assert "Drugi punkt regulaminu" in rows
    assert any("Kurs" in row and "ECTS" in row for row in rows)


def test_rows_with_nothing_to_match_on_are_ignored() -> None:
    assert extract_list_rows("- \n- x\n---\n") == []


def test_the_dropped_calendar_row_is_reported_as_missing() -> None:
    missing = rows_missing_from_cypher(extract_list_rows(CALENDAR_PAGE), INCOMPLETE_EXTRACTION)

    assert missing == [MISSED_ROW]


def test_a_complete_extraction_reports_nothing_missing() -> None:
    missing = rows_missing_from_cypher(extract_list_rows(CALENDAR_PAGE), COMPLETE_EXTRACTION)

    assert missing == []


def test_a_reworded_row_still_counts_as_covered() -> None:
    """The model may rephrase; only a row it never read leaves almost no wording behind."""
    rows = ["11 XI 2026 r. - Swieto Niepodleglosci"]
    statements = [
        "MERGE (n:DayOff {title: 'Swieto Niepodleglosci', context: 'Dzien wolny 11 XI 2026'})"
    ]

    assert rows_missing_from_cypher(rows, statements) == []


def test_an_empty_extraction_reports_every_row() -> None:
    rows = extract_list_rows(CALENDAR_PAGE)

    assert rows_missing_from_cypher(rows, []) == rows


def test_a_page_without_rows_is_never_flagged() -> None:
    assert rows_missing_from_cypher([], INCOMPLETE_EXTRACTION) == []


def test_missed_rows_trigger_a_second_extraction_pass(monkeypatch) -> None:
    second_pass_rows: list[list[str]] = []

    class FakePipe:
        def run(self, context: str, schema_context: str = "") -> list[str]:
            return list(INCOMPLETE_EXTRACTION)

        def run_missing_rows(self, context: str, rows: list[str]) -> list[str]:
            second_pass_rows.append(rows)
            return [
                "MERGE (extra1:DayOff {title: '2 XI 2026 r.', context: 'dzien wolny od zajec'})"
            ]

    monkeypatch.setattr(cypher_module, "LLMPipe", FakePipe)
    monkeypatch.setattr(cypher_module, "get_run_logger", MagicMock)

    result = cypher_module.generate_cypher_queries.fn(CALENDAR_PAGE)

    assert second_pass_rows == [[MISSED_ROW]]
    assert "'2 XI 2026 r.'" in result
    assert result.count("MERGE") == 5


def test_a_complete_first_pass_skips_the_second(monkeypatch) -> None:
    """The extra pass costs a model call; it must only run when something is actually missing."""
    second_pass_calls: list[int] = []

    class FakePipe:
        def run(self, context: str, schema_context: str = "") -> list[str]:
            return list(COMPLETE_EXTRACTION)

        def run_missing_rows(self, context: str, rows: list[str]) -> list[str]:
            second_pass_calls.append(1)
            return []

    monkeypatch.setattr(cypher_module, "LLMPipe", FakePipe)
    monkeypatch.setattr(cypher_module, "get_run_logger", MagicMock)

    cypher_module.generate_cypher_queries.fn(CALENDAR_PAGE)

    assert second_pass_calls == []


def test_a_failed_second_pass_keeps_the_first_pass_output(monkeypatch) -> None:
    class FakePipe:
        def run(self, context: str, schema_context: str = "") -> list[str]:
            return list(INCOMPLETE_EXTRACTION)

        def run_missing_rows(self, context: str, rows: list[str]) -> list[str]:
            return []

    monkeypatch.setattr(cypher_module, "LLMPipe", FakePipe)
    monkeypatch.setattr(cypher_module, "get_run_logger", MagicMock)

    result = cypher_module.generate_cypher_queries.fn(CALENDAR_PAGE)

    assert result.count("MERGE") == 4


# Review feedback on PR #58: the extra pass is one more model call per page that lost rows, so a
# run of list-heavy pages is a real cost bump. It stays on by default but is now boundable and
# always reported.
def test_the_extra_pass_budget_is_unlimited_by_default(monkeypatch) -> None:
    monkeypatch.delenv("DATA_PIPELINE_MAX_MISSED_ROW_PASSES", raising=False)

    assert cypher_module._get_missed_row_pass_budget() == 0


@pytest.mark.parametrize("raw_value", ["not-a-number", "-3"])
def test_an_unusable_budget_falls_back_to_unlimited(monkeypatch, raw_value) -> None:
    monkeypatch.setenv("DATA_PIPELINE_MAX_MISSED_ROW_PASSES", raw_value)

    assert cypher_module._get_missed_row_pass_budget() == 0


def test_pages_stop_getting_a_second_pass_once_the_budget_is_spent(monkeypatch) -> None:
    monkeypatch.setenv("DATA_PIPELINE_MAX_MISSED_ROW_PASSES", "1")
    cypher_module.reset_missed_row_passes()
    second_pass_calls: list[int] = []

    class FakePipe:
        def run(self, context: str, schema_context: str = "") -> list[str]:
            return list(INCOMPLETE_EXTRACTION)

        def run_missing_rows(self, context: str, rows: list[str]) -> list[str]:
            second_pass_calls.append(1)
            return ["MERGE (extra1:DayOff {title: '2 XI 2026 r.', context: 'dzien wolny'})"]

    monkeypatch.setattr(cypher_module, "LLMPipe", FakePipe)
    monkeypatch.setattr(cypher_module, "get_run_logger", MagicMock)

    cypher_module.generate_cypher_queries.fn(CALENDAR_PAGE)
    cypher_module.generate_cypher_queries.fn(CALENDAR_PAGE)

    assert second_pass_calls == [1]
    assert cypher_module.missed_row_passes_used() == 1


def test_the_run_reports_what_the_extra_passes_cost(monkeypatch) -> None:
    monkeypatch.delenv("DATA_PIPELINE_MAX_MISSED_ROW_PASSES", raising=False)
    cypher_module.reset_missed_row_passes()

    class FakePipe:
        def run(self, context: str, schema_context: str = "") -> list[str]:
            return list(INCOMPLETE_EXTRACTION)

        def run_missing_rows(self, context: str, rows: list[str]) -> list[str]:
            return ["MERGE (extra1:DayOff {title: '2 XI 2026 r.', context: 'dzien wolny'})"]

    monkeypatch.setattr(cypher_module, "LLMPipe", FakePipe)
    monkeypatch.setattr(cypher_module, "get_run_logger", MagicMock)

    cypher_module.generate_cypher_queries.fn(CALENDAR_PAGE)
    cypher_module.generate_cypher_queries.fn(CALENDAR_PAGE)

    assert cypher_module.missed_row_passes_used() == 2


def test_resetting_the_budget_starts_a_fresh_run(monkeypatch) -> None:
    cypher_module.reset_missed_row_passes()
    cypher_module._claim_missed_row_pass()

    cypher_module.reset_missed_row_passes()

    assert cypher_module.missed_row_passes_used() == 0


# The R1-R4 competency page from issue #78, in the shape a PDF text layer produces it: every
# bullet on a line of its own, and a long row wrapped across two lines. Before the markers were
# rejoined this page held no line that reads as a row at all, so the completeness check passed
# over it in silence while three of the four career stages kept none of their competencies.
COMPETENCY_PAGE = """Polska Rama Kompetencji Naukowca
Kompetencje pożądane dla naukowców na kolejnych etapach kariery (R1-R4)

R1 - Naukowiec początkujący
•
prowadzi badania naukowe pod nadzorem opiekuna naukowego
•
zna metody badawcze stosowane w swojej dyscyplinie
•
publikuje wyniki swoich badań w czasopismach naukowych:
a)
o zasięgu krajowym,
b)
o zasięgu międzynarodowym.
•
W grupie pracowników dydaktycznych (których podstawowym obowiązkiem jest kształcenie
studentów) prowadzi zajęcia pod opieką nauczyciela akademickiego.

Ocena kompetencji odbywa się raz w roku.
"""

COMPETENCY_ROWS = [
    "prowadzi badania naukowe pod nadzorem opiekuna naukowego",
    "zna metody badawcze stosowane w swojej dyscyplinie",
    "publikuje wyniki swoich badań w czasopismach naukowych:",
    "o zasięgu krajowym,",
    "o zasięgu międzynarodowym.",
    "W grupie pracowników dydaktycznych (których podstawowym obowiązkiem jest kształcenie "
    "studentów) prowadzi zajęcia pod opieką nauczyciela akademickiego.",
]

# What the model did to R1, R2 and R3: one node for the stage, with every bullet recited inside
# its context. Every row's wording is in the output, and not one row has a node.
CATEGORY_BLOB_EXTRACTION = [
    "MERGE (n1:CriterionCategory {title: 'R1 - Naukowiec poczatkujacy', "
    "context: 'prowadzi badania naukowe pod nadzorem opiekuna naukowego; zna metody badawcze "
    "stosowane w swojej dyscyplinie; publikuje wyniki swoich badan w czasopismach naukowych "
    "o zasiegu krajowym i o zasiegu miedzynarodowym; w grupie pracownikow dydaktycznych, "
    "ktorych podstawowym obowiazkiem jest ksztalcenie studentow, prowadzi zajecia pod opieka "
    "nauczyciela akademickiego'})"
]

# What R4 got, and what every stage should get: a node per row, titled from the row itself.
PER_ROW_EXTRACTION = [
    "MERGE (n1:CriterionCategory {title: 'R1 - Naukowiec poczatkujacy', "
    "context: 'Pierwszy etap kariery naukowej'})",
    "MERGE (n2:Competency {title: 'Prowadzi badania naukowe pod nadzorem opiekuna naukowego', "
    "context: 'Kompetencja naukowca R1'})",
    "MERGE (n3:Competency {title: 'Zna metody badawcze stosowane w swojej dyscyplinie', "
    "context: 'Kompetencja naukowca R1'})",
    "MERGE (n4:Competency {title: 'Publikuje wyniki swoich badan w czasopismach naukowych', "
    "context: 'Kompetencja naukowca R1'})",
    "MERGE (n5:Competency {title: 'Publikacje o zasiegu krajowym', context: 'Czasopisma krajowe'})",
    "MERGE (n6:Competency {title: 'Publikacje o zasiegu miedzynarodowym', "
    "context: 'Czasopisma miedzynarodowe'})",
    "MERGE (n7:Competency {title: 'W grupie pracownikow dydaktycznych prowadzi zajecia pod "
    "opieka nauczyciela akademickiego', context: 'Dotyczy pracownikow dydaktycznych, ktorych "
    "podstawowym obowiazkiem jest ksztalcenie studentow'})",
]


def test_bullets_on_their_own_line_are_counted_as_rows() -> None:
    """A PDF text layer detaches the marker from its text, and 168 of them counted as zero."""
    assert extract_list_rows(COMPETENCY_PAGE) == COMPETENCY_ROWS


def test_a_row_wrapped_across_lines_is_kept_whole() -> None:
    """The one row the old check found was half a sentence, and became a node in that state."""
    wrapped = [row for row in extract_list_rows(COMPETENCY_PAGE) if row.startswith("W grupie")]

    assert len(wrapped) == 1
    assert wrapped[0].endswith("prowadzi zajęcia pod opieką nauczyciela akademickiego.")


def test_the_paragraph_after_a_list_is_not_swallowed_by_its_last_row() -> None:
    assert not any("Ocena kompetencji" in row for row in extract_list_rows(COMPETENCY_PAGE))


def test_rows_recited_in_a_parent_context_have_no_node_of_their_own() -> None:
    """The R1-R3 shape: every row's wording is present, and every row is still missing."""
    missing = rows_missing_from_cypher(COMPETENCY_ROWS, CATEGORY_BLOB_EXTRACTION)

    assert missing == COMPETENCY_ROWS


def test_a_node_per_row_reports_nothing_missing() -> None:
    """The R4 shape, and the reason the check anchors on titles rather than banning context."""
    assert rows_missing_from_cypher(COMPETENCY_ROWS, PER_ROW_EXTRACTION) == []


def test_a_title_that_names_the_section_does_not_cover_its_rows() -> None:
    rows = ["prowadzi badania naukowe pod nadzorem opiekuna naukowego"]
    statements = [
        "MERGE (n:CriterionCategory {title: 'Kompetencje pozadane dla naukowca R2', "
        "context: 'prowadzi badania naukowe pod nadzorem opiekuna naukowego'})"
    ]

    assert rows_missing_from_cypher(rows, statements) == rows


def test_properties_set_after_the_merge_still_count_as_the_node_title() -> None:
    """The canonical-key rewrite moves title and context out of the MERGE pattern."""
    rows = ["zna metody badawcze stosowane w swojej dyscyplinie"]
    statements = [
        "MERGE (n1:Competency {key: 'zna metody badawcze stosowane w swojej dyscyplinie'}) "
        "ON CREATE SET n1.title = 'Zna metody badawcze stosowane w swojej dyscyplinie', "
        "n1.context = 'Kompetencja naukowca R1'"
    ]

    assert rows_missing_from_cypher(rows, statements) == []


def test_a_context_blob_sends_its_rows_to_the_second_pass(monkeypatch) -> None:
    """End to end: the shape that answered 'Nie wiem' now costs one extra pass and gets nodes."""
    second_pass_rows: list[list[str]] = []

    class FakePipe:
        def run(self, context: str, schema_context: str = "") -> list[str]:
            return list(CATEGORY_BLOB_EXTRACTION)

        def run_missing_rows(self, context: str, rows: list[str]) -> list[str]:
            second_pass_rows.append(rows)
            return list(PER_ROW_EXTRACTION[1:])

    monkeypatch.delenv("DATA_PIPELINE_MAX_MISSED_ROW_PASSES", raising=False)
    cypher_module.reset_missed_row_passes()
    monkeypatch.setattr(cypher_module, "LLMPipe", FakePipe)
    monkeypatch.setattr(cypher_module, "get_run_logger", MagicMock)

    result = cypher_module.generate_cypher_queries.fn(COMPETENCY_PAGE)

    assert second_pass_rows == [COMPETENCY_ROWS]
    assert "Zna metody badawcze stosowane w swojej dyscyplinie" in result
