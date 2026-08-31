"""The academic-calendar regression case from issue #53.

The page lists five days off. The model kept the ones with a proper name and dropped
"2 XI 2026 r. - dzien wolny od zajec", which is described only generically, so the answer looked
complete while a date was missing. These tests cover the check that makes such a miss visible and
the second pass that collects it.
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
