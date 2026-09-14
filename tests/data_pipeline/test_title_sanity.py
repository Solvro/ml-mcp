"""The titles issue #79 found in the graph, and the rules that keep them out.

Every string here is one the extraction actually produced: "a) doswiadczenie w kierowaniu i
pracy w zespolach naukowych.", "b) ksiazek,", "Odbyte szkolenia:", "Udzial w", "zagranicznych".
The enumerator and the punctuation split one entity across two keys; the fragments put nodes in
the graph that no question can reach.
"""

from unittest.mock import MagicMock

import pytest

from src.data_pipeline.canonical_nodes import canonical_entity_key
from src.data_pipeline.flows import llm_cypher_generation as cypher_module
from src.data_pipeline.title_sanity import (
    REASON_FRAGMENT,
    REASON_TRUNCATED,
    clean_title,
    sanitize_titles,
    title_rejection_reason,
)


@pytest.mark.parametrize(
    ("stored", "expected"),
    [
        (
            "a) doswiadczenie w kierowaniu i pracy w zespolach naukowych.",
            "doswiadczenie w kierowaniu i pracy w zespolach naukowych",
        ),
        ("b) ksiazek,", "ksiazek"),
        ("e) grantow.", "grantow"),
        ("Odbyte szkolenia:", "Odbyte szkolenia"),
        ("Dzialalnosc organizacyjna:", "Dzialalnosc organizacyjna"),
        ("1. Zasady rekrutacji", "Zasady rekrutacji"),
        ("(iii) Kryteria oceny", "Kryteria oceny"),
        ("- Wydzial Informatyki", "Wydzial Informatyki"),
        ("a) 1) Kryterium pierwsze", "Kryterium pierwsze"),
        ("Semestr   zimowy", "Semestr zimowy"),
    ],
)
def test_the_pages_layout_is_stripped_from_the_title(stored: str, expected: str) -> None:
    assert clean_title(stored) == expected


def test_an_abbreviation_keeps_its_full_stop() -> None:
    """ "2 XI 2026 r." is a date, not a sentence; the calendar reads wrong without the stop."""
    assert clean_title("2 XI 2026 r.") == "2 XI 2026 r."


def test_an_initial_is_not_an_enumerator() -> None:
    """A letter with a full stop is an initial. Dropping it would file two people as one node."""
    assert clean_title("W. Kowalski") == "W. Kowalski"
    assert clean_title("J. Nowak") == "J. Nowak"


def test_a_qualifier_in_brackets_survives() -> None:
    assert clean_title("Informatyka (studia I stopnia)") == "Informatyka (studia I stopnia)"


@pytest.mark.parametrize("title", ["Udzial w", "Wspolpraca z", "Publikacje i"])
def test_a_title_cut_mid_phrase_is_refused(title: str) -> None:
    assert title_rejection_reason(title) == REASON_TRUNCATED


@pytest.mark.parametrize("title", ["zagranicznych", "dydaktyczne", "ksiazek", "grantow"])
def test_a_fragment_lifted_out_of_a_row_is_refused(title: str) -> None:
    assert title_rejection_reason(title) == REASON_FRAGMENT


@pytest.mark.parametrize(
    "title",
    [
        "Informatyka",
        "Rektor",
        "Dziekanat",
        "R1",
        "W4",
        "K2A_W08",
        "Odbyte szkolenia",
        "doswiadczenie w kierowaniu i pracy w zespolach naukowych",
    ],
)
def test_a_name_is_kept(title: str) -> None:
    """The issue proposed refusing every title under two tokens; that would delete these."""
    assert title_rejection_reason(title) is None


def test_an_empty_title_is_refused() -> None:
    assert title_rejection_reason("") is not None


def test_the_enumerator_no_longer_splits_one_criterion_into_two_keys() -> None:
    enumerated = "a) doswiadczenie w organizowaniu i prowadzeniu badan."
    plain = "Doswiadczenie w organizowaniu i prowadzeniu badan"

    assert canonical_entity_key(enumerated) == canonical_entity_key(plain)


def test_a_heading_and_its_colon_key_the_same() -> None:
    assert canonical_entity_key("Odbyte szkolenia:") == canonical_entity_key("Odbyte szkolenia")


def test_the_title_stored_on_the_node_is_cleaned() -> None:
    statements = [
        "MERGE (n1:Criterion {title: 'a) doswiadczenie w kierowaniu zespolem.', "
        "context: 'Kryterium oceny'})"
    ]

    kept, report = sanitize_titles(statements)

    assert kept == [
        "MERGE (n1:Criterion {title: 'doswiadczenie w kierowaniu zespolem', "
        "context: 'Kryterium oceny'})"
    ]
    assert report.cleaned == [
        ("a) doswiadczenie w kierowaniu zespolem.", "doswiadczenie w kierowaniu zespolem")
    ]


def test_a_clean_title_is_left_exactly_as_it_was() -> None:
    statements = ["MERGE (n1:Course {title: 'Analiza matematyczna', context: 'Kurs'})"]

    kept, report = sanitize_titles(statements)

    assert kept == statements
    assert not report.changed


def test_an_escaped_quote_survives_the_rewrite() -> None:
    statements = [r"MERGE (n1:Topic {title: 'Prawo do \'wolnosci\' badan:', context: 'X'})"]

    kept, _ = sanitize_titles(statements)

    assert kept == [r"MERGE (n1:Topic {title: 'Prawo do \'wolnosci\' badan', context: 'X'})"]


def test_a_node_without_a_name_is_dropped() -> None:
    statements = [
        "MERGE (n1:Criterion {title: 'Odbyte szkolenia:', context: 'Naglowek'})",
        "MERGE (n2:Criterion {title: 'zagranicznych', context: 'Fragment wiersza'})",
    ]

    kept, report = sanitize_titles(statements)

    assert kept == ["MERGE (n1:Criterion {title: 'Odbyte szkolenia', context: 'Naglowek'})"]
    assert report.rejected == [("zagranicznych", REASON_FRAGMENT)]
    assert report.dropped_statements == 1


def test_dropping_a_node_takes_the_relationships_that_name_it() -> None:
    """A page runs as one query, so a relationship left pointing at nothing fails all of it."""
    statements = [
        "MERGE (n1:CriterionCategory {title: 'Dzialalnosc organizacyjna:', context: 'Kategoria'})",
        "MERGE (n2:Criterion {title: 'Udzial w', context: 'Uciety wiersz'})",
        "MERGE (n1)-[:HAS_CRITERION]->(n2)",
    ]

    kept, report = sanitize_titles(statements)

    assert kept == [
        "MERGE (n1:CriterionCategory {title: 'Dzialalnosc organizacyjna', context: 'Kategoria'})"
    ]
    assert report.rejected == [("Udzial w", REASON_TRUNCATED)]
    assert report.dropped_statements == 2


def test_a_variable_that_only_looks_like_another_is_not_dropped() -> None:
    statements = [
        "MERGE (n1:Criterion {title: 'zagranicznych', context: 'Fragment'})",
        "MERGE (n11:Course {title: 'Analiza matematyczna', context: 'Kurs'})",
    ]

    kept, _ = sanitize_titles(statements)

    assert kept == ["MERGE (n11:Course {title: 'Analiza matematyczna', context: 'Kurs'})"]


def test_statements_without_a_title_are_untouched() -> None:
    statements = ["MATCH (n:Course) RETURN n", "MERGE (n1)-[:BELONGS_TO]->(n2)"]

    kept, report = sanitize_titles(statements)

    assert kept == statements
    assert not report.changed


PAGE = "Kryteria oceny pracownikow naukowych opisane sa w zalaczniku do uchwaly.\n"

# One page of the criteria document as it came out of extraction: enumerated rows, a heading
# with its colon, and two fragments that name nothing.
EXTRACTED_STATEMENTS = [
    "MERGE (n1:CriterionCategory {title: 'Dzialalnosc organizacyjna:', context: 'Kategoria'})",
    "MERGE (n2:Criterion {title: 'a) doswiadczenie w kierowaniu zespolem.', context: 'Ocena'})",
    "MERGE (n3:Criterion {title: 'zagranicznych', context: 'Fragment wiersza'})",
    "MERGE (n1)-[:HAS_CRITERION]->(n2)",
    "MERGE (n1)-[:HAS_CRITERION]->(n3)",
]


def test_a_generated_page_reaches_the_graph_with_names_only(monkeypatch) -> None:
    class FakePipe:
        def run(self, context: str, schema_context: str = "") -> list[str]:
            return list(EXTRACTED_STATEMENTS)

        def run_missing_rows(self, context: str, rows: list[str]) -> list[str]:
            return []

    monkeypatch.setattr(cypher_module, "LLMPipe", FakePipe)
    monkeypatch.setattr(cypher_module, "get_run_logger", MagicMock)

    result = cypher_module.generate_cypher_queries.fn(PAGE)

    assert "'doswiadczenie w kierowaniu zespolem'" in result
    assert "a) doswiadczenie" not in result
    assert "Dzialalnosc organizacyjna:" not in result
    assert "zagranicznych" not in result
    # The category, the criterion, and the one relationship between them.
    assert result.count("MERGE") == 3
