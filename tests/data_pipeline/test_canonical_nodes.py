"""One canonical node per entity, whichever way a page spells it.

Issue #53: merging on ``title + context`` together produced a node per mention — a Semester
holding the dates and a second Semester holding nothing, "Cyberbezpieczenstwo" beside
"Cyberbezpieczenstwo (CBE)".
"""

from unittest.mock import MagicMock

import pytest

from src.data_pipeline.canonical_nodes import (
    canonical_entity_key,
    looks_like_abbreviation,
    rewrite_merge_to_canonical_key,
)
from src.data_pipeline.flows import llm_cypher_generation as cypher_module


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Cyberbezpieczenstwo", "cyberbezpieczenstwo"),
        ("Cyberbezpieczeństwo", "cyberbezpieczenstwo"),
        ("CYBERBEZPIECZENSTWO", "cyberbezpieczenstwo"),
        ("Cyberbezpieczenstwo (CBE)", "cyberbezpieczenstwo"),
        ("  Cyberbezpieczenstwo  ", "cyberbezpieczenstwo"),
    ],
)
def test_spellings_of_one_entity_share_a_key(title, expected) -> None:
    assert canonical_entity_key(title) == expected


# Review feedback on PR #58: dropping every trailing bracket fused entities that the bracket is
# the only thing distinguishing. Fusing two entities into one is worse than splitting one into
# two, because nothing in the graph shows it happened.
@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("Informatyka (studia I stopnia)", "Informatyka (studia II stopnia)"),
        ("Informatyka (I stopnia)", "Informatyka (II stopnia)"),
        ("Informatyka (stacjonarne)", "Informatyka (niestacjonarne)"),
        ("Informatyka (II)", "Informatyka (III)"),
        ("Rekrutacja (2024)", "Rekrutacja (2025)"),
    ],
)
def test_a_qualifying_bracket_keeps_two_entities_apart(first, second) -> None:
    assert canonical_entity_key(first) != canonical_entity_key(second)


@pytest.mark.parametrize(
    "title",
    [
        "Informatyka (studia I stopnia)",
        "Informatyka (stacjonarne)",
        "Informatyka (II)",
        "Rekrutacja (2024)",
    ],
)
def test_a_qualifying_bracket_stays_in_the_key(title) -> None:
    assert canonical_entity_key(title) != canonical_entity_key("Informatyka")
    assert canonical_entity_key(title) != canonical_entity_key("Rekrutacja")


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Cyberbezpieczenstwo (CBE)", "cyberbezpieczenstwo"),
        ("Politechnika Wroclawska (PWr)", "politechnika wroclawska"),
        ("Wydzial Informatyki (W8)", "wydzial informatyki"),
    ],
)
def test_an_abbreviating_bracket_is_still_dropped(title, expected) -> None:
    assert canonical_entity_key(title) == expected


@pytest.mark.parametrize(
    ("bracketed", "is_abbreviation"),
    [
        ("CBE", True),
        ("PWr", True),
        ("W8", True),
        ("studia I stopnia", False),
        ("stacjonarne", False),
        ("niestacjonarne", False),
        ("II", False),
        ("VIII", False),
        ("2024", False),
        ("", False),
        ("BardzoDlugiSkrot", False),
        ("Nowy", False),
        ("W4", True),
    ],
)
def test_only_a_short_capitalised_token_reads_as_an_abbreviation(
    bracketed, is_abbreviation
) -> None:
    assert looks_like_abbreviation(bracketed) is is_abbreviation


def test_distinct_entities_keep_distinct_keys() -> None:
    assert canonical_entity_key("Semestr zimowy 2026/2027") != canonical_entity_key(
        "Semestr letni 2026/2027"
    )


def test_punctuation_variants_of_a_calendar_row_share_a_key() -> None:
    assert canonical_entity_key("2 XI 2026 r. - dzien wolny od zajec") == canonical_entity_key(
        "2 XI 2026 r., dzien wolny od zajec"
    )


def test_title_that_is_only_brackets_still_produces_a_key() -> None:
    """Stripping the bracket must not leave an entity with no key at all."""
    assert canonical_entity_key("(CBE)") == "cbe"


def test_title_without_usable_characters_produces_no_key() -> None:
    assert canonical_entity_key("---") == ""


def test_merge_keys_on_the_canonical_key_not_on_title_and_context() -> None:
    statement = "MERGE (node1:Course {title: 'Analiza matematyczna', context: '8 ECTS'})"

    rewritten = rewrite_merge_to_canonical_key(statement)

    assert rewritten.startswith("MERGE (node1:Course {key: 'analiza matematyczna'})")
    assert "ON CREATE SET node1.title = 'Analiza matematyczna'" in rewritten
    assert "node1.context = '8 ECTS'" in rewritten
    assert "ON MATCH SET" in rewritten


def test_a_second_mention_keeps_the_fuller_title() -> None:
    rewritten = rewrite_merge_to_canonical_key(
        "MERGE (node1:StudyProgram {title: 'Cyberbezpieczenstwo (CBE)', context: 'Kierunek'})"
    )

    assert "size('Cyberbezpieczenstwo (CBE)') > size(coalesce(node1.title, ''))" in rewritten


def test_a_second_mention_appends_a_new_context_instead_of_replacing_it() -> None:
    """The Semester case: one mention carries the dates, the other does not."""
    rewritten = rewrite_merge_to_canonical_key(
        "MERGE (node1:Semester {title: 'Semestr zimowy', context: 'Trwa do 23 lutego 2027'})"
    )

    assert "CONTAINS 'Trwa do 23 lutego 2027'" in rewritten
    assert "node1.context + '; ' + 'Trwa do 23 lutego 2027'" in rewritten


def test_extra_properties_are_preserved_on_create_and_kept_on_match() -> None:
    rewritten = rewrite_merge_to_canonical_key(
        "MERGE (node1:Course {title: 'Analiza', context: 'x', ects: 8})"
    )

    assert "node1.ects = 8" in rewritten
    assert "node1.ects = coalesce(node1.ects, 8)" in rewritten


def test_relationship_merge_is_left_alone() -> None:
    statement = "MERGE (node1)-[:HAS_DAY_OFF]->(node2)"

    assert rewrite_merge_to_canonical_key(statement) == statement


def test_combined_pattern_is_left_alone() -> None:
    """Only a lone node MERGE can take ON CREATE / ON MATCH without changing the pattern."""
    statement = "MERGE (node1:Course {title: 'A'})-[:PART_OF]->(node2:StudyProgram {title: 'B'})"

    assert rewrite_merge_to_canonical_key(statement) == statement


def test_merge_without_a_title_is_left_alone() -> None:
    statement = "MERGE (node1:Course {code: 'W8-INF'})"

    assert rewrite_merge_to_canonical_key(statement) == statement


def test_merge_with_a_non_string_title_is_left_alone() -> None:
    statement = "MERGE (node1:Course {title: 42})"

    assert rewrite_merge_to_canonical_key(statement) == statement


def test_comma_inside_a_quoted_value_does_not_split_properties() -> None:
    rewritten = rewrite_merge_to_canonical_key(
        "MERGE (node1:DayOff {title: '2 XI 2026', context: 'dzien wolny, bez zajec'})"
    )

    assert "node1.context = 'dzien wolny, bez zajec'" in rewritten
    assert rewritten.startswith("MERGE (node1:DayOff {key: '2 xi 2026'})")


def test_generation_task_emits_canonical_key_merges(monkeypatch) -> None:
    """The two spellings from the issue must produce the same merge key end to end."""

    class FakePipe:
        def run(self, context: str, schema_context: str = "") -> list[str]:
            return [
                "MERGE (node1:Program {title: 'Cyberbezpieczeństwo', context: 'Kierunek W4'})",
                "MERGE (node2:StudyProgram {title: 'Cyberbezpieczenstwo (CBE)', "
                "context: 'Studia inzynierskie'})",
            ]

    monkeypatch.setattr(cypher_module, "LLMPipe", FakePipe)
    monkeypatch.setattr(cypher_module, "get_run_logger", MagicMock)

    result = cypher_module.generate_cypher_queries.fn("source text")

    assert result.count("{key: 'cyberbezpieczenstwo'}") == 2
    assert ":Program {" not in result


def test_no_generated_statement_contains_the_statement_delimiter() -> None:
    """The whole ingestion path splits on "|", so a pipe in a literal tears the statement apart."""
    rewritten = rewrite_merge_to_canonical_key(
        "MERGE (node1:Semester {title: 'Semestr zimowy', context: 'Trwa do 23 lutego 2027'})"
    )

    assert "|" not in rewritten


def test_appended_contexts_are_separated_without_a_pipe() -> None:
    rewritten = rewrite_merge_to_canonical_key(
        "MERGE (node1:Semester {title: 'Semestr zimowy', context: 'Trwa do 23 lutego 2027'})"
    )

    assert "node1.context + '; ' + 'Trwa do 23 lutego 2027'" in rewritten


def test_generated_output_survives_the_pipe_split(monkeypatch) -> None:
    """What generate_cypher_queries joins with "|", graph_populating must split back intact."""

    class FakePipe:
        def run(self, context: str, schema_context: str = "") -> list[str]:
            return [
                "MERGE (node1:Semester {title: 'Semestr zimowy', context: 'Trwa do 23 lutego'})",
                "MERGE (node2:Course {title: 'Kryptografia', context: 'Kurs'})",
            ]

    monkeypatch.setattr(cypher_module, "LLMPipe", FakePipe)
    monkeypatch.setattr(cypher_module, "get_run_logger", MagicMock)

    result = cypher_module.generate_cypher_queries.fn("source text")
    statements = [part.strip() for part in result.split("|") if part.strip()]

    assert len(statements) == 2
    assert all(statement.startswith("MERGE (") for statement in statements)
    assert all(statement.endswith("END") for statement in statements)
