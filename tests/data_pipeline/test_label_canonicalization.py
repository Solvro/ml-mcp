"""Node labels are forced into the configured vocabulary before anything reaches Neo4j.

Issue #53 reported one concept arriving under two labels (StudyProgram and Program), which
splits it into two nodes that no single query can find. The prompt asks for the closed set;
these tests cover the rewrite that guarantees it.
"""

from unittest.mock import MagicMock

import pytest

from src.config.config import get_config
from src.data_pipeline.flows import llm_cypher_generation as cypher_module
from src.data_pipeline.label_vocabulary import LabelVocabulary


@pytest.fixture
def vocabulary() -> LabelVocabulary:
    return LabelVocabulary(get_config().graph_schema)


@pytest.mark.parametrize(
    ("written", "expected"),
    [
        ("StudyProgram", "StudyProgram"),
        ("Program", "StudyProgram"),
        ("Programme", "StudyProgram"),
        ("CriterionItem", "Criterion"),
        ("CriterionGroup", "CriterionCategory"),
        ("CompetencyCriterionGroup", "CompetencyCategory"),
        ("Holiday", "DayOff"),
        ("Event", "CalendarEvent"),
    ],
)
def test_known_drift_resolves_to_one_canonical_label(vocabulary, written, expected) -> None:
    assert vocabulary.canonical_label(written) == expected


@pytest.mark.parametrize("written", ["studyprogram", "STUDYPROGRAM", "studyProgram"])
def test_label_matching_ignores_case(vocabulary, written) -> None:
    assert vocabulary.canonical_label(written) == "StudyProgram"


def test_label_matching_ignores_polish_diacritics(vocabulary) -> None:
    assert vocabulary.canonical_label("Wydział") == "Faculty"


def test_unknown_label_becomes_the_fallback(vocabulary) -> None:
    """An invented label must still produce a node, just not a new label."""
    assert vocabulary.canonical_label("SomethingTheModelMadeUp") == "Topic"


def test_empty_label_becomes_the_fallback(vocabulary) -> None:
    assert vocabulary.canonical_label("  ") == "Topic"


def test_statement_labels_are_rewritten_and_reported(vocabulary) -> None:
    statement = "MERGE (node1:Program {title: 'Cyberbezpieczenstwo', context: 'Kierunek'})"

    rewritten, rewrites = vocabulary.canonicalize_statement(statement)

    assert rewritten == (
        "MERGE (node1:StudyProgram {title: 'Cyberbezpieczenstwo', context: 'Kierunek'})"
    )
    assert rewrites == {"Program": "StudyProgram"}


def test_canonical_statement_is_left_untouched(vocabulary) -> None:
    statement = "MERGE (node1:Course {title: 'Analiza matematyczna', context: '8 ECTS'})"

    rewritten, rewrites = vocabulary.canonicalize_statement(statement)

    assert rewritten == statement
    assert rewrites == {}


def test_relationship_types_are_not_rewritten(vocabulary) -> None:
    """Only node labels are a closed set; a relationship type must survive verbatim."""
    statement = "MERGE (node1)-[:HAS_DAY_OFF]->(node2)"

    rewritten, rewrites = vocabulary.canonicalize_statement(statement)

    assert rewritten == statement
    assert rewrites == {}


def test_relationship_type_matching_a_label_name_is_not_rewritten(vocabulary) -> None:
    statement = "MERGE (node1:Semester)-[:Program]->(node2:Course)"

    rewritten, _ = vocabulary.canonicalize_statement(statement)

    assert "[:Program]" in rewritten


def test_labels_inside_string_values_are_not_rewritten(vocabulary) -> None:
    """A quoted value is data; rewriting inside it would corrupt the extracted text."""
    statement = "MERGE (node1:Document {title: 'Program studiow', context: 'Opis (:Program)'})"

    rewritten, _ = vocabulary.canonicalize_statement(statement)

    assert "'Program studiow'" in rewritten
    assert "'Opis (:Program)'" in rewritten
    assert rewritten.startswith("MERGE (node1:Document")


def test_multi_label_nodes_are_each_resolved(vocabulary) -> None:
    statement = "MERGE (node1:Program:Holiday {title: 'X'})"

    rewritten, _ = vocabulary.canonicalize_statement(statement)

    assert ":StudyProgram:DayOff" in rewritten


def test_generation_task_applies_the_vocabulary(monkeypatch) -> None:
    """End of the generation task: what leaves it is already on-vocabulary."""

    class FakePipe:
        def run(self, context: str, schema_context: str = "") -> list[str]:
            return [
                "MERGE (node1:Program {title: 'Cyberbezpieczeństwo', context: 'Kierunek'})",
                "MERGE (node2:Holiday {title: '2 XI 2026', context: 'dzień wolny od zajęć'})",
                "MERGE (node1)-[:HAS_DAY_OFF]->(node2)",
            ]

    monkeypatch.setattr(cypher_module, "LLMPipe", FakePipe)
    monkeypatch.setattr(cypher_module, "get_run_logger", MagicMock)

    result = cypher_module.generate_cypher_queries.fn("source text")

    assert ":StudyProgram" in result
    assert ":DayOff" in result
    assert ":Program {" not in result
    assert ":Holiday {" not in result
    assert "[:HAS_DAY_OFF]" in result
    # The existing diacritic folding still applies to values.
    assert "dzien wolny od zajec" in result
