import pytest

from src.config.config import get_config
from src.data_pipeline.relationship_vocabulary import (
    RelationshipVocabulary,
    render_allowed_relationship_types,
)


def _vocabulary() -> RelationshipVocabulary:
    return RelationshipVocabulary(get_config().graph_schema)


@pytest.mark.parametrize(
    ("input_type", "expected"),
    [
        ("WORKS_IN", "EMPLOYED_BY"),
        ("works-at", "EMPLOYED_BY"),
        ("is_part_of", "PART_OF"),
        ("TOTALLY_UNKNOWN_REL", "RELATED_TO"),
    ],
)
def test_relationship_types_resolve_to_canonical_or_fallback(
    input_type: str, expected: str
) -> None:
    assert _vocabulary().canonical_relationship_type(input_type) == expected


@pytest.mark.parametrize(
    ("statement", "rewritten", "rewrites", "fallbacks"),
    [
        (
            "MERGE (a)-[:WORKS_IN]->(b)",
            "MERGE (a)-[:EMPLOYED_BY]->(b)",
            {"WORKS_IN": "EMPLOYED_BY"},
            set(),
        ),
        (
            "MERGE (a)-[:`WORKS-IN`]->(b)",
            "MERGE (a)-[:EMPLOYED_BY]->(b)",
            {"WORKS-IN": "EMPLOYED_BY"},
            set(),
        ),
        (
            "MERGE (a)-[:Requires]->(b)",
            "MERGE (a)-[:REQUIRES]->(b)",
            {"Requires": "REQUIRES"},
            set(),
        ),
        (
            "MERGE (a)-[:HAS_COMPETENCY]->(b)",
            "MERGE (a)-[:RELATED_TO]->(b)",
            {"HAS_COMPETENCY": "RELATED_TO"},
            {"HAS_COMPETENCY"},
        ),
    ],
)
def test_statement_relationship_types_are_rewritten_and_tracked(
    statement: str,
    rewritten: str,
    rewrites: dict[str, str],
    fallbacks: set[str],
) -> None:
    result = _vocabulary().canonicalize_statement(statement)
    assert result == (rewritten, rewrites, fallbacks)


def test_relationship_names_inside_string_values_are_not_rewritten() -> None:
    statement = "MERGE (n:Document {title: 'WORKS_IN relation', context: '[:NEEDS]'})"
    rewritten, rewrites, fallback_rewrites = _vocabulary().canonicalize_statement(statement)

    assert rewritten == statement
    assert rewrites == {}
    assert fallback_rewrites == set()


def test_prompt_relationship_types_exclude_fallback_and_list_aliases() -> None:
    schema = get_config().graph_schema

    rendered = render_allowed_relationship_types(schema)

    assert "RELATED_TO" not in rendered
    assert "WORKS_IN -> EMPLOYED_BY" in rendered
