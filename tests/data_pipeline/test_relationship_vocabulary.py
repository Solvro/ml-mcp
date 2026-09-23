from src.config.config import get_config
from src.data_pipeline.relationship_vocabulary import (
    RelationshipVocabulary,
    render_allowed_relationship_types,
)


def _vocabulary() -> RelationshipVocabulary:
    return RelationshipVocabulary(get_config().graph_schema)


def test_known_relationship_aliases_resolve_to_canonical_types() -> None:
    vocabulary = _vocabulary()

    assert vocabulary.canonical_relationship_type("WORKS_IN") == "EMPLOYED_BY"
    assert vocabulary.canonical_relationship_type("works-at") == "EMPLOYED_BY"
    assert vocabulary.canonical_relationship_type("is_part_of") == "PART_OF"


def test_unknown_relationship_type_uses_fallback() -> None:
    vocabulary = _vocabulary()

    assert vocabulary.canonical_relationship_type("TOTALLY_UNKNOWN_REL") == "RELATED_TO"


def test_statement_relationship_types_are_rewritten_and_reported() -> None:
    vocabulary = _vocabulary()
    statement = "MERGE (a)-[:WORKS_IN]->(b)"

    rewritten, rewrites, fallback_rewrites = vocabulary.canonicalize_statement(statement)

    assert rewritten == "MERGE (a)-[:EMPLOYED_BY]->(b)"
    assert rewrites == {"WORKS_IN": "EMPLOYED_BY"}
    assert fallback_rewrites == set()


def test_backticked_relationship_type_is_rewritten() -> None:
    vocabulary = _vocabulary()
    statement = "MERGE (a)-[:`WORKS-IN`]->(b)"

    rewritten, rewrites, fallback_rewrites = vocabulary.canonicalize_statement(statement)

    assert rewritten == "MERGE (a)-[:EMPLOYED_BY]->(b)"
    assert rewrites == {"WORKS-IN": "EMPLOYED_BY"}
    assert fallback_rewrites == set()


def test_mixed_case_canonical_relationship_is_normalized() -> None:
    vocabulary = _vocabulary()
    statement = "MERGE (a)-[:Requires]->(b)"

    rewritten, rewrites, fallback_rewrites = vocabulary.canonicalize_statement(statement)

    assert rewritten == "MERGE (a)-[:REQUIRES]->(b)"
    assert rewrites == {"Requires": "REQUIRES"}
    assert fallback_rewrites == set()


def test_unknown_relationships_are_tracked_as_fallback_rewrites() -> None:
    vocabulary = _vocabulary()
    statement = "MERGE (a)-[:HAS_COMPETENCY]->(b)"

    rewritten, rewrites, fallback_rewrites = vocabulary.canonicalize_statement(statement)

    assert rewritten == "MERGE (a)-[:RELATED_TO]->(b)"
    assert rewrites == {"HAS_COMPETENCY": "RELATED_TO"}
    assert fallback_rewrites == {"HAS_COMPETENCY"}


def test_relationship_names_inside_string_values_are_not_rewritten() -> None:
    vocabulary = _vocabulary()
    statement = "MERGE (n:Document {title: 'WORKS_IN relation', context: '[:NEEDS]'})"

    rewritten, rewrites, fallback_rewrites = vocabulary.canonicalize_statement(statement)

    assert rewritten == statement
    assert rewrites == {}
    assert fallback_rewrites == set()


def test_prompt_relationship_types_exclude_fallback_and_list_aliases() -> None:
    schema = get_config().graph_schema

    rendered = render_allowed_relationship_types(schema)

    assert "RELATED_TO" not in rendered
    assert "WORKS_IN -> EMPLOYED_BY" in rendered
