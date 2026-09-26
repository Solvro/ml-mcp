from src.config.config import get_config
from src.data_pipeline.relationship_vocabulary import normalize_relationship_name
from src.text_normalization import normalize_search_text

EXPECTED_NODE_LABEL_COUNT = 27


def _schema():
    return get_config().graph_schema


def test_node_labels_are_unique_and_stable_in_size() -> None:
    """Growing the vocabulary is a deliberate decision, not a side effect of a page."""
    labels = _schema().node_labels

    assert len(labels) == len(set(labels))
    assert len(labels) == EXPECTED_NODE_LABEL_COUNT


def test_label_aliases_are_well_formed() -> None:
    """An alias must be unique, resolve to a configured label, and never rewrite a valid one."""
    aliases = _schema().label_aliases
    labels = set(_schema().node_labels)
    sources = [alias.invented for alias in aliases]

    assert len(sources) == len(set(sources))
    assert [alias.invented for alias in aliases if alias.canonical not in labels] == []
    assert [alias.invented for alias in aliases if alias.invented in labels] == []


def test_relationship_types_are_unique_upper_snake_case() -> None:
    relationship_types = _schema().relationship_types

    assert len(relationship_types) == len(set(relationship_types))
    assert all(name == name.upper() for name in relationship_types)
    assert all(name.replace("_", "").isalpha() for name in relationship_types)


def test_relationship_aliases_are_well_formed() -> None:
    """Same rules as labels, plus no alias may collide with a type once both are normalized."""
    aliases = _schema().relationship_aliases
    types = set(_schema().relationship_types)
    normalized_types = {normalize_relationship_name(name) for name in types}
    sources = [alias.invented for alias in aliases]

    assert len(sources) == len(set(sources))
    assert [alias.invented for alias in aliases if alias.canonical not in types] == []
    assert [alias.invented for alias in aliases if alias.invented in types] == []
    assert [
        alias.invented
        for alias in aliases
        if normalize_relationship_name(alias.invented) in normalized_types
    ] == []


def test_fallbacks_are_configured_entries() -> None:
    schema = _schema()

    assert schema.fallback_label in schema.node_labels
    assert schema.fallback_relationship_type in schema.relationship_types


def test_relationship_qualifier_rules_point_to_configured_relationship_types() -> None:
    schema = _schema()
    relationship_types = set(schema.relationship_types)

    assert schema.relationship_qualifier_rules
    for rule in schema.relationship_qualifier_rules:
        assert rule.relationship_type in relationship_types
        assert rule.token_stems
        assert rule.words
        assert all(stem == stem.lower() for stem in rule.token_stems)


def test_every_qualifier_word_is_matched_by_its_own_stems() -> None:
    for rule in _schema().relationship_qualifier_rules:
        stems = [normalize_search_text(stem) for stem in rule.token_stems]
        for word in rule.words:
            normalized_word = normalize_search_text(word)
            assert any(normalized_word.startswith(stem) for stem in stems), word


def test_category_item_pairs_point_to_configured_labels() -> None:
    schema = _schema()
    labels = set(schema.node_labels)

    assert schema.category_item_labels
    for pair in schema.category_item_labels:
        assert pair.category in labels
        assert pair.item in labels
        assert pair.category.endswith("Category")


def test_label_drift_reported_in_the_issue_is_covered() -> None:
    """The concrete duplicates from issue #53 must resolve to one canonical label each."""
    schema = _schema()
    aliases = {alias.invented: alias.canonical for alias in schema.label_aliases}

    assert aliases["Program"] == "StudyProgram"
    assert "StudyProgram" in schema.node_labels
    assert "Semester" in schema.node_labels
