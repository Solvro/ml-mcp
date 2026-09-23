from src.config.config import get_config
from src.data_pipeline.relationship_vocabulary import normalize_relationship_name

EXPECTED_NODE_LABEL_COUNT = 27


def _schema():
    return get_config().graph_schema


def test_node_labels_are_unique() -> None:
    labels = _schema().node_labels

    assert len(labels) == len(set(labels))


def test_node_label_set_stays_at_the_documented_size() -> None:
    """Growing the vocabulary is a deliberate decision, not a side effect of a page."""
    assert len(_schema().node_labels) == EXPECTED_NODE_LABEL_COUNT


def test_every_alias_resolves_to_a_configured_label() -> None:
    schema = _schema()
    labels = set(schema.node_labels)

    unresolved = [alias.invented for alias in schema.label_aliases if alias.canonical not in labels]

    assert unresolved == []


def test_no_alias_shadows_a_configured_label() -> None:
    """An alias for a label that is already canonical would rewrite valid output."""
    schema = _schema()
    labels = set(schema.node_labels)

    shadowing = [alias.invented for alias in schema.label_aliases if alias.invented in labels]

    assert shadowing == []


def test_alias_sources_are_unique() -> None:
    sources = [alias.invented for alias in _schema().label_aliases]

    assert len(sources) == len(set(sources))


def test_fallback_label_is_a_configured_label() -> None:
    schema = _schema()

    assert schema.fallback_label in schema.node_labels


def test_relationship_types_are_unique_upper_snake_case() -> None:
    relationship_types = _schema().relationship_types

    assert len(relationship_types) == len(set(relationship_types))
    assert all(name == name.upper() for name in relationship_types)
    assert all(name.replace("_", "").isalpha() for name in relationship_types)


def test_every_relationship_alias_resolves_to_a_configured_type() -> None:
    schema = _schema()
    relationship_types = set(schema.relationship_types)

    unresolved = [
        alias.invented
        for alias in schema.relationship_aliases
        if alias.canonical not in relationship_types
    ]

    assert unresolved == []


def test_relationship_aliases_do_not_shadow_canonical_types() -> None:
    schema = _schema()
    relationship_types = set(schema.relationship_types)

    shadowing = [
        alias.invented
        for alias in schema.relationship_aliases
        if alias.invented in relationship_types
    ]

    assert shadowing == []


def test_relationship_aliases_do_not_shadow_canonical_types_after_normalization() -> None:
    schema = _schema()
    normalized_canonical = {
        normalize_relationship_name(relationship_type)
        for relationship_type in schema.relationship_types
    }

    collisions = [
        alias.invented
        for alias in schema.relationship_aliases
        if normalize_relationship_name(alias.invented) in normalized_canonical
    ]

    assert collisions == []


def test_fallback_relationship_type_is_upper_snake_case() -> None:
    schema = _schema()
    fallback = schema.fallback_relationship_type

    assert fallback == fallback.upper()
    assert fallback.replace("_", "").isalpha()


def test_fallback_relationship_type_is_canonical() -> None:
    schema = _schema()
    assert schema.fallback_relationship_type in schema.relationship_types


def test_label_drift_reported_in_the_issue_is_covered() -> None:
    """The concrete duplicates from issue #53 must resolve to one canonical label each."""
    schema = _schema()
    aliases = {alias.invented: alias.canonical for alias in schema.label_aliases}

    assert aliases["Program"] == "StudyProgram"
    assert "StudyProgram" in schema.node_labels
    assert "Semester" in schema.node_labels
