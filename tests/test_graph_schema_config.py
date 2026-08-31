"""The ingestion label set is a closed vocabulary; these tests keep it internally consistent.

Issue #53: the extraction model invented a label per page, so one concept landed as several
nodes (StudyProgram and Program for the same programme). The fix only holds if the configured
vocabulary itself is coherent — an alias pointing at a label that does not exist would silently
reintroduce an off-list label.
"""

from src.config.config import get_config

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


def test_label_drift_reported_in_the_issue_is_covered() -> None:
    """The concrete duplicates from issue #53 must resolve to one canonical label each."""
    schema = _schema()
    aliases = {alias.invented: alias.canonical for alias in schema.label_aliases}

    assert aliases["Program"] == "StudyProgram"
    assert "StudyProgram" in schema.node_labels
    assert "Semester" in schema.node_labels
