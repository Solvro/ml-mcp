"""The Cypher model is shown entities, not the pipeline's ingestion bookkeeping (issue #86).

The input of every test here is built by the same ``neo4j_graphrag.schema.format_schema`` the
Neo4j driver calls, so these pin the filter to the layout it will actually meet rather than to
a hand-written guess at it.
"""

import logging

import pytest
from neo4j_graphrag.schema import format_schema

from src.config.system_labels import SYSTEM_LABELS
from src.mcp_server.tools.knowledge_graph.schema_visibility import hide_system_labels


def _structured_schema() -> dict:
    """A graph holding two real entities and everything the pipeline records about them."""
    return {
        "node_props": {
            "Course": [
                {"property": "title", "type": "STRING", "values": ["Analiza matematyczna"]},
                {"property": "key", "type": "STRING", "values": ["analiza matematyczna"]},
            ],
            "Professor": [
                {"property": "title", "type": "STRING", "values": ["Jan Kowalski"]},
            ],
            "ProcessedDocument": [
                {"property": "hash", "type": "STRING", "values": ["3f2a"]},
                {"property": "status", "type": "STRING", "values": ["completed"]},
                {"property": "claimed_at", "type": "STRING", "values": ["2026-09-15"]},
            ],
            "Source": [
                {"property": "source_id", "type": "STRING", "values": ["file://a.pdf#page=1"]},
            ],
            "PipelineRun": [
                {"property": "run_at", "type": "STRING", "values": ["2026-09-15T03:00:00Z"]},
            ],
        },
        "rel_props": {
            "TEACHES": [{"property": "since", "type": "STRING", "values": ["2024"]}],
            "FROM_SOURCE": [],
        },
        "relationships": [
            {"start": "Professor", "type": "TEACHES", "end": "Course"},
            {"start": "Course", "type": "FROM_SOURCE", "end": "Source"},
            {"start": "Professor", "type": "FROM_SOURCE", "end": "Source"},
            {"start": "ProcessedDocument", "type": "FROM_SOURCE", "end": "Source"},
        ],
    }


@pytest.fixture(params=[True, False], ids=["enhanced", "compact"])
def rendered_schema(request) -> str:
    """The schema text in both layouts ``format_schema`` produces."""
    return format_schema(schema=_structured_schema(), is_enhanced=request.param)


@pytest.mark.parametrize("label", sorted(SYSTEM_LABELS))
def test_a_bookkeeping_label_is_not_described_to_the_model(rendered_schema: str, label: str):
    assert label in rendered_schema

    assert label not in hide_system_labels(rendered_schema)


def test_the_properties_of_a_bookkeeping_label_go_with_it(rendered_schema: str):
    visible = hide_system_labels(rendered_schema)

    assert "claimed_at" not in visible
    assert "source_id" not in visible
    assert "run_at" not in visible
    assert "status" not in visible


def test_the_entities_a_question_is_about_survive(rendered_schema: str):
    visible = hide_system_labels(rendered_schema)

    assert "Course" in visible
    assert "Professor" in visible
    assert "TEACHES" in visible
    assert "since" in visible
    assert "(:Professor)-[:TEACHES]->(:Course)" in visible


def test_a_relationship_type_left_with_no_pattern_stops_being_described(rendered_schema: str):
    # FROM_SOURCE only ever ends at :Source, so once those patterns go it can no longer be
    # traversed. Describing it would be an invitation to write a query that matches nothing.
    assert "FROM_SOURCE" in rendered_schema

    assert "FROM_SOURCE" not in hide_system_labels(rendered_schema)


def test_the_three_section_headers_are_kept(rendered_schema: str):
    visible = hide_system_labels(rendered_schema)

    assert visible.startswith("Node properties:")
    assert "Relationship properties:" in visible
    assert "The relationships:" in visible


def test_a_graph_of_only_bookkeeping_reads_as_empty():
    # What the stack looks like after a pipeline run that ingested no entities. Leaving the
    # bookkeeping visible would have RAG generate Cypher against it; an empty schema abstains.
    bookkeeping_only = format_schema(
        schema={
            "node_props": {
                "ProcessedDocument": [{"property": "hash", "type": "STRING", "values": []}]
            },
            "rel_props": {},
            "relationships": [],
        },
        is_enhanced=True,
    )

    assert (
        hide_system_labels(bookkeeping_only)
        .replace("Node properties:", "")
        .replace("Relationship properties:", "")
        .replace("The relationships:", "")
        .strip()
        == ""
    )


def test_an_empty_schema_is_passed_through():
    empty = format_schema(
        schema={"node_props": {}, "rel_props": {}, "relationships": []}, is_enhanced=True
    )

    assert hide_system_labels(empty) == empty
    assert hide_system_labels("") == ""


def test_an_unreadable_layout_is_left_alone():
    # A driver upgrade that changes the layout must not cost every question its schema. The
    # filter says so in the log and serves what it was given.
    not_our_layout = "Labels: Course, Professor\nEdges: TEACHES"

    assert hide_system_labels(not_our_layout) == not_our_layout


def test_a_label_with_a_space_survives_after_a_bookkeeping_one() -> None:
    """Review of PR #89: the compact layout writes labels unquoted, spaces and all.

    `Study Program {title: STRING}` did not look like an entry, so it was read as a
    continuation of the `Source` entry above it and went out with it. The Course/Professor
    fixture cannot reach this - both names are a single token.
    """
    compact = format_schema(
        schema={
            "node_props": {
                "Source": [{"property": "source_id", "type": "STRING", "values": ["a.pdf"]}],
                "Study Program": [
                    {"property": "title", "type": "STRING", "values": ["Informatyka"]}
                ],
            },
            "rel_props": {},
            "relationships": [],
        },
        is_enhanced=False,
    )

    visible = hide_system_labels(compact)

    assert "Study Program" in visible
    assert "Source" not in visible


def test_a_line_the_layout_does_not_account_for_ends_the_drop(caplog) -> None:
    # Nothing should reach this once the label pattern reads the whole name, which is why it
    # has to fail towards keeping the line rather than silently swallowing it.
    schema = "\n".join(
        [
            "Node properties:",
            "- **Source**",
            "  - `source_id`: STRING",
            "something the layout does not describe",
            "- **Course**",
            "  - `title`: STRING",
            "Relationship properties:",
            "The relationships:",
        ]
    )

    with caplog.at_level(logging.WARNING):
        visible = hide_system_labels(schema)

    assert "something the layout does not describe" in visible
    assert "Course" in visible
    assert "Source" not in visible
    assert "source_id" not in visible
    assert "keeping it" in caplog.text
