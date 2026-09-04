"""The post-ingest repair for entities that were already split across several nodes.

Fixing generation only helps future runs; everything issue #53 describes is already stored.
These tests cover the repair pass with a fake graph, so they run without Neo4j or APOC.
"""

from typing import Any

import pytest

from src.config.config import get_config
from src.data_pipeline.flows import graph_dedup
from src.data_pipeline.label_vocabulary import LabelVocabulary


class FakeGraph:
    """Records every statement and answers the two queries the pass reads back."""

    def __init__(
        self,
        labels: list[str] | None = None,
        unkeyed_nodes: list[dict[str, Any]] | None = None,
        merge_result: list[dict[str, Any]] | Exception | None = None,
    ) -> None:
        self.labels = labels or []
        self.unkeyed_nodes = unkeyed_nodes or []
        self.merge_result = merge_result
        self.calls: list[tuple[str, dict[str, Any] | None]] = []

    def query(self, cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        self.calls.append((cypher, params))

        if "db.labels()" in cypher:
            return [{"label": label} for label in self.labels]
        if "node.key IS NULL" in cypher:
            return self.unkeyed_nodes
        if "apoc.refactor.mergeNodes" in cypher:
            if isinstance(self.merge_result, Exception):
                raise self.merge_result
            return self.merge_result or [{"merged_groups": 0}]
        return []


@pytest.fixture
def vocabulary() -> LabelVocabulary:
    return LabelVocabulary(get_config().graph_schema)


def test_off_vocabulary_labels_are_moved_to_their_canonical_label(vocabulary) -> None:
    graph = FakeGraph(labels=["Program", "Holiday", "Course"])

    rewrites = graph_dedup.relabel_off_vocabulary_nodes(graph, vocabulary)

    assert rewrites == {"Program": "StudyProgram", "Holiday": "DayOff"}
    relabel_statements = [call[0] for call in graph.calls if "REMOVE" in call[0]]
    assert any("REMOVE node:`Program` SET node:`StudyProgram`" in s for s in relabel_statements)
    assert any("REMOVE node:`Holiday` SET node:`DayOff`" in s for s in relabel_statements)


def test_configured_labels_are_left_alone(vocabulary) -> None:
    graph = FakeGraph(labels=["Course", "Semester", "DayOff"])

    assert graph_dedup.relabel_off_vocabulary_nodes(graph, vocabulary) == {}
    assert not [call for call in graph.calls if "REMOVE" in call[0]]


def test_bookkeeping_labels_are_never_relabelled(vocabulary) -> None:
    """ProcessedDocument drives idempotency; renaming it would replay every page."""
    graph = FakeGraph(labels=["ProcessedDocument", "PipelineRun", "Source"])

    assert graph_dedup.relabel_off_vocabulary_nodes(graph, vocabulary) == {}


def test_keys_are_backfilled_from_titles() -> None:
    graph = FakeGraph(
        unkeyed_nodes=[
            {"node_id": "4:a:1", "title": "Cyberbezpieczeństwo (CBE)"},
            {"node_id": "4:a:2", "title": "Semestr zimowy 2026/2027"},
        ]
    )

    assert graph_dedup.backfill_entity_keys(graph) == 2

    update_call = next(call for call in graph.calls if "SET node.key" in call[0])
    assert update_call[1]["updates"] == [
        {"node_id": "4:a:1", "key": "cyberbezpieczenstwo"},
        {"node_id": "4:a:2", "key": "semestr zimowy 2026 2027"},
    ]


def test_titles_with_no_usable_characters_are_skipped() -> None:
    graph = FakeGraph(unkeyed_nodes=[{"node_id": "4:a:1", "title": "---"}])

    assert graph_dedup.backfill_entity_keys(graph) == 0


def test_key_backfill_is_batched() -> None:
    graph = FakeGraph(
        unkeyed_nodes=[
            {"node_id": f"4:a:{index}", "title": f"Kurs {index}"}
            for index in range(graph_dedup.KEY_BACKFILL_BATCH_SIZE + 1)
        ]
    )

    graph_dedup.backfill_entity_keys(graph)

    update_calls = [call for call in graph.calls if "SET node.key" in call[0]]
    assert len(update_calls) == 2
    assert len(update_calls[0][1]["updates"]) == graph_dedup.KEY_BACKFILL_BATCH_SIZE
    assert len(update_calls[1][1]["updates"]) == 1


def test_duplicate_merge_reports_the_number_of_groups() -> None:
    graph = FakeGraph(merge_result=[{"merged_groups": 3}])

    assert graph_dedup.merge_duplicate_nodes(graph) == 3


def test_duplicate_merge_excludes_bookkeeping_nodes() -> None:
    graph = FakeGraph(merge_result=[{"merged_groups": 0}])

    graph_dedup.merge_duplicate_nodes(graph)

    merge_call = next(call for call in graph.calls if "apoc.refactor.mergeNodes" in call[0])
    assert merge_call[1]["internal_labels"] == ["PipelineRun", "ProcessedDocument", "Source"]


def test_missing_apoc_leaves_the_graph_untouched() -> None:
    """A half-finished merge is worse than a duplicate, so the pass gives up cleanly."""
    graph = FakeGraph(merge_result=RuntimeError("no procedure apoc.refactor.mergeNodes"))

    assert graph_dedup.merge_duplicate_nodes(graph) == 0


def test_deduplicate_graph_reports_every_stage() -> None:
    graph = FakeGraph(
        labels=["Program"],
        unkeyed_nodes=[{"node_id": "4:a:1", "title": "Kryptografia"}],
        merge_result=[{"merged_groups": 2}],
    )

    stats = graph_dedup.deduplicate_graph.fn(graph)

    assert stats == {"relabelled_labels": 1, "keys_backfilled": 1, "groups_merged": 2}


# Review feedback on PR #58: the repair walked the whole graph on every run, so its cost grew
# with the database rather than with what changed.
def test_a_run_scoped_pass_only_examines_the_keys_it_wrote() -> None:
    graph = FakeGraph(merge_result=[{"merged_groups": 1}])

    stats = graph_dedup.deduplicate_graph.fn(graph, ["analiza matematyczna", "semestr zimowy"])

    assert stats == {"relabelled_labels": 0, "keys_backfilled": 0, "groups_merged": 1}
    merge_call = next(call for call in graph.calls if "apoc.refactor.mergeNodes" in call[0])
    assert merge_call[1]["keys"] == ["analiza matematyczna", "semestr zimowy"]


def test_a_run_scoped_pass_skips_the_legacy_full_graph_repairs() -> None:
    """Relabelling and key backfill only ever apply to nodes written before the rules existed."""
    graph = FakeGraph(labels=["Program"], unkeyed_nodes=[{"node_id": "4:a:1", "title": "Kurs"}])

    graph_dedup.deduplicate_graph.fn(graph, ["kurs"])

    assert not [call for call in graph.calls if "db.labels()" in call[0]]
    assert not [call for call in graph.calls if "node.key IS NULL" in call[0]]


def test_a_run_that_wrote_nothing_does_not_touch_the_graph() -> None:
    graph = FakeGraph(merge_result=[{"merged_groups": 5}])

    stats = graph_dedup.deduplicate_graph.fn(graph, [])

    assert stats["groups_merged"] == 0
    assert graph.calls == []


def test_the_full_pass_still_walks_everything() -> None:
    graph = FakeGraph(
        labels=["Program"],
        unkeyed_nodes=[{"node_id": "4:a:1", "title": "Kryptografia"}],
        merge_result=[{"merged_groups": 2}],
    )

    stats = graph_dedup.deduplicate_graph.fn(graph)

    assert stats == {"relabelled_labels": 1, "keys_backfilled": 1, "groups_merged": 2}
    merge_call = next(call for call in graph.calls if "apoc.refactor.mergeNodes" in call[0])
    assert merge_call[1]["keys"] is None
