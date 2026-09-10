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
        fallback_merge_result: list[dict[str, Any]] | Exception | None = None,
    ) -> None:
        self.labels = labels or []
        self.unkeyed_nodes = unkeyed_nodes or []
        self.merge_result = merge_result
        self.fallback_merge_result = fallback_merge_result
        self.calls: list[tuple[str, dict[str, Any] | None]] = []

    def query(self, cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        self.calls.append((cypher, params))

        if "db.labels()" in cypher:
            return [{"label": label} for label in self.labels]
        if "node.key IS NULL" in cypher:
            return self.unkeyed_nodes
        if "apoc.create.removeLabels" in cypher:
            if isinstance(self.fallback_merge_result, Exception):
                raise self.fallback_merge_result
            return self.fallback_merge_result or [{"merged_groups": 0}]
        if "apoc.refactor.mergeNodes" in cypher:
            if isinstance(self.merge_result, Exception):
                raise self.merge_result
            return self.merge_result or [{"merged_groups": 0}]
        return []


def _same_label_merge_calls(graph: FakeGraph) -> list[tuple[str, dict[str, Any] | None]]:
    return [
        call
        for call in graph.calls
        if "apoc.refactor.mergeNodes" in call[0] and "removeLabels" not in call[0]
    ]


def _fallback_merge_calls(graph: FakeGraph) -> list[tuple[str, dict[str, Any] | None]]:
    return [call for call in graph.calls if "apoc.create.removeLabels" in call[0]]


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

    assert stats == {
        "relabelled_labels": 1,
        "keys_backfilled": 1,
        "groups_merged": 2,
        "fallback_merged": 0,
    }


# Review feedback on PR #58: the repair walked the whole graph on every run, so its cost grew
# with the database rather than with what changed.
def test_a_run_scoped_pass_only_examines_the_keys_it_wrote() -> None:
    graph = FakeGraph(merge_result=[{"merged_groups": 1}])

    stats = graph_dedup.deduplicate_graph.fn(graph, ["analiza matematyczna", "semestr zimowy"])

    assert stats == {
        "relabelled_labels": 0,
        "keys_backfilled": 0,
        "groups_merged": 1,
        "fallback_merged": 0,
    }
    merge_call = _same_label_merge_calls(graph)[0]
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

    assert stats == {
        "relabelled_labels": 1,
        "keys_backfilled": 1,
        "groups_merged": 2,
        "fallback_merged": 0,
    }
    merge_call = _same_label_merge_calls(graph)[0]
    assert merge_call[1]["keys"] is None
    assert _fallback_merge_calls(graph)[0][1]["keys"] is None


def test_a_fallback_node_is_folded_into_its_labelled_twin(vocabulary) -> None:
    graph = FakeGraph(fallback_merge_result=[{"merged_groups": 12}])

    assert graph_dedup.merge_fallback_nodes(graph, vocabulary.fallback_label) == 12

    (call,) = _fallback_merge_calls(graph)
    assert call[1]["fallback_label"] == "Topic"
    assert call[1]["keys"] is None
    assert call[1]["internal_labels"] == ["PipelineRun", "ProcessedDocument", "Source"]


def test_the_fallback_fold_targets_only_pure_fallback_nodes_and_one_real_label() -> None:
    cypher = graph_dedup.merge_fallback_cypher("Topic")

    assert "MATCH (fallback:`Topic`)" in cypher
    assert "size(labels(fallback)) = 1" in cypher
    assert "size(targets) = 1" in cypher
    assert "[targets[0], fallback] AS nodes" in cypher, "the labelled node's properties win"
    assert "apoc.create.removeLabels(merged, [$fallback_label])" in cypher, (
        "mergeNodes adds the absorbed node's labels to the survivor"
    )


def test_neither_merge_turns_a_relationship_between_the_pair_into_a_self_loop() -> None:
    for cypher in (graph_dedup.MERGE_DUPLICATES_CYPHER, graph_dedup.merge_fallback_cypher("Topic")):
        assert "produceSelfRel: false" in cypher
        assert "mergeRels: true" in cypher, "the absorbed node's other relationships must move"


def test_the_fallback_fold_runs_after_the_same_label_merge(vocabulary) -> None:
    graph = FakeGraph(
        merge_result=[{"merged_groups": 1}], fallback_merge_result=[{"merged_groups": 2}]
    )

    stats = graph_dedup.deduplicate_graph.fn(graph)

    assert stats["groups_merged"] == 1
    assert stats["fallback_merged"] == 2
    order = [
        "fallback" if "removeLabels" in call[0] else "same_label"
        for call in graph.calls
        if "apoc.refactor.mergeNodes" in call[0]
    ]
    assert order == ["same_label", "fallback"]


def test_a_run_scoped_pass_also_folds_fallback_nodes_for_its_keys() -> None:
    """Page one can write the Topic copy and page two the labelled one in a single run."""
    graph = FakeGraph(fallback_merge_result=[{"merged_groups": 1}])

    stats = graph_dedup.deduplicate_graph.fn(graph, ["opieka naukowa"])

    assert stats["fallback_merged"] == 1
    (call,) = _fallback_merge_calls(graph)
    assert call[1]["keys"] == ["opieka naukowa"]


def test_missing_apoc_leaves_fallback_nodes_untouched(vocabulary) -> None:
    graph = FakeGraph(fallback_merge_result=RuntimeError("no procedure apoc.create.removeLabels"))

    assert graph_dedup.merge_fallback_nodes(graph, vocabulary.fallback_label) == 0
