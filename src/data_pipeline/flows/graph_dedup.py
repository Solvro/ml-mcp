"""Collapse duplicate entities that were ingested before nodes had a canonical key.

Issue #53: merging on ``title + context`` and free label naming left the graph holding several
nodes for one entity. Fixing generation only helps future runs — everything already stored stays
split until it is repaired, so this pass runs after ingestion:

1. relabel nodes whose label is outside the configured vocabulary;
2. backfill the canonical key on nodes that predate it;
3. merge the nodes that end up sharing a label and a key.

Merging needs APOC. If the plugin is missing the pass reports what it found and changes nothing,
because a half-finished merge is worse than a duplicate.

Two modes. A pipeline run passes the keys it just wrote and only those groups are examined, so
the cost tracks what changed rather than how large the graph has grown. Relabelling and key
backfill repair nodes written before the rules existed, which no later run can reintroduce, so
they belong to the full pass — run once with ``uv run dedup-graph``.
"""

import logging

from langchain_neo4j import Neo4jGraph
from prefect import get_run_logger, task
from prefect.exceptions import MissingContextError

from src.config.config import get_config
from src.data_pipeline.canonical_nodes import CONTEXT_SEPARATOR, canonical_entity_key
from src.data_pipeline.label_vocabulary import LabelVocabulary

module_logger = logging.getLogger(__name__)

# Bookkeeping nodes the pipeline owns. They carry no title and must never be merged by key.
INTERNAL_LABELS = frozenset({"ProcessedDocument", "PipelineRun"})

KEY_BACKFILL_BATCH_SIZE = 500
MAX_CONTEXT_LENGTH = 2000

# $keys is null for a full pass and a list for a run-scoped one, so one query serves both and
# the two modes cannot drift apart. With a list the key index carries the lookup.
MERGE_DUPLICATES_CYPHER = """
MATCH (node)
WHERE node.key IS NOT NULL
  AND ($keys IS NULL OR node.key IN $keys)
  AND node.title IS NOT NULL
  AND NOT any(label IN labels(node) WHERE label IN $internal_labels)
WITH apoc.coll.sort(labels(node)) AS label_set, node.key AS entity_key, collect(node) AS nodes
WHERE size(nodes) > 1
WITH nodes,
     reduce(best = '', candidate IN [item IN nodes | coalesce(item.title, '')] |
            CASE WHEN size(candidate) > size(best) THEN candidate ELSE best END) AS best_title,
     reduce(kept = [], candidate IN [item IN nodes | coalesce(item.context, '')] |
            CASE WHEN candidate = '' OR candidate IN kept THEN kept ELSE kept + candidate END)
            AS contexts
CALL apoc.refactor.mergeNodes(nodes, {properties: 'discard', mergeRels: true})
YIELD node AS merged
SET merged.title = best_title,
    merged.context = substring(
        reduce(joined = '', part IN contexts |
               CASE WHEN joined = '' THEN part ELSE joined + $context_separator + part END),
        0, $max_context_length)
RETURN count(merged) AS merged_groups
"""


def _get_logger() -> logging.Logger:
    """Return Prefect run logger when available, otherwise the module logger."""
    try:
        return get_run_logger()
    except MissingContextError:
        return module_logger


def relabel_off_vocabulary_nodes(graph: Neo4jGraph, vocabulary: LabelVocabulary) -> dict[str, str]:
    """
    Move nodes stored under an off-vocabulary label onto their canonical label.

    Args:
        graph: Connected Neo4j graph
        vocabulary: Configured label vocabulary

    Returns:
        Map of the labels that were rewritten, from the stored label to the canonical one
    """
    logger = _get_logger()
    stored_labels = [
        row["label"] for row in graph.query("CALL db.labels() YIELD label RETURN label")
    ]

    rewrites: dict[str, str] = {}
    for stored_label in stored_labels:
        if stored_label in INTERNAL_LABELS or stored_label in vocabulary.node_labels:
            continue

        canonical = vocabulary.canonical_label(stored_label)
        graph.query(
            f"MATCH (node:`{stored_label}`) REMOVE node:`{stored_label}` SET node:`{canonical}`"
        )
        rewrites[stored_label] = canonical
        logger.info("Relabelled stored nodes %s -> %s", stored_label, canonical)

    return rewrites


def backfill_entity_keys(graph: Neo4jGraph) -> int:
    """
    Give pre-existing titled nodes the canonical key that new nodes are merged on.

    Keys are computed in Python rather than in Cypher so a backfilled node and a freshly
    extracted one can never disagree about what the key of a title is.

    Args:
        graph: Connected Neo4j graph

    Returns:
        Number of nodes given a key
    """
    logger = _get_logger()
    rows = graph.query(
        """
        MATCH (node)
        WHERE node.title IS NOT NULL
          AND node.key IS NULL
          AND NOT any(label IN labels(node) WHERE label IN $internal_labels)
        RETURN elementId(node) AS node_id, node.title AS title
        """,
        params={"internal_labels": sorted(INTERNAL_LABELS)},
    )

    updates = [
        {"node_id": row["node_id"], "key": canonical_entity_key(row["title"] or "")} for row in rows
    ]
    updates = [update for update in updates if update["key"]]

    for start in range(0, len(updates), KEY_BACKFILL_BATCH_SIZE):
        graph.query(
            """
            UNWIND $updates AS update
            MATCH (node) WHERE elementId(node) = update.node_id
            SET node.key = update.key
            """,
            params={"updates": updates[start : start + KEY_BACKFILL_BATCH_SIZE]},
        )

    logger.info("Backfilled canonical keys on %d nodes", len(updates))
    return len(updates)


def merge_duplicate_nodes(graph: Neo4jGraph, keys: list[str] | None = None) -> int:
    """
    Merge nodes that share a label set and a canonical key into one.

    The surviving node keeps the fullest title and every distinct context, and inherits the
    relationships of the nodes it absorbs.

    Args:
        graph: Connected Neo4j graph
        keys: Canonical keys to examine. None looks at the whole graph; a pipeline run passes
            the keys it wrote, so the work tracks what changed rather than the graph size.

    Returns:
        Number of duplicate groups merged, or 0 when APOC is unavailable
    """
    logger = _get_logger()
    if keys is not None and not keys:
        return 0

    try:
        rows = graph.query(
            MERGE_DUPLICATES_CYPHER,
            params={
                "internal_labels": sorted(INTERNAL_LABELS),
                "max_context_length": MAX_CONTEXT_LENGTH,
                "context_separator": CONTEXT_SEPARATOR,
                "keys": keys,
            },
        )
    except Exception as exc:
        logger.warning("Duplicate merge skipped (APOC required): %s", exc)
        return 0

    merged = int(rows[0]["merged_groups"]) if rows else 0
    logger.info("Merged %d duplicate node group(s)", merged)
    return merged


@task
def deduplicate_graph(
    graph: Neo4jGraph | None = None, keys: list[str] | None = None
) -> dict[str, int]:
    """
    Repair entities split across several nodes.

    Args:
        graph: Connected Neo4j graph; built from the environment when omitted
        keys: Canonical keys a run just wrote. Passing them keeps the pass proportional to what
            changed and skips the legacy repairs, which only ever apply to nodes written before
            the rules existed. None runs the full repair over the whole graph.

    Returns:
        Counts for each stage, for the pipeline summary log
    """
    logger = _get_logger()

    if graph is None:
        import os

        uri = os.getenv("NEO4J_URI")
        username = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER")
        password = os.getenv("NEO4J_PASSWORD")
        if not uri or not username or not password:
            logger.warning("Neo4j credentials not set - skipping deduplication")
            return {"relabelled_labels": 0, "keys_backfilled": 0, "groups_merged": 0}
        graph = Neo4jGraph(url=uri, username=username, password=password)

    if keys is not None:
        return {
            "relabelled_labels": 0,
            "keys_backfilled": 0,
            "groups_merged": merge_duplicate_nodes(graph, keys),
        }

    vocabulary = LabelVocabulary(get_config().graph_schema)

    relabelled = relabel_off_vocabulary_nodes(graph, vocabulary)
    keys_backfilled = backfill_entity_keys(graph)
    groups_merged = merge_duplicate_nodes(graph)

    return {
        "relabelled_labels": len(relabelled),
        "keys_backfilled": keys_backfilled,
        "groups_merged": groups_merged,
    }
