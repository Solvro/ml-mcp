"""Collapse duplicate entities that were ingested before nodes had a canonical key.

Issue #53: merging on ``title + context`` and free label naming left the graph holding several
nodes for one entity. Fixing generation only helps future runs — everything already stored stays
split until it is repaired, so this pass runs after ingestion:

1. relabel nodes whose label is outside the configured vocabulary;
2. take the page's layout back out of stored titles, and recompute the keys that were derived
   from it (issue #79);
3. backfill the canonical key on nodes that predate it;
4. merge the nodes that end up sharing a label and a key;
5. fold a node that carries only the fallback label into the node under a real label that
   shares its key (issue #8).

Merging needs APOC. If the plugin is missing the pass reports what it found and changes nothing,
because a half-finished merge is worse than a duplicate.

Two modes. A pipeline run passes the keys it just wrote and only those groups are examined, so
the cost tracks what changed rather than how large the graph has grown. Relabelling, title
repair and key backfill all fix nodes written before the rules existed, which no later run can
reintroduce, so they belong to the full pass — run once with ``uv run dedup-graph``.
"""

import logging

from langchain_neo4j import Neo4jGraph
from prefect import get_run_logger, task
from prefect.cache_policies import NO_CACHE
from prefect.exceptions import MissingContextError

from src.config.config import get_config
from src.config.system_labels import SYSTEM_LABELS
from src.data_pipeline.canonical_nodes import CONTEXT_SEPARATOR, canonical_entity_key
from src.data_pipeline.label_vocabulary import LabelVocabulary
from src.data_pipeline.title_sanity import clean_title, title_rejection_reason

module_logger = logging.getLogger(__name__)

# Bookkeeping/provenance nodes the pipeline owns. They carry no answer payload and must never be
# relabelled or merged by key.
INTERNAL_LABELS = SYSTEM_LABELS

KEY_BACKFILL_BATCH_SIZE = 500
MAX_CONTEXT_LENGTH = 2000

# Shared tail of both merge queries: `nodes` is the group to collapse, first node surviving.
# The survivor keeps the fullest title and every distinct context, and inherits the
# relationships of the nodes it absorbs. One fragment, so the two passes cannot disagree about
# what "merged" means.
_MERGE_GROUP_CYPHER = """
WITH nodes,
     reduce(best = '', candidate IN [item IN nodes | coalesce(item.title, '')] |
            CASE WHEN size(candidate) > size(best) THEN candidate ELSE best END) AS best_title,
     reduce(kept = [], candidate IN [item IN nodes | coalesce(item.context, '')] |
            CASE WHEN candidate = '' OR candidate IN kept THEN kept ELSE kept + candidate END)
            AS contexts
CALL apoc.refactor.mergeNodes(
    nodes, {properties: 'discard', mergeRels: true, produceSelfRel: false})
YIELD node AS merged
SET merged.title = best_title,
    merged.context = substring(
        reduce(joined = '', part IN contexts |
               CASE WHEN joined = '' THEN part ELSE joined + $context_separator + part END),
        0, $max_context_length)
"""

# $keys is null for a full pass and a list for a run-scoped one, so one query serves both and
# the two modes cannot drift apart. With a list the key index carries the lookup.
MERGE_DUPLICATES_CYPHER = (
    """
MATCH (node)
WHERE node.key IS NOT NULL
  AND ($keys IS NULL OR node.key IN $keys)
  AND node.title IS NOT NULL
  AND NOT any(label IN labels(node) WHERE label IN $internal_labels)
WITH apoc.coll.sort(labels(node)) AS label_set, node.key AS entity_key, collect(node) AS nodes
WHERE size(nodes) > 1
"""
    + _MERGE_GROUP_CYPHER
    + """
RETURN count(merged) AS merged_groups
"""
)


def merge_fallback_cypher(fallback_label: str) -> str:
    """
    Build the query that folds fallback-labelled nodes into their properly labelled twin.

    A key claimed by two different real labels is ambiguous and is left alone: fusing two
    entities is worse than a duplicate, and nothing here can tell which one the fallback node
    meant. The label has to be spliced in - a MATCH pattern cannot take a parameter - and it
    is the configured value, not user input.

    Args:
        fallback_label: The label ingestion assigns when nothing in the vocabulary fits

    Returns:
        Cypher taking $keys, $fallback_label, $internal_labels, $context_separator and
        $max_context_length, returning merged_groups
    """
    return (
        f"""
MATCH (fallback:`{fallback_label}`)
WHERE size(labels(fallback)) = 1
  AND fallback.key IS NOT NULL
  AND ($keys IS NULL OR fallback.key IN $keys)
  AND fallback.title IS NOT NULL
MATCH (labelled)
WHERE labelled.key = fallback.key
  AND NOT $fallback_label IN labels(labelled)
  AND labelled.title IS NOT NULL
  AND NOT any(label IN labels(labelled) WHERE label IN $internal_labels)
WITH fallback, collect(DISTINCT labelled) AS targets
WHERE size(targets) = 1
WITH [targets[0], fallback] AS nodes
"""
        + _MERGE_GROUP_CYPHER
        + """
WITH merged
CALL apoc.create.removeLabels(merged, [$fallback_label]) YIELD node AS relabelled
RETURN count(relabelled) AS merged_groups
"""
    )


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


def repair_stored_titles(graph: Neo4jGraph) -> dict[str, int]:
    """
    Take the page's layout back out of titles that were stored carrying it.

    Issue #79: nodes were stored as "a) doswiadczenie w organizowaniu ..." and "Odbyte
    szkolenia:", and their key was derived from that spelling. The enumerated copy and the plain
    one therefore sit under two keys, and the merge below cannot see that they are one entity.

    Title and key are recomputed in Python by the same functions ingestion uses, so a repaired
    node and a freshly written one cannot disagree. Nodes that have no key yet are left to
    backfill_entity_keys, which runs next and now derives a clean key of its own.

    Titles that name nothing at all - a fragment such as "zagranicznych" - are reported and left
    in place. Ingestion refuses to write new ones; deleting what is already stored is a call for
    whoever knows what else points at it.

    Args:
        graph: Connected Neo4j graph

    Returns:
        Counts of the titles rewritten and the keys recomputed
    """
    logger = _get_logger()
    rows = graph.query(
        """
        MATCH (node)
        WHERE node.title IS NOT NULL
          AND NOT any(label IN labels(node) WHERE label IN $internal_labels)
        RETURN elementId(node) AS node_id, node.title AS title, node.key AS key
        """,
        params={"internal_labels": sorted(INTERNAL_LABELS)},
    )

    updates: list[dict[str, str | None]] = []
    titles_cleaned = 0
    keys_repaired = 0
    unusable: list[str] = []

    for row in rows:
        stored_title = row["title"] or ""
        cleaned = clean_title(stored_title)
        if not cleaned:
            continue

        reason = title_rejection_reason(cleaned)
        if reason is not None:
            unusable.append(f"{stored_title!r} ({reason})")

        stored_key = row["key"]
        repaired_key = canonical_entity_key(cleaned) if stored_key is not None else None
        title_changed = cleaned != stored_title
        key_changed = bool(repaired_key) and repaired_key != stored_key

        if not title_changed and not key_changed:
            continue

        titles_cleaned += int(title_changed)
        keys_repaired += int(key_changed)
        updates.append(
            {
                "node_id": row["node_id"],
                "title": cleaned,
                "key": repaired_key if key_changed else stored_key,
            }
        )

    for start in range(0, len(updates), KEY_BACKFILL_BATCH_SIZE):
        graph.query(
            """
            UNWIND $updates AS update
            MATCH (node) WHERE elementId(node) = update.node_id
            SET node.title = update.title, node.key = update.key
            """,
            params={"updates": updates[start : start + KEY_BACKFILL_BATCH_SIZE]},
        )

    logger.info(
        "Cleaned %d stored title(s) and recomputed %d key(s)", titles_cleaned, keys_repaired
    )

    if unusable:
        logger.warning(
            "%d stored title(s) name nothing and were left in place: %s",
            len(unusable),
            "; ".join(unusable[:10]),
        )

    return {"titles_cleaned": titles_cleaned, "keys_repaired": keys_repaired}


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


def delete_self_relationships(graph: Neo4jGraph) -> int:
    """
    Remove relationships that already point from a node back to itself.

    Issue #87: the model related a category to a sub-item whose titles canonicalise to the same
    key, so the MERGE collapsed both ends onto one node and the page wrote a loop. Ingestion now
    drops that statement before it runs and the merge pass refuses to create one, but neither
    reaches a loop already in the database - this is what clears those.

    A relationship from a node to itself carries nothing: whatever it was meant to say about two
    entities is lost the moment they turn out to be one, and no traversal can use it.

    Internal labels are excluded like everywhere else in this pass, so it never writes anywhere
    near the pipeline's own records.

    Args:
        graph: Connected Neo4j graph

    Returns:
        Number of self-relationships deleted
    """
    logger = _get_logger()
    rows = graph.query(
        """
        MATCH (node)-[rel]->(node)
        WHERE NOT any(label IN labels(node) WHERE label IN $internal_labels)
        WITH collect(rel) AS loops
        FOREACH (rel IN loops | DELETE rel)
        RETURN size(loops) AS deleted
        """,
        params={"internal_labels": sorted(INTERNAL_LABELS)},
    )

    deleted = int(rows[0]["deleted"]) if rows else 0
    logger.info("Deleted %d self-relationship(s)", deleted)
    return deleted


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


def merge_fallback_nodes(
    graph: Neo4jGraph, fallback_label: str, keys: list[str] | None = None
) -> int:
    """
    Fold nodes that carry only the fallback label into the real-labelled node sharing their key.

    Runs after merge_duplicate_nodes, so each real label holds at most one node per key by the
    time this looks; a key still claimed by two different real labels is ambiguous and is
    skipped rather than guessed at.

    Args:
        graph: Connected Neo4j graph
        fallback_label: The label ingestion assigns when nothing in the vocabulary fits
        keys: Canonical keys to examine, or None for the whole graph (see merge_duplicate_nodes)

    Returns:
        Number of fallback nodes folded away, or 0 when APOC is unavailable
    """
    logger = _get_logger()
    if keys is not None and not keys:
        return 0

    try:
        rows = graph.query(
            merge_fallback_cypher(fallback_label),
            params={
                "fallback_label": fallback_label,
                "internal_labels": sorted(INTERNAL_LABELS),
                "max_context_length": MAX_CONTEXT_LENGTH,
                "context_separator": CONTEXT_SEPARATOR,
                "keys": keys,
            },
        )
    except Exception as exc:
        logger.warning("Fallback-label merge skipped (APOC required): %s", exc)
        return 0

    merged = int(rows[0]["merged_groups"]) if rows else 0
    logger.info("Folded %d %s node(s) into their labelled twin", merged, fallback_label)
    return merged


@task(cache_policy=NO_CACHE)
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
            return {
                "relabelled_labels": 0,
                "titles_cleaned": 0,
                "keys_repaired": 0,
                "keys_backfilled": 0,
                "self_relationships_deleted": 0,
                "groups_merged": 0,
                "fallback_merged": 0,
            }
        graph = Neo4jGraph(url=uri, username=username, password=password)

    vocabulary = LabelVocabulary(get_config().graph_schema)

    if keys is not None:
        # The fallback fold belongs here too: page one can write the fallback copy and page
        # two the labelled one within a single run, and nothing later would revisit the pair.
        groups_merged = merge_duplicate_nodes(graph, keys)
        return {
            "relabelled_labels": 0,
            "titles_cleaned": 0,
            "keys_repaired": 0,
            "keys_backfilled": 0,
            # Ingestion drops a self-relationship before it is written and the merges refuse to
            # create one, so a run has none to find. Clearing what predates those rules is the
            # full walk's job, like the other legacy repairs.
            "self_relationships_deleted": 0,
            "groups_merged": groups_merged,
            "fallback_merged": merge_fallback_nodes(graph, vocabulary.fallback_label, keys),
        }

    relabelled = relabel_off_vocabulary_nodes(graph, vocabulary)
    title_repairs = repair_stored_titles(graph)
    keys_backfilled = backfill_entity_keys(graph)
    # Before the merges, not after: merging two nodes moves the relationships of the one it
    # absorbs, so a loop left in place would be carried onto the survivor and counted again.
    self_relationships_deleted = delete_self_relationships(graph)
    groups_merged = merge_duplicate_nodes(graph)
    fallback_merged = merge_fallback_nodes(graph, vocabulary.fallback_label)

    return {
        "relabelled_labels": len(relabelled),
        "titles_cleaned": title_repairs["titles_cleaned"],
        "keys_repaired": title_repairs["keys_repaired"],
        "keys_backfilled": keys_backfilled,
        "self_relationships_deleted": self_relationships_deleted,
        "groups_merged": groups_merged,
        "fallback_merged": fallback_merged,
    }
