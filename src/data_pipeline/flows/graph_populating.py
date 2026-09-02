import json
import logging
import os
import re
import uuid

from langchain_neo4j import Neo4jGraph
from prefect import get_run_logger, task
from prefect.exceptions import MissingContextError

from src.config.config import get_config
from src.data_pipeline.canonical_nodes import extract_entity_keys

module_logger = logging.getLogger(__name__)


def _get_logger() -> logging.Logger:
    """Return Prefect run logger when available, otherwise the module logger."""
    try:
        return get_run_logger()
    except MissingContextError:
        return module_logger


def _get_claim_stale_minutes() -> int:
    """Read stale claim timeout in minutes from env with safe defaults."""
    raw_value = os.getenv("DATA_PIPELINE_CLAIM_STALE_MINUTES", "30").strip()
    try:
        parsed = int(raw_value)
    except ValueError:
        return 30
    return max(1, parsed)


_NODE_MERGE_VAR_RE = re.compile(
    r"^\s*MERGE\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*[A-Za-z_][A-Za-z0-9_]*"
)


def _extract_merged_node_vars(statements: list[str]) -> list[str]:
    """Extract unique node variables from MERGE (var:Label ...) clauses.

    Only the ``MERGE (var:Label ...)`` form the prompt mandates is recognised,
    and only the first variable of a clause. A clause that merges node and
    relationship at once (``MERGE (a:A)-[:R]->(b:B)``) therefore contributes
    ``a`` but not ``b``.
    """
    variables: list[str] = []
    seen: set[str] = set()

    for statement in statements:
        match = _NODE_MERGE_VAR_RE.match(statement)
        if not match:
            continue
        var_name = match.group(1)
        if var_name in seen:
            continue
        seen.add(var_name)
        variables.append(var_name)

    return variables


def _build_query_with_provenance(
    statements: list[str],
    source_id: str,
    logger: logging.Logger,
) -> tuple[str, dict[str, str]]:
    """Append FROM_SOURCE wiring when MERGE node variables are recoverable."""
    combined = "\n".join(statements)
    if not combined or not source_id:
        return combined, {}

    node_vars = _extract_merged_node_vars(statements)
    if not node_vars:
        logger.warning(
            "No MERGE node variables recovered for source_id=%s; "
            "executing query without provenance",
            source_id,
        )
        return combined, {}

    source_var = "prov_source"
    while source_var in node_vars:
        source_var += "_"

    with_clause = ", ".join(node_vars)
    provenance_lines = [
        f"WITH {with_clause}",
        f"MERGE ({source_var}:Source {{source_id: $source_id}})",
        *[f"MERGE ({var})-[:FROM_SOURCE]->({source_var})" for var in node_vars],
    ]
    return f"{combined}\n" + "\n".join(provenance_lines), {"source_id": source_id}


class GraphPopulator:
    def __init__(self):
        uri = os.getenv("NEO4J_URI")
        username = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER")
        password = os.getenv("NEO4J_PASSWORD")

        if not uri or not username or not password:
            raise ValueError("NEO4J connection settings are required")

        logger = _get_logger()
        logger.info(f"Connecting to Neo4j at {uri} as {username}")
        self.graph_db = Neo4jGraph(url=uri, username=username, password=password)

    def claim_document_hash(self, doc_hash: str) -> bool:
        """Atomically claim a document hash for processing.

        Returns True when processing is allowed for this worker:
        - first claim for a new hash
        - retry after a failed run
        - reclaim stale in-progress work (worker crash/timeout)

        Returns False for hashes already processed or currently claimed.
        """
        if not doc_hash:
            return False

        claim_token = str(uuid.uuid4())
        stale_minutes = _get_claim_stale_minutes()

        result = self.graph_db.query(
            """
            MERGE (doc:ProcessedDocument {hash: $doc_hash})
            ON CREATE SET
                doc.created_at = datetime(),
                doc.updated_at = datetime(),
                doc.claimed_at = datetime(),
                doc.status = 'processing',
                doc.claim_token = $claim_token,
                doc.attempt_count = 1
            WITH doc, doc.claim_token = $claim_token AS created_now, datetime() AS now
            WITH
                doc,
                created_now,
                now,
                CASE
                    WHEN created_now THEN true
                    WHEN doc.status = 'failed' THEN true
                    WHEN doc.status = 'processing'
                         AND coalesce(doc.claimed_at, datetime({epochMillis: 0}))
                             < now - duration({minutes: $stale_minutes})
                        THEN true
                    ELSE false
                END AS can_claim
            FOREACH (_ IN CASE WHEN can_claim AND NOT created_now THEN [1] ELSE [] END |
                SET doc.status = 'processing',
                    doc.claim_token = $claim_token,
                    doc.claimed_at = now,
                    doc.updated_at = now,
                    doc.attempt_count = coalesce(doc.attempt_count, 0) + 1
            )
            RETURN can_claim AS claimed
            """,
            params={
                "doc_hash": doc_hash,
                "claim_token": claim_token,
                "stale_minutes": stale_minutes,
            },
        )

        if not result:
            return False

        return bool(result[0].get("claimed"))

    def delete_sources_for_documents(self, document_source_ids: list[str]) -> set[str]:
        """Delete Source nodes for given document ids and clean now-orphaned nodes."""
        logger = _get_logger()
        deleted: set[str] = set()

        for document_source_id in sorted({sid for sid in document_source_ids if sid}):
            try:
                rows = self.graph_db.query(
                    """
                    MATCH (s:Source)
                    WHERE s.source_id = $doc_source_id
                       OR s.source_id STARTS WITH $doc_prefix
                    OPTIONAL MATCH (n)-[:FROM_SOURCE]->(s)
                    WITH collect(DISTINCT s) AS sources, collect(DISTINCT n) AS touched
                    FOREACH (s IN sources | DETACH DELETE s)
                    WITH size(sources) AS sources_deleted,
                         [n IN touched
                          WHERE n IS NOT NULL
                            AND NOT EXISTS { MATCH (n)-[:FROM_SOURCE]->(:Source) }]
                         AS orphans
                    FOREACH (n IN orphans | DETACH DELETE n)
                    RETURN sources_deleted, size(orphans) AS orphans_removed
                    """,
                    params={
                        "doc_source_id": document_source_id,
                        "doc_prefix": f"{document_source_id}#",
                    },
                )
            except Exception as exc:
                logger.error("Deletion failed for %s: %s", document_source_id, exc)
                continue

            row = rows[0] if rows else {}
            sources_deleted = row.get("sources_deleted", 0)
            orphans_removed = row.get("orphans_removed", 0)

            if sources_deleted:
                logger.info(
                    "Deleted %s: %s sources, %s orphaned nodes",
                    document_source_id,
                    sources_deleted,
                    orphans_removed,
                )
            else:
                logger.warning(
                    "Deleted %s but matched no Source nodes; provenance is missing, "
                    "relabelled, or was never written",
                    document_source_id,
                )
            deleted.add(document_source_id)

        return deleted

    def mark_document_processed(self, doc_hash: str) -> None:
        """Mark a previously claimed document hash as processed."""
        if not doc_hash:
            return

        self.graph_db.query(
            """
            MATCH (doc:ProcessedDocument {hash: $doc_hash})
            SET doc.status = 'processed',
                doc.processed_at = datetime(),
                doc.updated_at = datetime()
            REMOVE doc.claim_token
            """,
            params={"doc_hash": doc_hash},
        )

    def mark_document_failed(self, doc_hash: str, error_message: str) -> None:
        """Mark a claimed document hash as failed with last error details."""
        if not doc_hash:
            return

        self.graph_db.query(
            """
            MATCH (doc:ProcessedDocument {hash: $doc_hash})
            SET doc.status = 'failed',
                doc.failed_at = datetime(),
                doc.updated_at = datetime(),
                doc.last_error = $error_message
            REMOVE doc.claim_token
            """,
            params={
                "doc_hash": doc_hash,
                "error_message": error_message[:1000],
            },
        )

    def ensure_entity_key_indexes(self) -> int:
        """Index the canonical merge key on every configured label.

        Extraction merges on ``key``; without an index each MERGE scans the whole label.
        Creation is idempotent, so this is safe to call at the start of every run.

        Returns:
            Number of labels for which an index was requested
        """
        logger = _get_logger()
        labels = get_config().graph_schema.node_labels

        for label in labels:
            index_name = f"entity_key_{label.lower()}"
            try:
                self.graph_db.query(
                    f"CREATE INDEX {index_name} IF NOT EXISTS FOR (n:`{label}`) ON (n.key)"
                )
            except Exception as exc:
                logger.warning("Could not create key index for label %s: %s", label, exc)

        logger.info("Ensured canonical key indexes for %d labels", len(labels))
        return len(labels)

    def deduplicate_entities(self, keys: list[str] | None = None) -> dict[str, int]:
        """Run the post-ingest repair for entities split across several nodes.

        Args:
            keys: Canonical keys this run wrote. Passing them keeps the work proportional to
                what changed instead of to the size of the graph.

        Returns:
            Counts for each repair stage, for the pipeline summary log
        """
        from src.data_pipeline.flows.graph_dedup import deduplicate_graph

        return deduplicate_graph(self.graph_db, keys)

    def execute_cypher(self, query: str, params: dict[str, str] | None = None) -> None:
        logger = _get_logger()
        if not query or not query.strip():
            logger.error("Empty Cypher query")
            return
        try:
            logger.info("Executing Cypher query: %s", query)
            self.graph_db.query(query, params=params or {})
            logger.info("Cypher executed successfully")
        except Exception as e:
            logger.error("Failed to execute cypher: %s", e)
            raise

    def graph_has_data(self) -> bool:
        """Return True when the graph contains any node at all."""
        rows = self.graph_db.query("MATCH (n) RETURN count(n) AS total LIMIT 1")
        return bool(rows and rows[0].get("total"))

    def get_latest_pipeline_source_hashes(self) -> dict[str, str]:
        """Latest ``PipelineRun`` source id → content hash map."""
        rows = self.graph_db.query(
            """
            MATCH (pr:PipelineRun)
            WITH pr ORDER BY pr.run_at DESC LIMIT 1
            RETURN pr.source_hashes_json AS payload
            """
        )
        if not rows or not rows[0].get("payload"):
            return {}
        raw = rows[0]["payload"]
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {}
        return {str(k): str(v) for k, v in parsed.items()}

    def record_pipeline_run(self, source_hashes: dict[str, str], mode: str = "full") -> None:
        """Create a completed ``PipelineRun`` with source hashes JSON."""
        self.graph_db.query(
            """
            CREATE (pr:PipelineRun {
                run_id: $run_id,
                run_at: datetime(),
                status: 'completed',
                mode: $mode,
                source_hashes_json: $json
            })
            """,
            params={
                "run_id": str(uuid.uuid4()),
                "mode": mode,
                "json": json.dumps(source_hashes, sort_keys=True),
            },
        )

    def record_restore_run(self) -> None:
        """Create a ``PipelineRun`` row for a dump restore."""
        self.graph_db.query(
            """
            CREATE (pr:PipelineRun {
                run_id: $run_id,
                run_at: datetime(),
                status: 'completed',
                mode: 'restore'
            })
            """,
            params={"run_id": str(uuid.uuid4())},
        )

    def link_processed_document_to_source(self, doc_hash: str, source_id: str) -> None:
        """Attach processed-document hash bookkeeping to a source id."""
        if not doc_hash or not source_id:
            return
        self.graph_db.query(
            """
            MATCH (doc:ProcessedDocument {hash: $doc_hash})
            MERGE (source:Source {source_id: $source_id})
            MERGE (doc)-[:FROM_SOURCE]->(source)
            """,
            params={"doc_hash": doc_hash, "source_id": source_id},
        )


@task
def claim_document_for_processing(doc_hash: str) -> bool:
    """Claim document hash in Neo4j for idempotent processing."""
    logger = _get_logger()
    pop = GraphPopulator()
    claimed = pop.claim_document_hash(doc_hash)
    if claimed:
        logger.info("Claimed document hash %s", doc_hash)
    else:
        logger.info("Skipping hash %s (already processed or currently in progress)", doc_hash)
    return claimed


@task
def populate_graph(cypher_query: str, doc_hash: str = "", source_id: str = "") -> list[str]:
    """Execute pipe-separated cypher statements against the configured Neo4j instance.

    Args:
        cypher_query: Generated statements, separated by pipe
        doc_hash: Page hash to mark processed or failed
        source_id: Page source id recorded as provenance for the merged nodes

    Returns:
        The canonical keys this page wrote, so the post-ingest repair can look at only those
    """

    logger = _get_logger()
    logger.info(
        "populate_graph task received query of length %d for source %s",
        len(cypher_query or ""),
        source_id or "<unknown>",
    )
    statements = [part.strip() for part in (cypher_query or "").split("|") if part.strip()]
    query_to_execute, query_params = _build_query_with_provenance(
        statements,
        source_id,
        logger,
    )
    pop = GraphPopulator()
    try:
        if query_to_execute:
            pop.execute_cypher(query_to_execute, params=query_params)
        pop.mark_document_processed(doc_hash)
    except Exception as exc:
        pop.mark_document_failed(doc_hash, str(exc))
        raise
    finally:
        try:
            pop.link_processed_document_to_source(doc_hash, source_id)
        except Exception as link_exc:
            logger.warning("Failed to link hash %s to source %s: %s", doc_hash, source_id, link_exc)

    return extract_entity_keys("\n".join(statements))
