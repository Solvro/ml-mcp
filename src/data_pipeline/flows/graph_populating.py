import logging
import os
import uuid

from langchain_neo4j import Neo4jGraph
from prefect import get_run_logger, task
from prefect.exceptions import MissingContextError

module_logger = logging.getLogger(__name__)


def _get_logger() -> logging.Logger:
    """Return Prefect run logger when available, otherwise module logger."""
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

    def execute_cypher(self, query: str):
        logger = _get_logger()
        if not query or not query.strip():
            logger.error("Empty Cypher query")
            return
        try:
            logger.info("Executing Cypher query: %s", query)
            self.graph_db.query(query)
            logger.info("Cypher executed successfully")
        except Exception as e:
            logger.error("Failed to execute cypher: %s", e)
            raise


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
def populate_graph(cypher_query: str, doc_hash: str = ""):
    """Execute a cypher query against the configured Neo4j instance."""
    logger = _get_logger()
    logger.info("populate_graph task received query of length %d", len(cypher_query or ""))
    pop = GraphPopulator()
    try:
        pop.execute_cypher(cypher_query)
        pop.mark_document_processed(doc_hash)
    except Exception as exc:
        pop.mark_document_failed(doc_hash, str(exc))
        raise
