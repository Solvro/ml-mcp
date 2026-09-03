import logging

from dotenv import load_dotenv

from src.config.logging_config import configure_logging
from src.data_pipeline.flows.graph_dedup import deduplicate_graph
from src.data_pipeline.graph_dump import (
    ensure_host_dump_dir,
    export_graph_to_cypher,
    host_dump_path,
    import_graph_from_cypher_dump,
)

logger = logging.getLogger(__name__)


def dump_graph_main() -> None:
    """Export Neo4j graph to the configured dump path."""
    configure_logging()
    load_dotenv()
    ensure_host_dump_dir()
    export_graph_to_cypher()
    logger.info("Graph exported to %s", host_dump_path().resolve())


def restore_graph_main() -> None:
    """Import graph dump into Neo4j (APOC)."""
    configure_logging()
    load_dotenv()
    path = host_dump_path()
    if not path.is_file():
        raise SystemExit(f"missing dump: {path.resolve()}")
    import_graph_from_cypher_dump()
    logger.info("Graph imported from %s", path.resolve())


def dedup_graph_main() -> None:
    """Run the full post-ingest repair over the whole graph.

    Pipeline runs only repair the keys they wrote. This walks everything, including nodes
    written before labels were a closed set and before merges keyed on `key`, so it is the one
    to run after upgrading an existing database. It is idempotent.
    """
    configure_logging()
    load_dotenv()
    stats = deduplicate_graph.fn()
    logger.info(
        "Deduplication finished: relabelled=%s backfilled=%s merged=%s",
        stats["relabelled_labels"],
        stats["keys_backfilled"],
        stats["groups_merged"],
    )
