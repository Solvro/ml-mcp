import logging

from dotenv import load_dotenv

from src.data_pipeline.flows.graph_dedup import deduplicate_graph
from src.data_pipeline.graph_dump import (
    ensure_host_dump_dir,
    export_graph_to_cypher,
    host_dump_path,
    import_graph_from_cypher_dump,
)


def dump_graph_main() -> None:
    """Export Neo4j graph to the configured dump path."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    load_dotenv()
    ensure_host_dump_dir()
    export_graph_to_cypher()
    print("OK:", host_dump_path().resolve())


def restore_graph_main() -> None:
    """Import graph dump into Neo4j (APOC)."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    load_dotenv()
    path = host_dump_path()
    if not path.is_file():
        raise SystemExit(f"missing dump: {path.resolve()}")
    import_graph_from_cypher_dump()
    print("OK:", path.resolve())


def dedup_graph_main() -> None:
    """Run the full post-ingest repair over the whole graph.

    Pipeline runs only repair the keys they wrote. This walks everything, including nodes
    written before labels were a closed set and before merges keyed on `key`, so it is the one
    to run after upgrading an existing database. It is idempotent.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    load_dotenv()
    stats = deduplicate_graph.fn()
    print(
        "OK: relabelled={relabelled_labels} backfilled={keys_backfilled} "
        "merged={groups_merged}".format(**stats)
    )
