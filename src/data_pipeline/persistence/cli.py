import logging

from dotenv import load_dotenv

from src.data_pipeline.persistence.graph_dump import (
    ensure_host_dump_dir,
    export_graph_to_cypher,
    host_dump_path,
    host_nonempty_dump_exists,
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
    if not host_nonempty_dump_exists():
        raise SystemExit(f"missing or empty dump: {path.resolve()}")
    import_graph_from_cypher_dump()
    print("OK:", path.resolve())
