import logging

from dotenv import load_dotenv

from src.data_pipeline.graph_dump import (
    ensure_host_dump_dir,
    export_graph_to_cypher,
    host_dump_path,
    import_graph_from_cypher_dump,
)


def dump_graph_main() -> None:
    """CLI entrypoint: export Neo4j graph to ``dumps/graph_export.cypher``."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    load_dotenv()
    ensure_host_dump_dir()
    export_graph_to_cypher()
    print("OK:", host_dump_path().resolve())


def restore_graph_main() -> None:
    """CLI entrypoint: import ``dumps/graph_export.cypher`` into Neo4j via APOC."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    load_dotenv()
    path = host_dump_path()
    if not path.is_file():
        raise SystemExit(f"missing dump: {path.resolve()}")
    import_graph_from_cypher_dump()
    print("OK:", path.resolve())
