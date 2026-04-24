import logging
import os
import shutil
from pathlib import Path

from neo4j import GraphDatabase

logger = logging.getLogger(__name__)

# Relative to Neo4j import dir (e.g. ``/var/lib/neo4j/import`` in Docker).
NEO4J_IMPORT_REL_PATH = "dumps/graph_export.cypher"


def _auth() -> tuple[str, str, str]:
    uri = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")
    if not uri or not username or not password:
        raise ValueError("NEO4J connection settings are required")
    return uri, username, password


def host_dump_path() -> Path:
    return Path(os.getenv("PIPELINE_HOST_DUMP_DIR", "dumps")).expanduser() / "graph_export.cypher"


def ensure_host_dump_dir() -> Path:
    p = host_dump_path().parent
    p.mkdir(parents=True, exist_ok=True)
    return p


def import_graph_from_cypher_dump() -> None:
    """Import ``NEO4J_IMPORT_REL_PATH`` with ``apoc.cypher.runFile`` (caller checks host file)."""
    uri, username, password = _auth()
    with GraphDatabase.driver(uri, auth=(username, password)) as driver:
        with driver.session() as session:
            session.run(
                "CALL apoc.cypher.runFile($file, $config)",
                file=NEO4J_IMPORT_REL_PATH,
                config={"timeout": 600},
            ).consume()


def _copy_to_drive(out: Path) -> None:
    root = os.getenv("PIPELINE_DRIVE_OUT", "").strip()
    if not root:
        logger.info("Drive copy skipped (PIPELINE_DRIVE_OUT unset or empty)")
        return
    if not out.is_file():
        logger.warning("Drive copy skipped (dump missing at %s)", out.resolve())
        return
    dest = Path(root).expanduser() / out.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out, dest)
    logger.info("Drive copy: %s", dest)


def export_graph_to_cypher() -> None:
    uri, username, password = _auth()
    out = host_dump_path()
    with GraphDatabase.driver(uri, auth=(username, password)) as driver:
        with driver.session() as session:
            result = session.run(
                "CALL apoc.export.cypher.all($file, $config) "
                "YIELD file, batches, time RETURN file, batches, time",
                file=NEO4J_IMPORT_REL_PATH,
                config={"format": "cypher-shell"},
            )
            rec = result.single()
            if rec:
                logger.info("APOC export: %s", rec.data())
    if not out.is_file():
        logger.warning("Dump missing on host %s (see compose bind for import/dumps)", out)
    _copy_to_drive(out)
