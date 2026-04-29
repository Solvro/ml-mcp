import logging
import os
import shutil
from pathlib import Path

from neo4j import GraphDatabase
from neo4j.exceptions import ClientError

logger = logging.getLogger(__name__)

GRAPH_EXPORT_FILE_NAME = "graph_export.cypher"


def _neo4j_import_rel_path() -> str:
    """Path under Neo4j's import directory for APOC export/import.

    Default ``graph_export.cypher`` writes next to the import folder root (Desktop-friendly).
    Use ``dumps/graph_export.cypher`` when Docker Compose bind-mounts ``../dumps`` to
    ``import/dumps`` (set env ``NEO4J_IMPORT_REL_PATH`` in ``.env``).
    """
    raw = (os.getenv("NEO4J_IMPORT_REL_PATH") or "").strip()
    if raw:
        return raw.replace("\\", "/")
    return GRAPH_EXPORT_FILE_NAME


def _project_root() -> Path | None:
    """Root of the git project (directory containing ``pyproject.toml``), if known."""
    here = Path(__file__).resolve()
    for p in (here.parent, *here.parents):
        if (p / "pyproject.toml").is_file():
            return p
    return None


def _auth() -> tuple[str, str, str]:
    uri = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")
    if not uri or not username or not password:
        raise ValueError("NEO4J connection settings are required")
    return uri, username, password


def host_dump_path() -> Path:
    """Path to the Cypher dump file on the **development machine** (repo ``dumps/`` by default)."""
    raw = (os.getenv("PIPELINE_HOST_DUMP_DIR") or "").strip()
    if raw:
        return Path(raw).expanduser() / GRAPH_EXPORT_FILE_NAME
    root = _project_root()
    if root is not None:
        return root / "dumps" / GRAPH_EXPORT_FILE_NAME
    return Path("dumps") / GRAPH_EXPORT_FILE_NAME


def ensure_host_dump_dir() -> Path:
    p = host_dump_path().parent
    p.mkdir(parents=True, exist_ok=True)
    return p


def _neo4j_server_dump_path() -> Path | None:
    """Where APOC reads/writes the dump when ``NEO4J_LOCAL_IMPORT_DIR`` is the DBMS import dir."""
    root = os.getenv("NEO4J_LOCAL_IMPORT_DIR", "").strip()
    if not root:
        return None
    rel = _neo4j_import_rel_path()
    return Path(root).expanduser().joinpath(*rel.split("/"))


def _import_directory_from_server(session) -> Path | None:
    """Resolve the DBMS import directory (absolute on the server host; same as this machine for Desktop)."""
    for name in (
        "server.directories.import",
        "dbms.directories.import",
    ):
        for q in (
            f"CALL dbms.listConfig() YIELD name, value WHERE name = '{name}' RETURN value",
            f"SHOW SETTINGS YIELD name, value WHERE name = '{name}' RETURN value",
        ):
            try:
                rec = session.run(q).single()
            except Exception as exc:
                logger.debug("Import dir query failed: %s", exc)
                continue
            if not rec or rec.get("value") is None:
                continue
            raw = str(rec["value"]).strip().strip('"')
            path = Path(raw)
            if path.is_dir():
                return path.resolve()
            if not path.is_absolute():
                for home_key in ("server.directories.neo4j_home", "dbms.directories.neo4j_home"):
                    try:
                        hr = session.run(
                            f"CALL dbms.listConfig() YIELD name, value "
                            f"WHERE name = '{home_key}' RETURN value"
                        ).single()
                        if hr and hr.get("value"):
                            home = Path(str(hr["value"]).strip().strip('"'))
                            candidate = (home / path).resolve()
                            if candidate.is_dir():
                                return candidate
                    except Exception:
                        continue
    return None


def _detect_import_directory(driver) -> Path | None:
    """Try default DB session then ``system`` (Neo4j 5+) so ``dbms.listConfig`` resolves."""
    for db in (None, "system"):
        try:
            sess_kw: dict = {}
            if db is not None:
                sess_kw["database"] = db
            with driver.session(**sess_kw) as session:
                imp = _import_directory_from_server(session)
            if imp:
                return imp
        except Exception as exc:
            logger.debug("Import dir lookup (%s): %s", db or "default", exc)
            continue
    return None


def _copy_from_server_import_to_host(out: Path, rel: str) -> bool:
    """Copy dump from Neo4j's import directory into the repo (Neo4j Desktop: no shared bind mount)."""
    if out.is_file():
        return True
    uri, username, password = _auth()
    src_parts = tuple(rel.split("/"))
    try:
        with GraphDatabase.driver(uri, auth=(username, password)) as driver:
            imp = _detect_import_directory(driver)
            if not imp:
                return False
            src = imp.joinpath(*src_parts)
            if not src.is_file():
                return False
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out)
            logger.info("Copied graph dump from Neo4j import dir to repo: %s", out.resolve())
            return True
    except Exception as exc:
        logger.warning("Could not copy from Neo4j import dir: %s", exc)
    return False


def _copy_dump_to_repo_if_needed(out: Path, rel: str) -> None:
    """If APOC wrote under Neo4j's import tree, copy the file into the repo ``dumps/`` path."""
    if out.is_file():
        return
    server = _neo4j_server_dump_path()
    if server and server.is_file() and server.resolve() != out.resolve():
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(server, out)
        logger.info("Copied graph dump to repo: %s", out.resolve())
        return
    _copy_from_server_import_to_host(out, rel)


def _iter_cypher_shell_chunks(script: str):
    """Split a cypher-shell script (``apoc.export`` ``format: cypher-shell``) into runnable chunks.

    ``:begin`` / ``:commit`` blocks are yielded as one chunk each; standalone statements between
    blocks (e.g. ``CALL db.awaitIndexes``) are yielded separately.
    """
    lines = script.splitlines()
    buffer: list[str] = []
    in_tx = False
    for raw in lines:
        stripped = raw.strip()
        if stripped == ":begin":
            in_tx = True
            buffer = []
            continue
        if stripped == ":commit":
            in_tx = False
            body = "\n".join(buffer).strip()
            if body:
                yield body
            buffer = []
            continue
        if in_tx:
            buffer.append(raw)
            continue
        if stripped.startswith(":"):
            continue
        if stripped:
            yield stripped


def _run_many_tx(chunk: str):
    """Return a transaction callback that runs one cypher-shell chunk via ``apoc.cypher.runMany``."""

    def work(tx) -> None:
        tx.run("CALL apoc.cypher.runMany($cypher, {})", cypher=chunk)

    return work


def import_graph_from_cypher_dump() -> None:
    """Restore from the repo dump file using ``apoc.cypher.runMany`` (cypher-shell format).

    ``apoc.cypher.runFile`` is not available in some APOC builds; we read the file from disk and
    replay transaction chunks instead.
    """
    out = host_dump_path()
    if not out.is_file() or out.stat().st_size == 0:
        raise ValueError(f"missing or empty dump: {out.resolve()}")
    content = out.read_text(encoding="utf-8")
    uri, username, password = _auth()
    with GraphDatabase.driver(uri, auth=(username, password)) as driver:
        with driver.session() as session:
            try:
                for chunk in _iter_cypher_shell_chunks(content):
                    session.execute_write(_run_many_tx(chunk))
            except ClientError as exc:
                code = getattr(exc, "code", "") or ""
                if "ProcedureNotFound" in code or "ProcedureNotFound" in str(exc):
                    raise RuntimeError(
                        "APOC procedure apoc.cypher.runMany is not available. "
                        "Enable the full APOC plugin for this database in Neo4j Desktop, "
                        "or run the script manually with cypher-shell -f "
                        f'"{out.resolve()}"'
                    ) from exc
                raise


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
    ensure_host_dump_dir()
    uri, username, password = _auth()
    out = host_dump_path()
    rel = _neo4j_import_rel_path()
    with GraphDatabase.driver(uri, auth=(username, password)) as driver:
        with driver.session() as session:
            result = session.run(
                "CALL apoc.export.cypher.all($file, $config) "
                "YIELD file, batches, time RETURN file, batches, time",
                file=rel,
                config={"format": "cypher-shell"},
            )
            rec = result.single()
            if rec:
                logger.info("APOC export: %s", rec.data())
    _copy_dump_to_repo_if_needed(out, rel)
    if not out.is_file():
        logger.warning(
            "Dump missing on host %s (Docker with bind mount: set "
            "NEO4J_IMPORT_REL_PATH=dumps/graph_export.cypher so the file lands under repo dumps/)",
            out,
        )
    _copy_to_drive(out)


def host_nonempty_dump_exists() -> bool:
    """True if the repo dump file exists and is non-empty (placeholders do not trigger restore)."""
    p = host_dump_path()
    try:
        return p.is_file() and p.stat().st_size > 0
    except OSError:
        return False
