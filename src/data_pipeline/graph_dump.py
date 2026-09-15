import logging
import os
import re
import shutil
from pathlib import Path

from neo4j import GraphDatabase

logger = logging.getLogger(__name__)

# Relative to Neo4j import dir (e.g. ``/var/lib/neo4j/import`` in Docker).
NEO4J_IMPORT_REL_PATH = "dumps/graph_export.cypher"

# The three schema shapes ``apoc.export.cypher.all`` writes — ``CREATE RANGE INDEX FOR``,
# ``CREATE FULLTEXT INDEX entity_search FOR``, ``CREATE CONSTRAINT UNIQUE_IMPORT_NAME FOR`` —
# none of them idempotent as written. ``IF NOT EXISTS`` is inserted before ``FOR``.
_SCHEMA_CREATE_RE = re.compile(
    r"^(CREATE\s+(?:\w+\s+)?(?:INDEX|CONSTRAINT)(?:\s+`?[\w.]+`?)?)\s+FOR\b",
    re.IGNORECASE,
)


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


def _quotes_balanced(text: str) -> bool:
    """True when every double-quoted string literal in ``text`` is closed.

    APOC writes string values in double quotes and escapes with a backslash, so this is what
    tells a ``;`` that ends a statement from one that ends a line inside a value.
    """
    inside = False
    escaped = False
    for char in text:
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == '"':
            inside = not inside
    return not inside


def _make_schema_statement_idempotent(statement: str) -> str:
    if re.search(r"\bIF\s+NOT\s+EXISTS\b", statement, re.IGNORECASE):
        return statement
    return _SCHEMA_CREATE_RE.sub(r"\1 IF NOT EXISTS FOR", statement, count=1)


def parse_cypher_shell_dump(text: str) -> list[list[str]]:
    """Split a ``cypher-shell`` format dump into transactions of Cypher statements.

    ``apoc.export.cypher.all(..., {format: 'cypher-shell'})`` writes statements terminated by
    ``;`` at the end of a line, grouped by ``:begin`` / ``:commit`` into transactions, with a
    few (``CALL db.awaitIndexes``) outside any group. A statement may span several lines (the
    ``UNWIND`` / ``MATCH`` / ``CREATE`` relationship batches do) and a value may contain ``;``,
    so a statement ends only at a line-final ``;`` with every string literal closed.

    Args:
        text: The dump file's content

    Returns:
        One list of statements per transaction, in file order. A statement outside any
        ``:begin`` / ``:commit`` pair becomes a transaction of its own. Schema ``CREATE``
        statements are rewritten with ``IF NOT EXISTS`` so a dump loads over indexes the
        pipeline and the server already created.

    Raises:
        ValueError: On a cypher-shell command other than ``:begin`` / ``:commit``, an unclosed
            transaction, or a statement the file ends in the middle of
    """
    batches: list[list[str]] = []
    block: list[str] | None = None
    pending: list[str] = []

    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.rstrip()
        if not pending:
            if not line.strip():
                continue
            if line.startswith(":"):
                command = line.strip().lower()
                if command == ":begin":
                    if block is not None:
                        raise ValueError(f"line {lineno}: ':begin' inside an open transaction")
                    block = []
                elif command == ":commit":
                    if block is None:
                        raise ValueError(f"line {lineno}: ':commit' without ':begin'")
                    if block:
                        batches.append(block)
                    block = None
                else:
                    raise ValueError(f"line {lineno}: unsupported cypher-shell command {line!r}")
                continue
        pending.append(line)
        joined = "\n".join(pending)
        if joined.endswith(";") and _quotes_balanced(joined):
            statement = _make_schema_statement_idempotent(joined[:-1].strip())
            if block is not None:
                block.append(statement)
            else:
                batches.append([statement])
            pending = []

    if pending:
        raise ValueError("the dump ends in the middle of a statement (no closing ';')")
    if block is not None:
        raise ValueError("the dump ends inside a transaction (':begin' without ':commit')")
    return batches


def import_graph_from_cypher_dump() -> None:
    """Load the dump at ``host_dump_path()`` into Neo4j through the driver.

    The file is read here and its statements are sent over Bolt, one transaction per
    ``:begin`` / ``:commit`` group, so restore needs no file access on the Neo4j side and no
    APOC procedure: the ``runFile`` procedure this used to call ships in APOC Extended, not
    in the ``apoc`` plugin the image installs (#21). A failing statement rolls back its own
    transaction and raises; the transactions before it stay applied, as they would under
    ``cypher-shell``.

    Raises:
        ValueError: When the file is not a cypher-shell format dump, or the connection settings
            are missing
        neo4j.exceptions.Neo4jError: When Neo4j rejects a statement
    """
    path = host_dump_path()
    batches = parse_cypher_shell_dump(path.read_text(encoding="utf-8"))
    uri, username, password = _auth()
    statements = 0
    with GraphDatabase.driver(uri, auth=(username, password)) as driver:
        with driver.session() as session:
            for batch in batches:
                with session.begin_transaction() as tx:
                    for statement in batch:
                        tx.run(statement).consume()
                    tx.commit()
                statements += len(batch)
    logger.info("Restored %d statements in %d transactions from %s", statements, len(batches), path)


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
