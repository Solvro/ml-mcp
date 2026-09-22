import re

from ....text_normalization import CYPHER_STRING_LITERAL_RE

WRITE_KEYWORDS = frozenset(
    {
        "CREATE",
        "MERGE",
        "DELETE",
        "DETACH",
        "SET",
        "REMOVE",
        "DROP",
        "FOREACH",
        "LOAD",
        "ALTER",
        "GRANT",
        "DENY",
        "REVOKE",
    }
)

READ_ONLY_START_RE = re.compile(
    r"^\s*(MATCH|OPTIONAL\s+MATCH|WITH|UNWIND|CALL)\b",
    re.IGNORECASE,
)
# Every procedure a query calls by name. A CALL is only permitted when the caller passes that
# procedure in allowed_procedures, so generated Cypher - which passes none - still cannot call
# anything at all.
#
# The argument list is deliberately not part of the pattern: Cypher lets a no-argument procedure
# be called without parentheses, so requiring them meant `CALL db.labels YIELD label` captured no
# name, left the allowlist with nothing to vet, and passed.
PROCEDURE_CALL_RE = re.compile(r"\bCALL\s+(?P<procedure>[A-Za-z_][\w.]*)", re.IGNORECASE)
# A CALL subquery names no procedure, so the allowlist has nothing to vet.
CALL_SUBQUERY_RE = re.compile(r"\bCALL\s*\{", re.IGNORECASE)
CODE_FENCE_RE = re.compile(r"^\s*```\w*\s*\n?|\n?\s*```\s*$", re.MULTILINE)
# One left-to-right pass: the first alternative matching at a position wins, so `//` inside a
# string is not a comment and a quote inside a comment is not a string.
SCRUB_TOKEN_RE = re.compile(
    rf"(?P<literal>{CYPHER_STRING_LITERAL_RE.pattern})"
    r"|(?P<quoted>`[^`]+`)"
    r"|(?P<line_comment>//[^\r\n]*)"
    r"|(?P<block_comment>/\*.*?\*/)"
    r"|(?P<unterminated>['\"`]|/\*)",
    re.DOTALL,
)
# The cap on what a query may return is the LIMIT it ends on. One further in - `WITH n ORDER BY
# n.rank DESC LIMIT 100` - shapes an intermediate result, and rewriting it would change what the
# query means rather than how much of the answer comes back.
TRAILING_LIMIT_RE = re.compile(r"\bLIMIT\s+(?P<rows>\d+)\s*$", re.IGNORECASE)


class UnsafeCypherQueryError(ValueError):
    """Raised when generated Cypher contains a mutating operation or disallowed operation"""


def strip_code_fences(raw: str) -> str:
    """Remove markdown code fences that may wrap LLM output."""
    stripped = raw.strip()
    stripped = CODE_FENCE_RE.sub("", stripped)
    return stripped.strip()


def _scrub(cypher: str) -> tuple[str, str | None]:
    """Blank out comments, string literals and backticked names, keeping every offset.

    Args:
        cypher: Query to read

    Returns:
        The blanked query, and the opener that never closed (everything after it is blanked
        too), or None
    """
    pieces: list[str] = []
    cursor = 0

    for token in SCRUB_TOKEN_RE.finditer(cypher):
        pieces.append(cypher[cursor : token.start()])
        if token.lastgroup == "unterminated":
            pieces.append(" " * (len(cypher) - token.start()))
            return "".join(pieces), token.group(0)
        pieces.append(" " * (token.end() - token.start()))
        cursor = token.end()

    pieces.append(cypher[cursor:])
    return "".join(pieces), None


def _scrub_for_validation(cypher: str) -> str:
    """Blank out everything a keyword scan must not read."""
    return _scrub(cypher)[0]


def validate_read_only(cypher: str, allowed_procedures: frozenset[str] = frozenset()) -> None:
    """Reject Cypher that can mutate the graph or does not match the allowed read shape.

    Args:
        cypher: Query to validate
        allowed_procedures: Procedure names this query may call. Callers passing generated
            Cypher must leave this empty; it exists so an internally authored query can use one
            specific reviewed procedure without opening CALL up to the model.

    Raises:
        UnsafeCypherQueryError: If the query can mutate the graph, calls a procedure that was
            not allowlisted, leaves a quote or comment unterminated, or does not match the
            allowed read shape
    """
    cleaned = strip_code_fences(cypher)
    blanked, unterminated = _scrub(cleaned)
    if unterminated is not None:
        raise UnsafeCypherQueryError(f"unterminated {unterminated} in generated Cypher")
    scrubbed = blanked.strip()
    if not scrubbed:
        raise UnsafeCypherQueryError("generated Cypher query is empty")
    if ";" in scrubbed.rstrip(";"):
        raise UnsafeCypherQueryError("multiple Cypher statements are not allowed")
    if CALL_SUBQUERY_RE.search(scrubbed):
        raise UnsafeCypherQueryError("CALL subqueries are not allowed")
    permitted = {name.lower() for name in allowed_procedures}
    called = {match.group("procedure").lower() for match in PROCEDURE_CALL_RE.finditer(scrubbed)}
    forbidden = sorted(called - permitted)
    if forbidden:
        raise UnsafeCypherQueryError(f"blocked Cypher procedure call: {forbidden[0]}")
    if not READ_ONLY_START_RE.search(scrubbed):
        raise UnsafeCypherQueryError("Cypher must start with a read-only clause")
    normalized = scrubbed.upper()
    for keyword in WRITE_KEYWORDS:
        if re.search(rf"\b{keyword}\b", normalized):
            raise UnsafeCypherQueryError(f"blocked mutating Cypher keyword: {keyword}")
    if not re.search(r"\bRETURN\b", normalized):
        raise UnsafeCypherQueryError("read-only Cypher must return data")


def trailing_limit(cypher: str) -> int | None:
    """
    Read the row cap a query ends on, the one that decides how much of the answer comes back.

    Only the trailing clause is read, for the reason ``ensure_limit`` rewrites only that one: a
    ``WITH ... LIMIT n`` shapes an intermediate result and says nothing about the rows returned.

    Args:
        cypher: Query to read

    Returns:
        The number of rows the query ends on, or None when it ends without a LIMIT
    """
    trailing = TRAILING_LIMIT_RE.search(_scrub_for_validation(cypher.rstrip().rstrip(";")))
    return None if trailing is None else int(trailing.group("rows"))


def ensure_limit(cypher: str, max_results: int) -> str:
    """Cap the rows a query can return at ``max_results``.

    A query with no trailing LIMIT gets one. A query that ends on a LIMIT larger than
    ``max_results`` has it clamped down - the model asking for 10 rows, or for 999999999, does
    not get to decide how much of the graph comes back. A smaller one is left alone: a model
    narrowing its own result set is not the problem this cap exists for.

    Only the trailing LIMIT is read or rewritten, so a `WITH ... LIMIT n` that shapes an
    intermediate result keeps meaning what it said.

    **A UNION is capped on its last branch only**, so `... LIMIT 100 UNION ... LIMIT 100` with a
    cap of 5 returns up to 105 rows rather than 5 (review of PR #90). `validate_read_only` does
    not block UNION, and capping the whole thing would mean wrapping it in a `CALL` subquery,
    which the guardrail does block. Left as is: before this function clamped anything the case
    was uncapped entirely, so it is not a regression, and the Cypher prompt does not lead the
    model towards UNION.

    Args:
        cypher: Query to cap
        max_results: Most rows the query may return

    Returns:
        The query, ending on a LIMIT of at most ``max_results``

    Raises:
        ValueError: If max_results is not a positive integer
    """
    if max_results <= 0:
        raise ValueError("max_results must be a positive integer")

    base = cypher.rstrip().rstrip(";")
    trailing = TRAILING_LIMIT_RE.search(_scrub_for_validation(base))

    if trailing is None:
        # On its own line, so a query ending in a `//` comment does not swallow the clause.
        return f"{base}\nLIMIT {max_results}"

    if int(trailing.group("rows")) <= max_results:
        return base

    return f"{base[: trailing.start()]}LIMIT {max_results}{base[trailing.end() :]}"
