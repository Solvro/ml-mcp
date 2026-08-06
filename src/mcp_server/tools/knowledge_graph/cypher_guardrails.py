import re

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
    r"^\s*(MATCH|OPTIONAL\s+MATCH|WITH|UNWIND)\b",
    re.IGNORECASE,
)
STRING_LITERAL_RE = re.compile(r"'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\"")
COMMENT_RE = re.compile(r"//.*?$|/\*.*?\*/", re.MULTILINE | re.DOTALL)
CODE_FENCE_RE = re.compile(r"^\s*```\w*\s*\n?|\n?\s*```\s*$", re.MULTILINE)
LIMIT_CLAUSE_RE = re.compile(r"\bLIMIT\s+\d+\b", re.IGNORECASE)


class UnsafeCypherQueryError(ValueError):
    """Raised when generated Cypher contains a mutating operation or disallowed operation"""


def strip_code_fences(raw: str) -> str:
    """Remove markdown code fences that may wrap LLM output."""
    stripped = raw.strip()
    stripped = CODE_FENCE_RE.sub("", stripped)
    return stripped.strip()


def _scrub_for_validation(cypher: str) -> str:
    """Remove comments and string literals before keyword inspection."""
    without_comments = COMMENT_RE.sub(" ", cypher)
    return STRING_LITERAL_RE.sub(" ", without_comments)


def validate_read_only(cypher: str) -> None:
    """Reject Cypher that can mutate the graph or does not match the allowed read shape."""
    cleaned = strip_code_fences(cypher)
    scrubbed = _scrub_for_validation(cleaned).strip()
    if not scrubbed:
        raise UnsafeCypherQueryError("generated Cypher query is empty")
    if ";" in scrubbed.rstrip(";"):
        raise UnsafeCypherQueryError("multiple Cypher statements are not allowed")
    if not READ_ONLY_START_RE.search(scrubbed):
        raise UnsafeCypherQueryError("Cypher must start with a read-only clause")
    normalized = scrubbed.upper()
    for keyword in WRITE_KEYWORDS:
        if re.search(rf"\b{keyword}\b", normalized):
            raise UnsafeCypherQueryError(f"blocked mutating Cypher keyword: {keyword}")
    if not re.search(r"\bRETURN\b", normalized):
        raise UnsafeCypherQueryError("read-only Cypher must return data")


def ensure_limit(cypher: str, max_results: int) -> str:
    """Ensure the query has a LIMIT clause, appending one if missing."""
    if max_results <= 0:
        raise ValueError("max_results must be a positive integer")
    scrubbed = _scrub_for_validation(cypher)
    if LIMIT_CLAUSE_RE.search(scrubbed):
        return cypher.rstrip().rstrip(";")
    base = cypher.rstrip().rstrip(";")
    return f"{base} LIMIT {max_results}"
