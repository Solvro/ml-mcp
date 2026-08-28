"""Detect list and table rows the extraction model left out of its output.

Issue #53: on the academic-calendar page the model kept the days off that carry a proper name
and silently dropped "2 XI 2026 r. - dzien wolny od zajec", which is described only generically.
The prompt now forbids that, but a naming instruction is followed most of the time, and a page of
dates is exactly where the misses are invisible.

Rows are therefore counted before generation and checked against what was generated, so a miss
becomes a second extraction pass over the rows that were skipped instead of silent data loss.
"""

import re

from src.text_normalization import CYPHER_STRING_LITERAL_RE, normalize_search_text

# A list or table row: a bullet, a number, or a cell-separated line. These are the shapes the
# issue calls out as the ones where dropping an entry is actively harmful.
LIST_ROW_RE = re.compile(
    r"^\s*(?:[-*•–—>]|\d{1,3}[.)]|\|)\s*(?P<content>\S.*?)\s*$|"
    r"^\s*(?P<cells>[^|\t]*(?:[|\t][^|\t]*)+)\s*$"
)
TOKEN_RE = re.compile(r"[0-9a-z]+")

# Below this share of a row's tokens appearing in the generated values, the row counts as missed.
ROW_COVERAGE_THRESHOLD = 0.6
# One-character tokens carry no evidence either way.
MIN_TOKEN_LENGTH = 2
# A row needs some substance before its absence means anything.
MIN_ROW_TOKENS = 2


def extract_list_rows(text: str) -> list[str]:
    """
    Collect the list and table rows of a page, each of which should become its own node.

    Args:
        text: Page text as extracted from the source document

    Returns:
        Row contents in the order they appear, without their bullet or numbering
    """
    rows: list[str] = []

    for line in text.splitlines():
        match = LIST_ROW_RE.match(line)
        if match is None:
            continue

        content = match.group("content") or match.group("cells") or ""
        content = content.replace("|", " ").replace("\t", " ").strip()
        if len(_row_tokens(content)) < MIN_ROW_TOKENS:
            continue

        rows.append(content)

    return rows


def _row_tokens(row: str) -> list[str]:
    """Return the tokens of a row that carry enough substance to match on."""
    return [
        token
        for token in TOKEN_RE.findall(normalize_search_text(row))
        if len(token) >= MIN_TOKEN_LENGTH
    ]


def _generated_value_text(statements: list[str]) -> str:
    """Return every quoted value in the generated Cypher as one normalized haystack."""
    values = [
        literal.group(0)[1:-1]
        for statement in statements
        for literal in CYPHER_STRING_LITERAL_RE.finditer(statement)
    ]
    return normalize_search_text(" ".join(values))


def rows_missing_from_cypher(rows: list[str], statements: list[str]) -> list[str]:
    """
    Report the rows whose content did not make it into any generated node.

    A row counts as covered when most of its tokens appear somewhere in the generated values.
    Partial credit matters: the model is free to reword a row, but a row it never read leaves
    almost none of its wording behind.

    Args:
        rows: Rows found on the page
        statements: Generated Cypher statements

    Returns:
        The rows that are not represented in the generated output
    """
    if not rows:
        return []

    haystack_tokens = set(TOKEN_RE.findall(_generated_value_text(statements)))

    missing: list[str] = []
    for row in rows:
        tokens = _row_tokens(row)
        if not tokens:
            continue
        found = sum(1 for token in tokens if token in haystack_tokens)
        if found / len(tokens) < ROW_COVERAGE_THRESHOLD:
            missing.append(row)

    return missing
