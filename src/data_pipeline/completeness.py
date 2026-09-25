"""Detect list and table rows the extraction model left out of its output.

Issue #53: on the academic-calendar page the model kept the days off that carry a proper name
and silently dropped "2 XI 2026 r. - dzien wolny od zajec", which is described only generically.
The prompt now forbids that, but a naming instruction is followed most of the time, and a page of
dates is exactly where the misses are invisible.

Rows are therefore counted before generation and checked against what was generated, so a miss
becomes a second extraction pass over the rows that were skipped instead of silent data loss.

Issue #78 found the two ways this check could report success without meaning it:

* A PDF text layer puts each bullet on a line of its own, so a page of 168 bullets held no line
  that reads as a row and nothing was ever counted. The markers are rejoined before rows are
  read (:func:`~src.text_normalization.join_wrapped_list_rows`).
* A row counted as covered when its wording appeared in *any* quoted value, so one category node
  whose ``context`` recites sixteen bullets covered all sixteen of them. Coverage is therefore
  decided per node and anchored on the node's ``title``: the rule being enforced is that every
  row becomes a node of its own, and a node's title is what says which row it is.
"""

import re
from dataclasses import dataclass

from src.text_normalization import (
    CYPHER_STRING_LITERAL_RE,
    LIST_MARKER_PATTERN,
    join_wrapped_list_rows,
    normalize_search_text,
)

# A list or table row: a marker, or a cell-separated line. These are the shapes the issue calls
# out as the ones where dropping an entry is actively harmful. The marker is the one the joining
# pass recognises, so a row cannot be rebuilt and then go uncounted - lettered sub-entries such
# as "a) o zasiegu krajowym" were exactly that gap.
LIST_ROW_RE = re.compile(
    rf"^\s*(?:{LIST_MARKER_PATTERN}|\|)\s*(?P<content>\S.*?)\s*$|"
    r"^\s*(?P<cells>[^|\t]*(?:[|\t][^|\t]*)+)\s*$",
    re.IGNORECASE,
)
TOKEN_RE = re.compile(r"[0-9a-z]+")

_IDENTIFIER = r"(?:`[^`]+`|[A-Za-z_]\w*)"
# A node in a generated statement: "(var:Label {title: '...', context: '...'})". Relationship
# property maps sit in square brackets and are left out, as they describe no entity of their own.
NODE_PROPERTY_MAP_RE = re.compile(
    rf"\(\s*(?P<variable>[A-Za-z_]\w*)?\s*(?P<labels>(?::\s*{_IDENTIFIER}\s*)*)"
    rf"\{{(?P<properties>[^{{}}]*)\}}"
)
PROPERTY_ENTRY_RE = re.compile(
    rf"(?P<key>{_IDENTIFIER})\s*:\s*(?P<value>{CYPHER_STRING_LITERAL_RE.pattern})"
)
# The canonical-key rewrite moves properties out of the pattern: "ON CREATE SET n1.title = '...'".
SET_ASSIGNMENT_RE = re.compile(
    rf"(?P<variable>[A-Za-z_]\w*)\s*\.\s*(?P<key>{_IDENTIFIER})\s*=\s*"
    rf"(?P<value>{CYPHER_STRING_LITERAL_RE.pattern})"
)
TITLE_PROPERTY = "title"

# Below this share of a row's tokens appearing in one node's values, that node does not hold the
# row at all.
ROW_COVERAGE_THRESHOLD = 0.6
# How much of a node's title and the row must line up before the node counts as the row's own
# node rather than the node of the section the row sits in.
TITLE_MATCH_THRESHOLD = 0.6
# One-character tokens carry no evidence either way.
MIN_TOKEN_LENGTH = 2
# A row needs some substance before its absence means anything.
MIN_ROW_TOKENS = 2
# A row ending on this heads the rows beneath it instead of carrying an entry of its own.
HEADING_SUFFIX = ":"


@dataclass(frozen=True)
class GeneratedNode:
    """One node in the generated Cypher, reduced to the tokens it was given."""

    title_tokens: frozenset[str]
    value_tokens: frozenset[str]


def extract_list_rows(text: str) -> list[str]:
    """
    Collect the list and table rows of a page, each of which should become its own node.

    A row that ends on a colon introduces the rows beneath it rather than carrying an entry of
    its own ("... kryteriami doboru kandydatek/kandydatow sa:"). Demanding a node for it is how
    a lead-in sentence became a `CriterionCategory` titled with half a paragraph; the entries it
    introduces are the rows that hold the content, and they are counted.

    Args:
        text: Page text as extracted from the source document

    Returns:
        Row contents in the order they appear, without their bullet or numbering
    """
    rows: list[str] = []

    for line in join_wrapped_list_rows(text).splitlines():
        content = _extract_row_content(line)
        if content is None:
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


def _extract_row_content(line: str) -> str | None:
    """Return row content from one source line, or None when the line is not a row."""
    match = LIST_ROW_RE.match(line)
    if match is None:
        return None

    content = match.group("content") or match.group("cells") or ""
    content = content.replace("|", " ").replace("\t", " ").strip()
    if content.endswith(HEADING_SUFFIX):
        return None
    if len(_row_tokens(content)) < MIN_ROW_TOKENS:
        return None
    return content


def _token_set(values: list[str]) -> frozenset[str]:
    """Return the matchable tokens of a group of property values."""
    return frozenset(_row_tokens(" ".join(values)))


def _record_property(properties: dict[str, list[str]], key: str, value: str) -> None:
    """Add one property value to a node, keeping the title apart from the rest."""
    if key.strip("`").casefold() == TITLE_PROPERTY:
        properties[TITLE_PROPERTY].append(value)
    properties["values"].append(value)


def _shared_share(tokens: frozenset[str], other: frozenset[str]) -> float:
    """Return the share of ``tokens`` that also appears in ``other``."""
    if not tokens:
        return 0.0
    return len(tokens & other) / len(tokens)


def _node_holds_row(node: GeneratedNode, row_tokens: frozenset[str]) -> bool:
    """
    Report whether this node is the row's own node.

    Two things have to hold. The node has to carry most of the row's wording, which a node that
    merely mentions the row's subject does not. And its title has to line up with the row, in
    either direction: a title shorter than the row is the row's name ("Swieto Niepodleglosci"
    for a dated entry), a title longer than the row is the row's fuller form. A title matching
    in neither direction belongs to something else - typically the section heading whose context
    recites the row along with all its siblings, the shape issue #78 reported as covered while
    the graph held no node for the row at all.

    Args:
        node: One node from the generated Cypher
        row_tokens: Tokens of the row being looked for

    Returns:
        True when this node is the node the row was supposed to become
    """
    if not node.title_tokens:
        return False
    if _shared_share(row_tokens, node.value_tokens) < ROW_COVERAGE_THRESHOLD:
        return False
    return (
        _shared_share(node.title_tokens, row_tokens) >= TITLE_MATCH_THRESHOLD
        or _shared_share(row_tokens, node.title_tokens) >= TITLE_MATCH_THRESHOLD
    )


def node_holds_row_tokens(node: GeneratedNode, row_tokens: frozenset[str]) -> bool:
    """Report whether a node represents a row, using the same rule as the completeness check."""
    return _node_holds_row(node, row_tokens)


def extract_generated_nodes_by_variable(statements: list[str]) -> dict[str, GeneratedNode]:
    """Read generated nodes back as variable->tokens, merged across all statements.

    Args:
        statements: Generated Cypher statements

    Returns:
        Node token view keyed by variable name
    """
    groups: dict[str, dict[str, list[str]]] = {}

    for statement_index, statement in enumerate(statements):
        for order, node_match in enumerate(NODE_PROPERTY_MAP_RE.finditer(statement)):
            variable = node_match.group("variable") or f"#{statement_index}:{order}"
            properties = groups.setdefault(variable, {TITLE_PROPERTY: [], "values": []})
            for entry in PROPERTY_ENTRY_RE.finditer(node_match.group("properties")):
                _record_property(properties, entry.group("key"), entry.group("value")[1:-1])

        for assignment in SET_ASSIGNMENT_RE.finditer(statement):
            properties = groups.setdefault(
                assignment.group("variable"), {TITLE_PROPERTY: [], "values": []}
            )
            _record_property(properties, assignment.group("key"), assignment.group("value")[1:-1])

    return {
        variable: GeneratedNode(
            title_tokens=_token_set(properties[TITLE_PROPERTY]),
            value_tokens=_token_set(properties["values"]),
        )
        for variable, properties in groups.items()
        if properties["values"]
    }


def rows_missing_from_cypher(rows: list[str], statements: list[str]) -> list[str]:
    """
    Report the rows that did not get a node of their own.

    Partial credit matters: the model is free to reword a row, but a row it never read leaves
    almost none of its wording behind, and a row folded into a parent node's context leaves no
    title of its own.

    Args:
        rows: Rows found on the page
        statements: Generated Cypher statements

    Returns:
        The rows that are not represented by a node of their own
    """
    if not rows:
        return []

    nodes = list(extract_generated_nodes_by_variable(statements).values())

    missing: list[str] = []
    for row in rows:
        tokens = frozenset(_row_tokens(row))
        if not tokens:
            continue
        if not any(node_holds_row_tokens(node, tokens) for node in nodes):
            missing.append(row)

    return missing
