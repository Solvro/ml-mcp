"""Give every extracted entity one canonical node instead of one node per mention.

Issue #53: nodes were merged on ``title + context`` together, so the same entity described
slightly differently on two pages became two nodes — a Semester with the dates and a Semester
without them, "Cyberbezpieczenstwo" beside "Cyberbezpieczenstwo (CBE)". Retrieval then returns
whichever one the query happens to match.

Merging on a normalized key derived from the title alone collapses those. The extra properties
are applied afterwards with ``ON CREATE SET`` / ``ON MATCH SET``, so a later mention enriches the
node it already created rather than sitting next to it.
"""

import re

from src.text_normalization import CYPHER_STRING_LITERAL_RE, normalize_search_text

# One MERGE of a single node with a property map, which is the shape the extraction prompt asks
# for. Anything else (relationship MERGE, combined pattern) is deliberately left alone.
SINGLE_NODE_MERGE_RE = re.compile(
    r"^\s*MERGE\s*\(\s*(?P<variable>[A-Za-z_]\w*)\s*"
    r"(?P<labels>(?::\s*(?:`[^`]+`|[A-Za-z_]\w*)\s*)+)"
    r"\{(?P<properties>[^{}]*)\}\s*\)\s*$",
    re.IGNORECASE | re.DOTALL,
)

# A trailing abbreviation in brackets is a spelling of the same entity, not a different one.
PARENTHETICAL_SUFFIX_RE = re.compile(r"\s*\([^()]*\)\s*$")
NON_KEY_CHARACTERS_RE = re.compile(r"[^0-9a-z]+")

KEY_PROPERTY = "key"
TITLE_PROPERTY = "title"
CONTEXT_PROPERTY = "context"

# Guards the appended context against unbounded growth as more pages mention the same entity.
MAX_CONTEXT_LENGTH = 2000


def canonical_entity_key(title: str) -> str:
    """
    Derive the merge key that two spellings of one entity have in common.

    Case, Polish diacritics, punctuation and a trailing bracketed abbreviation are all dropped,
    so "Cyberbezpieczeństwo" and "Cyberbezpieczenstwo (CBE)" produce the same key.

    Args:
        title: Entity title as the extraction model wrote it

    Returns:
        Lowercase ASCII key, or an empty string when the title carries no usable characters
    """
    folded = normalize_search_text(title)
    without_abbreviation = PARENTHETICAL_SUFFIX_RE.sub("", folded).strip()
    return NON_KEY_CHARACTERS_RE.sub(" ", without_abbreviation or folded).strip()


def _split_properties(properties: str) -> list[tuple[str, str]]:
    """Split a Cypher property map into (name, raw value) pairs, ignoring commas in strings."""
    spans: list[tuple[int, int]] = [
        literal.span() for literal in CYPHER_STRING_LITERAL_RE.finditer(properties)
    ]

    def inside_literal(index: int) -> bool:
        return any(start <= index < end for start, end in spans)

    entries: list[str] = []
    start = 0
    for index, character in enumerate(properties):
        if character == "," and not inside_literal(index):
            entries.append(properties[start:index])
            start = index + 1
    entries.append(properties[start:])

    pairs: list[tuple[str, str]] = []
    for entry in entries:
        stripped = entry.strip()
        if not stripped:
            continue
        name, separator, value = stripped.partition(":")
        if not separator:
            continue
        pairs.append((name.strip().strip("`"), value.strip()))
    return pairs


def _quoted_value(raw_value: str) -> str | None:
    """Return the text of a quoted Cypher value, or None when the value is not a string."""
    match = CYPHER_STRING_LITERAL_RE.fullmatch(raw_value.strip())
    if match is None:
        return None
    return match.group(0)[1:-1]


def rewrite_merge_to_canonical_key(statement: str) -> str:
    """
    Rewrite a node MERGE so it keys on the canonical entity key.

    ``MERGE (n:Course {title: 'X', context: 'Y'})`` becomes a MERGE on the key with the title and
    context applied afterwards. A second mention of the same entity then matches the existing
    node: the more specific title wins and a new context is appended rather than starting a
    second node.

    Statements that are not a single-node MERGE with a property map — relationship MERGEs,
    combined patterns — are returned unchanged.

    Args:
        statement: One generated Cypher statement

    Returns:
        The rewritten statement, or the original when it cannot be rewritten safely
    """
    match = SINGLE_NODE_MERGE_RE.match(statement)
    if match is None:
        return statement

    properties = _split_properties(match.group("properties"))
    if not properties:
        return statement

    property_values = dict(properties)
    title_literal = property_values.get(TITLE_PROPERTY)
    if title_literal is None:
        return statement

    title_text = _quoted_value(title_literal)
    if title_text is None:
        return statement

    key = canonical_entity_key(title_text)
    if not key:
        return statement

    variable = match.group("variable")
    labels = match.group("labels").strip()

    create_assignments = [f"{variable}.{name} = {value}" for name, value in properties]
    match_assignments = [
        # A later page may name the entity more fully; the fuller title is the better one.
        f"{variable}.{TITLE_PROPERTY} = CASE "
        f"WHEN size({title_literal}) > size(coalesce({variable}.{TITLE_PROPERTY}, '')) "
        f"THEN {title_literal} ELSE {variable}.{TITLE_PROPERTY} END"
    ]

    context_literal = property_values.get(CONTEXT_PROPERTY)
    if context_literal is not None and _quoted_value(context_literal) is not None:
        match_assignments.append(
            # Keep both descriptions instead of letting the second mention overwrite the first,
            # which is what left one Semester holding the dates and another holding nothing.
            f"{variable}.{CONTEXT_PROPERTY} = CASE "
            f"WHEN coalesce({variable}.{CONTEXT_PROPERTY}, '') CONTAINS {context_literal} "
            f"THEN {variable}.{CONTEXT_PROPERTY} "
            f"WHEN size(coalesce({variable}.{CONTEXT_PROPERTY}, '')) + size({context_literal}) "
            f"> {MAX_CONTEXT_LENGTH} THEN {variable}.{CONTEXT_PROPERTY} "
            f"WHEN coalesce({variable}.{CONTEXT_PROPERTY}, '') = '' THEN {context_literal} "
            f"ELSE {variable}.{CONTEXT_PROPERTY} + ' | ' + {context_literal} END"
        )

    for name, value in properties:
        if name in (TITLE_PROPERTY, CONTEXT_PROPERTY):
            continue
        match_assignments.append(f"{variable}.{name} = coalesce({variable}.{name}, {value})")

    return (
        f"MERGE ({variable}{labels} {{{KEY_PROPERTY}: '{key}'}}) "
        f"ON CREATE SET {', '.join(create_assignments)} "
        f"ON MATCH SET {', '.join(match_assignments)}"
    )
