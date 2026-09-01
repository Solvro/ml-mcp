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

# A trailing bracket may be an abbreviation of the same entity — "Cyberbezpieczenstwo (CBE)" —
# or the very thing that tells two entities apart: "Informatyka (studia I stopnia)" against
# "(studia II stopnia)", "(stacjonarne)" against "(niestacjonarne)". Only the first kind may be
# dropped from the key. Fusing two entities into one is worse than splitting one into two,
# because the loss is silent.
PARENTHETICAL_SUFFIX_RE = re.compile(r"\s*\((?P<inner>[^()]*)\)\s*$")
# Roman numerals mark a degree level or an edition, never an abbreviation of the name.
ROMAN_NUMERAL_RE = re.compile(r"^[IVXLCDM]+$")
MAX_ABBREVIATION_LENGTH = 8
MIN_ABBREVIATION_UPPERCASE = 2
NON_KEY_CHARACTERS_RE = re.compile(r"[^0-9a-z]+")

KEY_PROPERTY = "key"
# Reads back the key a rewritten MERGE settled on. Keys are folded to [0-9a-z ] before they
# are written, so the literal can never contain a quote to escape past.
MERGE_KEY_RE = re.compile(r"\{\s*key\s*:\s*'(?P<key>[^']*)'\s*\}")
TITLE_PROPERTY = "title"
CONTEXT_PROPERTY = "context"

# Guards the appended context against unbounded growth as more pages mention the same entity.
MAX_CONTEXT_LENGTH = 2000
# Separates appended contexts. Must not contain a pipe: the whole ingestion path splits
# statements on "|", so a pipe inside a generated literal would tear the statement in half.
CONTEXT_SEPARATOR = "; "


def looks_like_abbreviation(bracketed_text: str) -> bool:
    """
    Report whether a bracketed suffix abbreviates the title rather than qualifying it.

    An abbreviation is one short token carrying capitals — "(CBE)", "(PWr)". Anything that
    describes a variant of the entity — "(studia I stopnia)", "(stacjonarne)", "(II)" — is what
    distinguishes two entities and must stay in the key.

    Args:
        bracketed_text: Text between the brackets, with its original case

    Returns:
        True when the suffix can be dropped without merging distinct entities
    """
    token = bracketed_text.strip()
    if not token or len(token) > MAX_ABBREVIATION_LENGTH:
        return False
    if not token.isalnum():
        return False
    if ROMAN_NUMERAL_RE.match(token):
        return False

    uppercase = sum(1 for character in token if character.isupper())
    if uppercase >= MIN_ABBREVIATION_UPPERCASE:
        return True
    # Faculty codes carry a single capital and a number: "(W8)", "(W4)".
    return uppercase == 1 and any(character.isdigit() for character in token)


def canonical_entity_key(title: str) -> str:
    """
    Derive the merge key that two spellings of one entity have in common.

    Case, Polish diacritics and punctuation are dropped, so "Cyberbezpieczeństwo" and
    "Cyberbezpieczenstwo" produce the same key. A trailing bracket is dropped only when it
    abbreviates the title — "Cyberbezpieczenstwo (CBE)" joins them — and kept when it tells two
    entities apart, so "Informatyka (studia I stopnia)" and "Informatyka (studia II stopnia)"
    stay two nodes.

    Args:
        title: Entity title as the extraction model wrote it

    Returns:
        Lowercase ASCII key, or an empty string when the title carries no usable characters
    """
    suffix = PARENTHETICAL_SUFFIX_RE.search(title)
    without_abbreviation = title
    if suffix is not None and looks_like_abbreviation(suffix.group("inner")):
        without_abbreviation = title[: suffix.start()]

    folded = normalize_search_text(without_abbreviation)
    key = NON_KEY_CHARACTERS_RE.sub(" ", folded).strip()
    if key:
        return key

    # A title that is nothing but the abbreviation still needs a key of its own.
    return NON_KEY_CHARACTERS_RE.sub(" ", normalize_search_text(title)).strip()


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
            f"ELSE {variable}.{CONTEXT_PROPERTY} + '{CONTEXT_SEPARATOR}' + {context_literal} END"
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


def extract_entity_keys(cypher: str) -> list[str]:
    """
    Collect the canonical keys a batch of generated statements merges on.

    The post-ingest repair uses these to look at only what a run touched, instead of scanning
    every node in the graph each time.

    Args:
        cypher: One or more generated statements, pipe-separated or already joined

    Returns:
        Deduplicated keys in the order they appear
    """
    keys: list[str] = []
    seen: set[str] = set()

    for match in MERGE_KEY_RE.finditer(cypher or ""):
        key = match.group("key")
        if key and key not in seen:
            seen.add(key)
            keys.append(key)

    return keys
