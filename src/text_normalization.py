"""Deterministic text normalization shared by ingestion and retrieval."""

import re
import unicodedata
from collections.abc import Callable

POLISH_DIACRITIC_TRANSLATION = str.maketrans({"ł": "l", "Ł": "L"})
CYPHER_STRING_LITERAL_RE = re.compile(r"'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\"")
LIST_PREFIX_KEYWORDS = {"IN", "RETURN", "WITH", "UNWIND", "AS", "THEN", "ELSE"}
_CYPHER_IDENTIFIER = r"(?:`[^`]+`|[A-Za-z_]\w*)"
_CYPHER_PROPERTY = (
    rf"{_CYPHER_IDENTIFIER}(?:\s*\.\s*{_CYPHER_IDENTIFIER}"
    rf"|\s*\[\s*(?:{CYPHER_STRING_LITERAL_RE.pattern})\s*\])"
)
_LOWERED_PROPERTY = rf"(?:toLower\s*\(\s*{_CYPHER_PROPERTY}\s*\)|{_CYPHER_PROPERTY})"
_LOWERED_LITERAL = (
    rf"(?:toLower\s*\(\s*(?:{CYPHER_STRING_LITERAL_RE.pattern})\s*\)"
    rf"|(?:{CYPHER_STRING_LITERAL_RE.pattern}))"
)
FUZZY_STRING_COMPARISON_RE = re.compile(
    rf"(?P<property>{_LOWERED_PROPERTY})"
    rf"(?P<before_operator>\s+)"
    rf"(?P<operator>CONTAINS|STARTS\s+WITH|ENDS\s+WITH)"
    rf"(?P<after_operator>\s+)"
    rf"(?P<literal>{_LOWERED_LITERAL})",
    re.IGNORECASE,
)


# Polish words that carry no entity of their own. Retrieval refuses to start or end a search
# phrase on any of them. Ingestion uses only the first group: a phrase that ends on a
# preposition or a conjunction was cut in half ("Udzial w"), while one ending on a copula is an
# ordinary Polish lead-in and a legitimate heading ("... kryteriami doboru kandydata sa:").
PREPOSITION_AND_CONJUNCTION_SOURCE = (
    "a",
    "aby",
    "albo",
    "ale",
    "bez",
    "dla",
    "do",
    "i",
    "jako",
    "lub",
    "między",
    "na",
    "nad",
    "o",
    "od",
    "oraz",
    "po",
    "pod",
    "przez",
    "przy",
    "u",
    "w",
    "we",
    "za",
    "z",
    "ze",
    "że",
)
COPULA_AND_PRONOUN_SOURCE = (
    "być",
    "jest",
    "ma",
    "mają",
    "nie",
    "są",
    "się",
    "ta",
    "te",
    "tego",
    "tej",
    "ten",
    "to",
    "tym",
)
FUNCTION_WORD_SOURCE = PREPOSITION_AND_CONJUNCTION_SOURCE + COPULA_AND_PRONOUN_SOURCE


def fold_diacritics(value: str) -> str:
    """Fold Polish and decomposable Unicode diacritics while preserving case."""
    translated = value.translate(POLISH_DIACRITIC_TRANSLATION)
    decomposed = unicodedata.normalize("NFKD", translated)
    return "".join(character for character in decomposed if not unicodedata.combining(character))


def normalize_search_text(value: str) -> str:
    """Return the canonical case- and diacritic-insensitive search representation."""
    return fold_diacritics(value).casefold()


def apply_outside_string_literals(cypher: str, transform: Callable[[str], str]) -> str:
    """Apply a rewrite to Cypher syntax only, preserving quoted values.

    Args:
        cypher: Cypher statement to transform
        transform: Function applied to every non-literal segment

    Returns:
        Cypher with rewritten syntax outside string literals
    """
    pieces: list[str] = []
    cursor = 0

    for literal in CYPHER_STRING_LITERAL_RE.finditer(cypher):
        pieces.append(transform(cypher[cursor : literal.start()]))
        pieces.append(literal.group(0))
        cursor = literal.end()

    pieces.append(transform(cypher[cursor:]))
    return "".join(pieces)


POLISH_FUNCTION_WORDS = frozenset(normalize_search_text(word) for word in FUNCTION_WORD_SOURCE)
# The subset whose presence at the end of a phrase means the phrase is unfinished.
POLISH_PHRASE_CUT_WORDS = frozenset(
    normalize_search_text(word) for word in PREPOSITION_AND_CONJUNCTION_SOURCE
)


def ensure_case_insensitive_fuzzy_matching(cypher: str) -> str:
    """Make human-readable Cypher fuzzy comparisons case-insensitive.

    Exact equality is deliberately left unchanged because the retrieval prompt reserves it for
    stable IDs, whose spelling and case may be significant.
    """

    def lower(expression: str) -> str:
        if re.match(r"toLower\s*\(", expression, re.IGNORECASE):
            return expression
        return f"toLower({expression})"

    def replace_comparison(match: re.Match[str]) -> str:
        return "".join(
            (
                lower(match.group("property")),
                match.group("before_operator"),
                match.group("operator"),
                match.group("after_operator"),
                lower(match.group("literal")),
            )
        )

    return FUZZY_STRING_COMPARISON_RE.sub(replace_comparison, cypher)


def normalize_cypher_string_literals(
    cypher: str,
    *,
    normalizer: Callable[[str], str] = fold_diacritics,
) -> str:
    """Normalize quoted Cypher values while preserving dynamic property keys."""

    def is_dynamic_property_key(match: re.Match[str]) -> bool:
        left = match.start() - 1
        while left >= 0 and cypher[left].isspace():
            left -= 1

        right = match.end()
        while right < len(cypher) and cypher[right].isspace():
            right += 1

        if left < 0 or right >= len(cypher) or cypher[left] != "[" or cypher[right] != "]":
            return False

        prefix = cypher[:left].rstrip()
        preceding_token = re.search(r"([A-Za-z_]\w*)$", prefix)
        if preceding_token and preceding_token.group(1).upper() in LIST_PREFIX_KEYWORDS:
            return False

        return bool(prefix) and (prefix[-1].isalnum() or prefix[-1] in "_)]")

    def replace_literal(match: re.Match[str]) -> str:
        if is_dynamic_property_key(match):
            return match.group(0)

        literal = match.group(0)
        quote = literal[0]
        return f"{quote}{normalizer(literal[1:-1])}{quote}"

    return CYPHER_STRING_LITERAL_RE.sub(replace_literal, cypher)


# A list marker: a bullet glyph, a number, or a letter, with or without a wrapping bracket.
# PDF text layers emit these on a line of their own, detached from the text they introduce.
LIST_MARKER_GLYPHS = "-*•▪◦‣·–—>+"
LIST_MARKER_PATTERN = (
    rf"(?:[{re.escape(LIST_MARKER_GLYPHS)}]"
    rf"|\(?(?:\d{{1,3}}|[a-z]|[ivx]{{1,4}})[.)])"
)
ORPHANED_LIST_MARKER_RE = re.compile(rf"^{LIST_MARKER_PATTERN}$", re.IGNORECASE)
INLINE_LIST_MARKER_RE = re.compile(rf"^{LIST_MARKER_PATTERN}\s+\S", re.IGNORECASE)
# A cell separator makes a line a table row of its own, never the tail of the row above it.
CELL_SEPARATOR_RE = re.compile(r"[|\t]")
# A row that ends on one of these is finished; anything after it starts something new.
SENTENCE_END_CHARACTERS = ".;:!?"


def _starts_a_row(line: str) -> bool:
    """Report whether a line opens a row of its own rather than continuing the one above."""
    return bool(
        ORPHANED_LIST_MARKER_RE.match(line)
        or INLINE_LIST_MARKER_RE.match(line)
        or CELL_SEPARATOR_RE.search(line)
    )


def _collect_row(lines: list[str], start_index: int, *, marker_only: bool) -> tuple[str, int]:
    """Rebuild the row that begins on ``start_index`` from the lines it wraps onto.

    Args:
        lines: All lines of the page
        start_index: Index of the line the row starts on
        marker_only: True when that line holds nothing but the marker, so its text is still to
            come; False when the marker and the start of the text share the line

    Returns:
        The rebuilt row and how many lines it consumed
    """
    head = lines[start_index].strip()
    if not marker_only and head.endswith(tuple(SENTENCE_END_CHARACTERS)):
        return head, 1

    parts: list[str] = []
    index = start_index + 1

    while index < len(lines):
        candidate = lines[index].strip()
        if not candidate or _starts_a_row(candidate):
            break

        parts.append(candidate)
        index += 1

        if candidate.endswith(tuple(SENTENCE_END_CHARACTERS)):
            break

    if not parts:
        return head, 1

    return f"{head} {' '.join(parts)}", index - start_index


def join_wrapped_list_rows(text: str) -> str:
    """Put each list row of a page back on one line.

    A PDF text layer breaks rows in two ways. It writes a bullet and its text as separate lines
    ("•\nprowadzi badania ..."), and it wraps a long row at the page width, whether or not the
    marker shares the first line. Either way the page holds no line that reads as a whole row -
    not for the model that has to turn each row into a node, and not for the completeness check
    that verifies it did. A marker standing alone is put back in front of its text, and the
    lines a row wraps onto are folded into it.

    Folding wrapped lines matters as much as the marker does: an inline-marked row left cut at
    the page width ("3. W grupie pracownikow dydaktycznych (ktorych podstawowym obowiazkiem
    jest ksztalcenie") is a fragment no node will ever carry, so it is reported missing on every
    run and the missed-row pass mints a node out of half a sentence.

    A row ends at a blank line, at the next row, or at sentence-ending punctuation, so the
    paragraph following a list is not swallowed by its last entry.

    Args:
        text: Extracted page text

    Returns:
        The text with every list row on a line of its own
    """
    lines = text.splitlines()
    rejoined: list[str] = []
    index = 0

    while index < len(lines):
        line = lines[index].strip()

        if ORPHANED_LIST_MARKER_RE.match(line):
            row, consumed = _collect_row(lines, index, marker_only=True)
        elif INLINE_LIST_MARKER_RE.match(line):
            row, consumed = _collect_row(lines, index, marker_only=False)
        else:
            rejoined.append(lines[index])
            index += 1
            continue

        rejoined.append(row)
        index += consumed

    return "\n".join(rejoined)
