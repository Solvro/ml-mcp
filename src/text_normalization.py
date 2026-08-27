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


def fold_diacritics(value: str) -> str:
    """Fold Polish and decomposable Unicode diacritics while preserving case."""
    translated = value.translate(POLISH_DIACRITIC_TRANSLATION)
    decomposed = unicodedata.normalize("NFKD", translated)
    return "".join(character for character in decomposed if not unicodedata.combining(character))


def normalize_search_text(value: str) -> str:
    """Return the canonical case- and diacritic-insensitive search representation."""
    return fold_diacritics(value).casefold()


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
