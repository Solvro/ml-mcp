"""Deterministic text normalization shared by ingestion and retrieval."""

import re
import unicodedata
from collections.abc import Callable

POLISH_DIACRITIC_TRANSLATION = str.maketrans({"ł": "l", "Ł": "L"})
CYPHER_STRING_LITERAL_RE = re.compile(r"'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\"")
LIST_PREFIX_KEYWORDS = {"IN", "RETURN", "WITH", "UNWIND", "AS", "THEN", "ELSE"}


def fold_diacritics(value: str) -> str:
    """Fold Polish and decomposable Unicode diacritics while preserving case."""
    translated = value.translate(POLISH_DIACRITIC_TRANSLATION)
    decomposed = unicodedata.normalize("NFKD", translated)
    return "".join(character for character in decomposed if not unicodedata.combining(character))


def normalize_search_text(value: str) -> str:
    """Return the canonical case- and diacritic-insensitive search representation."""
    return fold_diacritics(value).casefold()


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
