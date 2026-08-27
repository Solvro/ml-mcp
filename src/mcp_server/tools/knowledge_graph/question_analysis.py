"""Deterministic Polish question analysis used to repair Text2Cypher retrieval.

Two Text2Cypher failures produce an empty result on an otherwise valid query:

* the model copies the whole question into a ``CONTAINS`` literal, which matches no stored
  title (``CONTAINS 'jakie sa kryteria doboru kandydatki lub kandydata'``);
* the model picks one of several overlapping node labels, so the traversal starts from the
  wrong label even though the answer is in the graph under another one.

Prompt wording alone only reduces the frequency of both. The helpers here let the retrieval
step recognise a copied question and rebuild a label-agnostic search from the question itself,
so the repair happens the same way on every run.
"""

import re

from ....text_normalization import (
    CYPHER_STRING_LITERAL_RE,
    FUZZY_STRING_COMPARISON_RE,
    normalize_search_text,
)

# Interrogatives and question-shaped imperatives. A stored title practically never contains
# one, so their presence in a literal marks the literal as question text.
QUESTION_WORD_SOURCE = (
    "jaki",
    "jaka",
    "jakie",
    "jaką",
    "jakiego",
    "jakiej",
    "jakim",
    "jakich",
    "jakimi",
    "który",
    "która",
    "które",
    "którego",
    "której",
    "którym",
    "których",
    "którzy",
    "kto",
    "kogo",
    "komu",
    "kim",
    "co",
    "czego",
    "czemu",
    "czym",
    "czy",
    "gdzie",
    "dokąd",
    "skąd",
    "kiedy",
    "odkąd",
    "jak",
    "ile",
    "ilu",
    "dlaczego",
    "podaj",
    "wymień",
    "opisz",
    "wyjaśnij",
    "powiedz",
    "pokaż",
)

# Words that must not start or end a search phrase. Interior occurrences are kept, so
# "udzial w konferencjach" survives while "udzial w" does not.
FUNCTION_WORD_SOURCE = (
    "a",
    "aby",
    "albo",
    "ale",
    "bez",
    "być",
    "dla",
    "do",
    "i",
    "jest",
    "jako",
    "lub",
    "ma",
    "mają",
    "między",
    "na",
    "nad",
    "nie",
    "o",
    "od",
    "oraz",
    "po",
    "pod",
    "przez",
    "przy",
    "są",
    "się",
    "ta",
    "te",
    "tego",
    "tej",
    "ten",
    "to",
    "tym",
    "u",
    "w",
    "we",
    "za",
    "z",
    "ze",
    "że",
)

QUESTION_WORDS = frozenset(normalize_search_text(word) for word in QUESTION_WORD_SOURCE)
PHRASE_BOUNDARY_WORDS = QUESTION_WORDS | frozenset(
    normalize_search_text(word) for word in FUNCTION_WORD_SOURCE
)

SEARCH_TOKEN_RE = re.compile(r"[0-9a-z]+")

# A literal with no interrogative in it still counts as copied question text when it repeats a
# long contiguous run of the question verbatim.
COPIED_SPAN_MIN_TOKENS = 5
# Entity names in this graph are short noun phrases; longer spans are question text.
MAX_PHRASE_TOKENS = 4
# One-word phrases are only specific enough to search on when the word is reasonably long.
MIN_SINGLE_TOKEN_LENGTH = 5
MAX_SEARCH_PHRASES = 24


def tokenize_search_text(text: str) -> list[str]:
    """
    Split text into case- and diacritic-folded ASCII search tokens.

    Args:
        text: Arbitrary text, typically a question or a Cypher string literal

    Returns:
        Lowercase ASCII tokens with punctuation and diacritics removed
    """
    return SEARCH_TOKEN_RE.findall(normalize_search_text(text))


def _quoted_value(expression: str) -> str:
    """Return the text inside the first quoted literal of a Cypher expression."""
    match = CYPHER_STRING_LITERAL_RE.search(expression)
    if match is None:
        return ""
    return match.group(0)[1:-1]


def _contains_span(haystack: list[str], needle: list[str]) -> bool:
    """Report whether needle appears as a contiguous run of tokens inside haystack."""
    if not needle or len(needle) > len(haystack):
        return False
    return any(
        haystack[start : start + len(needle)] == needle
        for start in range(len(haystack) - len(needle) + 1)
    )


def is_question_like_literal(literal: str, user_question: str) -> bool:
    """
    Report whether a Cypher string literal holds question text instead of an entity name.

    Args:
        literal: Literal text without quotes, or the surrounding comparison expression
        user_question: The question the query was generated from

    Returns:
        True when the literal cannot plausibly match a stored entity name
    """
    tokens = tokenize_search_text(literal)
    if not tokens:
        return False
    if any(token in QUESTION_WORDS for token in tokens):
        return True
    return len(tokens) >= COPIED_SPAN_MIN_TOKENS and _contains_span(
        tokenize_search_text(user_question), tokens
    )


def strip_question_literal_filters(cypher: str, user_question: str) -> tuple[str, list[str]]:
    """
    Neutralize fuzzy predicates whose literal is question text rather than an entity name.

    The predicate is replaced by ``true`` instead of being deleted, so the surrounding boolean
    structure (``WHERE``, ``AND``, ``OR``, parentheses) stays valid without re-parsing the
    clause. What remains is the traversal the model wrote, unfiltered.

    Args:
        cypher: Generated Cypher query
        user_question: The question the query was generated from

    Returns:
        Tuple of the rewritten query and the literal values that were dropped
    """
    dropped: list[str] = []

    def replace_comparison(match: re.Match[str]) -> str:
        value = _quoted_value(match.group("literal"))
        if not is_question_like_literal(value, user_question):
            return match.group(0)
        dropped.append(value)
        return "true"

    return FUZZY_STRING_COMPARISON_RE.sub(replace_comparison, cypher), dropped


def extract_search_phrases(
    user_question: str,
    *,
    max_phrase_tokens: int = MAX_PHRASE_TOKENS,
    max_phrases: int = MAX_SEARCH_PHRASES,
) -> list[str]:
    """
    Build label-agnostic search phrases from a question, longest phrase first.

    Phrases never start or end with a question word or a function word, so
    "Co obejmuje udział w konferencjach?" yields "udzial w konferencjach" — the stored title —
    while never yielding the truncated "udzial w".

    Args:
        user_question: User's natural language question
        max_phrase_tokens: Longest phrase to emit, in tokens
        max_phrases: Cap on the number of phrases returned

    Returns:
        Deduplicated phrases ordered from most to least specific
    """
    tokens = tokenize_search_text(user_question)
    phrases: list[str] = []
    seen: set[str] = set()

    for length in range(min(max_phrase_tokens, len(tokens)), 0, -1):
        for start in range(len(tokens) - length + 1):
            span = tokens[start : start + length]
            if span[0] in PHRASE_BOUNDARY_WORDS or span[-1] in PHRASE_BOUNDARY_WORDS:
                continue
            if length == 1 and len(span[0]) < MIN_SINGLE_TOKEN_LENGTH:
                continue
            phrase = " ".join(span)
            if phrase in seen:
                continue
            seen.add(phrase)
            phrases.append(phrase)

    return phrases[:max_phrases]
