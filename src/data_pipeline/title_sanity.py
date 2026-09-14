"""Keep a node's title the name of an entity, not the line it was copied from.

Issue #79: the graph held nodes titled ``a) doswiadczenie w kierowaniu i pracy w zespolach
naukowych.``, ``b) ksiazek,``, ``Odbyte szkolenia:``, ``Udzial w``, ``zagranicznych``. Three
different faults, one symptom:

* **The enumerator is not part of the name.** It marks the row's position on the page, so
  ``a) doswiadczenie w organizowaniu ...`` and ``doswiadczenie w organizowaniu ...`` keyed as
  two entities and the graph kept both.
* **Trailing punctuation is not part of the name either.** The key already dropped it; the
  stored title, which is what a reader sees, kept it.
* **A fragment is not a name at all.** ``Udzial w`` was cut mid-phrase and ``zagranicznych`` is
  one inflected adjective lifted out of a row. Neither can be found by anyone asking a question,
  and both sit in the graph looking like entities.

The prompt asks for clean titles and mostly gets them. What is left is corrected here, where the
rule is deterministic and the result is the same on every run.
"""

import re
from dataclasses import dataclass, field

from src.text_normalization import (
    CYPHER_STRING_LITERAL_RE,
    POLISH_FUNCTION_WORDS,
    normalize_search_text,
)

LIST_MARKER_GLYPHS = "-*•▪◦‣·–—>+"

# The enumerator a row carries on the page: a bullet glyph, "a)", "(iii)", "1.", "2)".
#
# Deliberately narrower than a line-level list marker. Deciding that a *line* is a list row
# costs one extra node when it is wrong; deciding that a leading token is an enumerator rewrites
# an entity's name and its merge key, and a wrong rewrite here fuses two entities. So a single
# letter followed by a full stop is not an enumerator: "W. Kowalski" is an initial, and dropping
# it would file two people under one node. A letter needs a bracket, a digit does not.
TITLE_ENUMERATOR_RE = re.compile(
    rf"^\s*(?:[{re.escape(LIST_MARKER_GLYPHS)}]"
    r"|\(\s*(?:\d{1,3}|[a-z]|[ivx]{1,4})\s*\)"
    r"|(?:\d{1,3}|[a-z]|[ivx]{1,4})\)"
    r"|\d{1,3}\."
    r")\s+",
    re.IGNORECASE,
)
# Punctuation that ends the line a title was read from, never the title.
TRAILING_PUNCTUATION = ":;,"
SENTENCE_END = "."
TOKEN_RE = re.compile(r"[0-9a-z]+")
# Tokens shorter than this are prepositions and initials; they carry no name.
MIN_TOKEN_LENGTH = 2
# Below this many tokens a title is only a name when it is a code or a capitalised noun.
MIN_TITLE_TOKENS = 2
# Course and competence codes are single tokens that mean something: R1, W4, K2A_W08.
ENTITY_CODE_RE = re.compile(r"^(?=.*\d)[a-z][a-z0-9_]{1,7}$")

# One node MERGE with a property map, which is the shape the extraction prompt asks for.
NODE_PATTERN_RE = re.compile(
    r"\(\s*(?P<variable>[A-Za-z_]\w*)\s*"
    r"(?P<labels>(?::\s*(?:`[^`]+`|[A-Za-z_]\w*)\s*)+)"
    r"\{(?P<properties>[^{}]*)\}\s*\)"
)
TITLE_ENTRY_RE = re.compile(
    rf"(?P<prefix>\btitle\s*:\s*)(?P<value>{CYPHER_STRING_LITERAL_RE.pattern})",
    re.IGNORECASE,
)

REASON_EMPTY = "empty after cleaning"
REASON_TRUNCATED = "cut off mid-phrase"
REASON_FRAGMENT = "not a name, a fragment of one"


@dataclass
class TitleSanityReport:
    """What the pass changed, for the run log."""

    cleaned: list[tuple[str, str]] = field(default_factory=list)
    rejected: list[tuple[str, str]] = field(default_factory=list)
    dropped_statements: int = 0

    @property
    def changed(self) -> bool:
        """True when the pass had anything to correct."""
        return bool(self.cleaned or self.rejected)


def clean_title(title: str) -> str:
    """
    Strip what the page's layout left on a title.

    The leading enumerator and trailing punctuation are removed, and internal whitespace is
    collapsed. Nested enumerators ("a) 1) ...") are stripped in turn. A full stop is kept when
    it terminates an abbreviation rather than a sentence, so "2 XI 2026 r." does not become
    "2 XI 2026 r".

    Args:
        title: Title as the extraction model wrote it

    Returns:
        The title without the page furniture, possibly empty
    """
    cleaned = " ".join(title.split())

    while True:
        without_enumerator = TITLE_ENUMERATOR_RE.sub("", cleaned, count=1)
        if without_enumerator == cleaned:
            break
        cleaned = without_enumerator

    while cleaned:
        if cleaned[-1] in TRAILING_PUNCTUATION:
            cleaned = cleaned[:-1].rstrip()
            continue
        if cleaned[-1] == SENTENCE_END and _ends_a_sentence(cleaned):
            cleaned = cleaned[:-1].rstrip()
            continue
        break

    return cleaned


def _ends_a_sentence(title: str) -> bool:
    """Report whether a trailing full stop closes a sentence rather than an abbreviation.

    An abbreviation is short and keeps its stop: "r.", "ul.", "nr.". A whole word before the
    stop ends a sentence copied off the page, and the stop goes.
    """
    words = title[:-1].split()
    return bool(words) and len(words[-1]) > MIN_TOKEN_LENGTH


def _title_tokens(title: str) -> list[str]:
    """Return the tokens of a title that carry a name."""
    return [
        token
        for token in TOKEN_RE.findall(normalize_search_text(title))
        if len(token) >= MIN_TOKEN_LENGTH
    ]


def title_rejection_reason(title: str) -> str | None:
    """
    Report why a title is not usable as an entity name, or None when it is.

    Two shapes are refused. A title ending on a Polish function word was cut mid-phrase, which
    is how "Udzial w" reached the graph. A title carrying one token is a name only when it is a
    code (R1, W4) or a capitalised noun - "Informatyka" is an entity, "zagranicznych" is the
    tail of a row.

    The issue proposed refusing every title under two tokens. That would also delete
    "Informatyka", "Rektor" and "Dziekanat", so the capitalised single noun is kept: an entity
    name is written as one, a fragment lifted from mid-phrase is not.

    Args:
        title: Cleaned title

    Returns:
        A short reason for the log, or None when the title names something
    """
    if not title:
        return REASON_EMPTY

    words = title.split()
    if normalize_search_text(words[-1]) in POLISH_FUNCTION_WORDS:
        return REASON_TRUNCATED

    tokens = _title_tokens(title)
    if len(tokens) >= MIN_TITLE_TOKENS:
        return None
    if tokens and ENTITY_CODE_RE.match(tokens[0]):
        return None
    if title[0].isupper():
        return None

    return REASON_FRAGMENT


def sanitize_titles(statements: list[str]) -> tuple[list[str], TitleSanityReport]:
    """
    Clean the titles a generation pass produced, and drop the nodes that have no name.

    A dropped node takes with it every statement that names its variable: the pipeline runs a
    page's statements as one query, so a relationship left pointing at a variable nothing binds
    would fail the whole page rather than the one node.

    Args:
        statements: Generated Cypher statements

    Returns:
        The surviving statements and a report of what was corrected
    """
    report = TitleSanityReport()
    rejected_variables: set[str] = set()
    rewritten: list[str] = []

    for statement in statements:
        updated, rejected = _sanitize_statement(statement, report)
        rejected_variables.update(rejected)
        rewritten.append(updated)

    if not rejected_variables:
        return rewritten, report

    kept: list[str] = []
    for statement in rewritten:
        if _references_any(statement, rejected_variables):
            report.dropped_statements += 1
            continue
        kept.append(statement)

    return kept, report


def _sanitize_statement(statement: str, report: TitleSanityReport) -> tuple[str, set[str]]:
    """Rewrite the titles in one statement and report the variables whose title was refused."""
    rejected: set[str] = set()
    result: list[str] = []
    end_of_previous = 0

    for node in NODE_PATTERN_RE.finditer(statement):
        title_entry = TITLE_ENTRY_RE.search(node.group("properties"))
        if title_entry is None:
            continue

        literal = title_entry.group("value")
        title = literal[1:-1]
        cleaned = clean_title(title)

        reason = title_rejection_reason(cleaned)
        if reason is not None:
            report.rejected.append((title, reason))
            rejected.add(node.group("variable"))
            continue

        if cleaned == title:
            continue

        report.cleaned.append((title, cleaned))
        quote = literal[0]
        start = node.start("properties") + title_entry.start("value")
        result.append(statement[end_of_previous:start])
        result.append(f"{quote}{cleaned}{quote}")
        end_of_previous = node.start("properties") + title_entry.end("value")

    if not result:
        return statement, rejected

    result.append(statement[end_of_previous:])
    return "".join(result), rejected


def _references_any(statement: str, variables: set[str]) -> bool:
    """Report whether a statement names any of these variables."""
    return any(
        re.search(rf"(?<![A-Za-z0-9_]){re.escape(variable)}(?![A-Za-z0-9_])", statement)
        for variable in variables
    )
