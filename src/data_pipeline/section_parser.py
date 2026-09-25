"""Section parser for heading-plus-rows page shapes."""

import re
from dataclasses import dataclass

from src.data_pipeline.completeness import (
    HEADING_SUFFIX,
    LIST_ROW_RE,
    MIN_ROW_TOKENS,
    MIN_TOKEN_LENGTH,
    TOKEN_RE,
)
from src.text_normalization import (
    LIST_MARKER_GLYPHS,
    LIST_MARKER_PATTERN,
    is_code_token,
    join_wrapped_list_rows,
    normalize_search_text,
)

MAX_CONTEXT_HEADING_TOKENS = 8
HEADING_STAGE_RE = re.compile(r"^R\d+\b", re.IGNORECASE)
HEADING_PREFIX_RE = re.compile(
    rf"^\s*(?:[{re.escape(LIST_MARKER_GLYPHS)}]\s+|\d{{1,3}}[.)]\s+|[a-z][.)]\s+"
    r"|\(\s*(?:\d{1,3}|[a-z]|[ivx]{1,4})\s*\)\s+)",
    re.IGNORECASE,
)
SECTION_ROW_RE = re.compile(
    rf"^\s*(?P<marker>{LIST_MARKER_PATTERN})\s*(?P<content>\S.*?)\s*$", re.IGNORECASE
)
QUALIFIED_HEADING_RE = re.compile(
    r"\b(?:kompetencj\w*|kryteri\w*)\b.*\b(?:pozadan\w*|wymagan\w*|niezbedn\w*)\b",
    re.IGNORECASE,
)
EMBEDDED_HEADING_SPLIT_RE = re.compile(
    r"^(?P<row>.+?)\s+(?P<heading>(?:Kompetencje|Kryteria)\s+.+)$", re.IGNORECASE
)
EMBEDDED_STAGE_HEADING_SPLIT_RE = re.compile(
    r"^(?P<row>.+?)\s+(?P<heading>R\d+\s*-\s+\S.+)$", re.IGNORECASE
)


@dataclass(frozen=True)
class HeadingRowSection:
    """A section heading and the list rows it introduces."""

    heading: str
    rows: tuple[str, ...]
    context_headings: tuple[str, ...] = ()


def extract_heading_row_sections(text: str) -> list[HeadingRowSection]:
    """Collect heading->rows sections from list-shaped page fragments."""
    sections: list[HeadingRowSection] = []
    heading_stack: list[tuple[int, str]] = []
    current_heading: str | None = None
    current_rows: list[str] = []
    current_context: tuple[str, ...] = ()

    for raw_line in join_wrapped_list_rows(text).splitlines():
        line = " ".join(raw_line.split()).strip()
        if not line:
            if current_heading is not None and current_rows:
                sections.append(
                    HeadingRowSection(
                        heading=current_heading,
                        rows=tuple(current_rows),
                        context_headings=current_context,
                    )
                )
                current_heading = None
                current_rows = []
                current_context = ()
            continue

        row = _extract_section_row_content(line)
        if row is not None:
            if current_heading is not None:
                row_content, embedded_heading = _split_embedded_heading(row)
                if row_content:
                    current_rows.append(row_content)
                if embedded_heading:
                    embedded_level = _heading_level(embedded_heading, heading_stack)
                    sections.append(
                        HeadingRowSection(
                            heading=current_heading,
                            rows=tuple(current_rows),
                            context_headings=current_context,
                        )
                    )
                    normalized_embedded = _normalize_heading(embedded_heading)
                    _push_heading(heading_stack, normalized_embedded, embedded_level)
                    current_heading = normalized_embedded
                    current_context = tuple(text for _, text in heading_stack[:-1])
                    current_rows = []
            continue

        if _is_heading_candidate(line):
            normalized_heading = _normalize_heading(line)
            _push_heading(heading_stack, normalized_heading, _heading_level(line, heading_stack))
            if current_heading is not None and current_rows:
                sections.append(
                    HeadingRowSection(
                        heading=current_heading,
                        rows=tuple(current_rows),
                        context_headings=current_context,
                    )
                )
            current_heading = normalized_heading
            current_context = tuple(text for _, text in heading_stack[:-1])
            current_rows = []
            continue

        if _looks_like_context_heading(line):
            _push_heading(
                heading_stack,
                _normalize_heading(line),
                _heading_level(line, heading_stack),
            )

        if current_heading is not None and current_rows:
            sections.append(
                HeadingRowSection(
                    heading=current_heading,
                    rows=tuple(current_rows),
                    context_headings=current_context,
                )
            )
            current_heading = None
            current_rows = []
            current_context = ()

    if current_heading is not None and current_rows:
        sections.append(
            HeadingRowSection(
                heading=current_heading,
                rows=tuple(current_rows),
                context_headings=current_context,
            )
        )

    return sections


def _row_tokens(row: str) -> list[str]:
    return [
        token
        for token in TOKEN_RE.findall(normalize_search_text(row))
        if len(token) >= MIN_TOKEN_LENGTH
    ]


def _extract_section_row_content(line: str) -> str | None:
    row_match = SECTION_ROW_RE.match(line)
    if row_match is not None:
        content = row_match.group("content").replace("|", " ").replace("\t", " ").strip()
        if len(_row_tokens(content.rstrip(HEADING_SUFFIX).strip())) < MIN_ROW_TOKENS:
            return None
        return content

    match = LIST_ROW_RE.match(line)
    if match is None:
        return None
    cells = (match.group("cells") or "").replace("|", " ").replace("\t", " ").strip()
    if len(_row_tokens(cells)) < MIN_ROW_TOKENS:
        return None
    return cells


def _is_heading_candidate(line: str) -> bool:
    if line.endswith(HEADING_SUFFIX) and SECTION_ROW_RE.match(line) is None:
        return True
    if QUALIFIED_HEADING_RE.search(normalize_search_text(line)):
        return True
    return HEADING_STAGE_RE.match(line) is not None


def _normalize_heading(line: str) -> str:
    normalized = HEADING_PREFIX_RE.sub("", line).strip()
    if normalized.endswith(HEADING_SUFFIX):
        normalized = normalized[: -len(HEADING_SUFFIX)].rstrip()
    return normalized


def _heading_level(line: str, stack: list[tuple[int, str]]) -> int:
    normalized = normalize_search_text(line)
    if HEADING_STAGE_RE.match(line) is not None:
        return 2
    if QUALIFIED_HEADING_RE.search(normalized):
        stage_code = _nearest_stage_code(stack)
        heading_codes = _code_tokens(line)
        if stage_code and heading_codes == {stage_code}:
            return 3
        return 1
    if line.endswith(HEADING_SUFFIX):
        return 3
    return 0


def _push_heading(stack: list[tuple[int, str]], heading: str, level: int) -> None:
    if level <= 0:
        if stack and stack[0][0] == 0:
            stack[0] = (0, heading)
        else:
            stack.insert(0, (0, heading))
        return

    while stack and stack[-1][0] >= level:
        stack.pop()
    stack.append((level, heading))


def _nearest_stage_code(stack: list[tuple[int, str]]) -> str | None:
    for level, heading in reversed(stack):
        if level != 2:
            continue
        codes = _code_tokens(heading)
        if len(codes) == 1:
            return next(iter(codes))
    return None


def _code_tokens(text: str) -> set[str]:
    return {token for token in _row_tokens(text) if is_code_token(token)}


def _split_embedded_heading(row: str) -> tuple[str, str | None]:
    for regex in (EMBEDDED_HEADING_SPLIT_RE, EMBEDDED_STAGE_HEADING_SPLIT_RE):
        match = regex.match(row)
        if match is None:
            continue
        heading = match.group("heading").strip()
        if regex is EMBEDDED_HEADING_SPLIT_RE and not QUALIFIED_HEADING_RE.search(
            normalize_search_text(heading)
        ):
            continue
        return match.group("row").rstrip(), heading
    return row, None


def _looks_like_context_heading(line: str) -> bool:
    if _extract_section_row_content(line) is not None:
        return False
    stripped = line.strip()
    if stripped.endswith((".", "!", "?")):
        return False
    token_count = len(_row_tokens(stripped))
    return MIN_ROW_TOKENS <= token_count <= MAX_CONTEXT_HEADING_TOKENS
