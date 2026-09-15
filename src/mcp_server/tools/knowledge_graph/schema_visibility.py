"""Keep the pipeline's own bookkeeping out of the schema the Cypher model is shown.

Issue #86: ``Neo4jGraph.refresh_schema()`` describes every label the database holds, so
``ProcessedDocument {hash, status, claimed_at, ...}``, ``Source {source_id}`` and
``PipelineRun {run_at, source_hashes}`` arrive in the Cypher prompt looking exactly like the
entities a question is about. They are not answerable: they record which page was ingested
when, and a question about "dokumenty" answered from them returns ingestion bookkeeping
presented as university data.

``SYSTEM_LABELS`` already draws this line for the full-text index and for the dedup pass. This
draws it for the prompt too, and it does so by removing the labels rather than by asking the
model not to use them: a prompt instruction is something the model can talk itself out of on
any given run, and the schema is the one place where an unavailable label is simply unavailable.

Filtering the rendered text rather than the structured schema keeps the change to the single
line in ``RAG._fetch_schema`` that reads it, and keeps the cache, the emptiness check and the
fallback-to-last-good-schema path working on the one type they already work on.
``tests/test_schema_visibility.py`` builds its input by calling the same
``neo4j_graphrag.schema.format_schema`` the driver calls, so a library that changes the layout
fails there instead of quietly letting the bookkeeping back in.
"""

import logging
import re

from ....config.system_labels import SYSTEM_LABELS

logger = logging.getLogger(__name__)

NODE_SECTION_HEADER = "Node properties:"
REL_PROPERTY_SECTION_HEADER = "Relationship properties:"
RELATIONSHIP_SECTION_HEADER = "The relationships:"

SECTION_HEADERS = (
    NODE_SECTION_HEADER,
    REL_PROPERTY_SECTION_HEADER,
    RELATIONSHIP_SECTION_HEADER,
)

# One entry of a properties section, in either layout the driver emits: `- **Course**` with the
# enhanced schema, `Course {title: STRING}` without it. Indented lines belong to the entry above
# them, so only an unindented line can open a new one.
ENHANCED_ENTRY_RE = re.compile(r"^- \*\*(?P<name>.+?)\*\*\s*$")
COMPACT_ENTRY_RE = re.compile(r"^(?P<name>[^\s{]+)\s*\{")

# `(:Professor)-[:TEACHES]->(:Course)`
RELATIONSHIP_RE = re.compile(
    r"^\(:(?P<start>[^)]+)\)-\[:(?P<type>[^\]]+)\]->\(:(?P<end>[^)]+)\)\s*$"
)


def _entry_name(line: str) -> str | None:
    """Return the label or relationship type a properties line opens, if it opens one."""
    if line[:1].isspace():
        return None
    enhanced = ENHANCED_ENTRY_RE.match(line)
    if enhanced:
        return enhanced.group("name").strip()
    compact = COMPACT_ENTRY_RE.match(line)
    return compact.group("name").strip() if compact else None


def _split_sections(schema: str) -> dict[str, list[str]] | None:
    """
    Cut the schema text into its three sections.

    Returns:
        The lines under each header, or None when the text is not the three-section layout this
        module knows how to read. The caller then leaves the schema alone: an unrecognised
        layout means the filter cannot tell bookkeeping from entities, and a mangled schema
        costs every question an answer.
    """
    sections: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in schema.splitlines():
        if line.strip() in SECTION_HEADERS:
            current = sections.setdefault(line.strip(), [])
            continue
        if current is None:
            return None
        current.append(line)

    return sections if all(header in sections for header in SECTION_HEADERS) else None


def _drop_entries(lines: list[str], unwanted: frozenset[str] | set[str]) -> list[str]:
    """Remove each named entry of a properties section along with the lines it owns."""
    kept: list[str] = []
    dropping = False
    for line in lines:
        name = _entry_name(line)
        if name is not None:
            dropping = name in unwanted
        if not dropping:
            kept.append(line)
    return kept


def hide_system_labels(schema: str) -> str:
    """
    Remove the pipeline's bookkeeping labels from a rendered Neo4j schema.

    Node entries for ``SYSTEM_LABELS`` go, every relationship pattern with one of those labels
    at either end goes, and a relationship type left without a single surviving pattern loses
    its properties entry too - it can no longer be traversed, so describing it only invites the
    model to try.

    A line the layout does not account for is kept: leaving one stray line in beats cutting a
    real entity out of the only description of the graph the model gets.

    Args:
        schema: Schema text as ``Neo4jGraph.get_schema`` renders it

    Returns:
        The same text with the bookkeeping removed, or the input unchanged when it is empty or
        is not the layout this module reads
    """
    if not schema.strip():
        return schema

    sections = _split_sections(schema)
    if sections is None:
        logger.warning(
            "Unrecognised Neo4j schema layout; serving it without hiding %s", sorted(SYSTEM_LABELS)
        )
        return schema

    kept_relationships: list[str] = []
    types_before: set[str] = set()
    types_after: set[str] = set()
    for line in sections[RELATIONSHIP_SECTION_HEADER]:
        pattern = RELATIONSHIP_RE.match(line.strip())
        if pattern is None:
            kept_relationships.append(line)
            continue
        types_before.add(pattern.group("type"))
        if {pattern.group("start"), pattern.group("end")} & SYSTEM_LABELS:
            continue
        types_after.add(pattern.group("type"))
        kept_relationships.append(line)

    filtered = {
        NODE_SECTION_HEADER: _drop_entries(sections[NODE_SECTION_HEADER], SYSTEM_LABELS),
        REL_PROPERTY_SECTION_HEADER: _drop_entries(
            sections[REL_PROPERTY_SECTION_HEADER], types_before - types_after
        ),
        RELATIONSHIP_SECTION_HEADER: kept_relationships,
    }

    rendered: list[str] = []
    for header in SECTION_HEADERS:
        rendered.append(header)
        rendered.extend(filtered[header])
    text = "\n".join(rendered)
    # splitlines() drops the empty string after a final newline, so put it back: a schema
    # holding no bookkeeping at all must come out of here byte for byte as it went in.
    return f"{text}\n" if schema.endswith("\n") else text
