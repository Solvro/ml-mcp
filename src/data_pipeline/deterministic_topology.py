import re
from dataclasses import dataclass

from src.config.config import get_config
from src.config.relationship_qualifiers import get_relationship_qualifier_rules
from src.data_pipeline.completeness import (
    LIST_ROW_RE,
    NODE_PROPERTY_MAP_RE,
    PROPERTY_ENTRY_RE,
    SET_ASSIGNMENT_RE,
    TITLE_PROPERTY,
)
from src.text_normalization import CYPHER_STRING_LITERAL_RE, normalize_search_text

CONTROLLED_RELATIONSHIP_TYPES = frozenset({"HAS_CRITERION", "REQUIRES", "RECOMMENDS", "RELATED_TO"})
CRITERION_CATEGORY_LABEL = "CriterionCategory"
CRITERION_RELATIONSHIP_TYPE = "HAS_CRITERION"
NEGATION = "nie"
MAX_HEADING_TOKENS = 12
MAX_CATEGORY_TOKEN_INDEX = 2

TOKEN_RE = re.compile(r"[0-9a-z]+")
NODE_LABEL_RE = re.compile(r"\(\s*(?P<variable>[A-Za-z_]\w*)\s*:\s*`?(?P<label>[A-Za-z_]\w*)")
# One hop of a pattern, read over text with its string literals blanked so a bracket inside a
# title cannot end a node early. The right node is only looked ahead at, so the next hop of a
# chain starts where this one ends.
HOP_RE = re.compile(
    r"\(\s*(?P<left>[A-Za-z_]\w*)[^()]*\)\s*"
    r"(?P<hop><?-\s*\[\s*(?P<variable>[A-Za-z_]\w*)?\s*:\s*(?P<type>[A-Za-z_]\w*)"
    r"(?P<tail>[^\]]*)\]\s*->?)"
    r"(?=\s*\(\s*(?P<right>[A-Za-z_]\w*))"
)


@dataclass(frozen=True)
class _PageContext:
    normalized: str
    qualifiers: list[tuple[int, str]]
    page_type: str | None


def page_relationship_type(page_text: str) -> str | None:
    """
    Return the relationship type the page's qualifier names, when it names exactly one.

    Args:
        page_text: Source page text

    Returns:
        The configured type for the one qualifier on the page, or None for none or several
    """
    found = {relationship_type for _, relationship_type in _qualifier_spans(page_text)}
    return found.pop() if len(found) == 1 else None


def stabilize_category_item_edges(
    page_text: str, statements: list[str]
) -> tuple[list[str], list[str]]:
    """
    Give every category -> item edge on the page one shape, whichever the model wrote.

    Three steps, in order: a ``Topic`` at one end of such an edge takes the other half of its
    configured label pair, every edge between a paired category and item is rewritten in place
    to point from the category with the type the qualifier above the item names, and an item no
    category claims is attached to the nearest category title above its row. Statements are
    rewritten or appended, never dropped, so one that also binds a node keeps binding it.

    Args:
        page_text: Source page text the statements were generated from
        statements: Generated Cypher statements for this page

    Returns:
        The stabilized statements, and one description per relabelled node, rewritten edge
        and added edge
    """
    labels = _node_labels(statements)
    titles = _node_titles(statements)
    page = _PageContext(
        normalized=normalize_search_text(page_text),
        qualifiers=_qualifier_spans(page_text),
        page_type=page_relationship_type(page_text),
    )
    rewrites: list[str] = []

    stabilized = _resolve_topic_ends(statements, labels, rewrites)
    stabilized = [
        _rewrite_hops(
            part,
            labels=labels,
            titles=titles,
            page=page,
            rewrites=rewrites,
        )
        for part in stabilized
    ]
    stabilized = _attach_unlinked_items(
        stabilized, labels=labels, titles=titles, page=page, rewrites=rewrites
    )
    return stabilized, rewrites


def _blank_literals(cypher: str) -> str:
    """Replace every string literal's content with spaces, keeping every offset."""
    return CYPHER_STRING_LITERAL_RE.sub(
        lambda match: match.group(0)[0] + " " * (len(match.group(0)) - 2) + match.group(0)[-1],
        cypher,
    )


def _node_labels(statements: list[str]) -> dict[str, str]:
    """Map each variable to the first label it was bound with."""
    labels: dict[str, str] = {}
    for statement in statements:
        for match in NODE_LABEL_RE.finditer(_blank_literals(statement)):
            labels.setdefault(match.group("variable"), match.group("label"))
    return labels


def _node_titles(statements: list[str]) -> dict[str, str]:
    """Map each variable to its title property, if present."""
    titles: dict[str, str] = {}
    for statement in statements:
        for node_match in NODE_PROPERTY_MAP_RE.finditer(statement):
            variable = node_match.group("variable")
            if not variable:
                continue
            properties = node_match.group("properties") or ""
            for entry in PROPERTY_ENTRY_RE.finditer(properties):
                if entry.group("key").strip("`").casefold() != TITLE_PROPERTY:
                    continue
                titles.setdefault(variable, entry.group("value")[1:-1])

        for assignment in SET_ASSIGNMENT_RE.finditer(statement):
            if assignment.group("key").strip("`").casefold() != TITLE_PROPERTY:
                continue
            titles[assignment.group("variable")] = assignment.group("value")[1:-1]

    return titles


def _category_item_maps() -> tuple[dict[str, str], dict[str, str]]:
    schema = get_config().graph_schema
    item_of = {pair.category: pair.item for pair in schema.category_item_labels}
    category_of = {pair.item: pair.category for pair in schema.category_item_labels}
    return item_of, category_of


def _rewrite_hops(
    statement: str,
    *,
    labels: dict[str, str],
    titles: dict[str, str],
    page: _PageContext,
    rewrites: list[str],
) -> str:
    """Rewrite the category -> item hops of one statement, leaving everything else verbatim.

    The label pair decides which hops those are, whatever type the model gave them: an item hung
    under its category by HAS_SUBCOMPETENCY is the same edge drifted, and leaving it would add a
    second one beside it.
    """
    pieces: list[str] = []
    cursor = 0
    item_of, _ = _category_item_maps()

    for hop in HOP_RE.finditer(_blank_literals(statement)):
        left, right = hop.group("left"), hop.group("right")
        left_label = labels.get(left)
        right_label = labels.get(right)
        if item_of.get(left_label or "") == right_label:
            category_label, from_left = left_label, True
        elif item_of.get(right_label or "") == left_label:
            category_label, from_left = right_label, False
        else:
            continue

        item_variable = right if from_left else left
        relationship_type = _edge_type(category_label, titles.get(item_variable), page)
        if relationship_type is None:
            continue

        tail = statement[hop.start("tail") : hop.end("tail")]
        inner = f"{hop.group('variable') or ''}:{relationship_type}{tail}"
        replacement = f"-[{inner}]->" if from_left else f"<-[{inner}]-"
        original = statement[hop.start("hop") : hop.end("hop")]
        if replacement == original:
            continue

        pieces.append(statement[cursor : hop.start("hop")])
        pieces.append(replacement)
        cursor = hop.end("hop")
        rewrites.append(f"({left}){original}({right}) -> ({left}){replacement}({right})")

    pieces.append(statement[cursor:])
    return "".join(pieces)


def _resolve_topic_ends(
    statements: list[str], labels: dict[str, str], rewrites: list[str]
) -> list[str]:
    """Give a Topic on one end of a category-item edge the matching pair label.

    Only when the edge points the way a category -> item edge does: a Topic pointing at an item
    is its category, and a Topic a category points at is its item. A Topic pointing at a category
    is the group heading above it, and relabelling it would invert the hierarchy.
    """
    schema = get_config().graph_schema
    item_of, category_of = _category_item_maps()
    wanted: dict[str, set[str]] = {}

    for statement in statements:
        for hop in HOP_RE.finditer(_blank_literals(statement)):
            if hop.group("type") not in CONTROLLED_RELATIONSHIP_TYPES:
                continue
            arrow = hop.group("hop")
            if arrow.endswith(">") and not arrow.startswith("<"):
                source, target = hop.group("left"), hop.group("right")
            elif arrow.startswith("<") and not arrow.endswith(">"):
                source, target = hop.group("right"), hop.group("left")
            else:
                continue

            source_label, target_label = labels.get(source), labels.get(target)
            if source_label == schema.fallback_label and target_label in category_of:
                wanted.setdefault(source, set()).add(category_of[target_label])
            elif target_label == schema.fallback_label and source_label in item_of:
                wanted.setdefault(target, set()).add(item_of[source_label])

    resolved = {
        variable: next(iter(new_labels))
        for variable, new_labels in wanted.items()
        if len(new_labels) == 1
    }
    if not resolved:
        return statements

    updated = statements
    for variable, new_label in resolved.items():
        binding = re.compile(
            rf"\(\s*{re.escape(variable)}\s*:\s*`?{re.escape(schema.fallback_label)}`?\b"
        )
        updated = [binding.sub(f"({variable}:{new_label}", part, count=1) for part in updated]
        labels[variable] = new_label
        rewrites.append(f"({variable}:{schema.fallback_label}) -> ({variable}:{new_label})")
    return updated


def _edge_type(
    category_label: str | None, item_title: str | None, page: _PageContext
) -> str | None:
    """Pick the edge type: always HAS_CRITERION for criteria, the qualifier for competencies.

    Retrieval tells RECOMMENDS from REQUIRES only for competencies, and criterion pages such as
    the OTM-R policy say "wymagane dokumenty" in prose, so a qualifier never retypes a criterion.
    """
    if category_label == CRITERION_CATEGORY_LABEL:
        return CRITERION_RELATIONSHIP_TYPE
    relationship_type, title_found = _nearest_qualifier_for_item_title(
        item_title,
        normalized_page=page.normalized,
        qualifiers=page.qualifiers,
    )
    return relationship_type if title_found else page.page_type


def _attach_unlinked_items(
    statements: list[str],
    *,
    labels: dict[str, str],
    titles: dict[str, str],
    page: _PageContext,
    rewrites: list[str],
) -> list[str]:
    """Attach unlinked items to the nearest category title above them on the page."""
    item_of, _ = _category_item_maps()
    item_labels = set(item_of.values())
    linked = {
        end
        for statement in statements
        for hop in HOP_RE.finditer(_blank_literals(statement))
        for category, end in (
            (hop.group("left"), hop.group("right")),
            (hop.group("right"), hop.group("left")),
        )
        if item_of.get(labels.get(category) or "") == labels.get(end)
    }

    categories = sorted(
        (position, variable)
        for variable, label in labels.items()
        if label in item_of
        and (title := (titles.get(variable) or "").strip())
        and (normalized_title := normalize_search_text(title).strip())
        and (position := page.normalized.find(normalized_title)) >= 0
    )

    added: list[str] = []
    for variable, label in labels.items():
        if variable in linked or label not in item_labels:
            continue
        title = (titles.get(variable) or "").strip()
        if not title:
            continue
        normalized_title = normalize_search_text(title).strip()
        if not normalized_title:
            continue

        position = page.normalized.find(normalized_title)
        above = [
            category
            for category_position, category in categories
            if 0 <= category_position < position
        ]
        if position < 0 or not above:
            continue

        category = above[-1]
        category_label = labels.get(category)
        if category_label is None or item_of.get(category_label) != label:
            continue
        relationship_type = _edge_type(category_label, titles.get(variable), page)
        if relationship_type is None:
            continue
        added.append(f"MERGE ({category})-[:{relationship_type}]->({variable})")

    rewrites.extend(f"added {statement}" for statement in added)
    return statements + added


def _is_heading_line(line: str) -> bool:
    """Report whether a line reads as a heading: short, not a list row, not a sentence."""
    text = line.strip()
    tokens = TOKEN_RE.findall(text)
    has_near_category_token = any(
        token.startswith(("kompetencj", "kryteri"))
        for token in tokens[: MAX_CATEGORY_TOKEN_INDEX + 1]
    )
    return (
        bool(text)
        and LIST_ROW_RE.match(text) is None
        and not text.endswith((".", "!", "?"))
        and len(tokens) <= MAX_HEADING_TOKENS
        and (text.endswith(":") or has_near_category_token)
    )


def _qualifier_spans(page_text: str) -> list[tuple[int, str]]:
    """Return qualifier positions and types, read from the page's heading lines only.

    Prose carries the same words ("niezbedne dokumenty"), so a qualifier counts only where a
    heading states it. The R1-R4 page title counts: it is short and does not end a sentence.
    """
    rules = get_relationship_qualifier_rules(get_config().graph_schema)
    spans: list[tuple[int, str]] = []
    offset = 0
    for line in normalize_search_text(page_text).splitlines(keepends=True):
        if _is_heading_line(line):
            tokens = list(TOKEN_RE.finditer(line))
            for index, match in enumerate(tokens):
                previous = tokens[index - 1].group(0) if index > 0 else None
                relationship_type = _relationship_type_for_token(match.group(0), previous, rules)
                if relationship_type is not None:
                    spans.append((offset + match.start(), relationship_type))
        offset += len(line)
    return spans


def _relationship_type_for_token(
    token: str,
    previous_token: str | None,
    rules: tuple[tuple[str, tuple[str, ...]], ...],
) -> str | None:
    for relationship_type, stems in rules:
        for stem in stems:
            if not token.startswith(stem):
                continue
            if stem.startswith(NEGATION):
                return relationship_type
            if previous_token == NEGATION:
                continue
            return relationship_type
    return None


def _nearest_qualifier_for_item_title(
    item_title: str | None,
    *,
    normalized_page: str,
    qualifiers: list[tuple[int, str]],
) -> tuple[str | None, bool]:
    """Return the nearest qualifier before item title position, and whether title was found."""
    if not item_title:
        return None, False
    normalized_title = normalize_search_text(item_title).strip()
    if not normalized_title:
        return None, False

    title_position = normalized_page.find(normalized_title)
    if title_position < 0:
        return None, False

    nearest: str | None = None
    for qualifier_position, qualifier_type in qualifiers:
        if qualifier_position >= title_position:
            break
        nearest = qualifier_type
    return nearest, True
