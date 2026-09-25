"""Build deterministic category->item edges for heading+rows page shapes.

Issue #104: a single page can produce different category-item relationship topology run to run.
For sections that follow a known heading + list-rows structure, this pass stabilizes relationship
type and direction deterministically while preserving statements that also create node variables.
"""

import re
from dataclasses import dataclass, field

from src.config.config import get_config
from src.config.relationship_qualifiers import get_relationship_qualifier_rules
from src.data_pipeline.completeness import (
    NODE_PROPERTY_MAP_RE,
    PROPERTY_ENTRY_RE,
    SET_ASSIGNMENT_RE,
    TITLE_PROPERTY,
    GeneratedNode,
    extract_generated_nodes_by_variable,
    node_holds_row_tokens,
)
from src.data_pipeline.ingestion_guardrails import TOKEN_RE as INGESTION_TOKEN_RE
from src.data_pipeline.section_parser import HeadingRowSection, extract_heading_row_sections
from src.text_normalization import is_code_token, normalize_search_text

TOKEN_RE = re.compile(r"[0-9a-z]+")
MIN_TOKEN_LENGTH = 2
HEADING_MATCH_THRESHOLD = 0.55

LABEL_RE = re.compile(r":\s*(?P<label>`[^`]+`|[A-Za-z_]\w*)")
RELATIONSHIP_INNER_RE = re.compile(
    r"^\s*(?:(?P<variable>[A-Za-z_]\w*)\s*)?:\s*"
    r"(?P<types>(?:`[^`]+`|[A-Za-z_]\w*)(?:\s*\|\s*(?:`[^`]+`|[A-Za-z_]\w*))*)"
    r"(?P<tail>.*)$"
)
CONTROLLED_RELATIONSHIP_TYPES = frozenset({"HAS_CRITERION", "REQUIRES", "RECOMMENDS", "RELATED_TO"})


@dataclass(frozen=True)
class _TokenSpan:
    kind: str
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class _NodeSpan:
    variable: str | None
    start: int
    end: int


@dataclass(frozen=True)
class _PatternSpan:
    start: int
    end: int
    left_var: str
    right_var: str
    left_arrow: str
    right_arrow: str
    inner_text: str
    left_node_text: str
    right_node_text: str


@dataclass(frozen=True)
class NodeFacts:
    """What this pass needs to know about one generated node."""

    variable: str
    label: str | None
    title: str | None
    title_tokens: frozenset[str]
    code_tokens: frozenset[str]
    generated_node: GeneratedNode


@dataclass
class TopologyRewriteReport:
    """How many deterministic repairs this pass applied."""

    matched_sections: int = 0
    rewritten_relationships: int = 0
    added_relationships: int = 0
    rewritten_examples: list[str] = field(default_factory=list)
    added_examples: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        """True when the pass rewrote or added at least one relationship."""
        return bool(self.rewritten_relationships or self.added_relationships)


def normalize_tokens(value: str) -> frozenset[str]:
    """Return matchable tokens from a title or a row string."""
    return frozenset(
        token
        for token in TOKEN_RE.findall(normalize_search_text(value))
        if len(token) >= MIN_TOKEN_LENGTH
    )


def relationship_type_for_heading(heading: str) -> str | None:
    """Pick relationship type from qualifier words in a heading string.

    Token stems are matched (`pozadan*`, `wymagan*`, `niezbedn*`) so inflected forms are covered.
    A negated form (`niepozadane`) does not count as a recommends qualifier.
    """
    schema = get_config().graph_schema
    for relationship_type, stems in get_relationship_qualifier_rules(schema):
        if _matches_any_qualifier_stem(heading, stems):
            return relationship_type
    return None


def enforce_deterministic_category_item_topology(
    page_text: str, statements: list[str]
) -> tuple[list[str], TopologyRewriteReport]:
    """Rewrite category-item relationships to deterministic type and direction.

    Args:
        page_text: Source page text the statements were generated from
        statements: Generated Cypher statements for this page

    Returns:
        Statements with deterministic category->item edges and a rewrite report
    """
    report = TopologyRewriteReport()
    sections = extract_heading_row_sections(page_text)
    if not sections:
        return statements, report

    nodes = _extract_node_facts(statements)
    if not nodes:
        return statements, report

    desired_edges: dict[tuple[str, str], str] = {}
    used_categories: set[str] = set()

    for section in sections:
        category = _match_category(section, nodes, used_categories)
        if category is None:
            continue

        relationship_type = _relationship_type_for_section(section, category)
        if relationship_type is None:
            continue

        used_items: set[str] = set()
        matched_items: list[str] = []
        for row in section.rows:
            item = _match_item(row, nodes, category.variable, used_items)
            if item is None:
                continue
            used_items.add(item.variable)
            matched_items.append(item.variable)

        if not matched_items:
            continue

        used_categories.add(category.variable)
        report.matched_sections += 1
        for item_variable in matched_items:
            desired_edges[(category.variable, item_variable)] = relationship_type

    if not desired_edges:
        return statements, report

    rewritten_statements: list[str] = []
    present_edges: set[tuple[str, str, str]] = set()

    for statement in statements:
        rewritten, rewritten_count, rewritten_examples, statement_edges = (
            _rewrite_statement_relationships(statement, desired_edges)
        )
        rewritten_statements.append(rewritten)
        report.rewritten_relationships += rewritten_count
        report.rewritten_examples.extend(rewritten_examples)
        present_edges.update(statement_edges)

    for (source, target), relationship_type in desired_edges.items():
        edge = (source, relationship_type, target)
        if edge in present_edges:
            continue
        statement = f"MERGE ({source})-[:{relationship_type}]->({target})"
        rewritten_statements.append(statement)
        present_edges.add(edge)
        report.added_relationships += 1
        report.added_examples.append(statement)

    return rewritten_statements, report


def _extract_node_facts(statements: list[str]) -> list[NodeFacts]:
    """Collect node labels and titles by variable from generated statements."""
    grouped: dict[str, dict[str, str | None]] = {}
    generated_by_variable = extract_generated_nodes_by_variable(statements)

    for statement in statements:
        for node_match in NODE_PROPERTY_MAP_RE.finditer(statement):
            variable = node_match.group("variable")
            if not variable:
                continue
            slot = grouped.setdefault(variable, {"label": None, "title": None})
            labels = node_match.group("labels") or ""
            label_match = LABEL_RE.search(labels)
            if slot["label"] is None and label_match is not None:
                slot["label"] = label_match.group("label").strip("`")

            properties = node_match.group("properties") or ""
            for entry in PROPERTY_ENTRY_RE.finditer(properties):
                if entry.group("key").strip("`").casefold() != TITLE_PROPERTY:
                    continue
                slot["title"] = entry.group("value")[1:-1]

        for assignment in SET_ASSIGNMENT_RE.finditer(statement):
            if assignment.group("key").strip("`").casefold() != TITLE_PROPERTY:
                continue
            variable = assignment.group("variable")
            slot = grouped.setdefault(variable, {"label": None, "title": None})
            slot["title"] = assignment.group("value")[1:-1]

    return [
        NodeFacts(
            variable=variable,
            label=values["label"],
            title=values["title"],
            title_tokens=normalize_tokens(values["title"] or ""),
            code_tokens=frozenset(
                token for token in normalize_tokens(values["title"] or "") if is_code_token(token)
            ),
            generated_node=generated_by_variable.get(
                variable, GeneratedNode(title_tokens=frozenset(), value_tokens=frozenset())
            ),
        )
        for variable, values in grouped.items()
        if variable in generated_by_variable
    ]


def _match_category(
    section: HeadingRowSection, nodes: list[NodeFacts], used_categories: set[str]
) -> NodeFacts | None:
    """Choose the category node whose title best matches a section heading."""
    heading_tokens = normalize_tokens(section.heading)
    if not heading_tokens:
        return None
    heading_codes = frozenset(token for token in heading_tokens if is_code_token(token))

    best_node: NodeFacts | None = None
    best_score = 0.0
    for node in nodes:
        if node.variable in used_categories or not node.title_tokens:
            continue
        if node.label is None or not node.label.endswith("Category"):
            continue
        if heading_codes and node.code_tokens != heading_codes:
            continue
        score = _bidirectional_overlap(heading_tokens, node.title_tokens)
        if score > best_score:
            best_score = score
            best_node = node

    if best_score < HEADING_MATCH_THRESHOLD:
        return None
    return best_node


def _match_item(
    row: str, nodes: list[NodeFacts], category_variable: str, used_items: set[str]
) -> NodeFacts | None:
    """Choose the item node whose title and values satisfy the completeness row-matching rule."""
    row_tokens = normalize_tokens(row)
    if not row_tokens:
        return None

    best_node: NodeFacts | None = None
    best_score = 0.0
    for node in nodes:
        if node.variable == category_variable or node.variable in used_items:
            continue
        if node.label is not None and (node.label.endswith("Category") or node.label == "Topic"):
            continue
        if not node_holds_row_tokens(node.generated_node, row_tokens):
            continue
        score = _overlap_share(row_tokens, node.title_tokens)
        if score > best_score:
            best_score = score
            best_node = node

    return best_node


def _relationship_type_for_section(section: HeadingRowSection, category: NodeFacts) -> str | None:
    """Resolve deterministic relationship type from qualifier context and category label."""
    context_text = " ".join((*section.context_headings, section.heading))
    by_qualifier = relationship_type_for_heading(context_text)
    if by_qualifier is not None:
        return by_qualifier
    if category.label == "CriterionCategory":
        return "HAS_CRITERION"
    return None


def _rewrite_statement_relationships(
    statement: str, desired_edges: dict[tuple[str, str], str]
) -> tuple[str, int, list[str], set[tuple[str, str, str]]]:
    """Rewrite controlled relationship segments in one statement to desired type/direction."""
    pieces: list[str] = []
    cursor = 0
    rewritten_count = 0
    rewritten_examples: list[str] = []
    present_edges: set[tuple[str, str, str]] = set()

    for segment in _iter_pattern_spans(statement):
        pieces.append(statement[cursor : segment.start])

        left_var = segment.left_var
        right_var = segment.right_var
        desired_type = desired_edges.get((left_var, right_var))
        desired_source = left_var
        desired_target = right_var
        if desired_type is None:
            desired_type = desired_edges.get((right_var, left_var))
            desired_source = right_var
            desired_target = left_var

        parsed_inner = RELATIONSHIP_INNER_RE.match(segment.inner_text or "")
        original_types = (
            _relationship_types_from_inner(parsed_inner.group("types"))
            if parsed_inner is not None
            else ()
        )
        can_rewrite = bool(
            desired_type is not None
            and parsed_inner is not None
            and set(original_types).intersection(CONTROLLED_RELATIONSHIP_TYPES)
        )

        if not can_rewrite:
            rewritten_segment = statement[segment.start : segment.end]
            present_edges.update(
                _present_edges_for_segment(
                    left_var=left_var,
                    right_var=right_var,
                    left_arrow=segment.left_arrow,
                    right_arrow=segment.right_arrow,
                    relationship_types=original_types,
                )
            )
        else:
            relation_variable = (parsed_inner.group("variable") or "").strip()
            tail = parsed_inner.group("tail") or ""
            inner = (
                f"{relation_variable}:{desired_type}{tail}"
                if relation_variable
                else f":{desired_type}{tail}"
            )
            if desired_source == left_var and desired_target == right_var:
                left_arrow, right_arrow = "-", "->"
            else:
                left_arrow, right_arrow = "<-", "-"
            rewritten_segment = (
                f"{segment.left_node_text}{left_arrow}[{inner}]"
                f"{right_arrow}{segment.right_node_text}"
            )
            rewritten_count += 1
            rewritten_examples.append(
                f"{left_var}-{sorted(original_types)}-{right_var} -> "
                f"{desired_source}-[{desired_type}]->{desired_target}"
            )
            present_edges.update(
                _present_edges_for_segment(
                    left_var=left_var,
                    right_var=right_var,
                    left_arrow=left_arrow,
                    right_arrow=right_arrow,
                    relationship_types=(desired_type,),
                )
            )

        pieces.append(rewritten_segment)
        cursor = segment.end

    pieces.append(statement[cursor:])
    return "".join(pieces), rewritten_count, rewritten_examples, present_edges


def _present_edges_for_segment(
    *,
    left_var: str,
    right_var: str,
    left_arrow: str,
    right_arrow: str,
    relationship_types: tuple[str, ...],
) -> set[tuple[str, str, str]]:
    """Return all edges represented by one relationship segment."""
    if not relationship_types:
        return set()
    if left_arrow == "-" and right_arrow == "->":
        source, target = left_var, right_var
    elif left_arrow == "<-" and right_arrow == "-":
        source, target = right_var, left_var
    else:
        return set()
    return {(source, relationship_type, target) for relationship_type in relationship_types}


def _relationship_types_from_inner(types: str) -> tuple[str, ...]:
    """Read relationship type names from a parsed relationship inner-type expression."""
    return tuple(rel_type.strip().strip("`") for rel_type in types.split("|") if rel_type.strip())


def _iter_pattern_spans(statement: str) -> list[_PatternSpan]:
    """Return relationship pattern spans from a statement using token-aware parsing."""
    tokens = _tokenize_with_spans(statement)
    spans: list[_PatternSpan] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.kind == "operator" and token.text == "(":
            left_node, next_index = _parse_node_span(tokens, index)
            if left_node is None:
                index += 1
                continue
            cursor = next_index
            current_left = left_node
            while True:
                relationship, after_relationship = _parse_relationship_span(
                    tokens, cursor, statement
                )
                if relationship is None:
                    break
                right_node, after_right = _parse_node_span(tokens, after_relationship)
                if right_node is None:
                    break
                if current_left.variable and right_node.variable:
                    spans.append(
                        _PatternSpan(
                            start=current_left.start,
                            end=right_node.end,
                            left_var=current_left.variable,
                            right_var=right_node.variable,
                            left_arrow=relationship["left_arrow"],
                            right_arrow=relationship["right_arrow"],
                            inner_text=relationship["inner_text"],
                            left_node_text=statement[current_left.start : current_left.end],
                            right_node_text=statement[right_node.start : right_node.end],
                        )
                    )
                current_left = right_node
                cursor = after_right
            index = next_index
            continue
        index += 1
    return spans


def _tokenize_with_spans(statement: str) -> list[_TokenSpan]:
    """Tokenize Cypher with source spans."""
    tokens: list[_TokenSpan] = []
    position = 0
    while position < len(statement):
        match = INGESTION_TOKEN_RE.match(statement, position)
        if match is None:
            return []
        kind = match.lastgroup or ""
        if kind != "space":
            tokens.append(
                _TokenSpan(
                    kind=kind,
                    text=match.group(0),
                    start=match.start(),
                    end=match.end(),
                )
            )
        position = match.end()
    return tokens


def _parse_node_span(tokens: list[_TokenSpan], start_index: int) -> tuple[_NodeSpan | None, int]:
    """Parse one node pattern starting at ``(``."""
    if start_index >= len(tokens):
        return None, start_index
    if tokens[start_index].kind != "operator" or tokens[start_index].text != "(":
        return None, start_index

    variable: str | None = None
    if start_index + 1 < len(tokens) and tokens[start_index + 1].kind in ("name", "quoted"):
        variable = tokens[start_index + 1].text.strip("`")

    depth = 0
    index = start_index
    while index < len(tokens):
        token = tokens[index]
        if token.kind == "operator" and token.text == "(":
            depth += 1
        elif token.kind == "operator" and token.text == ")":
            depth -= 1
            if depth == 0:
                return _NodeSpan(
                    variable=variable, start=tokens[start_index].start, end=token.end
                ), index + 1
        index += 1

    return None, start_index + 1


def _parse_relationship_span(
    tokens: list[_TokenSpan], start_index: int, statement: str
) -> tuple[dict[str, str] | None, int]:
    """Parse one relationship segment between two node spans."""
    index = start_index
    left_arrow = "-"
    if index < len(tokens) and tokens[index].kind == "operator" and tokens[index].text == "<":
        left_arrow = "<-"
        index += 1
    if index >= len(tokens) or tokens[index].kind != "operator" or tokens[index].text != "-":
        return None, start_index
    index += 1
    if index >= len(tokens) or tokens[index].kind != "operator" or tokens[index].text != "[":
        return None, start_index
    index += 1

    inner_start = tokens[index].start if index < len(tokens) else tokens[index - 1].end
    bracket_depth = 1
    inner_end = inner_start
    while index < len(tokens):
        token = tokens[index]
        if token.kind == "operator" and token.text == "[":
            bracket_depth += 1
        elif token.kind == "operator" and token.text == "]":
            bracket_depth -= 1
            if bracket_depth == 0:
                inner_end = token.start
                index += 1
                break
        index += 1
    else:
        return None, start_index

    if index >= len(tokens) or tokens[index].kind != "operator" or tokens[index].text != "-":
        return None, start_index
    index += 1
    right_arrow = "-"
    if index < len(tokens) and tokens[index].kind == "operator" and tokens[index].text == ">":
        right_arrow = "->"
        index += 1

    return {
        "left_arrow": left_arrow,
        "right_arrow": right_arrow,
        "inner_text": statement[inner_start:inner_end],
    }, index


def _matches_any_qualifier_stem(text: str, stems: tuple[str, ...]) -> bool:
    """Report whether text tokens contain any stem, excluding negated forms like 'niepozadane'."""
    tokens = [token for token in TOKEN_RE.findall(normalize_search_text(text)) if token]
    for index, token in enumerate(tokens):
        for stem in stems:
            if not token.startswith(stem):
                continue
            if stem.startswith("nie"):
                return True
            if index > 0 and tokens[index - 1] == "nie":
                continue
            return True
    return False


def _bidirectional_overlap(left: frozenset[str], right: frozenset[str]) -> float:
    """Return the stronger directional overlap between two token sets."""
    return max(_overlap_share(left, right), _overlap_share(right, left))


def _overlap_share(left: frozenset[str], right: frozenset[str]) -> float:
    """Return the share of left tokens that also appears in right."""
    if not left:
        return 0.0
    return len(left & right) / len(left)
