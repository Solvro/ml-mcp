"""The closed node-label vocabulary shared by the extraction prompt and the ingestion pipeline.

Issue #53: the extraction model named labels freely, so one real entity arrived as
``StudyProgram`` on one page and ``Program`` on the next. Two nodes, and a query naming either
label finds only half the graph. The vocabulary lives in ``graph_config.yaml`` under
``graph_schema``.

The prompt is told the vocabulary and the rewrite enforces it, because a model follows a naming
instruction most of the time and "most of the time" is what splits an entity in two.
"""

import re
from collections.abc import Callable

from src.config.config_models import GraphSchema
from src.text_normalization import CYPHER_STRING_LITERAL_RE, normalize_search_text

# A node pattern: an optional variable followed by one or more :Label parts, inside parentheses.
# Relationship types live in square brackets and are deliberately not matched.
NODE_LABELS_RE = re.compile(
    r"\(\s*(?:[A-Za-z_]\w*)?\s*(?P<labels>(?::\s*(?:`[^`]+`|[A-Za-z_]\w*)\s*)+)"
)
SINGLE_LABEL_RE = re.compile(r":\s*(?P<label>`[^`]+`|[A-Za-z_]\w*)")


def render_allowed_labels(schema: GraphSchema) -> str:
    """
    Render the closed label set for the extraction prompt.

    Known drift is listed alongside the labels so the model is corrected before it writes, not
    only afterwards by the ingestion rewrite.

    Args:
        schema: Graph schema section of the loaded configuration

    Returns:
        Prompt-ready text listing the canonical labels and the aliases they absorb
    """
    lines = [", ".join(schema.node_labels)]

    if schema.label_aliases:
        redirects = ", ".join(
            f"{alias.invented} -> {alias.canonical}" for alias in schema.label_aliases
        )
        lines.append("")
        lines.append(f"Never use these; write the canonical label instead: {redirects}")

    return "\n".join(lines)


def _apply_outside_string_literals(cypher: str, transform: Callable[[str], str]) -> str:
    """Run a rewrite over Cypher syntax only, leaving quoted values untouched."""
    pieces: list[str] = []
    cursor = 0

    for literal in CYPHER_STRING_LITERAL_RE.finditer(cypher):
        pieces.append(transform(cypher[cursor : literal.start()]))
        pieces.append(literal.group(0))
        cursor = literal.end()

    pieces.append(transform(cypher[cursor:]))
    return "".join(pieces)


class LabelVocabulary:
    """Resolves any label the extraction model produced to a configured canonical label."""

    def __init__(self, schema: GraphSchema) -> None:
        """
        Build the lookup from the configured labels and aliases.

        Args:
            schema: Graph schema section of the loaded configuration
        """
        self.fallback_label = schema.fallback_label
        self.node_labels = tuple(schema.node_labels)

        # Case- and diacritic-insensitive, so "wydzial", "Wydzial" and "WYDZIAL" all land on
        # Faculty without needing an alias row each.
        self._by_normalized_form = {
            normalize_search_text(label): label for label in self.node_labels
        }
        for alias in schema.label_aliases:
            self._by_normalized_form.setdefault(
                normalize_search_text(alias.invented), alias.canonical
            )

    def canonical_label(self, label: str) -> str:
        """
        Resolve one label to its canonical spelling.

        Args:
            label: Label as written by the extraction model, with or without backticks

        Returns:
            A label from the configured set, or the fallback label when nothing matches
        """
        stripped = label.strip().strip("`")
        if not stripped:
            return self.fallback_label
        return self._by_normalized_form.get(normalize_search_text(stripped), self.fallback_label)

    def canonicalize_statement(self, cypher: str) -> tuple[str, dict[str, str]]:
        """
        Rewrite every node label in a Cypher statement to the configured vocabulary.

        Args:
            cypher: One generated MERGE statement

        Returns:
            Tuple of the rewritten statement and a map of the labels that were changed,
            from the original spelling to the canonical one
        """
        rewrites: dict[str, str] = {}

        def replace_label(match: re.Match[str]) -> str:
            original = match.group("label").strip("`")
            canonical = self.canonical_label(original)
            if canonical != original:
                rewrites[original] = canonical
            return f":{canonical}"

        def replace_node(match: re.Match[str]) -> str:
            rewritten_labels = SINGLE_LABEL_RE.sub(replace_label, match.group("labels"))
            return match.group(0).replace(match.group("labels"), rewritten_labels, 1)

        rewritten = _apply_outside_string_literals(
            cypher, lambda segment: NODE_LABELS_RE.sub(replace_node, segment)
        )
        return rewritten, rewrites
