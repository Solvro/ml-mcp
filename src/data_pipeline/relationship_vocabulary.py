import re

from src.config.config_models import GraphSchema
from src.text_normalization import apply_outside_string_literals, normalize_search_text

RELATIONSHIP_TYPES_PREFIX_RE = re.compile(
    r"\[\s*(?P<variable>[A-Za-z_]\w*\s*)?:\s*"
    r"(?P<types>(?:`[^`]+`|[A-Za-z_]\w*)(?:\s*\|\s*(?:`[^`]+`|[A-Za-z_]\w*))*)"
)
SINGLE_RELATIONSHIP_TYPE_RE = re.compile(r"`[^`]+`|[A-Za-z_]\w*")


def normalize_relationship_name(name: str) -> str:
    """Normalize relationship names for vocabulary matching.

    Args:
        name: Relationship name as written in generated Cypher

    Returns:
        Case- and diacritic-insensitive form without separators (`_`, `-`, whitespace)
    """
    return normalize_search_text(name).replace("_", "").replace("-", "").replace(" ", "")


def render_allowed_relationship_types(schema: GraphSchema) -> str:
    """Render allowed relationship types for extraction prompts.

    Args:
        schema: Graph schema section of the loaded configuration

    Returns:
        Prompt-ready text listing canonical relationship types (excluding fallback)
        and known alias redirects
    """
    allowed = [
        relationship_type
        for relationship_type in schema.relationship_types
        if relationship_type != schema.fallback_relationship_type
    ]
    lines = [", ".join(allowed)]

    if schema.relationship_aliases:
        redirects = ", ".join(
            f"{alias.invented} -> {alias.canonical}" for alias in schema.relationship_aliases
        )
        lines.append("")
        lines.append(f"Never use these; write the canonical relationship type instead: {redirects}")

    return "\n".join(lines)


class RelationshipVocabulary:
    """Resolves generated relationship types to configured canonical names."""

    def __init__(self, schema: GraphSchema) -> None:
        """Build relationship-type lookup from canonical names and aliases.

        Args:
            schema: Graph schema section of the loaded configuration
        """
        self.fallback_relationship_type = schema.fallback_relationship_type
        self.relationship_types = tuple(schema.relationship_types)

        self._by_normalized_form = {
            normalize_relationship_name(relationship_type): relationship_type
            for relationship_type in self.relationship_types
        }
        for alias in schema.relationship_aliases:
            self._by_normalized_form.setdefault(
                normalize_relationship_name(alias.invented),
                alias.canonical,
            )

    def resolve_relationship_type(self, relationship_type: str) -> tuple[str, bool]:
        """Resolve a relationship type and report whether fallback was used.

        Args:
            relationship_type: Relationship type as written by the extraction model

        Returns:
            Tuple of (canonical_relationship_type, used_fallback)
        """
        stripped = relationship_type.strip().strip("`")
        if not stripped:
            return self.fallback_relationship_type, True

        normalized = normalize_relationship_name(stripped)
        canonical = self._by_normalized_form.get(normalized)
        if canonical is None:
            return self.fallback_relationship_type, True
        return canonical, False

    def canonical_relationship_type(self, relationship_type: str) -> str:
        """Resolve one relationship type to canonical name or fallback.

        Args:
            relationship_type: Relationship type as written by the extraction model

        Returns:
            Canonical relationship type from configuration
        """
        canonical, _ = self.resolve_relationship_type(relationship_type)
        return canonical

    def canonicalize_statement(self, cypher: str) -> tuple[str, dict[str, str], set[str]]:
        """Rewrite relationship types in one Cypher statement.

        Args:
            cypher: One generated MERGE statement

        Returns:
            Tuple of:
            - rewritten statement,
            - map of rewritten types (original -> canonical),
            - set of original types that were downgraded to fallback
        """
        rewrites: dict[str, str] = {}
        fallback_rewrites: set[str] = set()

        def replace_relationship(match: re.Match[str]) -> str:
            original_types = match.group("types")

            rewritten_types: list[str] = []
            seen: set[str] = set()
            for relationship_type_match in SINGLE_RELATIONSHIP_TYPE_RE.finditer(original_types):
                original_type = relationship_type_match.group(0).strip().strip("`")
                canonical_type, used_fallback = self.resolve_relationship_type(original_type)
                if canonical_type != original_type:
                    rewrites[original_type] = canonical_type
                if used_fallback and original_type != canonical_type:
                    fallback_rewrites.add(original_type)
                if canonical_type not in seen:
                    seen.add(canonical_type)
                    rewritten_types.append(canonical_type)

            variable = (match.group("variable") or "").strip()
            return f"[{variable}:{'|'.join(rewritten_types)}"

        rewritten = apply_outside_string_literals(
            cypher,
            lambda segment: RELATIONSHIP_TYPES_PREFIX_RE.sub(replace_relationship, segment),
        )
        return rewritten, rewrites, fallback_rewrites
