from src.config.config_models import GraphSchema
from src.text_normalization import normalize_search_text


def get_relationship_qualifier_rules(
    schema: GraphSchema,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Return qualifier->relationship mapping from config.

    Args:
        schema: ``graph_schema`` section of loaded config

    Returns:
        Tuple of ``(relationship_type, token_stems)``
    """
    rules: list[tuple[str, tuple[str, ...]]] = []
    for rule in schema.relationship_qualifier_rules:
        rules.append(
            (
                rule.relationship_type,
                tuple(normalize_search_text(stem).strip() for stem in rule.token_stems),
            )
        )
    return tuple(rules)


def render_relationship_qualifier_guidance(schema: GraphSchema) -> str:
    """Render one prompt sentence describing qualifier->relationship mapping.

    Args:
        schema: ``graph_schema`` section of loaded config

    Returns:
        One sentence suitable for ``prompts.cypher_search``
    """
    mappings: list[str] = []
    for rule in schema.relationship_qualifier_rules:
        words = ", ".join(f'"{word}"' for word in rule.words)
        mappings.append(f"{words} -> {rule.relationship_type}")
    return (
        "The question also says which items it wants: qualifiers map as "
        f"{'; '.join(mappings)}; traverse only that type, never a union of types, "
        "which mixes the two lists into one:"
    )
