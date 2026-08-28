"""The closed node-label vocabulary shared by the extraction prompt and the ingestion pipeline.

Issue #53: the extraction model named labels freely, so one real entity arrived as
``StudyProgram`` on one page and ``Program`` on the next. Two nodes, and a query naming either
label finds only half the graph. The vocabulary lives in ``graph_config.yaml`` under
``graph_schema`` and is rendered into the prompt from here.
"""

from src.config.config_models import GraphSchema


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
