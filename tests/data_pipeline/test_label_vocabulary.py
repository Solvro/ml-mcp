"""The extraction prompt must carry the closed label set and the completeness rule."""

from langchain_core.prompts import PromptTemplate

from src.config.config import get_config
from src.data_pipeline.label_vocabulary import render_allowed_labels

CALENDAR_PAGE = (
    "Dni wolne od zajec w semestrze zimowym 2026/2027:\n"
    "- 1 XI 2026 r. - Wszystkich Swietych\n"
    "- 2 XI 2026 r. - dzien wolny od zajec\n"
    "- 11 XI 2026 r. - Swieto Niepodleglosci\n"
    "- 16 XI 2026 r. - Obchody Swieta PWr\n"
    "- 24 XII 2026 r. - Wigilia\n"
)


def _render_prompt() -> str:
    config = get_config()
    template = PromptTemplate(
        input_variables=["context", "schema_context", "node_labels", "relationship_types"],
        template=config.prompts.cypher_insert,
    )
    return template.format(
        context=CALENDAR_PAGE,
        schema_context="(empty)",
        node_labels=render_allowed_labels(config.graph_schema),
        relationship_types=", ".join(config.graph_schema.relationship_types),
    )


def test_rendered_labels_list_every_canonical_label() -> None:
    config = get_config()

    rendered = render_allowed_labels(config.graph_schema)

    for label in config.graph_schema.node_labels:
        assert label in rendered


def test_rendered_labels_redirect_known_drift() -> None:
    rendered = render_allowed_labels(get_config().graph_schema)

    assert "Program -> StudyProgram" in rendered
    assert "CriterionItem -> Criterion" in rendered


def test_prompt_template_renders_with_the_pipeline_payload() -> None:
    """The four variables here are exactly what LLMPipe passes; a rename breaks ingestion."""
    prompt = _render_prompt()

    assert CALENDAR_PAGE in prompt
    assert "StudyProgram" in prompt
    assert "HAS_DAY_OFF" in prompt


def test_prompt_states_the_completeness_rule() -> None:
    """Issue #53: the model dropped the calendar row that had no proper name."""
    prompt = _render_prompt().lower()

    assert "every row of a list or table becomes its own node" in prompt
    assert "do not summarise" in prompt
    assert "dzien wolny od zajec" in prompt


def test_prompt_forbids_inventing_labels() -> None:
    prompt = _render_prompt().lower()

    assert "never invent a new one" in prompt
    assert "one node per real entity" in prompt


def test_prompt_keeps_the_pipe_separated_output_contract() -> None:
    """graph_populating splits on the pipe; losing it would break every ingestion run."""
    prompt = _render_prompt()

    assert "Separate multiple statements with PIPE character (|)" in prompt
    assert 'Begin with "MERGE"' in prompt
