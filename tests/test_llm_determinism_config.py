import re
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.config.config import get_config
from src.config.relationship_qualifiers import render_relationship_qualifier_guidance
from src.mcp_server.tools.knowledge_graph.rag import RAG, LLMProvider


def test_configured_models_do_not_sample() -> None:
    config = get_config()
    assert config.llm.fast_model.temperature == 0
    assert config.llm.accurate_model.temperature == 0


@pytest.mark.parametrize("use_accurate", [False, True])
def test_openai_client_is_built_with_the_configured_temperature(
    monkeypatch: pytest.MonkeyPatch, use_accurate: bool
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    rag = object.__new__(RAG)
    rag.api_key = "test-key"
    rag.llm_timeout_sec = 30.0
    rag.config = SimpleNamespace(
        llm=SimpleNamespace(
            fast_model=SimpleNamespace(name="fast-model", temperature=0.0),
            accurate_model=SimpleNamespace(name="accurate-model", temperature=0.0),
        )
    )

    with patch("src.mcp_server.tools.knowledge_graph.rag.BaseChatOpenAI") as chat_openai:
        rag._build_chat_model(LLMProvider.OPENAI, use_accurate=use_accurate)

    assert chat_openai.call_args.kwargs["temperature"] == 0.0


def test_the_cypher_prompt_asks_for_the_limit_the_code_enforces() -> None:
    """Review of PR #90: a prompt asking for more rows than the clamp allows is a trap.

    `ensure_limit` rewrites a larger trailing LIMIT down to `rag.max_results`, so a prompt naming
    a different number does not produce more rows - it just means the model is told to ask for
    something the code silently overrides, and whoever reads the prompt is misled about how many
    rows an answer is built from.
    """
    config = get_config()
    limit_lines = [
        line for line in config.prompts.cypher_search.splitlines() if "LIMIT" in line.upper()
    ]
    assert limit_lines, "the Cypher prompt no longer names a LIMIT; this test needs updating"

    asked_for = [int(number) for line in limit_lines for number in re.findall(r"\d+", line)]
    assert asked_for == [config.rag.max_results] * len(asked_for)


def test_the_cypher_prompt_uses_relationship_qualifier_rules_from_config() -> None:
    config = get_config()
    guidance = render_relationship_qualifier_guidance(config.graph_schema)

    payload = RAG._build_cypher_prompt_payload(
        "Jakie sa kompetencje pozadane dla naukowca R2?", "(:X)"
    )

    assert "{relationship_qualifier_guidance}" in config.prompts.cypher_search
    assert payload["relationship_qualifier_guidance"] == guidance
    assert '"pozadane" -> RECOMMENDS' in guidance
    assert '"wymagane", "niezbedne" -> REQUIRES' in guidance

    rendered = config.prompts.cypher_search.format(**payload)
    assert guidance in rendered
