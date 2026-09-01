"""The routing and Cypher models must be deterministic.

Issue #52 measured the cost of sampling: the same question was routed to `end` on some runs and
to `generate_cypher` on others (~40% false rejects), and the generated Cypher differed run to
run. Both models are therefore pinned to temperature 0 in graph_config.yaml, and this test keeps
a future config edit from quietly reintroducing the sampling.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.config.config import get_config
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
