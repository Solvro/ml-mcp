from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    RateLimitError,
)

from src.mcp_server.tools.knowledge_graph.rag import (
    PROVIDER_FALLBACK_EXCEPTIONS,
    RAG,
    LLMProvider,
)


def _rag_stub() -> RAG:
    """Build a RAG instance without running __init__ (no network, no database)."""
    rag = object.__new__(RAG)
    rag.api_key = "test-key"
    rag.llm_timeout_sec = 30.0
    rag.config = SimpleNamespace(
        llm=SimpleNamespace(
            fast_model=SimpleNamespace(name="fast-model", temperature=0.1),
            accurate_model=SimpleNamespace(name="accurate-model", temperature=0),
            gemini=SimpleNamespace(name="gemini-test"),
            deepseek=SimpleNamespace(
                base_url="https://api.deepseek.test",
                fast_model="deepseek-fast",
                accurate_model="deepseek-accurate",
            ),
            provider_fallback_order=["openai", "deepseek", "google"],
        )
    )
    return rag


@pytest.fixture(autouse=True)
def _clear_provider_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep provider selection independent of the developer's own environment."""
    for name in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize(
    ("configured_order", "expected_providers"),
    [
        (["openai", "deepseek", "google"], [LLMProvider.OPENAI, LLMProvider.GOOGLE]),
        (["google", "openai"], [LLMProvider.GOOGLE, LLMProvider.OPENAI]),
    ],
)
def test_providers_follow_config_order_and_skip_missing_keys(
    monkeypatch: pytest.MonkeyPatch,
    configured_order: list[str],
    expected_providers: list[LLMProvider],
) -> None:
    """Order comes from config; providers without an API key are dropped."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("GOOGLE_API_KEY", "sk-google")

    rag = _rag_stub()
    rag.config.llm.provider_fallback_order = configured_order

    assert rag._get_configured_providers() == expected_providers


def test_single_provider_is_not_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    """With one provider there is nothing to fall back to - skip the wrapper."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")

    rag = _rag_stub()
    primary = MagicMock(name="primary")

    with patch.object(rag, "_build_chat_model", return_value=primary):
        result = rag._build_llm_with_fallback(use_accurate=False)

    assert result is primary
    primary.with_fallbacks.assert_not_called()


def test_providers_are_chained_in_configured_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek")

    rag = _rag_stub()
    primary = MagicMock(name="primary")
    secondary = MagicMock(name="secondary")
    primary.with_fallbacks.return_value = MagicMock(name="chained")

    with patch.object(rag, "_build_chat_model", side_effect=[primary, secondary]):
        result = rag._build_llm_with_fallback(use_accurate=True)

    args, kwargs = primary.with_fallbacks.call_args
    assert result is primary.with_fallbacks.return_value
    assert args[0] == [secondary]
    assert kwargs["exceptions_to_handle"] == PROVIDER_FALLBACK_EXCEPTIONS


@pytest.mark.parametrize("transient_error", [APITimeoutError, RateLimitError])
def test_transient_errors_trigger_provider_switch(transient_error: type[Exception]) -> None:
    assert issubclass(transient_error, PROVIDER_FALLBACK_EXCEPTIONS)


@pytest.mark.parametrize("client_error", [AuthenticationError, BadRequestError])
def test_client_errors_never_trigger_provider_switch(client_error: type[Exception]) -> None:
    """A bad key or a malformed request must fail loudly, not silently cost twice."""
    assert not issubclass(client_error, PROVIDER_FALLBACK_EXCEPTIONS)


def test_missing_all_keys_raises_runtime_error() -> None:
    with pytest.raises(RuntimeError, match="No LLM provider"):
        _rag_stub()._build_llm_with_fallback()


@patch("src.mcp_server.tools.knowledge_graph.rag.BaseChatOpenAI")
def test_deepseek_client_uses_its_own_model_and_base_url(
    mock_openai: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DeepSeek must not inherit OpenAI model names - they do not exist there."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek")

    _rag_stub()._build_chat_model(LLMProvider.DEEPSEEK, use_accurate=True)

    kwargs = mock_openai.call_args.kwargs
    assert kwargs["model"] == "deepseek-accurate"
    assert kwargs["base_url"] == "https://api.deepseek.test"
    assert kwargs["timeout"] == 30.0


@pytest.mark.parametrize("provider", list(LLMProvider))
@patch("src.mcp_server.tools.knowledge_graph.rag.ChatGoogleGenerativeAI")
@patch("src.mcp_server.tools.knowledge_graph.rag.BaseChatOpenAI")
def test_clients_keep_the_sdk_retry_budget_at_zero(
    mock_openai: MagicMock, mock_google: MagicMock, provider: LLMProvider
) -> None:
    _rag_stub()._build_chat_model(provider)

    client = mock_google if provider is LLMProvider.GOOGLE else mock_openai
    assert client.call_args.kwargs["max_retries"] == 0


ANSWER = "answer"
TRANSIENT_ERROR = APIConnectionError(request=httpx.Request("POST", "https://api.openai.test/v1"))


class CountingChain:
    """Chain stand-in that fails a fixed number of times, then answers."""

    def __init__(self, failures: int, error: Exception = TRANSIENT_ERROR) -> None:
        self.failures = failures
        self.error = error
        self.calls = 0

    def invoke(self, payload: dict[str, Any], config: dict[str, Any] | None = None) -> str:
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error
        return ANSWER


def _invoke_with_retry(chain: CountingChain, *, single_provider: bool) -> str:
    rag = _rag_stub()
    rag.single_provider = single_provider
    return rag._invoke_with_single_provider_retry(
        chain=chain, payload={}, invoke_config={}, operation_name="Guardrails"
    )


def test_a_blip_is_retried_when_no_provider_can_take_over() -> None:
    chain = CountingChain(failures=1)

    assert _invoke_with_retry(chain, single_provider=True) == ANSWER
    assert chain.calls == 2


def test_the_retry_is_made_once_and_then_gives_up() -> None:
    chain = CountingChain(failures=2)

    with pytest.raises(APIConnectionError):
        _invoke_with_retry(chain, single_provider=True)

    assert chain.calls == 2


def test_a_chain_that_can_switch_providers_is_left_alone() -> None:
    chain = CountingChain(failures=1)

    with pytest.raises(APIConnectionError):
        _invoke_with_retry(chain, single_provider=False)

    assert chain.calls == 1


def test_a_timed_out_call_is_not_repeated() -> None:
    chain = CountingChain(
        failures=1,
        error=APITimeoutError(request=httpx.Request("POST", "https://api.openai.test/v1")),
    )

    with pytest.raises(APITimeoutError):
        _invoke_with_retry(chain, single_provider=True)

    assert chain.calls == 1


def test_a_rate_limit_is_not_repeated_immediately() -> None:
    chain = CountingChain(
        failures=1,
        error=RateLimitError(
            "rate limited",
            response=httpx.Response(
                429, request=httpx.Request("POST", "https://api.openai.test/v1")
            ),
            body=None,
        ),
    )

    with pytest.raises(RateLimitError):
        _invoke_with_retry(chain, single_provider=True)

    assert chain.calls == 1


def test_a_client_error_is_never_repeated() -> None:
    chain = CountingChain(
        failures=1,
        error=AuthenticationError(
            "invalid api key",
            response=httpx.Response(
                401, request=httpx.Request("POST", "https://api.openai.test/v1")
            ),
            body=None,
        ),
    )

    with pytest.raises(AuthenticationError):
        _invoke_with_retry(chain, single_provider=True)

    assert chain.calls == 1
