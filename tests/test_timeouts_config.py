import logging

import pytest

from src.config.config import get_config
from src.config.timeouts import (
    ENV_NEO4J_CONNECTION_TIMEOUT,
    ENV_NEO4J_MAX_RETRY,
    ENV_NEO4J_QUERY_TIMEOUT,
    get_neo4j_connection_timeout_seconds,
    get_neo4j_max_transaction_retry_seconds,
    get_neo4j_query_timeout_seconds,
)


@pytest.fixture(autouse=True)
def _clear_timeout_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    for env_name in (
        ENV_NEO4J_QUERY_TIMEOUT,
        ENV_NEO4J_CONNECTION_TIMEOUT,
        ENV_NEO4J_MAX_RETRY,
    ):
        monkeypatch.delenv(env_name, raising=False)


def test_neo4j_timeout_defaults_come_from_yaml() -> None:
    config = get_config().rag

    assert get_neo4j_query_timeout_seconds() == float(config.neo4j_query_timeout_seconds)
    assert get_neo4j_connection_timeout_seconds() == float(config.neo4j_connection_timeout_seconds)
    assert get_neo4j_max_transaction_retry_seconds() == float(
        config.neo4j_max_transaction_retry_seconds
    )


@pytest.mark.parametrize(
    ("env_name", "env_value", "getter"),
    [
        (ENV_NEO4J_QUERY_TIMEOUT, "3.5", get_neo4j_query_timeout_seconds),
        (ENV_NEO4J_CONNECTION_TIMEOUT, "2.25", get_neo4j_connection_timeout_seconds),
        (ENV_NEO4J_MAX_RETRY, "4.75", get_neo4j_max_transaction_retry_seconds),
    ],
)
def test_neo4j_timeout_overrides_are_read_from_env(
    monkeypatch: pytest.MonkeyPatch, env_name: str, env_value: str, getter
) -> None:
    monkeypatch.setenv(env_name, env_value)

    assert getter() == float(env_value)


@pytest.mark.parametrize("bad_value", ["0", "-1", "nope"])
def test_invalid_neo4j_timeout_override_falls_back_to_yaml(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, bad_value: str
) -> None:
    fallback = float(get_config().rag.neo4j_query_timeout_seconds)
    monkeypatch.setenv(ENV_NEO4J_QUERY_TIMEOUT, bad_value)

    with caplog.at_level(logging.WARNING):
        value = get_neo4j_query_timeout_seconds()

    assert value == fallback
    assert any(ENV_NEO4J_QUERY_TIMEOUT in message for message in caplog.messages)
