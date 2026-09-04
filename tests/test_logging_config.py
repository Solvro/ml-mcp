"""Log level comes from the environment, and a bad value must not silence the process.

Issue #63 replaced every print with a logger, so the only remaining way to see what a service
is doing is LOG_LEVEL. A typo there falling back to silence would be worse than the prints it
replaced, which is what these tests pin down.
"""

import logging

import pytest

from src.config import logging_config
from src.config.logging_config import (
    DEFAULT_LOG_FORMAT,
    DEFAULT_LOG_LEVEL,
    configure_logging,
    get_log_format,
    get_log_level,
)


@pytest.fixture(autouse=True)
def _reset_configured_flag():
    """configure_logging is process-wide; keep each test from seeing the previous one."""
    logging_config._configured = False
    yield
    logging_config._configured = False


def test_missing_log_level_falls_back_to_info(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LOG_LEVEL", raising=False)

    assert get_log_level() == DEFAULT_LOG_LEVEL


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("DEBUG", logging.DEBUG),
        ("debug", logging.DEBUG),
        (" warning ", logging.WARNING),
        ("ERROR", logging.ERROR),
        ("CRITICAL", logging.CRITICAL),
    ],
)
def test_log_level_is_read_from_the_environment(
    monkeypatch: pytest.MonkeyPatch, raw: str, expected: int
) -> None:
    monkeypatch.setenv("LOG_LEVEL", raw)

    assert get_log_level() == expected


@pytest.mark.parametrize("raw", ["", "   ", "LOUD", "42"])
def test_unusable_log_level_falls_back_to_info(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    monkeypatch.setenv("LOG_LEVEL", raw)

    assert get_log_level() == DEFAULT_LOG_LEVEL


def test_log_format_is_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOG_FORMAT", "%(levelname)s %(message)s")

    assert get_log_format() == "%(levelname)s %(message)s"


def test_empty_log_format_falls_back_to_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOG_FORMAT", "   ")

    assert get_log_format() == DEFAULT_LOG_FORMAT


def test_configure_logging_applies_the_level_to_the_root_logger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    root = logging.getLogger()
    original = root.level
    try:
        assert configure_logging() == logging.WARNING
        assert root.level == logging.WARNING
    finally:
        root.setLevel(original)


def test_configure_logging_runs_once_per_process(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "ERROR")
    root = logging.getLogger()
    original = root.level
    try:
        configure_logging()

        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        assert configure_logging() == logging.DEBUG
        assert root.level == logging.ERROR, "second call must not reconfigure the root logger"

        assert configure_logging(force=True) == logging.DEBUG
        assert root.level == logging.DEBUG
    finally:
        root.setLevel(original)


def test_a_bad_level_still_configures_logging_and_warns(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("LOG_LEVEL", "LOUD")
    root = logging.getLogger()
    original = root.level
    try:
        with caplog.at_level(logging.WARNING, logger="src.config.logging_config"):
            assert configure_logging() == DEFAULT_LOG_LEVEL

        assert root.level == DEFAULT_LOG_LEVEL
        assert "LOUD" in caplog.text
    finally:
        root.setLevel(original)
