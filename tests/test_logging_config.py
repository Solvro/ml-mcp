"""Log level comes from the environment, and a bad value must not silence the process.

Issue #63 replaced every print with a logger, so the only remaining way to see what a service
is doing is LOG_LEVEL. A typo there falling back to silence would be worse than the prints it
replaced, which is what these tests pin down.
"""

import logging
import logging.config

import pytest
from uvicorn.config import LOGGING_CONFIG as UVICORN_LOGGING_CONFIG

from src.config import logging_config
from src.config.logging_config import (
    DEFAULT_LOG_FORMAT,
    DEFAULT_LOG_LEVEL,
    HEALTH_CHECK_PATH,
    UVICORN_ACCESS_LOGGER,
    HealthCheckAccessFilter,
    configure_logging,
    get_log_format,
    get_log_level,
)

# The exact call uvicorn's h11/httptools protocols make for every request.
UVICORN_ACCESS_FORMAT = '%s - "%s %s HTTP/%s" %d'


def _health_filters() -> list[logging.Filter]:
    access_logger = logging.getLogger(UVICORN_ACCESS_LOGGER)
    return [f for f in access_logger.filters if isinstance(f, HealthCheckAccessFilter)]


@pytest.fixture(autouse=True)
def _reset_configured_flag():
    """configure_logging is process-wide; keep each test from seeing the previous one."""
    logging_config._configured = False
    yield
    logging_config._configured = False
    access_logger = logging.getLogger(UVICORN_ACCESS_LOGGER)
    for health_filter in _health_filters():
        access_logger.removeFilter(health_filter)


def _access_record(path: str) -> logging.LogRecord:
    return logging.LogRecord(
        name=UVICORN_ACCESS_LOGGER,
        level=logging.INFO,
        pathname=__file__,
        lineno=0,
        msg=UVICORN_ACCESS_FORMAT,
        args=("127.0.0.1:54321", "GET", path, "1.1", 200),
        exc_info=None,
    )


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


def test_the_health_probe_is_dropped_from_the_access_log() -> None:
    # Built from bytes, the way uvicorn builds the path: equal to the constant, never the
    # same object, so an identity comparison would let every probe through.
    path = b"/health".decode()

    assert not HealthCheckAccessFilter().filter(_access_record(path))


@pytest.mark.parametrize("path", ["/mcp", "/healthz", "/mcp?next=/health", "/"])
def test_every_other_request_stays_in_the_access_log(path: str) -> None:
    assert HealthCheckAccessFilter().filter(_access_record(path))


@pytest.mark.parametrize("args", [(), ("only", "two"), None, {"path": HEALTH_CHECK_PATH}])
def test_a_record_of_another_shape_is_kept_rather_than_raising(args: object) -> None:
    record = logging.LogRecord(
        UVICORN_ACCESS_LOGGER, logging.INFO, __file__, 0, "plain message", None, None
    )
    record.args = args

    assert HealthCheckAccessFilter().filter(record)


def test_configure_logging_attaches_the_filter_to_the_access_logger() -> None:
    configure_logging()

    assert len(_health_filters()) == 1


def test_reconfiguring_does_not_stack_the_filter() -> None:
    configure_logging()
    configure_logging(force=True)
    configure_logging(force=True)

    assert len(_health_filters()) == 1


def test_the_filter_survives_uvicorn_configuring_its_own_loggers(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """configure_logging runs at import; uvicorn applies its dictConfig later, on mcp.run()."""
    access_logger = logging.getLogger(UVICORN_ACCESS_LOGGER)
    touched = [logging.getLogger(name) for name in UVICORN_LOGGING_CONFIG["loggers"]]
    saved = [(lg, lg.handlers[:], lg.level, lg.propagate) for lg in touched]
    try:
        configure_logging()
        logging.config.dictConfig(UVICORN_LOGGING_CONFIG)
        access_logger.addHandler(caplog.handler)

        access_logger.info(UVICORN_ACCESS_FORMAT, "127.0.0.1:1", "GET", "/health", "1.1", 200)
        access_logger.info(UVICORN_ACCESS_FORMAT, "127.0.0.1:1", "POST", "/mcp", "1.1", 200)

        messages = [record.getMessage() for record in caplog.records]
        assert not [message for message in messages if "/health" in message]
        assert [message for message in messages if "POST /mcp" in message]
    finally:
        for lg, handlers, level, propagate in saved:
            lg.handlers[:] = handlers
            lg.setLevel(level)
            lg.propagate = propagate
