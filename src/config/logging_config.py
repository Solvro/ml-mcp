"""Process-wide logging setup: level and format come from the environment.

Entry points call :func:`configure_logging` once, before anything worth logging happens.
Every module keeps using ``logging.getLogger(__name__)``; nothing else needs to know where
the level came from.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

ENV_LOG_LEVEL = "LOG_LEVEL"
ENV_LOG_FORMAT = "LOG_FORMAT"

DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

_configured = False


def _resolve_level() -> tuple[int, str | None]:
    """Read LOG_LEVEL. Return the level plus the raw value when it was unusable."""
    raw = os.getenv(ENV_LOG_LEVEL, "").strip()
    if not raw:
        return DEFAULT_LOG_LEVEL, None

    level = logging.getLevelNamesMapping().get(raw.upper())
    if level is None:
        return DEFAULT_LOG_LEVEL, raw
    return level, None


def get_log_level() -> int:
    """Log level for this process. Env LOG_LEVEL wins; invalid values fall back to INFO."""
    level, _ = _resolve_level()
    return level


def get_log_format() -> str:
    """Log line format for this process. Env LOG_FORMAT wins."""
    return os.getenv(ENV_LOG_FORMAT, "").strip() or DEFAULT_LOG_FORMAT


def configure_logging(*, force: bool = False) -> int:
    """
    Configure root logging from the environment.

    Idempotent: later calls are no-ops, so an entry point that imports another one does not
    reconfigure it. An existing handler (uvicorn, Prefect) is left in place and only the
    level is applied, because stealing it would drop those frameworks' formatting.

    Args:
        force: Re-run the configuration even if this process already did it

    Returns:
        The level that was applied
    """
    global _configured

    level, invalid = _resolve_level()

    if _configured and not force:
        return level

    logging.basicConfig(level=level, format=get_log_format())
    logging.getLogger().setLevel(level)
    _configured = True

    if invalid:
        logger.warning(
            "Invalid %s=%r; falling back to %s",
            ENV_LOG_LEVEL,
            invalid,
            logging.getLevelName(DEFAULT_LOG_LEVEL),
        )

    return level
