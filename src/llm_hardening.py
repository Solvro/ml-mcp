"""Shared timeout settings for LLM calls."""

from __future__ import annotations

import os

DEFAULT_LLM_TIMEOUT_SECONDS = 30.0


def get_llm_timeout_seconds() -> float:
    """Return the configured positive LLM timeout in seconds."""
    raw_value = os.getenv("LLM_TIMEOUT_SECONDS", "").strip()
    if not raw_value:
        return DEFAULT_LLM_TIMEOUT_SECONDS

    try:
        timeout = float(raw_value)
    except ValueError:
        return DEFAULT_LLM_TIMEOUT_SECONDS

    return timeout if timeout > 0 else DEFAULT_LLM_TIMEOUT_SECONDS
