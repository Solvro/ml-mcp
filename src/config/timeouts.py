"""Resolve LLM/graph timeouts: optional env override, else graph_config.yaml."""

from __future__ import annotations

import logging
import os

from src.config.config import get_config

logger = logging.getLogger(__name__)

ENV_LLM_TIMEOUT = "LLM_TIMEOUT_SECONDS"
ENV_GRAPH_TIMEOUT = "GRAPH_TIMEOUT_SECONDS"


def _parse_positive_float(raw: str, *, env_name: str) -> float | None:
    """Parse env value; return None if empty. Warn and return None if invalid."""
    value = raw.strip()
    if not value:
        return None
    try:
        parsed = float(value)
    except ValueError:
        logger.warning(
            "Invalid %s=%r; falling back to graph_config.yaml",
            env_name,
            raw,
        )
        return None
    if parsed <= 0:
        logger.warning(
            "Non-positive %s=%r; falling back to graph_config.yaml",
            env_name,
            raw,
        )
        return None
    return parsed


def get_llm_timeout_seconds() -> float:
    """Per-call LLM HTTP timeout. Env overrides yaml rag.llm_timeout_seconds."""
    from_env = _parse_positive_float(
        os.getenv(ENV_LLM_TIMEOUT, ""),
        env_name=ENV_LLM_TIMEOUT,
    )
    if from_env is not None:
        return from_env
    return float(get_config().rag.llm_timeout_seconds)


def get_graph_timeout_seconds() -> float:
    """Full LangGraph RAG wall-clock budget. Env overrides yaml."""
    from_env = _parse_positive_float(
        os.getenv(ENV_GRAPH_TIMEOUT, ""),
        env_name=ENV_GRAPH_TIMEOUT,
    )
    if from_env is not None:
        return from_env
    return float(get_config().rag.graph_timeout_seconds)
