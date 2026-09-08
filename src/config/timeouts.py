"""Resolve timeout/runtime knobs: env override, else yaml."""

from __future__ import annotations

import logging
import os

from .config import get_config

logger = logging.getLogger(__name__)

ENV_LLM_TIMEOUT = "LLM_TIMEOUT_SECONDS"
ENV_GRAPH_TIMEOUT = "GRAPH_TIMEOUT_SECONDS"
ENV_NEO4J_QUERY_TIMEOUT = "NEO4J_QUERY_TIMEOUT_SECONDS"
ENV_NEO4J_CONNECTION_TIMEOUT = "NEO4J_CONNECTION_TIMEOUT_SECONDS"
ENV_NEO4J_MAX_RETRY = "NEO4J_MAX_TRANSACTION_RETRY_SECONDS"
ENV_SCHEMA_REFRESH = "SCHEMA_REFRESH_SECONDS"
ENV_SCHEMA_VERSION_PROBE = "SCHEMA_VERSION_PROBE_SECONDS"


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


def get_neo4j_query_timeout_seconds() -> float:
    """Per-query Neo4j timeout. Env overrides yaml rag.neo4j_query_timeout_seconds."""
    from_env = _parse_positive_float(
        os.getenv(ENV_NEO4J_QUERY_TIMEOUT, ""),
        env_name=ENV_NEO4J_QUERY_TIMEOUT,
    )
    if from_env is not None:
        return from_env
    return float(get_config().rag.neo4j_query_timeout_seconds)


def get_neo4j_connection_timeout_seconds() -> float:
    """Neo4j connection timeout. Env overrides yaml rag.neo4j_connection_timeout_seconds."""
    from_env = _parse_positive_float(
        os.getenv(ENV_NEO4J_CONNECTION_TIMEOUT, ""),
        env_name=ENV_NEO4J_CONNECTION_TIMEOUT,
    )
    if from_env is not None:
        return from_env
    return float(get_config().rag.neo4j_connection_timeout_seconds)


def get_neo4j_max_transaction_retry_seconds() -> float:
    """Neo4j driver retry budget. Env overrides yaml rag.neo4j_max_transaction_retry_seconds."""
    from_env = _parse_positive_float(
        os.getenv(ENV_NEO4J_MAX_RETRY, ""),
        env_name=ENV_NEO4J_MAX_RETRY,
    )
    if from_env is not None:
        return from_env
    return float(get_config().rag.neo4j_max_transaction_retry_seconds)


def get_schema_refresh_seconds() -> float:
    """How long a fetched Neo4j schema stays usable. Env overrides yaml."""
    from_env = _parse_positive_float(
        os.getenv(ENV_SCHEMA_REFRESH, ""),
        env_name=ENV_SCHEMA_REFRESH,
    )
    if from_env is not None:
        return from_env
    return float(get_config().rag.schema_refresh_seconds)


def get_schema_version_probe_seconds() -> float:
    """How often the graph-version marker may be re-read. Env overrides yaml."""
    from_env = _parse_positive_float(
        os.getenv(ENV_SCHEMA_VERSION_PROBE, ""),
        env_name=ENV_SCHEMA_VERSION_PROBE,
    )
    if from_env is not None:
        return from_env
    return float(get_config().rag.schema_version_probe_seconds)
