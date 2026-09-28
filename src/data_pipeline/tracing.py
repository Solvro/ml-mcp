"""Langfuse tracing for the pipeline's LLM calls, so each run's token use and cost is visible."""

import logging
import os
from functools import lru_cache
from typing import Any

from prefect.runtime import flow_run

logger = logging.getLogger(__name__)

PIPELINE_TAG = "data-pipeline"


@lru_cache(maxsize=1)
def _langfuse_handler() -> Any | None:
    """Build the shared callback handler, or None when Langfuse is not configured."""
    if not (os.getenv("LANGFUSE_SECRET_KEY") and os.getenv("LANGFUSE_PUBLIC_KEY")):
        return None
    try:
        from langfuse.langchain import CallbackHandler

        return CallbackHandler()
    except Exception as exc:
        logger.warning("Failed to initialize Langfuse, pipeline tracing disabled: %s", exc)
        return None


def llm_run_config(name: str) -> dict[str, Any]:
    """LangChain run config for one pipeline LLM call.

    Calls from the same Prefect run share a Langfuse session keyed on the root flow run id, so
    the session view shows what a whole refresh or pipeline run spent.
    """
    config: dict[str, Any] = {"run_name": name}
    handler = _langfuse_handler()
    if handler is None:
        return config
    metadata: dict[str, Any] = {"langfuse_tags": [PIPELINE_TAG]}
    session_id = flow_run.root_flow_run_id or flow_run.id
    if session_id:
        metadata["langfuse_session_id"] = str(session_id)
    config["callbacks"] = [handler]
    config["metadata"] = metadata
    return config
