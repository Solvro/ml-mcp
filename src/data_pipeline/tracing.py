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
        from langfuse import Langfuse
        from langfuse.langchain import CallbackHandler

        # Langfuse installs the global OpenTelemetry tracer provider, and Prefect emits a span
        # for every flow and task run through it. Unblocked, each run lands in Langfuse as its
        # own trace named after the run (careful-mantis, ...). Blocking the scope keeps only the
        # LLM calls, which still share the flow run's trace id, so one run stays one trace.
        Langfuse(blocked_instrumentation_scopes=["prefect"])
        return CallbackHandler(update_trace=True)
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
