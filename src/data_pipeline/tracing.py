"""Langfuse tracing for the pipeline's LLM calls, so each run's token use and cost is visible."""

import logging
import os
from functools import lru_cache
from typing import Any

from prefect.runtime import flow_run

logger = logging.getLogger(__name__)

PIPELINE_TAG = "data-pipeline"
# Langfuse user the pipeline's calls are filed under, so its total cost sits beside real users.
PIPELINE_USER_ID = "data-pipeline"


@lru_cache(maxsize=1)
def _langfuse_enabled() -> bool:
    """Initialize the Langfuse client once; False when it is not configured or fails."""
    if not (os.getenv("LANGFUSE_SECRET_KEY") and os.getenv("LANGFUSE_PUBLIC_KEY")):
        return False
    try:
        from langfuse import Langfuse

        # Langfuse installs the global OpenTelemetry tracer provider, and Prefect emits a span
        # for every flow and task run through it. Unblocked, each run lands in Langfuse as its
        # own trace named after the run (careful-mantis, ...).
        Langfuse(blocked_instrumentation_scopes=["prefect"])
        return True
    except Exception as exc:
        logger.warning("Failed to initialize Langfuse, pipeline tracing disabled: %s", exc)
        return False


def llm_run_config(name: str) -> dict[str, Any]:
    """LangChain run config for one pipeline LLM call.

    Each call is its own trace, started from a fresh trace id rather than from the Prefect span
    around it: that span is never exported, so a call parented on it would sit under a parent
    Langfuse never sees, and calls sharing one trace would each overwrite its name and input.
    Calls from the same flow run share a Langfuse session, so the session view shows what a
    whole pipeline run spent. Tracing never fails the call: any error here drops the callbacks.
    """
    config: dict[str, Any] = {"run_name": name}
    if not _langfuse_enabled():
        return config
    try:
        from langfuse import Langfuse
        from langfuse.langchain import CallbackHandler

        handler = CallbackHandler(
            update_trace=True, trace_context={"trace_id": Langfuse.create_trace_id()}
        )
        metadata: dict[str, Any] = {
            "langfuse_tags": [PIPELINE_TAG],
            "langfuse_user_id": PIPELINE_USER_ID,
        }
        if flow_run.id:
            metadata["langfuse_session_id"] = str(flow_run.id)
    except Exception as exc:
        logger.warning("Langfuse tracing skipped for %s: %s", name, exc)
        return config
    config["callbacks"] = [handler]
    config["metadata"] = metadata
    return config
