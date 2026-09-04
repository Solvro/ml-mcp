import asyncio
import atexit
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from langfuse.langchain import CallbackHandler
from starlette.requests import Request
from starlette.responses import JSONResponse

from ..config.config import get_config
from ..config.logging_config import configure_logging
from ..config.messages import GRAPH_PIPELINE_TIMEOUT_MESSAGE
from ..config.timeouts import get_graph_timeout_seconds, get_llm_timeout_seconds
from .tools.knowledge_graph.rag import RAG

load_dotenv()
configure_logging()

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)

# The health route answers within this budget even when Neo4j has stopped responding, so the
# container reports unhealthy with a reason instead of having its probe killed mid-query. Keep
# it below the healthcheck timeout in docker/compose.stack.yml.
HEALTH_PING_TIMEOUT_SECONDS = 5.0

rag = None
langfuse = None


def close_rag() -> None:
    """Release the Neo4j driver held by the RAG instance. Safe to call more than once."""
    global rag

    if rag is None:
        return
    try:
        rag.close()
        logger.info("Closed the Neo4j driver")
    except Exception as exc:
        logger.warning("Failed to close the Neo4j driver: %s", exc)
    finally:
        rag = None


@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[None]:
    """Close the graph driver when the server stops.

    RAG opens a Neo4j driver in its constructor and holds it for the process lifetime. Without
    this, every restart leaks one.
    """
    try:
        yield
    finally:
        close_rag()


mcp = FastMCP("SOLVRO MCP", lifespan=lifespan)

# Initialize Langfuse only if credentials are configured
_langfuse_secret = os.getenv("LANGFUSE_SECRET_KEY")
_langfuse_public = os.getenv("LANGFUSE_PUBLIC_KEY")
_langfuse_host = os.getenv("LANGFUSE_HOST")

if _langfuse_secret and _langfuse_public:
    try:
        from langfuse import Langfuse

        langfuse = Langfuse(
            secret_key=_langfuse_secret,
            public_key=_langfuse_public,
            host=_langfuse_host,
        )
        atexit.register(langfuse.flush)
    except Exception as e:
        logger.warning("Failed to initialize Langfuse: %s", e)
else:
    logger.info("Langfuse credentials not configured. Tracing disabled.")


def initialize_rag():
    """Initialize RAG instance with environment variables."""
    global rag

    api_key = (
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("DEEPSEEK_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
    )
    neo4j_uri = os.environ.get("NEO4J_URI")
    neo4j_username = os.environ.get("NEO4J_USER")
    neo4j_password = os.environ.get("NEO4J_PASSWORD")

    if not all([api_key, neo4j_uri, neo4j_username, neo4j_password]):
        raise ValueError("Missing required environment variables. Check .env file.")

    rag = RAG(
        api_key=api_key,
        neo4j_url=neo4j_uri,
        neo4j_username=neo4j_username,
        neo4j_password=neo4j_password,
        graph_timeout_sec=get_graph_timeout_seconds(),
        llm_timeout_sec=get_llm_timeout_seconds(),
    )

    return rag


def _unhealthy(reason: str, detail: str | None = None) -> JSONResponse:
    """Report a server that is running but cannot answer, with why."""
    logger.warning("Health check failed: %s%s", reason, f" ({detail})" if detail else "")
    payload = {"status": "unhealthy", "reason": reason}
    if detail:
        payload["detail"] = detail
    return JSONResponse(payload, status_code=503)


@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> JSONResponse:
    """
    Report whether this server can actually answer, not merely that its port is open.

    A bound socket says nothing about the graph behind it. Anything wiring itself to this
    service with `depends_on: service_healthy` needs the difference, so the check runs a
    trivial query against Neo4j and fails the probe when it cannot.
    """
    if rag is None:
        return _unhealthy("rag_not_initialized")

    try:
        await asyncio.wait_for(
            asyncio.to_thread(rag.ping_database), timeout=HEALTH_PING_TIMEOUT_SECONDS
        )
    except TimeoutError:
        return _unhealthy("neo4j_timeout", f"no answer within {HEALTH_PING_TIMEOUT_SECONDS:g}s")
    except Exception as exc:
        return _unhealthy("neo4j_unreachable", str(exc))

    return JSONResponse({"status": "healthy", "neo4j": "reachable"})


@mcp.tool
async def knowledge_graph_tool(
    user_input: str, trace_id: str = None, session_id: str = None
) -> str:
    """
    Query the knowledge graph with natural language.

    Args:
        user_input: User's question or query
        trace_id: Trace identifier for this single request
        session_id: Conversation session identifier from the calling API

    Returns:
        AI-generated instructions based on knowledge graph data

    Raises:
        ToolError: The graph could not be consulted at all. Reported as a failed tool call so
            the caller can tell it apart from an answer; "no data in the graph" is an answer
            and comes back normally.
    """
    if rag is None:
        raise ToolError("Knowledge graph unavailable: the server has no initialized RAG.")

    per_request_handler = None
    if langfuse is not None:
        per_request_handler = CallbackHandler(
            trace_context={"trace_id": trace_id} if trace_id else None,
        )
    try:
        result = await rag.ainvoke(
            message=user_input,
            session_id=session_id,
            trace_id=trace_id,
            callback_handler=per_request_handler,
        )
    except TimeoutError as exc:
        raise ToolError(GRAPH_PIPELINE_TIMEOUT_MESSAGE) from exc

    metadata = result.get("metadata", {})
    logger.info("Guardrail decision: %s", metadata.get("guardrail_decision"))
    logger.info("Retrieval strategy: %s", metadata.get("retrieval_strategy"))
    logger.debug("Generated Cypher:\n%s", metadata.get("cypher_query"))
    logger.debug("Graph context:\n%s", metadata.get("context"))

    # Return the answer directly (already a JSON string from rag.py)
    return result["answer"]


def main():
    """Main entry point for the MCP server."""
    import os

    global rag

    rag = initialize_rag()

    config = get_config()

    # Use 0.0.0.0 in Docker, config host otherwise
    host = os.getenv("MCP_BIND_HOST", config.servers.mcp.host)

    mcp.run(transport=config.servers.mcp.transport, host=host, port=config.servers.mcp.port)


if __name__ == "__main__":
    main()
