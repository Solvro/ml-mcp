import atexit
import logging
import os

from dotenv import load_dotenv
from fastmcp import FastMCP
from langfuse.langchain import CallbackHandler

from ..config.config import get_config
from ..config.logging_config import configure_logging
from ..config.messages import GRAPH_PIPELINE_TIMEOUT_MESSAGE
from ..config.timeouts import get_graph_timeout_seconds, get_llm_timeout_seconds
from .tools.knowledge_graph.rag import RAG

load_dotenv()
configure_logging()

logger = logging.getLogger(__name__)

mcp = FastMCP("SOLVRO MCP")

rag = None
langfuse = None

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
    """
    if rag is None:
        return "Error: RAG not initialized. Please start the server first."

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
    except TimeoutError:
        return f"Error: {GRAPH_PIPELINE_TIMEOUT_MESSAGE}"

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
