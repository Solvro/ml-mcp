import os

from dotenv import load_dotenv
from fastmcp import FastMCP

from ..config.config import get_config
from .tools.knowledge_graph.rag import RAG

load_dotenv()

mcp = FastMCP("SOLVRO MCP")

rag = None
langfuse = None
handler = None

# Initialize Langfuse only if credentials are configured
_langfuse_secret = os.getenv("LANGFUSE_SECRET_KEY")
_langfuse_public = os.getenv("LANGFUSE_PUBLIC_KEY")
_langfuse_host = os.getenv("LANGFUSE_HOST")

if _langfuse_secret and _langfuse_public:
    try:
        from langfuse import Langfuse
        from langfuse.langchain import CallbackHandler

        langfuse = Langfuse(
            secret_key=_langfuse_secret,
            public_key=_langfuse_public,
            host=_langfuse_host,
        )
        handler = CallbackHandler()
    except Exception as e:
        print(f"Warning: Failed to initialize Langfuse: {e}")
else:
    print("Langfuse credentials not configured. Tracing disabled.")


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
    )

    return rag


@mcp.tool
async def knowledge_graph_tool(user_input: str, trace_id: str = None) -> str:
    """
    Query the knowledge graph with natural language.

    Args:
        user_input: User's question or query
        trace_id: Optional trace ID for tracking

    Returns:
        AI-generated instructions based on knowledge graph data
    """
    if rag is None:
        return "Error: RAG not initialized. Please start the server first."

    result = await rag.ainvoke(message=user_input, trace_id=trace_id, callback_handler=handler)
    print(rag.visualizer.draw_mermaid())

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
