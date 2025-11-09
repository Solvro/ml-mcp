import os

from dotenv import load_dotenv
from fastmcp import FastMCP
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

from .tools.knowledge_graph.rag import RAG

load_dotenv()

mcp = FastMCP("SOLVRO MCP")

rag = None

langfuse = Langfuse(
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
)


handler = CallbackHandler()


def initialize_rag():
    """Initialize RAG instance with environment variables."""
    global rag

    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
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
    global rag

    rag = initialize_rag()

    mcp.run(transport="http", port=8005)


if __name__ == "__main__":
    main()
