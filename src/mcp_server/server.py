import os

from dotenv import load_dotenv
from fastmcp import FastMCP
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

from .tools.knowledge_graph.rag import RAG
from .tools.offers_db.rag import RAG as Karierownik # xd
load_dotenv()

mcp = FastMCP("SOLVRO MCP")

rag = None
karierownik = None

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

def initialize_karierownik():
    """Initialize Karierownik instance with environment variables."""
    global karierownik

    karierownik = Karierownik()

    return karierownik


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

@mcp.tool
async def karierownik_tool(user_input: str, trace_id: str = None) -> str:
    """
    Retrieve internship and apprenticeship offers with natural language.

    This tool searches a vector database containing internship and apprenticeship
    offers. It should always be used whenever a recommendation of relevant offers
    is requested.

    Returns:
    - Ranked list of internship or apprenticeship offers from the vector database
      that are most semantically similar to the input description, optionally
      filtered by company inclusion or exclusion.  
      The returned offer links can later be used with the `get_offer_details` tool
      to retrieve detailed information about each offer.
    """
    if karierownik is None:
        return "Error: Karierownik not initialized. Please start the server first."

    result = await karierownik.ainvoke(message=user_input, trace_id=trace_id, callback_handler=handler)
    return result["answer"]

def main():
    """Main entry point for the MCP server."""
    global rag
    global karierownik

    # rag = initialize_rag()
    karierownik = initialize_karierownik()

    mcp.run(transport="http", port=8005)


if __name__ == "__main__":
    main()
