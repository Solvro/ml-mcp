import json
import os

from dotenv import load_dotenv
from fastmcp import FastMCP

from ..config.config import get_config
from .tools.knowledge_graph.tool import Tool as KnowledgeGraphTool
from .tools.offers_db.tool import Tool as OffersDBTool

load_dotenv()

mcp = FastMCP("SOLVRO MCP")

knowledge_graph_tool = None
offers_db_tool = None


def initialize_knowledge_graph_tool():
    """Initialize Knowledge Graph Tool instance with environment variables."""
    global knowledge_graph_tool


def initialize_offers_db_tool():
    """Initialize Offers DB Tool instance."""
    global offers_db_tool

    offers_db_tool = OffersDBTool()

    return offers_db_tool


@mcp.tool
async def knowledge_graph_tool(cypher_query: str) -> str:
    """
    Execute a Cypher query against the Neo4j knowledge graph.
    This tool should be used when the user asks about the Wrocław University of Science and Technology.
    It should always be used whenever a question about the university is asked.

    Args:
        cypher_query: Cypher query string to execute

    Returns:
        JSON string with query results
    """
    if knowledge_graph_tool is None:
        return json.dumps({"error": "Knowledge Graph Tool not initialized. Please start the server first."})

    try:
        results = await knowledge_graph_tool.ainvoke(cypher_query)
        return json.dumps(results, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool
async def offers_db_tool(
    internship_info: str,
    include_companies: list[str] | None = None,
    exclude_companies: list[str] | None = None,
    limit: int = 5,
    offset: int = 0,
) -> str:
    """
    This tool searches a vector database containing internship and apprenticeship
    offers. Always use this tool when the user asks for offers.

    Parameters:
    - internship_info: Free-text input describing an internship or apprenticeship,
      similar in style to a job posting (e.g., required skills, role, responsibilities).
      The tool will use this description to find semantically matching offers
      from the indexed dataset.

    - include_companies: Optional. A list of company names to explicitly include 
      in the search results.  
      Use this parameter **only when the user explicitly requests offers from one 
      or more specific companies** (e.g., “show me offers from Sii and Nokia”).  
      Each listed company name will be used as a filter to include only offers 
      from those companies.

    - exclude_companies: Optional. A list of company names to explicitly exclude 
      from the search results.  
      Use this parameter **only when the user asks to exclude offers from certain 
      companies** (e.g., “show me offers not from Sii” or “show me offers from 
      other companies than Nokia”).  
      Offers from any company in this list will be filtered out.

    - limit: Optional. The maximum number of offers to return (default = 5).  
      Use this parameter **only if the user explicitly specifies** how many offers 
      they want to see (e.g., “show me 10 offers”).  
      Otherwise, do not include it in the call — the default value of 5 will be used automatically.

    - offset: Optional. Used to skip a given number of top-ranked results (default = 0).  
      Use this parameter **when the user asks for other or new offers** after already 
      receiving some (e.g., “show me different ones” or “what else do you have?”).  
      In such cases, pass an offset equal to the number of previously shown offers 
      (e.g., offset = 5 if the previous call returned 5 offers).

    Returns:
    - Ranked list of internship or apprenticeship offers from the vector database
      that are most semantically similar to the input description, optionally
      filtered by company inclusion or exclusion.  
      The returned offer links can later be used with the `get_offer_details` tool
      to retrieve detailed information about each offer.
    """
    if offers_db_tool is None:
        return json.dumps({"error": "Offers DB Tool not initialized. Please start the server first."})

    try:
        results = await offers_db_tool.ainvoke(
            internship_info=internship_info,
            include_companies=include_companies,
            exclude_companies=exclude_companies,
            limit=limit,
            offset=offset,
        )
        return json.dumps(results, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


def main():
    """Main entry point for the MCP server."""
    global knowledge_graph_tool
    global offers_db_tool

    # Initialize tools
    knowledge_graph_tool = initialize_knowledge_graph_tool()
    offers_db_tool = initialize_offers_db_tool()

    # Load configuration
    config = get_config()
    
    # Use config for server settings (host can be overridden by env for Docker)
    host = os.getenv("MCP_SERVER_HOST", config.servers.mcp.host)
    port = int(os.getenv("MCP_SERVER_PORT", str(config.servers.mcp.port)))
    transport = config.servers.mcp.transport
    
    # Run the MCP server
    mcp.run(transport=transport, host=host, port=port)


if __name__ == "__main__":
    main()
