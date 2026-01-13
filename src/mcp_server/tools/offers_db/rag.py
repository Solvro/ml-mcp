import json
from datetime import date, datetime
from typing import Any, Dict

from langchain_openai.chat_models.base import BaseChatOpenAI
from langfuse.langchain import CallbackHandler
from langgraph.graph import END, START, StateGraph
from langchain_core.tools import tool

from src.mcp_server.tools.offers_db.settings import Settings
from src.mcp_server.tools.offers_db.offers_db import OffersDB
from src.mcp_server.tools.offers_db.state import State


@tool
async def retrieve_offers(internship_info: str, 
                          include_companies: list[str] | None = None,
                          exclude_companies: list[str] | None = None,
                          limit: int = 5, offset: int = 0):
    """
    Retrieve internship and apprenticeship offers based on semantic similarity.

    This tool searches a vector database containing internship and apprenticeship
    offers. It should always be used whenever a recommendation of relevant offers
    is requested.

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
    print('Querying with description:', internship_info, 'Include companies:', include_companies, 'Exclude companies:', exclude_companies, 'Limit:', limit, 'Offset:', offset)

    if include_companies:
        include_filters = {'company': include_companies}
    else:
        include_filters = None
    if exclude_companies:
        exclude_filters = {'company': exclude_companies}
    else:
        exclude_filters = None

    results = OffersDB.similarity_search_cosine(query=internship_info, k=limit, offset=offset, include_filters=include_filters, exclude_filters=exclude_filters)
    print('Found results:', results)
    return results


class RAG:
    settings = Settings()
    def __init__(self):

        self.llm = BaseChatOpenAI(
            model="gpt-5-mini",
            api_key=self.settings.OPENAI_API_KEY.get_secret_value(),
            temperature=0,
        )
        tools = [retrieve_offers]

        self.tools_map = {tool.name: tool for tool in tools}
        self.llm_w_tools = self.llm.bind_tools(tools)

        self.graph = self._build_processing_graph()

        self.handler = None

    @property
    def system_message(self):
        return """
You are a helpful assistant whose goal is to find the best internship or apprenticeship offers for the user.

When recommending an offer, always include:
- Link to the offer in markdown format: [Offer title] (offer_link)  
  (If no title is available, use the company name instead.)
- Company name
- Short description of the offer (max 2-3 sentences)

If available, you may also include additional details:
- Location
- Contract type
- Date posted
- Closing date

When showing offer details, always include full information about the offer as returned by the tool.

Instructions:
- Always use the `retrieve_offers` tool when asked to find or recommend internship or apprenticeship offers.
- Always use the `get_offer_details` tool when asked to get details about an offer that was previously recommended.
- In all other cases, respond based on your knowledge without using any tools.
- When the user asks for the highest-paying or best-paid offers (either in general or in a specific company):
  * Do NOT use any tools.
  * Always respond that salary information is not available and that you do not have access to salary data.
  * Suggest instead that you can help find internship or apprenticeship offers that best match the user's skills, interests, or goals.
- When the user asks about salary, pay, or compensation for a specific offer that has already been retrieved:
  * You may use the `get_offer_details` tool to check whether salary information is available.
  * If the salary information is not present in the returned data, inform the user that this information is not available and that you do not have access to it.
  * Then respond to the user accordingly.
"""

    def _build_processing_graph(self):
        """Construct the state machine graph for the RAG pipeline."""
        builder = StateGraph(State)

        nodes = [
            ("retrieve", self.retrieve),
        ]

        for node_name, node_func in nodes:
            builder.add_node(node_name, node_func)

        builder.add_edge(START, "retrieve")

        builder.add_edge("retrieve", END)

        return builder.compile()


    async def retrieve(self, state: State):
        """
        Execute query against offers database and retrieve results.
        If query fails, return error message.

        Args:
            state: Current pipeline state

        Returns:
            Updated state with retrieved context
        """
        query = state.get("user_question", "")

        # Helper function to serialize dates
        def serialize_dates(obj):
            if isinstance(obj, date):
                return obj.isoformat()
            elif isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: serialize_dates(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [serialize_dates(item) for item in obj]
            return obj

        try:
            messages = [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": query},
            ]
            response = await self.llm_w_tools.ainvoke(messages)
            context = []
            if tool_calls:=response.tool_calls:
                for tool_call in tool_calls:
                    tool = self.tools_map.get(tool_call["name"]) 
                    try:
                        tool_message = await tool.ainvoke(tool_call)
                        context = tool_message.content
                        # Serialize dates immediately when we get the context
                        if isinstance(context, (list, dict)):
                            context = serialize_dates(context)
                    except Exception as e:
                        context = f"Couldn't use tool: {tool_call['name']}, because of {e}."
                        return {"context": context}

            return {"context": context}

        except Exception as e:
            error_msg = str(e)

            return {"context": [], "generated_cypher": f"Query failed: {error_msg}"}


    def invoke(self, message: str, session_id: str = "default") -> Dict[str, Any]:
        """
        Execute the RAG pipeline with user message.

        Args:
            message: User's question/input
            session_id: Session identifier for tracking

        Returns:
            Dictionary with context from graph or "W bazie danych nie ma informacji"
        """
        result = self.graph.invoke({"user_question": message})

        context_data = result.get("context", [])
        context_json = json.dumps(context_data, ensure_ascii=False, indent=2)

        return {
            "answer": context_json,
            "metadata": {
                "context": context_data,
            },
        }

    async def ainvoke(
        self,
        message: str,
        session_id: str = "default",
        trace_id: str = "default",
        callback_handler: CallbackHandler = None,
    ) -> Dict[str, Any]:
        """
        Async version of invoke for better performance in concurrent scenarios.

        Args:
            message: User's question/input
            session_id: Session identifier for tracking

        Returns:
            Dictionary with context from graph or "W bazie danych nie ma informacji"
        """
        self.handler = callback_handler

        result = await self.graph.ainvoke({"user_question": message, "trace_id": trace_id})

        context_data = result.get("context", [])    
        context_json = json.dumps(context_data, ensure_ascii=False, indent=2)

        return {
            "answer": context_json,
            "metadata": {
                "context": context_data,
            },
        }
