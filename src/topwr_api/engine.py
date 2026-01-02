import os
import logging
from dotenv import load_dotenv
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

logger = logging.getLogger(__name__)

class LLMEngine:
    def __init__(self):
        api_key = os.getenv("CLARIN_API_KEY")
        if not api_key:
            logger.warning("CLARIN_API_KEY is missing in env var")

        self.llm= ChatOpenAI(
            model_name = "pllum",
            base_url="https://services.clarin-pl.eu/api/v1/oapi",
            api_key=os.getenv("CLARIN_API_KEY"),
            temperature=0
        )

    def _get_mock_mcp_data(self, query: str) -> str:
        """
        Docstring for _get_mock_mcp_data
        Simulates the MCP Client call to the Knownledge Graph
        """
        logger.info(f"Mocking MCP data fetch for query: {query}")

        return """
        {
            "source": "KnowledgeGraph_Mock",
            "entities": [
                {
                    "name": XXX Office",
                    "opening_hours": "08:00 - 15:00",
                    "location": "Building C-13, Room 1.14",
                    "days": "Monday to Friday"
                },
                {
                    "name": "Rector's Scholarship",
                    "requirement": "Grade average above 5.49",
                    "deadline": "October 15th"
                }
            ],
            "note": "This is mock data for development purposes."
        }
        """
    async def genetate_response(self, user_message: str, history: List[Dict[str, str]]) -> str:
    
        try: 
            context_data = self._get_mock_mcp_data(user_message)

            system_prompt = f"""You are a helpful AI assistant for the ToPWR application at Wroclaw University of Science and Technology.

            You are a helpful AI assistant for the ToPWR application at Wroclaw University of Science and Technology.
            Your task is to answer the user's question using strictly this context.
            
            CONTEXT DATA (JSON):
            {context_data}

            INSTRUCTIONS:
                - Use the provided context to answer the question.
                - If the context does not contain the answer, use your general knowledge but explicitly state: "I don't have information about this in the university database, but generally..."
                - Be concise and natural.
                - Answer in the language of the user's question (predominantly Polish)
            """

            messages = [SystemMessage(content=system_prompt)]

            for msg in history[-5:]:
                if msg['role'] == 'user':
                    messages.append(HumanMessage(content=msg['context']))
                elif msg['role'] == 'assistant':
                    messages.append(AIMessage(content=msg['content']))
            
            messages.append(HumanMessage(content=user_message))

            response = await self.llm.ainvoke(messages)
            return response.content
    
        except Exception:
            logger.error(f"LLM generation error: {str(Exception)}", exc_info=True)
