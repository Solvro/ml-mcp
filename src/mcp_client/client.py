import asyncio
import os
import sys
import uuid

from dotenv import load_dotenv
from fastmcp import Client
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langfuse import Langfuse, observe
from langfuse.langchain import CallbackHandler
from mcp.types import ContentBlock

load_dotenv()


client = Client("http://localhost:8005/mcp")

llm = ChatOpenAI(
    model_name="pllum",
    base_url="https://services.clarin-pl.eu/api/v1/oapi",
    api_key=os.getenv("CLARIN_API_KEY"),
)

langfuse = Langfuse(
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
)

handler = CallbackHandler()


@observe(name="Knowledge Graph Tool Query")
async def get_knowledge_graph_data(
    user_input: str,
    trace_id: str | None = None,
    **langfuse_kwargs: dict,
) -> list[ContentBlock]:
    async with client:
        result = await client.call_tool(
            "knowledge_graph_tool",
            {
                "user_input": user_input,
                "trace_id": trace_id,
            },
        )
        return result.content


async def query_knowledge_graph(user_input: str, trace_id: str | None = None) -> BaseMessage:
    """Query the knowledge graph with user input."""

    trace_id = str(uuid.uuid4().hex)

    data = await get_knowledge_graph_data(
        user_input,
        trace_id,
        session_id=trace_id,
    )

    final_prompt = f"""Otrzymujesz informacje w postaci JSON w od innego LLM
    z danymi pochodzącymi z bazy wiedzy.

    Pytanie użytkownika: {user_input}

    Informacje z bazy wiedzy (w formacie JSON):
    {data}

    Twoim zadaniem jest odpowiedzieć użytkownikowi na pytanie w oparciu
    o te informacje - musisz wykorzystywać wszystkie dane z JSONa,
    aby udzielić kompletnej odpowiedzi.
    Odpowiedz w języku pytania, w sposób naturalny i zwięzły.
    Jeśli nie dostaniesz informacji na temat pytania,
    udziel odpowiedzi z wiedzy ogólnej na dany temat.
    """

    llm_response = await llm.ainvoke(
        final_prompt,
        config={
            "callbacks": [handler],
            "metadata": {
                "langfuse_session_id": trace_id,
                "langfuse_tags": ["mcp_client", "final_answer"],
                "run_name": "Final Answer",
            },
        },
    )

    return llm_response


def call_knowledge_graph_tool() -> None:
    """CLI entry point for knowledge graph tool."""
    if len(sys.argv) < 2:
        print("Usage: kg <question>")
        print("Example: kg 'Czym jest nagroda dziekana?'")
        sys.exit(1)

    user_input = " ".join(sys.argv[1:])
    asyncio.run(query_knowledge_graph(user_input))


if __name__ == "__main__":
    call_knowledge_graph_tool()
