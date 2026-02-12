import asyncio
import os
import sys
import uuid

from dotenv import load_dotenv
from fastmcp import Client
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from ..config.config import get_config

load_dotenv()

config = get_config()
mcp_host = os.getenv("MCP_HOST", config.servers.mcp.host)
mcp_port = os.getenv("MCP_PORT", config.servers.mcp.port)
mcp_transport = config.servers.mcp.transport
mcp_url = f"{mcp_transport}://{mcp_host}:{mcp_port}/mcp"

client = Client(mcp_url)

clarin_api_key = os.getenv("CLARIN_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

if clarin_api_key:
    llm = ChatOpenAI(
        model_name=config.llm.clarin.name,
        base_url=config.llm.clarin.base_url,
        api_key=clarin_api_key,
    )
elif google_api_key:
    llm = ChatGoogleGenerativeAI(
        model=config.llm.gemini.name,
        google_api_key=google_api_key,
        temperature=1.0,
    )
else:
    raise ValueError(
        "No LLM API key found. Please set either CLARIN_API_KEY or GOOGLE_API_KEY in your .env file"
    )

# Initialize Langfuse only if credentials are configured
langfuse = None
handler = None
observe = None

_langfuse_secret = os.getenv("LANGFUSE_SECRET_KEY")
_langfuse_public = os.getenv("LANGFUSE_PUBLIC_KEY")
_langfuse_host = os.getenv("LANGFUSE_HOST")

if _langfuse_secret and _langfuse_public:
    try:
        from langfuse import Langfuse
        from langfuse import observe as langfuse_observe
        from langfuse.langchain import CallbackHandler

        langfuse = Langfuse(
            secret_key=_langfuse_secret,
            public_key=_langfuse_public,
            host=_langfuse_host,
        )
        handler = CallbackHandler()
        observe = langfuse_observe
    except Exception as e:
        print(f"Warning: Failed to initialize Langfuse: {e}")
else:
    print("Langfuse credentials not configured. Tracing disabled.")


def optional_observe(name: str):
    """Decorator that applies Langfuse observe if available, otherwise no-op."""

    def decorator(func):
        if observe is not None:
            return observe(name=name)(func)
        return func

    return decorator


@optional_observe(name="Knowledge Graph Tool Query")
async def get_knowledge_graph_data(
    user_input: str,
    trace_id: str = None,
    **langfuse_kwargs,
):
    async with client:
        result = await client.call_tool(
            "knowledge_graph_tool",
            {
                "user_input": user_input,
                "trace_id": trace_id,
            },
        )
        return result.content


async def query_knowledge_graph(user_input: str, trace_id: str = None):
    """Query the knowledge graph with user input."""

    trace_id = str(uuid.uuid4().hex)

    data = await get_knowledge_graph_data(
        user_input,
        trace_id,
        session_id=trace_id,
    )

    final_prompt = config.prompts.final_answer.format(user_input=user_input, data=data)

    # Build config with optional handler
    invoke_config = {
        "metadata": {
            "langfuse_session_id": trace_id,
            "langfuse_tags": ["mcp_client", "final_answer"],
            "run_name": "Final Answer",
        },
    }
    if handler is not None:
        invoke_config["callbacks"] = [handler]

    llm_response = await llm.ainvoke(final_prompt, config=invoke_config)

    print(llm_response.content)
    return llm_response


def call_knowledge_graph_tool():
    """CLI entry point for knowledge graph tool."""
    if len(sys.argv) < 2:
        print("Usage: kg <question>")
        print("Example: kg 'Czym jest nagroda dziekana?'")
        sys.exit(1)

    user_input = " ".join(sys.argv[1:])
    asyncio.run(query_knowledge_graph(user_input))


if __name__ == "__main__":
    call_knowledge_graph_tool()
