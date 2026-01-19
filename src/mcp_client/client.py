"""LangGraph Agent for MCP tool integration."""

import json
import logging
import os
import uuid
from typing import List, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from langgraph.graph import END, START, StateGraph

from src.topwr_api.models import Message, MessageRole

from ..config.config import get_config

load_dotenv()

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State for the LangGraph agent."""

    messages: List[BaseMessage]
    trace_id: str


class Agent:
    """LangGraph Agent that uses GPT with MCP tools and CLARIN LLM for final answers."""

    def __init__(
        self,
        mcp_server_url: str | None = None,
        gpt_model: str | None = None,
        clarin_model: str | None = None,
    ):
        """
        Initialize the Agent.

        Args:
            mcp_server_url: Optional URL of the MCP server (overrides config)
            gpt_model: Optional GPT model name for tool calling (overrides config)
            clarin_model: Optional CLARIN model name for final answers (overrides config)
        """
        # Load configuration
        config = get_config()
        
        # MCP Server URL - construct from config or use override
        if mcp_server_url is None:
            mcp_host = config.servers.mcp.host
            mcp_port = config.servers.mcp.port
            mcp_server_url = f"http://{mcp_host}:{mcp_port}/mcp"
        
        # MCP Client using langchain-mcp-adapters
        self.mcp_client = MultiServerMCPClient(
            {
                "mcp-server": {
                    "transport": "streamable_http",
                    "url": mcp_server_url,
                }
            }
        )

        # LLM Models - use config or override
        fast_model = config.llm.fast_model
        clarin_config = config.llm.clarin
        
        gpt_model = gpt_model or fast_model.name
        clarin_model = clarin_model or clarin_config.name
        
        # LLMs - API keys still from environment for security
        self.gpt_llm = ChatOpenAI(
            model=gpt_model,
            api_key=os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY"),
            temperature=fast_model.temperature,
        )

        self.clarin_llm = ChatOpenAI(
            model_name=clarin_model,
            base_url=clarin_config.base_url,
            api_key=os.getenv("CLARIN_API_KEY"),
        )

        # Langfuse setup from config (credentials resolved from env vars)
        langfuse_config = config.observability.langfuse
        langfuse_secret = langfuse_config.secret_key
        langfuse_public = langfuse_config.public_key
        langfuse_host = langfuse_config.host
        
        if langfuse_secret and langfuse_public and langfuse_secret != "your_langfuse_secret_key" and "${" not in langfuse_secret:
            self.langfuse = Langfuse(
                secret_key=langfuse_secret,
                public_key=langfuse_public,
                host=langfuse_host,
            )
            self.handler = CallbackHandler()
            logger.info("Langfuse observability enabled")
        else:
            self.langfuse = None
            self.handler = None
            logger.info("Langfuse observability disabled (no valid credentials)")

        # Build the graph
        self.graph = self._build_agent_graph()

    def _convert_messages_to_langchain(self, messages: List[Message]) -> List[BaseMessage]:
        """
        Convert messages from models.Message format to LangChain message format.

        Args:
            messages: List of Message objects from models.py

        Returns:
            List of LangChain BaseMessage objects
        """
        langchain_messages = []
        for msg in messages:
            if msg.role == MessageRole.USER:
                langchain_messages.append(HumanMessage(content=msg.content))
            elif msg.role == MessageRole.ASSISTANT:
                langchain_messages.append(AIMessage(content=msg.content))
            elif msg.role == MessageRole.SYSTEM:
                langchain_messages.append(SystemMessage(content=msg.content))
            elif msg.role == MessageRole.TOOL:
                # Tool messages need tool_call_id and name
                tool_call_id = msg.metadata.get("tool_call_id", "")
                tool_name = msg.metadata.get("tool_name", "")
                langchain_messages.append(
                    ToolMessage(content=msg.content, tool_call_id=tool_call_id, name=tool_name)
                )
        return langchain_messages

    async def _get_mcp_tools(self):
        """
        Get MCP tools as LangChain tools using langchain-mcp-adapters.

        Returns:
            List of LangChain tools from MCP server
        """
        try:
            # langchain-mcp-adapters automatically converts MCP tools to LangChain tools
            tools = await self.mcp_client.get_tools()
            logger.info(f"Loaded {len(tools)} tools from MCP server")
            for tool in tools:
                logger.info(f"Tool: {tool.name}")
            return tools
        except Exception as e:
            # Log error but don't fail - return empty tools list
            logger.error(f"Failed to get MCP tools: {e}", exc_info=True)
            return []

    async def _call_tool_node(self, state: AgentState) -> AgentState:
        """
        Execute tool calls from the agent.

        Args:
            state: Current agent state

        Returns:
            Updated state with tool results
        """
        last_message = state["messages"][-1]
        if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
            return state

        # Get MCP tools
        tools = await self._get_mcp_tools()
        tool_map = {tool.name: tool for tool in tools}

        # Execute tool calls
        tool_messages = []
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call.get("args", {})

            if tool_name in tool_map:
                try:
                    # Execute the tool using LangChain tool's ainvoke
                    print("tool_args")
                    print(tool_args)
                    result = await tool_map[tool_name].ainvoke(tool_args)
                    print("result")
                    print(result)

                    # Convert result to string if needed
                    if isinstance(result, (dict, list)):
                        result = json.dumps(result, ensure_ascii=False, indent=2)

                    tool_messages.append(
                        ToolMessage(
                            content=str(result),
                            tool_call_id=tool_call["id"],
                            name=tool_name,
                        )
                    )
                except Exception as e:
                    logger.error(f"Error executing tool {tool_name}: {e}", exc_info=True)
                    tool_messages.append(
                        ToolMessage(
                            content=f"Error: {str(e)}",
                            tool_call_id=tool_call["id"],
                            name=tool_name,
                        )
                    )
            else:
                logger.warning(f"Tool {tool_name} not found in available tools")
                tool_messages.append(
                    ToolMessage(
                        content=f"Tool {tool_name} is not available",
                        tool_call_id=tool_call["id"],
                        name=tool_name,
                    )
                )

        return {"messages": state["messages"] + tool_messages}

    async def _agent_with_tools(self, state: AgentState) -> AgentState:
        """
        Agent node that uses GPT with MCP tools.

        Args:
            state: Current agent state

        Returns:
            Updated state with agent response
        """
        tools = await self._get_mcp_tools()
        llm_with_tools = self.gpt_llm.bind_tools(tools)

        config = {
            "metadata": {
                "langfuse_session_id": state["trace_id"],
                "langfuse_tags": ["agent", "gpt_with_tools"],
                "run_name": "Agent with Tools",
            },
        }
        if self.handler:
            config["callbacks"] = [self.handler]
        
        response = await llm_with_tools.ainvoke(
            state["messages"],
            config=config,
        )

        return {"messages": state["messages"] + [response]}

    async def _final_answer_node(self, state: AgentState) -> AgentState:
        """
        Final answer node that uses CLARIN LLM to generate the final response.

        Args:
            state: Current agent state

        Returns:
            Updated state with final answer
        """
        # Build context from conversation history
        conversation_context = []
        for msg in state["messages"]:
            if isinstance(msg, HumanMessage):
                conversation_context.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                if msg.tool_calls:
                    # Include tool calls in context
                    for tool_call in msg.tool_calls:
                        tool_name = tool_call['name']
                        tool_args = tool_call.get('args', {})
                        conversation_context.append(
                            f"Assistant called tool {tool_name} with args: {tool_args}"
                        )
                else:
                    conversation_context.append(f"Assistant: {msg.content}")
            elif isinstance(msg, ToolMessage):
                conversation_context.append(f"Tool {msg.name}: {msg.content}")

        context_text = "\n".join(conversation_context)

        # Use prompt from config
        config = get_config()
        final_prompt_template = config.prompts.final_answer
        
        # Format the prompt with user input and data
        # Extract the last user message for user_input
        user_input = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                user_input = msg.content
                break
        
        # Format the prompt template with the conversation context
        # The template expects {user_input} and {data} placeholders
        try:
            final_prompt = final_prompt_template.format(
                user_input=user_input,
                data=context_text
            )
        except KeyError:
            # Fallback if template doesn't have expected placeholders
            final_prompt = f"""{final_prompt_template}

Historia konwersacji:
{context_text}

Pytanie użytkownika: {user_input}
"""

        config = {
            "metadata": {
                "langfuse_session_id": state["trace_id"],
                "langfuse_tags": ["final_answer", "clarin"],
                "run_name": "Final Answer",
            },
        }
        if self.handler:
            config["callbacks"] = [self.handler]
        
        response = await self.clarin_llm.ainvoke(
            final_prompt,
            config=config,
        )

        return {"messages": state["messages"] + [response]}

    def _should_continue(self, state: AgentState) -> str:
        """
        Determine if we should continue with tools or move to final answer.

        Args:
            state: Current agent state

        Returns:
            Next node name
        """
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "call_tools"
        return "final_answer"

    def _build_agent_graph(self):
        """Build the LangGraph agent graph."""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("agent", self._agent_with_tools)
        workflow.add_node("call_tools", self._call_tool_node)
        workflow.add_node("final_answer", self._final_answer_node)

        # Set entry point
        workflow.add_edge(START, "agent")

        # Add conditional edge after agent
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "call_tools": "call_tools",
                "final_answer": "final_answer",
            },
        )

        # After tools, go back to agent
        workflow.add_edge("call_tools", "agent")

        # Final answer is the end
        workflow.add_edge("final_answer", END)

        return workflow.compile()

    async def process_messages(
        self, messages: List[Message], trace_id: str | None = None
    ) -> str:
        """
        Process a list of messages through the LangGraph agent.

        Args:
            messages: List of Message objects from models.py
            trace_id: Optional trace ID for observability

        Returns:
            Final answer as string
        """
        if trace_id is None:
            trace_id = str(uuid.uuid4().hex)

        # Convert messages to LangChain format
        langchain_messages = self._convert_messages_to_langchain(messages)

        # Create initial state
        initial_state: AgentState = {
            "messages": langchain_messages,
            "trace_id": trace_id,
        }

        # Run the graph
        final_state = await self.graph.ainvoke(initial_state)

        # Get the final answer (last message from CLARIN LLM)
        final_messages = final_state["messages"]
        final_answer = final_messages[-1].content if final_messages else ""

        return final_answer
