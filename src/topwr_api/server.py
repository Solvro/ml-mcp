"""FastAPI application for ToPWR MCP integration."""

import logging
import os
import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastmcp import Client
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from ..config.config import get_config
from .models import ChatRequest, ChatResponse, MessageRole
from .session_manager import SessionManager

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration
config = get_config()

# MCP Client setup
mcp_host = os.getenv("MCP_HOST", config.servers.mcp.host)
mcp_port = os.getenv("MCP_PORT", config.servers.mcp.port)
mcp_transport = config.servers.mcp.transport
mcp_url = f"{mcp_transport}://{mcp_host}:{mcp_port}/mcp"

mcp_client = Client(mcp_url)

# LLM setup for final answer generation
clarin_api_key = os.getenv("CLARIN_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

llm = None
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
    logger.warning("No LLM API key found. Chat will return raw knowledge graph data.")

# Global session manager
session_manager: SessionManager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global session_manager

    # Startup
    logger.info("Starting ToPWR API service...")
    logger.info(f"MCP Server URL: {mcp_url}")
    session_manager = SessionManager()
    logger.info("Session manager initialized")

    yield

    # Shutdown
    logger.info("Shutting down ToPWR API service...")
    stats = session_manager.get_stats()
    logger.info(f"Final stats: {stats}")


# Initialize FastAPI app
app = FastAPI(
    title="ToPWR MCP Integration API",
    description="API for integrating ToPWR application with MCP Knowledge Graph",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS
config = get_config()
cors_origins = config.servers.topwr_api.cors_origins
if cors_origins == "*":
    allow_origins = ["*"]
else:
    allow_origins = [origin.strip() for origin in cors_origins.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def query_mcp_knowledge_graph(user_input: str, trace_id: str = None) -> str:
    """
    Query the MCP server's knowledge graph tool.

    Args:
        user_input: User's question
        trace_id: Optional trace ID for tracking

    Returns:
        Knowledge graph data as JSON string
    """
    async with mcp_client:
        result = await mcp_client.call_tool(
            "knowledge_graph_tool",
            {
                "user_input": user_input,
                "trace_id": trace_id,
            },
        )
        return result.content


async def generate_final_answer(user_input: str, kg_data: str) -> str:
    """
    Generate a final answer using the LLM with knowledge graph context.

    Args:
        user_input: Original user question
        kg_data: Knowledge graph data from MCP server

    Returns:
        LLM-generated answer
    """
    if llm is None:
        return f"Dane z grafu wiedzy: {kg_data}"

    final_prompt = config.prompts.final_answer.format(user_input=user_input, data=kg_data)

    response = await llm.ainvoke(final_prompt)
    return response.content


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "ToPWR MCP Integration API",
        "version": "1.0.0",
        "status": "running",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    stats = session_manager.get_stats()
    return {"status": "healthy", "session_stats": stats}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint for ToPWR integration.

    Args:
        request: ChatRequest with user_id, message, optional session_id

    Returns:
        ChatResponse with session_id and AI response
    """
    try:
        # Get or create session
        if request.session_id:
            session = session_manager.get_session(request.session_id)
            if not session:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Session {request.session_id} not found",
                )
            logger.info(f"Using existing session {session.session_id} for user {request.user_id}")
        else:
            session = session_manager.create_session(
                user_id=request.user_id, metadata=request.metadata
            )
            logger.info(f"Created new session {session.session_id} for user {request.user_id}")

        # Add user message to conversation
        session_manager.add_message(
            session_id=session.session_id,
            role=MessageRole.USER,
            content=request.message,
            metadata=request.metadata,
        )

        # Query MCP knowledge graph
        trace_id = str(uuid.uuid4().hex)
        try:
            kg_data = await query_mcp_knowledge_graph(
                user_input=request.message,
                trace_id=trace_id,
            )
            logger.info(f"Retrieved knowledge graph data for session {session.session_id}")

            # Generate final answer with LLM
            response_message = await generate_final_answer(
                user_input=request.message,
                kg_data=kg_data,
            )
            source = "mcp_knowledge_graph"

        except Exception as mcp_error:
            logger.error(f"MCP query failed: {mcp_error}", exc_info=True)
            response_message = (
                f"Przepraszam, nie mogłem uzyskać odpowiedzi z bazy wiedzy. Błąd: {str(mcp_error)}"
            )
            source = "error"

        # Add assistant response to conversation
        session_manager.add_message(
            session_id=session.session_id,
            role=MessageRole.ASSISTANT,
            content=response_message,
            metadata={"source": source, "trace_id": trace_id},
        )

        return ChatResponse(
            session_id=session.session_id,
            message=response_message,
            metadata={
                "message_count": len(session.messages),
                "source": source,
                "trace_id": trace_id,
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}",
        )


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session information by ID."""
    session_info = session_manager.get_session_info(session_id)
    if not session_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )
    return session_info


@app.get("/api/sessions/{session_id}/history")
async def get_session_history(session_id: str, limit: int = None):
    """Get conversation history for a session."""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )

    messages = session.get_conversation_history(limit=limit)
    return {
        "session_id": session_id,
        "messages": messages,
        "total_messages": len(session.messages),
    }


@app.get("/api/users/{user_id}/sessions")
async def get_user_sessions(user_id: str, active_only: bool = True):
    """Get all sessions for a user."""
    sessions = session_manager.get_user_sessions(user_id, active_only=active_only)
    return {
        "user_id": user_id,
        "session_count": len(sessions),
        "sessions": [
            {
                "session_id": s.session_id,
                "message_count": len(s.messages),
                "created_at": s.created_at,
                "updated_at": s.updated_at,
                "is_active": s.is_active,
            }
            for s in sessions
        ],
    }


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session."""
    if not session_manager.delete_session(session_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )
    return {"message": f"Session {session_id} deleted successfully"}


@app.post("/api/sessions/{session_id}/deactivate")
async def deactivate_session(session_id: str):
    """Deactivate a session."""
    if not session_manager.deactivate_session(session_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )
    return {"message": f"Session {session_id} deactivated successfully"}


@app.get("/api/stats")
async def get_stats():
    """Get system statistics."""
    return session_manager.get_stats()


def main():
    """Run the FastAPI server."""
    import uvicorn

    config = get_config()
    port = config.servers.topwr_api.port
    host = config.servers.topwr_api.host

    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
