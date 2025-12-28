"""FastAPI application for ToPWR MCP integration."""

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from .models import ChatRequest, ChatResponse, MessageRole
from .session_manager import SessionManager

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global session manager
session_manager: SessionManager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global session_manager

    # Startup
    logger.info("Starting ToPWR API service...")
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
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {"service": "ToPWR MCP Integration API", "version": "1.0.0", "status": "running"}


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

        # TODO: Replace with actual MCP client call
        # For now, return mock response
        mock_response = (
            f"[MOCK] Otrzymałem pytanie: '{request.message}'. "
            "To jest tymczasowa odpowiedź. Integracja z MCP będzie dodana wkrótce."
        )

        # Add assistant response to conversation
        session_manager.add_message(
            session_id=session.session_id,
            role=MessageRole.ASSISTANT,
            content=mock_response,
            metadata={"source": "mock"},
        )

        return ChatResponse(
            session_id=session.session_id,
            message=mock_response,
            metadata={"message_count": len(session.messages), "mock_mode": True},
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
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Session {session_id} not found"
        )
    return session_info


@app.get("/api/sessions/{session_id}/history")
async def get_session_history(session_id: str, limit: int = None):
    """Get conversation history for a session."""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Session {session_id} not found"
        )

    messages = session.get_conversation_history(limit=limit)
    return {"session_id": session_id, "messages": messages, "total_messages": len(session.messages)}


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
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Session {session_id} not found"
        )
    return {"message": f"Session {session_id} deleted successfully"}


@app.post("/api/sessions/{session_id}/deactivate")
async def deactivate_session(session_id: str):
    """Deactivate a session."""
    if not session_manager.deactivate_session(session_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Session {session_id} not found"
        )
    return {"message": f"Session {session_id} deactivated successfully"}


@app.get("/api/stats")
async def get_stats():
    """Get system statistics."""
    return session_manager.get_stats()


def main():
    """Run the FastAPI server."""
    import uvicorn

    port = int(os.getenv("TOPWR_API_PORT", 8000))
    host = os.getenv("TOPWR_API_HOST", "0.0.0.0")

    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
