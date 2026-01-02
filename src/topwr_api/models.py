"""Data models for conversation management."""

from datetime import datetime
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field


class MessageRole(StrEnum):
    """Message role in conversation."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(BaseModel):
    """Single message in a conversation."""

    id: str = Field(default_factory=lambda: uuid4().hex)
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict | None = Field(default_factory=dict)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class ConversationSession(BaseModel):
    """Conversation session with full history and state."""

    session_id: str = Field(default_factory=lambda: uuid4().hex)
    user_id: str
    messages: list[Message] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict = Field(default_factory=dict)
    is_active: bool = True

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}

    def add_message(
        self, role: MessageRole, content: str, metadata: dict | None = None
    ) -> Message:
        """Add a new message to the conversation."""
        message = Message(role=role, content=content, metadata=metadata or {})
        self.messages.append(message)
        self.updated_at = datetime.utcnow()
        return message

    def get_conversation_history(self, limit: int | None = None) -> list[Message]:
        """Get conversation history, optionally limited to last N messages."""
        if limit:
            return self.messages[-limit:]
        return self.messages

    def get_context_window(self, max_messages: int = 10) -> str:
        """Get formatted conversation context for LLM."""
        recent_messages = self.get_conversation_history(limit=max_messages)
        context = []
        for msg in recent_messages:
            context.append(f"{msg.role.value}: {msg.content}")
        return "\n".join(context)


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    user_id: str = Field(..., description="Unique identifier for the user")
    message: str = Field(..., min_length=1, description="User's question or message")
    session_id: str | None = Field(
        None, description="Session ID to continue existing conversation"
    )
    metadata: dict | None = Field(default_factory=dict, description="Additional metadata")


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""

    session_id: str
    message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict | None = Field(default_factory=dict)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class SessionInfo(BaseModel):
    """Session information model."""

    session_id: str
    user_id: str
    message_count: int
    created_at: datetime
    updated_at: datetime
    is_active: bool

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}
