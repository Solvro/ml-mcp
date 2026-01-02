"""Session management for conversation storage and retrieval."""

import logging
from datetime import datetime
from threading import Lock
from typing import Dict, List, Optional

from .models import ConversationSession, Message, MessageRole, SessionInfo

logger = logging.getLogger(__name__)


class SessionManager:
    """Thread-safe in-memory session manager for conversations."""

    def __init__(self):
        """Initialize session manager with thread-safe storage."""
        self._sessions: Dict[str, ConversationSession] = {}
        self._user_sessions: Dict[str, List[str]] = {}  # user_id -> [session_ids]
        self._lock = Lock()
        logger.info("SessionManager initialized with in-memory storage")

    def create_session(self, user_id: str, metadata: Optional[dict] = None) -> ConversationSession:
        """
        Create a new conversation session for a user.

        Args:
            user_id: Unique identifier for the user
            metadata: Optional metadata for the session

        Returns:
            New ConversationSession instance
        """
        with self._lock:
            session = ConversationSession(user_id=user_id, metadata=metadata or {})
            self._sessions[session.session_id] = session

            # Track session by user_id
            if user_id not in self._user_sessions:
                self._user_sessions[user_id] = []
            self._user_sessions[user_id].append(session.session_id)

            logger.info(f"Created session {session.session_id} for user {user_id}")
            return session

    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """
        Retrieve a session by ID.

        Args:
            session_id: Session identifier

        Returns:
            ConversationSession if found, None otherwise
        """
        with self._lock:
            return self._sessions.get(session_id)

    def update_session(self, session: ConversationSession) -> bool:
        """
        Update an existing session.

        Args:
            session: Updated session object

        Returns:
            True if successful, False if session not found
        """
        with self._lock:
            if session.session_id in self._sessions:
                session.updated_at = datetime.utcnow()
                self._sessions[session.session_id] = session
                logger.debug(f"Updated session {session.session_id}")
                return True
            return False

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session by ID.

        Args:
            session_id: Session identifier

        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            if session_id in self._sessions:
                session = self._sessions.pop(session_id)
                # Remove from user sessions tracking
                if session.user_id in self._user_sessions:
                    self._user_sessions[session.user_id].remove(session_id)
                logger.info(f"Deleted session {session_id}")
                return True
            return False

    def get_user_sessions(
        self, user_id: str, active_only: bool = True
    ) -> List[ConversationSession]:
        """
        Get all sessions for a specific user.

        Args:
            user_id: User identifier
            active_only: If True, return only active sessions

        Returns:
            List of ConversationSession objects
        """
        with self._lock:
            session_ids = self._user_sessions.get(user_id, [])
            sessions = [self._sessions[sid] for sid in session_ids if sid in self._sessions]

            if active_only:
                sessions = [s for s in sessions if s.is_active]

            return sessions

    def get_active_session(self, user_id: str) -> Optional[ConversationSession]:
        """
        Get the most recent active session for a user.

        Args:
            user_id: User identifier

        Returns:
            Most recent active ConversationSession or None
        """
        sessions = self.get_user_sessions(user_id, active_only=True)
        if not sessions:
            return None

        # Return most recently updated session
        return sorted(sessions, key=lambda s: s.updated_at, reverse=True)[0]

    def add_message(
        self, session_id: str, role: MessageRole, content: str, metadata: Optional[dict] = None
    ) -> Optional[Message]:
        """
        Add a message to a session.

        Args:
            session_id: Session identifier
            role: Message role (user/assistant/system)
            content: Message content
            metadata: Optional message metadata

        Returns:
            Created Message object or None if session not found
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                logger.warning(f"Session {session_id} not found")
                return None

            message = session.add_message(role, content, metadata)
            logger.debug(f"Added {role.value} message to session {session_id}")
            return message

    def deactivate_session(self, session_id: str) -> bool:
        """
        Mark a session as inactive.

        Args:
            session_id: Session identifier

        Returns:
            True if successful, False if session not found
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.is_active = False
                session.updated_at = datetime.utcnow()
                logger.info(f"Deactivated session {session_id}")
                return True
            return False

    def get_session_info(self, session_id: str) -> Optional[SessionInfo]:
        """
        Get session information without full message history.

        Args:
            session_id: Session identifier

        Returns:
            SessionInfo object or None if not found
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return None

            return SessionInfo(
                session_id=session.session_id,
                user_id=session.user_id,
                message_count=len(session.messages),
                created_at=session.created_at,
                updated_at=session.updated_at,
                is_active=session.is_active,
            )

    def get_all_sessions(self) -> List[SessionInfo]:
        """
        Get information about all sessions.

        Returns:
            List of SessionInfo objects
        """
        with self._lock:
            return [
                SessionInfo(
                    session_id=s.session_id,
                    user_id=s.user_id,
                    message_count=len(s.messages),
                    created_at=s.created_at,
                    updated_at=s.updated_at,
                    is_active=s.is_active,
                )
                for s in self._sessions.values()
            ]

    def clear_all_sessions(self) -> int:
        """
        Clear all sessions (for testing/reset).

        Returns:
            Number of sessions cleared
        """
        with self._lock:
            count = len(self._sessions)
            self._sessions.clear()
            self._user_sessions.clear()
            logger.warning(f"Cleared all {count} sessions")
            return count

    def get_stats(self) -> dict:
        """
        Get statistics about current sessions.

        Returns:
            Dictionary with session statistics
        """
        with self._lock:
            total_sessions = len(self._sessions)
            active_sessions = sum(1 for s in self._sessions.values() if s.is_active)
            total_messages = sum(len(s.messages) for s in self._sessions.values())
            unique_users = len(self._user_sessions)

            return {
                "total_sessions": total_sessions,
                "active_sessions": active_sessions,
                "inactive_sessions": total_sessions - active_sessions,
                "total_messages": total_messages,
                "unique_users": unique_users,
                "avg_messages_per_session": (
                    total_messages / total_sessions if total_sessions > 0 else 0
                ),
            }
