from typing import List, Optional, Dict

from langchain_core.documents import Document
from langgraph.graph import MessagesState


class State(MessagesState):
    """Represents the state of the RAG pipeline with all necessary components."""

    user_question: str
    context: Optional[List[Dict]] = None
    answer: Optional[str] = None
    trace_id: Optional[str] = None
