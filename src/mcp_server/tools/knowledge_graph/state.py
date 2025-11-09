from typing import List, Optional

from langchain_core.documents import Document
from langgraph.graph import MessagesState


class State(MessagesState):
    """Represents the state of the RAG pipeline with all necessary components."""

    user_question: str
    context: Optional[List[Document]] = None
    answer: Optional[str] = None
    next_node: str
    generated_cypher: Optional[str] = None
    guardrail_decision: Optional[str] = None
    trace_id: Optional[str] = None
