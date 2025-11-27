from langchain_core.documents import Document
from langgraph.graph import MessagesState


class State(MessagesState):
    """Represents the state of the RAG pipeline with all necessary components."""

    user_question: str
    context: list[Document] | None = None
    answer: str | None = None
    next_node: str
    generated_cypher: str | None = None
    guardrail_decision: str | None = None
    trace_id: str | None = None
