"""Shared user-facing messages for LLM/pipeline timeouts and empty retrievals."""

GRAPH_PIPELINE_TIMEOUT_MESSAGE = (
    "The knowledge graph pipeline exceeded the maximum allowed wait time."
)
LLM_CALL_TIMEOUT_MESSAGE = "The language model request exceeded the maximum allowed wait time."

# Raised as a ToolError when Neo4j could not be consulted at all. Fixed on purpose: the driver's
# own text names codes, hosts and auth failures, and this reaches a student-facing API.
GRAPH_UNAVAILABLE_MESSAGE = "The knowledge graph database could not be reached."

# Returned when the guardrail routes a question away from graph retrieval.
OFF_TOPIC_MESSAGE = "W bazie danych nie ma informacji"

# Returned when retrieval ran but found nothing. Kept distinct from OFF_TOPIC_MESSAGE and stated
# explicitly rather than as an empty JSON list, so the answering model abstains instead of
# filling the gap from its own knowledge.
NO_GRAPH_DATA_MESSAGE = "Brak danych w grafie wiedzy dla tego pytania."
