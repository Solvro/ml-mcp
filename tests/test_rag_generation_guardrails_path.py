from typing import Any

import httpx
import pytest
from langchain_core.runnables import RunnableLambda
from neo4j.exceptions import ClientError, ServiceUnavailable
from openai import APIConnectionError

from src.mcp_server.tools.knowledge_graph.rag import (
    RAG,
    KnowledgeGraphUnavailableError,
    LLMUnavailableError,
)

QUESTION = "Kto wykłada analizę matematyczną?"
SCHEMA_TEXT = "Node properties: Course\nRelationship properties: TEACHES\nThe relationships: X"
GENERATED_CYPHER = "MATCH (c:Course) RETURN c.name"


class FakeSchemaDatabase:
    """Stands in for Neo4jGraph: serves a schema string and nothing else."""

    def __init__(self, schema: str = SCHEMA_TEXT, refresh_error: Exception | None = None) -> None:
        self.get_schema = schema
        self.refresh_error = refresh_error
        self.refresh_calls = 0

    def refresh_schema(self) -> None:
        """Neo4jGraph re-reads the graph here; the fake's schema is already current."""
        self.refresh_calls += 1
        if self.refresh_error is not None:
            raise self.refresh_error


class RecordingLLM:
    """Chat-model stand-in that records the rendered prompt and returns a canned reply."""

    def __init__(self, reply: str | Exception) -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def as_runnable(self) -> RunnableLambda:
        def _invoke(prompt_value: Any, config: dict[str, Any] | None = None) -> str:
            self.prompts.append(prompt_value.to_string())
            if isinstance(self.reply, Exception):
                raise self.reply
            return self.reply

        return RunnableLambda(_invoke)


def _rag_stub(
    reply: str | Exception, schema: str = SCHEMA_TEXT, refresh_error: Exception | None = None
) -> tuple[RAG, RecordingLLM, list[dict[str, Any]]]:
    """Build a RAG instance without running __init__ (no network, no LLM clients).

    Prompt templates come from the real graph_config.yaml, so a prompt that stops
    taking {schema} or {user_question} breaks these tests too.
    """
    rag = object.__new__(RAG)
    rag._init_schema_cache()
    rag.database = FakeSchemaDatabase(schema, refresh_error=refresh_error)
    rag._initialize_prompt_templates()
    rag.single_provider = False

    llm = RecordingLLM(reply)
    rag.fast_llm = llm.as_runnable()
    rag.cypher_llm = llm.as_runnable()

    invoke_configs: list[dict[str, Any]] = []

    def _record_invoke_config(**kwargs: Any) -> dict[str, Any]:
        invoke_configs.append(kwargs)
        return {}

    rag._get_invoke_config = _record_invoke_config

    return rag, llm, invoke_configs


def test_generate_cypher_feeds_schema_and_question_into_prompt():
    rag, llm, _ = _rag_stub(reply=GENERATED_CYPHER)

    result = rag.generate_cypher({"user_question": QUESTION})

    assert len(llm.prompts) == 1
    assert SCHEMA_TEXT in llm.prompts[0]
    assert QUESTION in llm.prompts[0]
    assert result == {"generated_cypher": GENERATED_CYPHER, "next_node": "retrieve"}


def test_generate_cypher_returns_llm_output_untouched():
    """Cleaning and validation belong to retrieve() - generate_cypher must not pre-sanitize."""
    rag, _, _ = _rag_stub(reply="  MATCH (n) RETURN n  ")

    result = rag.generate_cypher({"user_question": QUESTION})

    assert result["generated_cypher"] == "  MATCH (n) RETURN n  "


def test_generate_cypher_raises_when_schema_refresh_hits_outage_without_cache() -> None:
    rag, _, _ = _rag_stub(
        reply=GENERATED_CYPHER,
        refresh_error=ServiceUnavailable("neo4j unavailable"),
    )

    with pytest.raises(KnowledgeGraphUnavailableError, match="neo4j unavailable"):
        rag.generate_cypher({"user_question": QUESTION})


def test_generate_cypher_raises_when_the_schema_read_times_out() -> None:
    rag, _, _ = _rag_stub(
        reply=GENERATED_CYPHER,
        refresh_error=ClientError._hydrate_neo4j(
            code="Neo.ClientError.Transaction.TransactionTimedOutClientConfiguration",
            message="The transaction has been terminated.",
        ),
    )

    with pytest.raises(KnowledgeGraphUnavailableError, match="terminated"):
        rag.generate_cypher({"user_question": QUESTION})


def test_generate_cypher_keeps_cached_schema_when_refresh_fails() -> None:
    rag, _, _ = _rag_stub(
        reply=GENERATED_CYPHER,
        refresh_error=RuntimeError("temporary schema refresh error"),
    )
    rag._cached_schema = SCHEMA_TEXT
    rag._schema_fetched_at = 0.0
    rag._version_probed_at = 0.0
    rag._graph_version_seen = "v-1"

    result = rag.generate_cypher({"user_question": QUESTION})

    assert result["generated_cypher"] == GENERATED_CYPHER
    assert result["next_node"] == "retrieve"


def test_generate_cypher_propagates_tracing_context():
    rag, _, invoke_configs = _rag_stub(reply=GENERATED_CYPHER)
    handler = object()

    rag.generate_cypher(
        {
            "user_question": QUESTION,
            "trace_id": "trace-1",
            "session_id": "session-1",
            "callback_handler": handler,
        }
    )

    assert invoke_configs == [
        {
            "trace_id": "trace-1",
            "tags": ["knowledge_graph", "generated_cypher"],
            "run_name": "Generate Cypher",
            "handler": handler,
            "session_id": "session-1",
        }
    ]


def test_generate_cypher_raises_llm_unavailable_on_provider_error() -> None:
    rag, _, _ = _rag_stub(
        reply=APIConnectionError(request=httpx.Request("POST", "https://api.openai.test/v1"))
    )
    rag.single_provider = False

    with pytest.raises(LLMUnavailableError, match="Generate Cypher") as raised:
        rag.generate_cypher({"user_question": QUESTION})

    assert isinstance(raised.value.__cause__, APIConnectionError)


@pytest.mark.parametrize(
    ("llm_reply", "expected_decision"),
    [
        ('{"decision": "generate"}', "generate_cypher"),
        ('{"decision": "generate_cypher"}', "generate_cypher"),
        ('{"decision": "GENERATE"}', "generate_cypher"),
        ('```json\n{"decision": "generate"}\n```', "generate_cypher"),
        ('{"decision": "end"}', "end"),
        ('Sure! {"decision": "end"} hope that helps', "end"),
    ],
)
def test_guardrails_normalizes_decision(llm_reply, expected_decision):
    rag, _, _ = _rag_stub(reply=llm_reply)

    result = rag.guardrails_system({"user_question": QUESTION})

    assert result["next_node"] == expected_decision
    assert result["guardrail_decision"] == expected_decision


@pytest.mark.parametrize(
    "llm_reply",
    [
        "to nie jest JSON",
        "",
        '{"decision": "DROP TABLE"}',
        '{"nie_ma": "decision"}',
    ],
)
def test_guardrails_fails_closed_on_unusable_output(llm_reply):
    """Output we cannot read as a known decision must end the run, never reach Cypher."""
    rag, _, _ = _rag_stub(reply=llm_reply)

    result = rag.guardrails_system({"user_question": QUESTION})

    assert result["next_node"] == "end"


def test_guardrails_feeds_question_into_prompt():
    rag, llm, _ = _rag_stub(reply='{"decision": "end"}')

    rag.guardrails_system({"user_question": QUESTION})

    assert len(llm.prompts) == 1
    assert QUESTION in llm.prompts[0]


def test_guardrails_propagates_tracing_context():
    rag, _, invoke_configs = _rag_stub(reply='{"decision": "end"}')
    handler = object()

    rag.guardrails_system(
        {
            "user_question": QUESTION,
            "trace_id": "trace-1",
            "session_id": "session-1",
            "callback_handler": handler,
        }
    )

    assert invoke_configs == [
        {
            "trace_id": "trace-1",
            "tags": ["knowledge_graph", "guardrails"],
            "run_name": "Guardrails",
            "handler": handler,
            "session_id": "session-1",
        }
    ]


def test_guardrails_raises_llm_unavailable_on_provider_error() -> None:
    rag, _, _ = _rag_stub(
        reply=APIConnectionError(request=httpx.Request("POST", "https://api.openai.test/v1"))
    )
    rag.single_provider = False

    with pytest.raises(LLMUnavailableError, match="Guardrails") as raised:
        rag.guardrails_system({"user_question": QUESTION})

    assert isinstance(raised.value.__cause__, APIConnectionError)
