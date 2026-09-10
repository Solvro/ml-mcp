from typing import Any

import pytest
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

from src.config.messages import NO_GRAPH_DATA_MESSAGE
from src.mcp_server.tools.knowledge_graph.graph_visualizer import GraphVisualizer
from src.mcp_server.tools.knowledge_graph.rag import (
    RAG,
    KnowledgeGraphUnavailableError,
    RetrievalStrategy,
)

QUESTION = "Kto wyklada analize matematyczna?"
EMPTY_SCHEMA = "Node properties:\nRelationship properties:\nThe relationships:"
POPULATED_SCHEMA = "Node properties: Course\nRelationship properties: TEACHES\nThe relationships: X"
INGESTED_SCHEMA = f"{POPULATED_SCHEMA}\nNode properties: Professor"


class FakeGraphDatabase:
    def __init__(self, schema: str = POPULATED_SCHEMA, *, refresh_error: Exception | None = None):
        self.live_schema = schema
        self.get_schema = schema
        self.refresh_error = refresh_error
        self.version_error: Exception | None = None
        self.refresh_calls = 0
        self.version_calls = 0
        self.version = "2026-09-07T03:00:00Z"
        self.labels: list[str] = []
        self.queries: list[str] = []

    def ingest(self, schema: str, *, records_a_run: bool = True) -> None:
        self.live_schema = schema
        if records_a_run:
            self.version = f"{self.version}+1"

    def refresh_schema(self) -> None:
        self.refresh_calls += 1
        if self.refresh_error is not None:
            raise self.refresh_error
        self.get_schema = self.live_schema

    def query(self, cypher_query: str, params: dict[str, Any] | None = None):
        self.queries.append(cypher_query)
        if "PipelineRun" in cypher_query:
            self.version_calls += 1
            if self.version_error is not None:
                raise self.version_error
            return [{"version": self.version}]
        if "db.labels()" in cypher_query:
            return [{"label": label} for label in self.labels]
        if "SHOW INDEXES" in cypher_query:
            return []
        return []


def _rag_stub(database: FakeGraphDatabase) -> RAG:
    rag = object.__new__(RAG)
    rag.database = database
    rag.enable_fallback_search = True
    rag.max_results = 5
    rag.fallback_min_score = 0.5
    rag._init_schema_cache()
    return rag


def _expire_the_cache(rag: RAG) -> None:
    rag._schema_fetched_at -= rag._schema_ttl_sec + 1


def _allow_a_probe(rag: RAG) -> None:
    rag._version_probed_at -= rag._version_probe_interval_sec + 1


def test_reading_the_schema_goes_back_to_the_database():
    database = FakeGraphDatabase(schema=POPULATED_SCHEMA)
    rag = _rag_stub(database)

    assert rag.schema == POPULATED_SCHEMA
    assert database.refresh_calls == 1


def test_an_empty_graph_at_startup_is_re_read_on_the_next_call():
    database = FakeGraphDatabase(schema=EMPTY_SCHEMA)
    rag = _rag_stub(database)

    assert rag.schema == ""

    database.ingest(POPULATED_SCHEMA)

    assert rag.schema == POPULATED_SCHEMA
    assert database.refresh_calls == 2


def test_a_new_pipeline_run_is_picked_up_without_waiting_out_the_ttl():
    database = FakeGraphDatabase(schema=POPULATED_SCHEMA)
    rag = _rag_stub(database)

    assert rag.schema == POPULATED_SCHEMA
    database.ingest(INGESTED_SCHEMA)
    _allow_a_probe(rag)

    assert rag.schema == INGESTED_SCHEMA
    assert database.refresh_calls == 2


def test_an_unmoved_marker_serves_the_cache():
    database = FakeGraphDatabase(schema=POPULATED_SCHEMA)
    rag = _rag_stub(database)

    assert rag.schema == POPULATED_SCHEMA

    for _ in range(5):
        _allow_a_probe(rag)
        assert rag.schema == POPULATED_SCHEMA

    assert database.refresh_calls == 1
    assert database.version_calls == 6, "one probe per read, plus the one before the refresh"


def test_repeated_reads_inside_the_probe_interval_ask_the_graph_nothing():
    database = FakeGraphDatabase(schema=POPULATED_SCHEMA)
    rag = _rag_stub(database)

    assert rag.schema == POPULATED_SCHEMA
    probes_after_the_first_read = database.version_calls

    for _ in range(5):
        assert rag.schema == POPULATED_SCHEMA

    assert database.version_calls == probes_after_the_first_read
    assert database.refresh_calls == 1


def test_a_change_that_recorded_no_run_is_caught_by_the_backstop():
    database = FakeGraphDatabase(schema=POPULATED_SCHEMA)
    rag = _rag_stub(database)

    assert rag.schema == POPULATED_SCHEMA
    database.ingest(INGESTED_SCHEMA, records_a_run=False)
    _allow_a_probe(rag)

    assert rag.schema == POPULATED_SCHEMA, "an unmoved marker is believed until the TTL"

    _expire_the_cache(rag)

    assert rag.schema == INGESTED_SCHEMA


def test_a_failed_probe_keeps_serving_the_cached_schema():
    database = FakeGraphDatabase(schema=POPULATED_SCHEMA)
    rag = _rag_stub(database)

    assert rag.schema == POPULATED_SCHEMA

    database.version_error = RuntimeError("neo4j unreachable")
    database.ingest(INGESTED_SCHEMA)
    _allow_a_probe(rag)

    assert rag.schema == POPULATED_SCHEMA
    assert database.refresh_calls == 1

    database.version_error = None
    _allow_a_probe(rag)

    assert rag.schema == INGESTED_SCHEMA, "the next probe recovers"


def test_the_version_is_read_before_the_refresh_not_after():
    database = FakeGraphDatabase(schema=POPULATED_SCHEMA)
    rag = _rag_stub(database)

    def _refresh_and_ingest() -> None:
        database.refresh_calls += 1
        database.get_schema = database.live_schema
        # A pipeline run completes while the refresh is in flight.
        database.ingest(INGESTED_SCHEMA)

    database.refresh_schema = _refresh_and_ingest

    assert rag.schema == POPULATED_SCHEMA
    _allow_a_probe(rag)

    assert rag.schema == INGESTED_SCHEMA, "the run that landed mid-refresh is still seen"


def test_a_failed_refresh_keeps_the_last_good_schema():
    database = FakeGraphDatabase(schema=POPULATED_SCHEMA)
    rag = _rag_stub(database)

    assert rag.schema == POPULATED_SCHEMA

    database.refresh_error = RuntimeError("neo4j unreachable")
    _expire_the_cache(rag)

    assert rag.schema == POPULATED_SCHEMA


def test_a_failed_refresh_with_no_cached_schema_is_a_failure_not_an_empty_graph():
    database = FakeGraphDatabase(refresh_error=RuntimeError("neo4j unreachable"))
    rag = _rag_stub(database)

    with pytest.raises(KnowledgeGraphUnavailableError, match="neo4j unreachable"):
        rag.schema


def test_a_changed_label_set_drops_the_cached_schema():
    database = FakeGraphDatabase(schema=POPULATED_SCHEMA)
    database.labels = ["Course"]
    rag = _rag_stub(database)

    rag.ensure_fulltext_index()
    assert rag.schema == POPULATED_SCHEMA

    database.labels = ["Course", "Professor"]
    database.ingest(INGESTED_SCHEMA)
    rag.ensure_fulltext_index()

    assert rag.schema == INGESTED_SCHEMA, "a new label must not wait out the TTL"


def test_an_unchanged_label_set_leaves_the_cache_alone():
    database = FakeGraphDatabase(schema=POPULATED_SCHEMA)
    database.labels = ["Course"]
    rag = _rag_stub(database)

    rag.ensure_fulltext_index()
    assert rag.schema == POPULATED_SCHEMA

    rag.ensure_fulltext_index()

    assert rag.schema == POPULATED_SCHEMA
    assert database.refresh_calls == 1


def _cypher_stub(database: FakeGraphDatabase) -> tuple[RAG, list[str]]:
    rag = _rag_stub(database)
    rag.visualizer = GraphVisualizer()
    rag.graph_timeout_sec = 5.0

    cypher_prompts: list[str] = []

    def _invoke(prompt_value: Any, config: dict[str, Any] | None = None) -> str:
        cypher_prompts.append(prompt_value.to_string())
        return "MATCH (n) RETURN n"

    rag.generate_cypher_template = PromptTemplate(
        input_variables=["user_question", "normalized_question", "schema"],
        template="Q: {user_question} / {normalized_question}\nSchema:\n{schema}",
    )
    rag.guard_rails_template = PromptTemplate(
        input_variables=["user_question"],
        template="Question: {user_question}",
    )
    rag.cypher_llm = RunnableLambda(_invoke)
    rag.fast_llm = RunnableLambda(lambda _prompt, config=None: '{"decision": "generate"}')
    rag._get_invoke_config = lambda **kwargs: {}
    rag.graph = rag._build_processing_graph()

    return rag, cypher_prompts


def test_generate_cypher_abstains_instead_of_writing_a_query_against_nothing():
    rag, cypher_prompts = _cypher_stub(FakeGraphDatabase(schema=EMPTY_SCHEMA))

    result = rag.generate_cypher({"user_question": QUESTION})

    assert cypher_prompts == [], "an empty graph must not cost a Cypher model call"
    assert result["next_node"] == "end"
    assert result["generated_cypher"] is None
    assert result["context"] == []
    assert result["retrieval_strategy"] == RetrievalStrategy.EMPTY.value


def test_a_run_against_an_empty_graph_answers_no_data():
    database = FakeGraphDatabase(schema=EMPTY_SCHEMA)
    rag, cypher_prompts = _cypher_stub(database)

    result = rag.invoke(QUESTION)

    assert result["answer"] == NO_GRAPH_DATA_MESSAGE
    assert cypher_prompts == []
    assert all("PipelineRun" in query for query in database.queries), (
        "no generated Cypher may be executed against an empty graph"
    )
    assert result["metadata"]["retrieval_strategy"] == RetrievalStrategy.EMPTY.value


def test_a_run_against_a_populated_graph_still_reaches_retrieval():
    rag, cypher_prompts = _cypher_stub(FakeGraphDatabase(schema=POPULATED_SCHEMA))

    rag.invoke(QUESTION)

    assert len(cypher_prompts) == 1
    assert POPULATED_SCHEMA in cypher_prompts[0]
