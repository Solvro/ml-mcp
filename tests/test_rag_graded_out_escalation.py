"""A query that matched the wrong rows gets the same second chance as one that matched nothing.

Issue #99: `Jakie kryteria oceniają działalność dydaktyczną?` generated a traversal that returned
one row, the guideline's own preamble. A non-empty primary result skipped the grader and never
escalated, so the answer was "Nie wiem". The same question asked a minute later matched nothing,
went to the full-text search and was answered. These tests run the whole graph on that shape.
"""

from typing import Any

import pytest
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from neo4j.exceptions import ServiceUnavailable

from src.config.config import get_config
from src.config.messages import NO_GRAPH_DATA_MESSAGE
from src.mcp_server.tools.knowledge_graph.graph_visualizer import GraphVisualizer
from src.mcp_server.tools.knowledge_graph.rag import RAG, KnowledgeGraphUnavailableError

QUESTION = "Jakie kryteria oceniają działalność dydaktyczną?"
SCHEMA_TEXT = (
    "Node properties:\nGuideline {title: STRING}\nRelationship properties:\n"
    "The relationships:\n(:Guideline)-[:CONSIDERS]->(:Criterion)"
)
WRONG_TRAVERSAL = "MATCH (g:Guideline)-[:CONSIDERS]->(c:Criterion) RETURN g.title, g.context"
QUESTION_LITERAL_CYPHER = (
    "MATCH (g:Guideline)-[:CONSIDERS]->(c:Criterion) "
    "WHERE toLower(c.title) CONTAINS toLower('jakie kryteria oceniaja dzialalnosc dydaktyczna') "
    "RETURN g.title, g.context"
)
PREAMBLE_ROWS = [{"g.title": "Wytyczne", "g.context": "Niniejsze wytyczne okreslaja tryb oceny"}]
FULLTEXT_ROWS = [
    {
        "labels": ["CriterionCategory"],
        "title": "Dzialalnosc dydaktyczna",
        "context": "prowadzenie zajec, opieka nad dyplomantami",
        "score": 20.9,
        "related": ["HAS_CRITERION: Hospitacje zajec"],
    }
]
KEEP_FIRST = '{"relevant": [1]}'
KEEP_NONE = '{"relevant": []}'
GRADER_OPENING = "You are a retrieval grader"


class ScriptedDatabase:
    """Neo4j stand-in with separate scripts for the model's queries and the full-text search."""

    def __init__(
        self,
        query_results: list[list[dict[str, Any]]],
        fulltext_results: list[list[dict[str, Any]] | Exception],
    ) -> None:
        self.get_schema = SCHEMA_TEXT
        self.query_results = list(query_results)
        self.fulltext_results = list(fulltext_results)
        self.queries: list[str] = []
        self.fulltext_queries: list[dict[str, Any] | None] = []

    def refresh_schema(self) -> None:
        """The fake's schema is already current."""

    def query(
        self,
        cypher_query: str,
        params: dict[str, Any] | None = None,
        session_params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if "PipelineRun" in cypher_query:
            return [{"version": "2026-09-18T03:00:00Z"}]
        if "db.labels()" in cypher_query:
            return [{"label": "Guideline"}, {"label": "CriterionCategory"}]
        if "SHOW INDEXES" in cypher_query:
            return [{"labels": ["CriterionCategory", "Guideline"], "properties": []}]
        if "db.index.fulltext.queryNodes" in cypher_query:
            self.fulltext_queries.append(params)
            result = self.fulltext_results.pop(0) if self.fulltext_results else []
            if isinstance(result, Exception):
                raise result
            return result

        self.queries.append(cypher_query)
        return self.query_results.pop(0) if self.query_results else []


def _build_graph(
    *,
    cypher_reply: str,
    query_results: list[list[dict[str, Any]]],
    fulltext_results: list[list[dict[str, Any]] | Exception],
    grader_replies: list[str],
    enable_fallback_search: bool = True,
) -> tuple[RAG, ScriptedDatabase, list[str]]:
    """Build the real graph, with the real grader prompt, around scripted models and data."""
    rag = object.__new__(RAG)
    rag.max_results = 10
    rag.enable_fallback_search = enable_fallback_search
    rag.fallback_min_score = 0.5
    rag.graph_timeout_sec = 5.0
    rag._init_schema_cache()
    rag.visualizer = GraphVisualizer()

    database = ScriptedDatabase(query_results, fulltext_results)
    rag.database = database

    rag.guard_rails_template = PromptTemplate(
        input_variables=["user_question"], template="Route: {user_question}"
    )
    rag.generate_cypher_template = PromptTemplate(
        input_variables=["user_question", "schema"], template="Cypher: {user_question}\n{schema}"
    )
    rag.context_grader_template = PromptTemplate(
        input_variables=["user_question", "retrieval", "candidates"],
        template=get_config().prompts.context_grader,
    )

    replies = list(grader_replies)
    grader_prompts: list[str] = []

    def _fast(prompt_value: Any, config: dict[str, Any] | None = None) -> str:
        prompt = prompt_value.to_string()
        if prompt.startswith(GRADER_OPENING):
            grader_prompts.append(prompt)
            return replies.pop(0)
        return '{"decision": "generate"}'

    rag.fast_llm = RunnableLambda(_fast)
    rag.cypher_llm = RunnableLambda(lambda prompt_value, config=None: cypher_reply)
    rag.graph = rag._build_processing_graph()
    return rag, database, grader_prompts


def test_a_query_that_matched_the_wrong_rows_is_answered_from_the_full_text_search() -> None:
    rag, database, grader_prompts = _build_graph(
        cypher_reply=WRONG_TRAVERSAL,
        query_results=[PREAMBLE_ROWS],
        fulltext_results=[FULLTEXT_ROWS],
        grader_replies=[KEEP_NONE, KEEP_FIRST],
    )

    result = rag.invoke(QUESTION)

    assert result["metadata"]["retrieval_strategy"] == "label_agnostic_after_grading"
    assert result["metadata"]["context"] == FULLTEXT_ROWS
    assert result["metadata"]["context_graded"] is True
    assert "Dzialalnosc dydaktyczna" in result["answer"]
    assert len(database.fulltext_queries) == 1
    assert len(grader_prompts) == 2
    assert WRONG_TRAVERSAL in grader_prompts[0]
    assert "Niniejsze wytyczne" in grader_prompts[0]
    assert "Dzialalnosc dydaktyczna" in grader_prompts[1]


def test_rows_the_grader_confirms_never_reach_the_full_text_search() -> None:
    rag, database, grader_prompts = _build_graph(
        cypher_reply=WRONG_TRAVERSAL,
        query_results=[PREAMBLE_ROWS],
        fulltext_results=[FULLTEXT_ROWS],
        grader_replies=[KEEP_FIRST],
    )

    result = rag.invoke(QUESTION)

    assert result["metadata"]["retrieval_strategy"] == "primary"
    assert result["metadata"]["context"] == PREAMBLE_ROWS
    assert database.fulltext_queries == []
    assert len(grader_prompts) == 1


def test_a_search_that_finds_nothing_leaves_the_run_graded_out() -> None:
    rag, database, grader_prompts = _build_graph(
        cypher_reply=WRONG_TRAVERSAL,
        query_results=[PREAMBLE_ROWS],
        fulltext_results=[],
        grader_replies=[KEEP_NONE],
    )

    result = rag.invoke(QUESTION)

    assert result["answer"] == NO_GRAPH_DATA_MESSAGE
    assert result["metadata"]["retrieval_strategy"] == "graded_out"
    assert result["metadata"]["context"] == []
    assert database.fulltext_queries
    assert len(grader_prompts) == 1


def test_rejected_full_text_rows_end_the_run_instead_of_searching_again() -> None:
    rag, database, grader_prompts = _build_graph(
        cypher_reply=WRONG_TRAVERSAL,
        query_results=[PREAMBLE_ROWS],
        fulltext_results=[FULLTEXT_ROWS, FULLTEXT_ROWS],
        grader_replies=[KEEP_NONE, KEEP_NONE],
    )

    result = rag.invoke(QUESTION)

    assert result["answer"] == NO_GRAPH_DATA_MESSAGE
    assert result["metadata"]["retrieval_strategy"] == "graded_out"
    assert len(database.fulltext_queries) == 1
    assert len(grader_prompts) == 2


def test_rejected_rows_from_the_literal_repair_also_reach_the_full_text_search() -> None:
    """The repair keeps the model's traversal, so its rows can be wrong in the same way."""
    rag, database, grader_prompts = _build_graph(
        cypher_reply=QUESTION_LITERAL_CYPHER,
        query_results=[[], PREAMBLE_ROWS],
        fulltext_results=[FULLTEXT_ROWS],
        grader_replies=[KEEP_NONE, KEEP_FIRST],
    )

    result = rag.invoke(QUESTION)

    assert len(database.queries) == 2
    assert "jakie kryteria" not in database.queries[1]
    assert result["metadata"]["retrieval_strategy"] == "label_agnostic_after_grading"
    assert result["metadata"]["context"] == FULLTEXT_ROWS
    assert len(grader_prompts) == 2


def test_with_the_search_off_rejected_rows_end_in_no_data() -> None:
    rag, database, grader_prompts = _build_graph(
        cypher_reply=WRONG_TRAVERSAL,
        query_results=[PREAMBLE_ROWS],
        fulltext_results=[FULLTEXT_ROWS],
        grader_replies=[KEEP_NONE],
        enable_fallback_search=False,
    )

    result = rag.invoke(QUESTION)

    assert result["answer"] == NO_GRAPH_DATA_MESSAGE
    assert result["metadata"]["retrieval_strategy"] == "graded_out"
    assert database.fulltext_queries == []


def test_an_unreachable_graph_during_the_search_is_still_an_outage() -> None:
    """A search that could not run says nothing about the graph, so it is not "no data"."""
    rag, _, _ = _build_graph(
        cypher_reply=WRONG_TRAVERSAL,
        query_results=[PREAMBLE_ROWS],
        fulltext_results=[ServiceUnavailable("connection refused")],
        grader_replies=[KEEP_NONE],
    )

    with pytest.raises(KnowledgeGraphUnavailableError):
        rag.invoke(QUESTION)
