"""Tests for what happens when a generated Cypher query executes but matches nothing.

The scenarios are the ones reported in issue #52: the question copied into a CONTAINS literal,
and the answer stored under a label the model did not pick. Both must be recovered rather than
reported as missing data, and a genuinely empty result must be reported as an explicit
"no data" answer instead of an empty JSON list.
"""

from typing import Any

from src.config.messages import NO_GRAPH_DATA_MESSAGE, OFF_TOPIC_MESSAGE
from src.mcp_server.tools.knowledge_graph.rag import RAG

CRITERIA_QUESTION = "Jakie są kryteria doboru kandydatki lub kandydata?"
CONFERENCE_QUESTION = "Co obejmuje udział w konferencjach?"

QUESTION_LITERAL_CYPHER = (
    "MATCH (g:Guideline)-[:RECOMMENDS]->(c:Committee)-[:CONSIDERS]->(comp:Competency) "
    "WHERE toLower(g.title) CONTAINS "
    "toLower('jakie sa kryteria doboru kandydatki lub kandydata') "
    "RETURN g.title, c.title, comp.title"
)
WRONG_LABEL_CYPHER = (
    "MATCH (cc:CriterionCategory)-[:HAS_ITEM]->(ci:CriterionItem) RETURN cc.title, ci.title"
)

CRITERIA_ROWS = [{"g.title": "Kryteria doboru", "comp.title": "Praca zespolowa"}]
CONFERENCE_ROWS = [{"title": "Udzial w konferencjach", "related": ["HAS_SUBCOMPETENCY: Panel"]}]


class ScriptedDatabase:
    """Neo4j stand-in that answers each retrieval query with a scripted result in order.

    Schema statements (label listing, index inspection, index creation) are answered separately
    so they do not consume a scripted retrieval result or distort the call count.
    """

    def __init__(self, results: list[list[dict[str, Any]] | Exception]) -> None:
        self.results = list(results)
        self.calls: list[tuple[str, dict[str, Any] | None]] = []
        self.schema_calls: list[str] = []

    def query(
        self, cypher_query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        if (
            "db.labels()" in cypher_query
            or "SHOW INDEXES" in cypher_query
            or cypher_query.startswith(("CREATE FULLTEXT", "DROP INDEX"))
        ):
            self.schema_calls.append(cypher_query)
            return [{"label": "Course"}] if "db.labels()" in cypher_query else []

        self.calls.append((cypher_query, params))
        result = self.results.pop(0) if self.results else []
        if isinstance(result, Exception):
            raise result
        return result


def _rag_stub(
    results: list[list[dict[str, Any]] | Exception],
    *,
    enable_fallback_search: bool = True,
    max_results: int = 5,
) -> tuple[RAG, ScriptedDatabase]:
    database = ScriptedDatabase(results)

    rag = object.__new__(RAG)
    rag.database = database
    rag.max_results = max_results
    rag.enable_debug = False
    rag.enable_fallback_search = enable_fallback_search
    rag.fallback_min_score = 0.5

    return rag, database


def test_rows_on_the_first_attempt_skip_every_retry() -> None:
    rag, database = _rag_stub([CRITERIA_ROWS])

    result = rag.retrieve(
        {"generated_cypher": QUESTION_LITERAL_CYPHER, "user_question": CRITERIA_QUESTION}
    )

    assert result["retrieval_strategy"] == "primary"
    assert result["context"] == CRITERIA_ROWS
    assert len(database.calls) == 1


def test_copied_question_literal_is_retried_without_the_filter() -> None:
    rag, database = _rag_stub([[], CRITERIA_ROWS])

    result = rag.retrieve(
        {"generated_cypher": QUESTION_LITERAL_CYPHER, "user_question": CRITERIA_QUESTION}
    )

    assert result["retrieval_strategy"] == "repaired_literals"
    assert result["context"] == CRITERIA_ROWS
    assert len(database.calls) == 2

    retried_query = database.calls[1][0]
    assert "WHERE true" in retried_query
    assert "jakie sa kryteria" not in retried_query
    assert "[:RECOMMENDS]" in retried_query, "the traversal the model wrote must be kept"
    assert result["generated_cypher"] == retried_query


def test_wrong_label_falls_back_to_searching_every_label() -> None:
    rag, database = _rag_stub([[], CONFERENCE_ROWS])

    result = rag.retrieve(
        {"generated_cypher": WRONG_LABEL_CYPHER, "user_question": CONFERENCE_QUESTION}
    )

    assert result["retrieval_strategy"] == "label_agnostic_phrases"
    assert result["context"] == CONFERENCE_ROWS
    assert len(database.calls) == 2, "nothing to repair, so the literal retry is skipped"

    fallback_query, params = database.calls[1]
    assert "db.index.fulltext.queryNodes" in fallback_query
    assert "CriterionCategory" not in fallback_query
    assert "udzial w konferencjach" in params["lucene_query"]
    assert params["min_score"] == 0.5
    assert fallback_query.count("LIMIT") == 1
    assert "LIMIT 5" in fallback_query


def test_both_retries_run_before_giving_up() -> None:
    rag, database = _rag_stub([[], [], CRITERIA_ROWS])

    result = rag.retrieve(
        {"generated_cypher": QUESTION_LITERAL_CYPHER, "user_question": CRITERIA_QUESTION}
    )

    assert result["retrieval_strategy"] == "label_agnostic_phrases"
    assert result["context"] == CRITERIA_ROWS
    assert len(database.calls) == 3


def test_giving_up_reports_the_query_that_was_generated() -> None:
    rag, database = _rag_stub([[], [], [], []])

    result = rag.retrieve(
        {"generated_cypher": QUESTION_LITERAL_CYPHER, "user_question": CRITERIA_QUESTION}
    )

    assert result["retrieval_strategy"] == "empty"
    assert result["context"] == []
    assert "jakie sa kryteria" in result["generated_cypher"]
    # primary, repaired, label-agnostic, then label-agnostic once more after refreshing the
    # index - an empty result can mean the index is missing rather than the answer is.
    assert len(database.calls) == 4
    assert any("CREATE FULLTEXT" in statement for statement in database.schema_calls)


def test_a_failing_retry_never_turns_an_empty_result_into_an_error() -> None:
    rag, database = _rag_stub([[], RuntimeError("syntax error"), CONFERENCE_ROWS])

    result = rag.retrieve(
        {"generated_cypher": QUESTION_LITERAL_CYPHER, "user_question": CRITERIA_QUESTION}
    )

    assert result["retrieval_strategy"] == "label_agnostic_phrases"
    assert result["context"] == CONFERENCE_ROWS
    assert len(database.calls) == 3


def test_disabled_fallback_search_stops_after_the_literal_retry() -> None:
    rag, database = _rag_stub([[], []], enable_fallback_search=False)

    result = rag.retrieve(
        {"generated_cypher": QUESTION_LITERAL_CYPHER, "user_question": CRITERIA_QUESTION}
    )

    assert result["retrieval_strategy"] == "empty"
    assert len(database.calls) == 2


def test_missing_question_skips_the_retries() -> None:
    """retrieve() is also called without a question by benchmarks and older callers."""
    rag, database = _rag_stub([[]])

    result = rag.retrieve({"generated_cypher": WRONG_LABEL_CYPHER})

    assert result["retrieval_strategy"] == "empty"
    assert len(database.calls) == 1


def test_blocked_query_is_not_retried() -> None:
    """Escalation repairs a bad match, not a rejected query."""
    rag, database = _rag_stub([CRITERIA_ROWS])

    result = rag.retrieve(
        {
            "generated_cypher": "MATCH (n) DETACH DELETE n RETURN n",
            "user_question": CRITERIA_QUESTION,
        }
    )

    assert database.calls == []
    assert result["context"] == []
    assert result["retrieval_strategy"] == "empty"
    assert result["generated_cypher"].startswith("Blocked unsafe Cypher")


def test_database_failure_is_not_retried() -> None:
    rag, database = _rag_stub([RuntimeError("neo4j unavailable")])

    result = rag.retrieve(
        {"generated_cypher": WRONG_LABEL_CYPHER, "user_question": CONFERENCE_QUESTION}
    )

    assert len(database.calls) == 1
    assert result["retrieval_strategy"] == "empty"
    assert result["generated_cypher"] == "Query failed: neo4j unavailable"


def test_empty_context_is_reported_as_no_data_not_as_an_empty_list() -> None:
    """An empty JSON list invites the answering model to fill the gap; a sentence does not."""
    result = RAG._format_result(
        {
            "context": [],
            "generated_cypher": QUESTION_LITERAL_CYPHER,
            "guardrail_decision": "generate_cypher",
            "retrieval_strategy": "empty",
        }
    )

    assert result["answer"] == NO_GRAPH_DATA_MESSAGE
    assert result["metadata"]["retrieval_strategy"] == "empty"
    assert result["metadata"]["cypher_query"] == QUESTION_LITERAL_CYPHER
    assert result["metadata"]["context"] == []


def test_recovered_context_is_serialized_with_polish_characters_intact() -> None:
    result = RAG._format_result(
        {
            "context": [{"title": "Udział w konferencjach"}],
            "generated_cypher": "MATCH (node) RETURN node.title",
            "guardrail_decision": "generate_cypher",
            "retrieval_strategy": "label_agnostic_phrases",
        }
    )

    assert "Udział w konferencjach" in result["answer"]
    assert result["metadata"]["retrieval_strategy"] == "label_agnostic_phrases"


def test_off_topic_answer_keeps_reporting_no_cypher() -> None:
    result = RAG._format_result(
        {
            "answer": OFF_TOPIC_MESSAGE,
            "context": [],
            "generated_cypher": None,
            "guardrail_decision": "end",
            "retrieval_strategy": "empty",
        }
    )

    assert result["answer"] == OFF_TOPIC_MESSAGE
    assert result["metadata"]["cypher_query"] is None
    assert result["metadata"]["context"] == []
