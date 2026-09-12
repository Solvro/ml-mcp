"""Tests for what happens when a generated Cypher query executes but matches nothing.

The scenarios are the ones reported in issue #52: the question copied into a CONTAINS literal,
and the answer stored under a label the model did not pick. Both must be recovered rather than
reported as missing data, and a genuinely empty result must be reported as an explicit
"no data" answer instead of an empty JSON list.
"""

from typing import Any

import pytest
from neo4j.exceptions import ClientError, ServiceUnavailable
from neo4j.time import DateTime

from src.config.messages import NO_GRAPH_DATA_MESSAGE, OFF_TOPIC_MESSAGE
from src.mcp_server.tools.knowledge_graph.rag import (
    RAG,
    KnowledgeGraphQueryError,
    KnowledgeGraphUnavailableError,
)

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


def _statement_error(message: str = "bad query") -> ClientError:
    return ClientError._hydrate_neo4j(
        code="Neo.ClientError.Statement.SyntaxError",
        message=message,
    )


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
    rag.enable_fallback_search = enable_fallback_search
    rag.fallback_min_score = 0.5
    rag._init_schema_cache()

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

    with pytest.raises(KnowledgeGraphQueryError, match="blocked"):
        rag.retrieve(
            {
                "generated_cypher": "MATCH (n) DETACH DELETE n RETURN n",
                "user_question": CRITERIA_QUESTION,
            }
        )

    assert database.calls == []


def test_database_outage_is_not_retried_and_is_propagated() -> None:
    rag, database = _rag_stub([ServiceUnavailable("neo4j unavailable")])

    with pytest.raises(KnowledgeGraphUnavailableError, match="neo4j unavailable"):
        rag.retrieve({"generated_cypher": WRONG_LABEL_CYPHER, "user_question": CONFERENCE_QUESTION})

    assert len(database.calls) == 1


# Issue #3: a syntax or type error in the model's Cypher was the most common way a question
# whose answer is in the graph came back as "no data", and the same question succeeded on the
# next run. Neo4j rejecting the statement says nothing about the graph, so the question goes to
# the full-text search instead - the one retry that does not depend on the model's query.
BROKEN_CYPHER = "MATCH (n) RETURN r.title"


def test_a_rejected_statement_is_escalated_to_the_label_agnostic_search() -> None:
    rag, database = _rag_stub([_statement_error("Variable `r` not defined"), CONFERENCE_ROWS])

    result = rag.retrieve({"generated_cypher": BROKEN_CYPHER, "user_question": CONFERENCE_QUESTION})

    assert result["context"] == CONFERENCE_ROWS
    assert result["retrieval_strategy"] == "label_agnostic_after_error"
    assert len(database.calls) == 2
    assert "db.index.fulltext.queryNodes" in database.calls[1][0]
    assert result["generated_cypher"] != BROKEN_CYPHER, "the rows came from the search query"


def test_the_literal_repair_retry_is_skipped_after_a_rejected_statement() -> None:
    """That retry is derived from the failed statement and would fail the same way."""
    rag, database = _rag_stub([_statement_error("invalid input"), CRITERIA_ROWS])

    result = rag.retrieve(
        {"generated_cypher": QUESTION_LITERAL_CYPHER, "user_question": CRITERIA_QUESTION}
    )

    assert result["retrieval_strategy"] == "label_agnostic_after_error"
    assert len(database.calls) == 2
    assert "db.index.fulltext.queryNodes" in database.calls[1][0]


def test_a_rejected_statement_whose_search_finds_nothing_is_an_empty_result() -> None:
    """The search ran against the graph and matched nothing: that is "no data", honestly."""
    rag, database = _rag_stub([_statement_error("invalid input"), [], []])

    result = rag.retrieve({"generated_cypher": BROKEN_CYPHER, "user_question": CONFERENCE_QUESTION})

    assert result["context"] == []
    assert result["retrieval_strategy"] == "empty"
    assert len(database.calls) == 3, "primary, search, search again after the index re-check"


def test_a_rejected_statement_stays_a_query_error_when_the_search_is_disabled() -> None:
    """With nothing to escalate to, the failure stands: nothing was put to the graph."""
    rag, database = _rag_stub([_statement_error("invalid input")], enable_fallback_search=False)

    with pytest.raises(KnowledgeGraphQueryError, match="invalid input"):
        rag.retrieve({"generated_cypher": BROKEN_CYPHER, "user_question": CONFERENCE_QUESTION})

    assert len(database.calls) == 1


def test_a_rejected_statement_stays_a_query_error_without_a_question_to_search() -> None:
    rag, database = _rag_stub([_statement_error("invalid input"), CONFERENCE_ROWS])

    with pytest.raises(KnowledgeGraphQueryError, match="invalid input"):
        rag.retrieve({"generated_cypher": BROKEN_CYPHER})

    assert len(database.calls) == 1


def test_a_database_error_outside_the_statement_family_is_not_escalated() -> None:
    """Schema.* names something missing in the database, not a mistake in the statement."""
    missing_index = ClientError._hydrate_neo4j(
        code="Neo.ClientError.Schema.IndexNotFound", message="no such index"
    )
    rag, database = _rag_stub([missing_index, CONFERENCE_ROWS])

    with pytest.raises(KnowledgeGraphQueryError, match="no such index"):
        rag.retrieve({"generated_cypher": BROKEN_CYPHER, "user_question": CONFERENCE_QUESTION})

    assert len(database.calls) == 1


def test_infrastructure_failure_during_retry_is_propagated() -> None:
    rag, database = _rag_stub([[], ServiceUnavailable("neo4j unavailable")])

    with pytest.raises(KnowledgeGraphUnavailableError, match="neo4j unavailable"):
        rag.retrieve(
            {"generated_cypher": QUESTION_LITERAL_CYPHER, "user_question": CRITERIA_QUESTION}
        )

    assert len(database.calls) == 2


def test_timeout_during_retry_is_propagated() -> None:
    timed_out = ClientError._hydrate_neo4j(
        code="Neo.ClientError.Transaction.TransactionTimedOutClientConfiguration",
        message="The transaction has been terminated.",
    )
    rag, database = _rag_stub([[], timed_out])

    with pytest.raises(KnowledgeGraphUnavailableError, match="terminated"):
        rag.retrieve(
            {"generated_cypher": QUESTION_LITERAL_CYPHER, "user_question": CRITERIA_QUESTION}
        )

    assert len(database.calls) == 2


def test_missing_fulltext_index_is_repaired_rather_than_reported_as_an_outage() -> None:
    missing_index = ClientError._hydrate_neo4j(
        code="Neo.ClientError.Procedure.ProcedureCallFailed",
        message=(
            "Failed to invoke procedure `db.index.fulltext.queryNodes`: "
            "There is no such fulltext schema index: entity_search"
        ),
    )
    rag, database = _rag_stub([[], missing_index, CONFERENCE_ROWS])

    result = rag.retrieve(
        {"generated_cypher": WRONG_LABEL_CYPHER, "user_question": CONFERENCE_QUESTION}
    )

    assert result["retrieval_strategy"] == "label_agnostic_phrases"
    assert result["context"] == CONFERENCE_ROWS
    assert len(database.calls) == 3, "the fallback runs again once the index has been rebuilt"
    assert any(call.startswith("CREATE FULLTEXT") for call in database.schema_calls)


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


def test_rows_carrying_neo4j_temporal_values_still_serialise() -> None:
    """Seen live on 2026-09-10: a query touched a ProcessedDocument, whose `created_at` is a
    Cypher datetime(), and the whole call failed with "Object of type DateTime is not JSON
    serializable". json has no encoder for the driver's types; the answer must not depend on
    which properties a generated query happens to return."""
    result = RAG._format_result(
        {
            "context": [{"d.title": "Zal. nr 2", "d.created_at": DateTime(2026, 9, 10, 11, 42)}],
            "retrieval_strategy": "primary",
            "generated_cypher": "MATCH (d:Document) RETURN d.title, d.created_at",
        }
    )

    assert "2026-09-10" in result["answer"]
    assert result["metadata"]["retrieval_strategy"] == "primary"
