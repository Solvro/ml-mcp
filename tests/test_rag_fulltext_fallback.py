"""The label-agnostic rescue is a scored, index-backed lookup, not a full-graph scan.

Review feedback on PR #57: abstention should be a retrieval decision. A CONTAINS scan gives no
signal about how good a match is, so "good enough" was left to the answering model. A Lucene
index returns a relevance score per hit, which makes the gate a number and keeps the lookup off
a full scan.
"""

from typing import Any

import pytest

from src.mcp_server.tools.knowledge_graph.cypher_guardrails import (
    UnsafeCypherQueryError,
    validate_read_only,
)
from src.mcp_server.tools.knowledge_graph.question_analysis import build_lucene_query
from src.mcp_server.tools.knowledge_graph.rag import (
    ALLOWED_RETRIEVAL_PROCEDURES,
    FALLBACK_SEARCH_CYPHER,
    FULLTEXT_INDEX_NAME,
    RAG,
)

QUESTION = "Co obejmuje udział w konferencjach?"
FALLBACK_ROWS = [{"title": "Udzial w konferencjach", "score": 4.2, "related": []}]


class ScriptedDatabase:
    """Neo4j stand-in that answers each query with a scripted result in order."""

    def __init__(
        self,
        results: list[list[dict[str, Any]] | Exception],
        labels: list[str] | None = None,
        existing_index: list[dict[str, Any]] | None = None,
    ) -> None:
        self.results = list(results)
        self.labels = labels or []
        self.existing_index = existing_index
        self.calls: list[tuple[str, dict[str, Any] | None]] = []

    def query(
        self, cypher_query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        self.calls.append((cypher_query, params))

        if "db.labels()" in cypher_query:
            return [{"label": label} for label in self.labels]
        if "SHOW INDEXES" in cypher_query:
            return list(self.existing_index or [])
        if cypher_query.startswith(("CREATE FULLTEXT", "DROP INDEX")):
            return []

        result = self.results.pop(0) if self.results else []
        if isinstance(result, Exception):
            raise result
        return result


def _rag_stub(
    results: list[list[dict[str, Any]] | Exception],
    *,
    labels: list[str] | None = None,
    existing_index: list[dict[str, Any]] | None = None,
    min_score: float = 0.5,
) -> tuple[RAG, ScriptedDatabase]:
    database = ScriptedDatabase(results, labels=labels, existing_index=existing_index)

    rag = object.__new__(RAG)
    rag.database = database
    rag.max_results = 5
    rag.enable_debug = False
    rag.enable_fallback_search = True
    rag.fallback_min_score = min_score

    return rag, database


def _exact_clauses(query: str) -> list[str]:
    """The quoted phrase clauses, without the inflection expansions added for issue #59."""
    return [clause for clause in query.split(" OR ") if clause.startswith('"')]


def test_lucene_query_boosts_longer_phrases() -> None:
    """A node matching the whole noun phrase must outrank one sharing a single word."""
    query = build_lucene_query(["udzial w konferencjach", "konferencjach"])

    assert _exact_clauses(query) == ['"udzial w konferencjach"^3', '"konferencjach"']


def test_lucene_query_drops_phrases_carrying_metacharacters() -> None:
    """A malformed query fails the whole lookup, so an odd phrase is dropped, not escaped."""
    query = build_lucene_query(["udzial w konferencjach", 'title:"x" OR *'])

    assert _exact_clauses(query) == ['"udzial w konferencjach"^3']
    assert "title" not in query


def test_lucene_query_is_empty_when_no_phrase_is_usable() -> None:
    assert build_lucene_query(["*", ""]) == ""


def test_fallback_search_passes_the_score_threshold_to_the_index() -> None:
    rag, database = _rag_stub([FALLBACK_ROWS], min_score=1.5)

    result = rag._search_every_label(QUESTION)

    assert result["retrieval_strategy"] == "label_agnostic_phrases"
    assert result["context"] == FALLBACK_ROWS

    _, params = database.calls[0]
    assert params["min_score"] == 1.5
    assert params["index_name"] == FULLTEXT_INDEX_NAME
    assert "udzial w konferencjach" in params["lucene_query"]


def test_fallback_search_is_index_backed_not_a_scan() -> None:
    """The whole point of the change: no unlabelled MATCH walking every node."""
    assert "db.index.fulltext.queryNodes" in FALLBACK_SEARCH_CYPHER
    assert "CONTAINS" not in FALLBACK_SEARCH_CYPHER
    assert "MATCH (node)\n" not in FALLBACK_SEARCH_CYPHER


def test_fallback_search_returns_nothing_when_every_hit_scores_low() -> None:
    """The database applies the threshold, so a filtered-out hit reaches us as no rows."""
    rag, _ = _rag_stub([[]], existing_index=[{"labels": ["Course"]}], labels=["Course"])

    assert rag._search_every_label(QUESTION) is None


def test_a_missing_index_is_created_and_the_search_retried() -> None:
    rag, database = _rag_stub([[], FALLBACK_ROWS], labels=["Course", "Semester"])

    result = rag._search_every_label(QUESTION)

    assert result["context"] == FALLBACK_ROWS
    created = [call[0] for call in database.calls if call[0].startswith("CREATE FULLTEXT")]
    assert len(created) == 1
    assert "`Course`|`Semester`" in created[0]
    assert "ON EACH [n.title, n.context]" in created[0]


def test_a_stale_index_is_rebuilt_when_labels_changed() -> None:
    """Ingestion can add a label between restarts; the rescue must not go blind to it."""
    rag, database = _rag_stub(
        [[], FALLBACK_ROWS],
        labels=["Course", "Semester"],
        existing_index=[{"labels": ["Course"]}],
    )

    rag._search_every_label(QUESTION)

    assert any(call[0].startswith("DROP INDEX") for call in database.calls)
    assert any(call[0].startswith("CREATE FULLTEXT") for call in database.calls)


def test_bookkeeping_labels_are_left_out_of_the_index() -> None:
    rag, database = _rag_stub(
        [[], []], labels=["Course", "ProcessedDocument", "PipelineRun", "Source"]
    )

    rag.ensure_fulltext_index()

    created = [call[0] for call in database.calls if call[0].startswith("CREATE FULLTEXT")]
    assert "ProcessedDocument" not in created[0]
    assert "PipelineRun" not in created[0]
    assert "Source" not in created[0]
    assert "`Course`" in created[0]


def test_an_empty_graph_needs_no_index() -> None:
    rag, database = _rag_stub([], labels=[])

    assert rag.ensure_fulltext_index() is False
    assert not [call for call in database.calls if call[0].startswith("CREATE FULLTEXT")]


def test_disabled_fallback_never_touches_the_index() -> None:
    rag, database = _rag_stub([FALLBACK_ROWS])
    rag.enable_fallback_search = False

    assert rag._search_every_label(QUESTION) is None
    assert database.calls == []


def test_the_fallback_query_passes_the_read_only_guardrail() -> None:
    validate_read_only(
        FALLBACK_SEARCH_CYPHER.format(max_nodes=5),
        allowed_procedures=ALLOWED_RETRIEVAL_PROCEDURES,
    )


def test_generated_cypher_still_cannot_call_any_procedure() -> None:
    """The allowlist is for our own query only; the model gets an empty one."""
    with pytest.raises(UnsafeCypherQueryError, match="blocked Cypher procedure call"):
        validate_read_only(FALLBACK_SEARCH_CYPHER.format(max_nodes=5))


def test_the_allowlist_covers_exactly_one_procedure() -> None:
    assert ALLOWED_RETRIEVAL_PROCEDURES == frozenset({"db.index.fulltext.queryNodes"})


def test_an_allowlisted_procedure_does_not_permit_a_different_one() -> None:
    with pytest.raises(UnsafeCypherQueryError, match="apoc.cypher.runwrite"):
        validate_read_only(
            "CALL apoc.cypher.runWrite('CREATE (n)', {}) YIELD value RETURN value",
            allowed_procedures=ALLOWED_RETRIEVAL_PROCEDURES,
        )


def test_call_subqueries_are_rejected_even_with_an_allowlist() -> None:
    with pytest.raises(UnsafeCypherQueryError, match="CALL subqueries"):
        validate_read_only(
            "MATCH (n) CALL { WITH n RETURN n AS m } RETURN m",
            allowed_procedures=ALLOWED_RETRIEVAL_PROCEDURES,
        )
