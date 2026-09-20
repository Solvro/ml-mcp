"""Whether to answer is decided on retrieval evidence, before the answer is written.

Review feedback on PR #57: the abstain/answer decision was happening in generation. It now
happens in a grader node between retrieve and answer — rows recovered by a widened search are
candidates, and a candidate that does not address the question is dropped before it can become
a confident wrong answer.
"""

import json
from typing import Any

import pytest
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

from src.config.config import get_config
from src.mcp_server.tools.knowledge_graph.rag import (
    RAG,
    GraderVerdict,
    RetrievalStrategy,
    locate_anchor,
)

QUESTION = "Kiedy jest pierwszy dzień wolny w semestrze zimowym?"
ROWS = [
    {"title": "2 XI 2026 r.", "context": "dzien wolny od zajec"},
    {"title": "Przerwa miedzysemestralna", "context": "23 lutego 2027"},
]
KEPT_AS_RETRIEVED = {"context_graded": False, "next_node": "end"}

# The question from the PR #102 review, and the row shapes its eight kg runs came back with.
CRITERIA_QUESTION = "Jakie kryteria oceniają działalność dydaktyczną?"
TEACHING = "działalność dydaktyczna"
ANCHORED_CYPHER = (
    "MATCH (cc:CriterionCategory)-[:HAS_CRITERION]->(c:Criterion) "
    "WHERE toLower(cc.title) CONTAINS toLower('dzialalnosc dydaktyczna') RETURN c.title"
)
UNANCHORED_CYPHER = (
    "MATCH (cc:CriterionCategory)-[:HAS_CRITERION]->(c:Criterion) "
    "RETURN cc.title AS category, c.title AS criterion"
)
TEACHING_ROWS = [{"c.title": "prowadzenie zajec"}, {"c.title": "opieka nad pracami dyplomowymi"}]
TRAINING_ROWS = [
    {"category": "Odbyte szkolenia", "criterion": "naukowe"},
    {"category": "Odbyte szkolenia", "criterion": "dydaktyczne"},
]
MIXED_ROWS = [
    {"category": "Odbyte szkolenia", "criterion": "dydaktyczne"},
    {"category": "Dzialalnosc dydaktyczna", "criterion": "prowadzenie zajec"},
    {"category": "Dzialalnosc organizacyjna", "criterion": "doswiadczenie w organizowaniu"},
]


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


def _grader_stub(reply: str | Exception) -> tuple[RAG, RecordingLLM]:
    """Build a RAG with the real grader prompt from graph_config.yaml and a stubbed model."""
    rag = object.__new__(RAG)
    rag.context_grader_template = PromptTemplate(
        input_variables=["user_question", "retrieval", "candidates"],
        template=get_config().prompts.context_grader,
    )
    llm = RecordingLLM(reply)
    rag.fast_llm = llm.as_runnable()
    rag._get_invoke_config = lambda **kwargs: {}
    return rag, llm


def _state(**overrides: Any) -> dict[str, Any]:
    state = {
        "user_question": QUESTION,
        "context": list(ROWS),
        "retrieval_strategy": "label_agnostic_phrases",
    }
    state.update(overrides)
    return state


def test_only_the_rows_that_answer_the_question_survive() -> None:
    rag, _ = _grader_stub('{"relevant": [1]}')

    result = rag.grade_context(_state())

    assert result["context"] == [ROWS[0]]
    assert result["context_graded"] is True


def test_rejecting_every_row_abstains_deterministically() -> None:
    """The nearest-looking row is exactly what produced the confidently wrong date."""
    rag, _ = _grader_stub('{"relevant": []}')

    result = rag.grade_context(_state())

    assert result["context"] == []
    assert result["retrieval_strategy"] == "graded_out"
    assert result["context_graded"] is True
    assert result["next_node"] == "end"


def test_a_graded_out_result_reaches_the_caller_as_no_data() -> None:
    formatted = RAG._format_result(
        {
            "context": [],
            "generated_cypher": "CALL db.index.fulltext.queryNodes($index_name, $lucene_query)",
            "guardrail_decision": "generate_cypher",
            "retrieval_strategy": "graded_out",
        }
    )

    assert formatted["answer"] == "Brak danych w grafie wiedzy dla tego pytania."
    assert formatted["metadata"]["retrieval_strategy"] == "graded_out"


def _verdict(*, anchor: str | None, relevant: list[int], entity: str = TEACHING) -> str:
    return json.dumps({"entity": entity, "anchor": anchor, "relevant": relevant})


def _criteria_state(
    cypher: str, rows: list[dict[str, Any]], retrieval_strategy: str = "primary"
) -> dict[str, Any]:
    return _state(
        user_question=CRITERIA_QUESTION,
        retrieval_strategy=retrieval_strategy,
        generated_cypher=cypher,
        context=list(rows),
    )


def test_a_primary_query_that_filters_on_the_entity_keeps_its_whole_list() -> None:
    """Everything it returned sits under the entity; the grader does not get to edit the list."""
    rag, llm = _grader_stub(_verdict(anchor="dzialalnosc dydaktyczna", relevant=[1]))

    result = rag.grade_context(_criteria_state(ANCHORED_CYPHER, TEACHING_ROWS))

    assert len(llm.prompts) == 1
    assert "context" not in result
    assert result == {"context_graded": True, "next_node": "end"}


def test_an_anchor_is_found_whatever_case_and_diacritics_the_grader_copied() -> None:
    rag, _ = _grader_stub(_verdict(anchor="Działalność dydaktyczna", relevant=[1]))

    result = rag.grade_context(_criteria_state(ANCHORED_CYPHER, TEACHING_ROWS))

    assert result == {"context_graded": True, "next_node": "end"}


def test_a_filter_on_the_teacher_anchors_course_titles_that_never_name_them() -> None:
    rag, _ = _grader_stub(
        _verdict(entity="dr Jan Kowalski", anchor="jan kowalski", relevant=[1, 2])
    )

    result = rag.grade_context(
        _state(
            user_question="Jakie kursy prowadzi dr Jan Kowalski?",
            retrieval_strategy="primary",
            generated_cypher=(
                "MATCH (p:Person)-[:TEACHES]->(c:Course) "
                "WHERE toLower(p.title) CONTAINS toLower('jan kowalski') RETURN c.title"
            ),
            context=[{"c.title": "Analiza matematyczna 1"}, {"c.title": "Algebra liniowa"}],
        )
    )

    assert result == {"context_graded": True, "next_node": "end"}


def test_an_anchored_primary_list_stays_when_the_grader_drops_every_row() -> None:
    """Measured on the fast model: right anchor, and every course title dropped anyway."""
    rag, _ = _grader_stub(_verdict(entity="dr Jan Kowalski", anchor="jan kowalski", relevant=[]))

    result = rag.grade_context(
        _state(
            user_question="Jakie kursy prowadzi dr Jan Kowalski?",
            retrieval_strategy="primary",
            generated_cypher=(
                "MATCH (p:Person)-[:TEACHES]->(c:Course) "
                "WHERE toLower(p.title) CONTAINS toLower('jan kowalski') RETURN c.title"
            ),
            context=[{"c.title": "Analiza matematyczna 1"}, {"c.title": "Algebra liniowa"}],
        )
    )

    assert result == {"context_graded": True, "next_node": "end"}


def test_the_entity_anchors_when_the_grader_says_null_but_the_query_filters_on_it() -> None:
    """The fast model did this about one run in six; the filter is evidence, the null is not."""
    rag, _ = _grader_stub(_verdict(anchor=None, relevant=[1, 2]))

    result = rag.grade_context(_criteria_state(ANCHORED_CYPHER, TEACHING_ROWS))

    assert result == {"context_graded": True, "next_node": "end"}


def test_a_label_is_the_anchor_when_the_question_names_only_a_kind() -> None:
    rag, _ = _grader_stub(_verdict(entity="dni wolne", anchor="DayOff", relevant=[1]))

    result = rag.grade_context(
        _state(
            user_question="Jakie są dni wolne od zajęć?",
            retrieval_strategy="primary",
            generated_cypher="MATCH (d:DayOff) RETURN d.title",
        )
    )

    assert result == {"context_graded": True, "next_node": "end"}


@pytest.mark.parametrize("strategy", ["primary", "repaired_literals"])
def test_a_word_that_resembles_the_entity_is_not_an_anchor(strategy) -> None:
    """The PR #102 review: "dydaktyczne" was kept as if it named "działalność dydaktyczna"."""
    rag, _ = _grader_stub(_verdict(anchor="dydaktyczne", relevant=[1]))

    result = rag.grade_context(
        _criteria_state(UNANCHORED_CYPHER, TRAINING_ROWS, retrieval_strategy=strategy)
    )

    assert result["context"] == []
    assert result["retrieval_strategy"] == "graded_out"
    assert result["next_node"] == "search_after_grading"


@pytest.mark.parametrize(
    "reply",
    [
        _verdict(anchor=None, relevant=[1]),
        '{"relevant": [1]}',
        _verdict(anchor="dzialalnosc dydaktyczna", relevant=[1]),
    ],
    ids=["null-anchor", "no-anchor", "anchor-not-in-query-or-rows"],
)
def test_model_query_rows_without_a_real_anchor_are_rejected(reply) -> None:
    rag, _ = _grader_stub(reply)

    result = rag.grade_context(_criteria_state(UNANCHORED_CYPHER, TRAINING_ROWS))

    assert result["retrieval_strategy"] == "graded_out"
    assert result["next_node"] == "search_after_grading"


def test_an_anchor_found_only_in_the_rows_keeps_just_the_rows_the_grader_picked() -> None:
    """The query did not select the entity, so its rows mix categories."""
    rag, _ = _grader_stub(_verdict(anchor="Dzialalnosc dydaktyczna", relevant=[2]))

    result = rag.grade_context(_criteria_state(UNANCHORED_CYPHER, MIXED_ROWS))

    assert result["context"] == [MIXED_ROWS[1]]
    assert result["context_graded"] is True
    assert result["next_node"] == "end"


@pytest.mark.parametrize(
    ("entity", "anchor", "expected"),
    [
        ("działalność dydaktyczną", "Dzialalnosc dydaktyczna", "rows"),
        ("działalność dydaktyczna", "Odbyte szkolenia dydaktyczne", None),
        ("dni wolne", "dzien wolny od zajec", "rows"),
        ("rok akademicki 2026/2027", "rok akademicki 2025/2026", None),
    ],
)
def test_an_anchor_has_to_cover_every_word_of_the_entity(entity, anchor, expected) -> None:
    """A case ending may differ; a missing word or a different year may not."""
    rows = [{"title": anchor}]
    verdict = GraderVerdict(kept=[0], entity=entity, anchor=anchor)

    assert locate_anchor(verdict, "MATCH (n:Topic) RETURN n.title", rows) == expected


def _filter_query(literal: str) -> str:
    return (
        "MATCH (cc:CriterionCategory)-[:HAS_CRITERION]->(c:Criterion) "
        f"WHERE toLower(cc.title) CONTAINS '{literal}' RETURN cc.title, c.title LIMIT 10"
    )


@pytest.mark.parametrize(
    ("entity", "anchor", "literal"),
    [
        ("kryteria w kategorii Dorobek naukowy", "Dorobek naukowy", "dorobek naukowy"),
        ("kompetencje pożądane dla naukowca R2", "R2", "r2"),
        (
            "działalność dydaktyczna na Politechnice Wrocławskiej",
            "Dzialalnosc dydaktyczna",
            "dzialalnosc dydaktyczna",
        ),
    ],
    ids=["category-name", "code", "institution-appended"],
)
def test_a_query_filter_on_the_name_inside_a_wider_entity_phrase_anchors_it(
    entity, anchor, literal
) -> None:
    """Ihe grader names the question's noun phrase; the query filters on the name."""
    verdict = GraderVerdict(kept=[0], entity=entity, anchor=anchor)

    assert locate_anchor(verdict, _filter_query(literal), [{"c.title": "patenty"}]) == "query"


@pytest.mark.parametrize(
    ("entity", "anchor", "literal"),
    [
        ("kursy prowadzone przez dr Jan Kowalski", "kursy prowadzone", "kursy prowadzone"),
        ("kompetencje pożądane dla naukowca R2", "kompetencje pożądane", "kompetencje pozadane"),
        ("kryteria oceny działalności dydaktycznej", "kryteria oceny", "kryteria oceny"),
        ("Kryteria w kategorii Dorobek naukowy", "Kryteria", "kryteria"),
        ("dr Jan Kowalski", "jan", "jan"),
    ],
    ids=["head-not-name", "head-not-code", "head-not-qualifier", "capital-at-word-0", "one-name"],
)
def test_a_query_filter_on_the_generic_head_of_the_entity_is_not_an_anchor(
    entity, anchor, literal
) -> None:
    """The head of the phrase is two content words too, and the name it leaves out is nowhere 
    in the query, and "query" would keep every row without escalating."""
    verdict = GraderVerdict(kept=[0], entity=entity, anchor=anchor)

    assert locate_anchor(verdict, _filter_query(literal), [{"n.title": "x"}]) is None


@pytest.mark.parametrize(
    ("entity", "anchor", "literal"),
    [
        ("Kryteria w kategorii Dorobek naukowy", "Dorobek naukowy", "dorobek naukowy"),
        ("kursy prowadzone przez dr Jan Kowalski", "Jan Kowalski", "jan kowalski"),
    ],
    ids=["capitalised-head-left-out", "name-inside-qualifier"],
)
def test_the_generic_head_may_be_left_out_however_the_grader_spelled_it(
    entity, anchor, literal
) -> None:
    verdict = GraderVerdict(kept=[0], entity=entity, anchor=anchor)

    assert locate_anchor(verdict, _filter_query(literal), [{"n.title": "x"}]) == "query"


def test_a_query_filter_on_a_lone_adjective_of_the_entity_is_not_an_anchor() -> None:
    """The #99 leak, moved into the query: "dydaktyczne" still names nothing."""
    verdict = GraderVerdict(kept=[0], entity=TEACHING, anchor="dydaktyczne")
    rows = [{"c.title": "Odbyte szkolenia dydaktyczne"}]

    assert locate_anchor(verdict, _filter_query("dydaktyczne"), rows) is None


def test_an_anchor_found_only_in_a_row_still_has_to_cover_the_whole_entity() -> None:
    verdict = GraderVerdict(
        kept=[0], entity="kryteria w kategorii Dorobek naukowy", anchor="Dorobek naukowy"
    )
    rows = [{"category": "Dorobek naukowy", "criterion": "patenty"}]

    assert locate_anchor(verdict, UNANCHORED_CYPHER, rows) is None


def test_a_code_in_the_entity_is_a_word_the_anchor_has_to_match() -> None:
    """R1 and R2 differ in a character ANCHOR_MIN_WORD_CHARS used to drop."""
    verdict = GraderVerdict(kept=[0], entity="kompetencje R2", anchor="kompetencje R1")
    rows = [{"title": "kompetencje R1"}]

    assert locate_anchor(verdict, UNANCHORED_CYPHER, rows) is None


def test_a_wider_entity_phrase_keeps_the_whole_filtered_primary_list() -> None:
    """The end-to-end shape: 3 of 6 runs threw this list away and re-found it."""
    rag, llm = _grader_stub(
        _verdict(
            entity="kryteria w kategorii Dorobek naukowy", anchor="Dorobek naukowy", relevant=[1]
        )
    )
    rows = [
        {"cc.title": "Dorobek naukowy", "c.title": "patenty"},
        {"cc.title": "Dorobek naukowy", "c.title": "grantow"},
    ]

    result = rag.grade_context(_criteria_state(_filter_query("dorobek naukowy"), rows))

    assert len(llm.prompts) == 1
    assert result == {"context_graded": True, "next_node": "end"}


@pytest.mark.parametrize("strategy", ["primary", "repaired_literals"])
def test_a_model_query_the_grader_rejects_moves_on_to_the_full_text_search(strategy) -> None:
    """Issue #99: a wrong result used to end the run, where an empty one would have escalated."""
    rag, _ = _grader_stub('{"relevant": []}')

    result = rag.grade_context(_state(retrieval_strategy=strategy))

    assert result["context"] == []
    assert result["retrieval_strategy"] == "graded_out"
    assert result["next_node"] == "search_after_grading"


@pytest.mark.parametrize(
    "strategy",
    ["label_agnostic_phrases", "label_agnostic_after_error", "label_agnostic_after_grading"],
)
def test_a_rejected_full_text_result_ends_the_run(strategy) -> None:
    """Nothing is left to try after the full-text search, and the run must not loop."""
    rag, _ = _grader_stub('{"relevant": []}')

    result = rag.grade_context(_state(retrieval_strategy=strategy))

    assert result["retrieval_strategy"] == "graded_out"
    assert result["next_node"] == "end"


def test_the_grader_sees_the_query_behind_primary_rows() -> None:
    """A bare course title only answers "what does X teach" next to the traversal from X."""
    rag, llm = _grader_stub('{"relevant": [1]}')
    cypher = "MATCH (p:Person)-[:TEACHES]->(c:Course) RETURN c.title"

    rag.grade_context(_state(retrieval_strategy="primary", generated_cypher=cypher))

    assert cypher in llm.prompts[0]


def test_the_grader_is_told_full_text_rows_may_only_share_a_word() -> None:
    rag, llm = _grader_stub('{"relevant": [1]}')

    rag.grade_context(_state(generated_cypher="CALL db.index.fulltext.queryNodes(...)"))

    assert "full-text search" in llm.prompts[0]
    assert "db.index.fulltext" not in llm.prompts[0]


@pytest.mark.parametrize("strategy", ["primary", "repaired_literals"])
def test_a_failing_grader_keeps_model_query_rows(strategy) -> None:
    rag, _ = _grader_stub(RuntimeError("provider down"))

    result = rag.grade_context(_state(retrieval_strategy=strategy))

    assert result == KEPT_AS_RETRIEVED


@pytest.mark.parametrize(
    "strategy",
    ["label_agnostic_phrases", "label_agnostic_after_error", "label_agnostic_after_grading"],
)
def test_full_text_rows_are_graded_row_by_row_without_an_anchor(strategy) -> None:
    """The anchor rule is for model queries; a full-text row is judged on its own."""
    rag, llm = _grader_stub('{"relevant": [2]}')

    result = rag.grade_context(_state(retrieval_strategy=strategy))

    assert len(llm.prompts) == 1
    assert result["context"] == [ROWS[1]]


def test_repaired_rows_with_an_anchor_are_graded_row_by_row() -> None:
    rag, _ = _grader_stub(_verdict(anchor="Dzialalnosc dydaktyczna", relevant=[2]))

    result = rag.grade_context(
        _criteria_state(ANCHORED_CYPHER, MIXED_ROWS, retrieval_strategy="repaired_literals")
    )

    assert result["context"] == [MIXED_ROWS[1]]


def test_an_empty_retrieval_skips_the_grader() -> None:
    rag, llm = _grader_stub('{"relevant": []}')

    result = rag.grade_context(_state(context=[], retrieval_strategy="empty"))

    assert llm.prompts == []
    assert result == KEPT_AS_RETRIEVED


def test_the_grader_sees_the_question_and_the_numbered_rows() -> None:
    rag, llm = _grader_stub('{"relevant": [1]}')

    rag.grade_context(_state())

    prompt = llm.prompts[0]
    assert QUESTION in prompt
    assert "1. " in prompt and "2. " in prompt
    assert "dzien wolny od zajec" in prompt
    assert "Przerwa miedzysemestralna" in prompt


def test_a_failing_grader_keeps_the_rows() -> None:
    """A model outage must not be indistinguishable from an empty graph."""
    rag, _ = _grader_stub(RuntimeError("provider down"))

    result = rag.grade_context(_state())

    assert result == KEPT_AS_RETRIEVED


@pytest.mark.parametrize(
    "reply",
    ["not json at all", "", '{"nope": [1]}', '{"relevant": "1"}'],
)
def test_an_unusable_grader_reply_keeps_the_rows(reply) -> None:
    rag, _ = _grader_stub(reply)

    result = rag.grade_context(_state())

    assert result == KEPT_AS_RETRIEVED


def test_out_of_range_and_duplicate_indices_are_ignored() -> None:
    rag, _ = _grader_stub('{"relevant": [0, 1, 1, 9, -3]}')

    result = rag.grade_context(_state())

    assert result["context"] == [ROWS[0]]


def test_a_fenced_reply_is_still_read() -> None:
    rag, _ = _grader_stub('```json\n{"relevant": [2]}\n```')

    result = rag.grade_context(_state())

    assert result["context"] == [ROWS[1]]


def test_long_rows_are_truncated_before_grading() -> None:
    """A wide fallback result must stay one cheap call."""
    rag, llm = _grader_stub('{"relevant": []}')
    long_row = {"title": "Kurs", "context": "x" * 5000}

    rag.grade_context(_state(context=[long_row]))

    assert "..." in llm.prompts[0]
    assert len(llm.prompts[0]) < 3000


def test_the_answer_payload_carries_how_the_rows_were_found() -> None:
    """The answering model cannot tell an answer from a candidate without the strategy."""
    formatted = RAG._format_result(
        {
            "context": [{"title": "Udział w konferencjach"}],
            "generated_cypher": "MATCH (n) RETURN n.title",
            "guardrail_decision": "generate_cypher",
            "retrieval_strategy": "label_agnostic_phrases",
            "context_graded": True,
        }
    )

    payload = json.loads(formatted["answer"])
    assert payload["retrieval_strategy"] == "label_agnostic_phrases"
    assert payload["context_graded"] is True
    assert payload["rows"] == [{"title": "Udział w konferencjach"}]
    assert formatted["metadata"]["context_graded"] is True


def test_a_trusted_primary_result_says_so_in_the_payload() -> None:
    formatted = RAG._format_result(
        {
            "context": [{"title": "Analiza matematyczna"}],
            "generated_cypher": "MATCH (n) RETURN n.title",
            "guardrail_decision": "generate_cypher",
            "retrieval_strategy": "primary",
            "context_graded": False,
        }
    )

    payload = json.loads(formatted["answer"])
    assert payload["retrieval_strategy"] == "primary"
    assert payload["context_graded"] is False


def test_the_answer_prompt_explains_both_kinds_of_row() -> None:
    prompt = get_config().prompts.final_answer

    assert "retrieval_strategy" in prompt
    assert "CANDIDATES" in prompt


@pytest.mark.parametrize(
    "strategy",
    [
        strategy.value
        for strategy in RetrievalStrategy
        if strategy not in (RetrievalStrategy.GRADED_OUT, RetrievalStrategy.EMPTY)
    ],
)
def test_the_answer_prompt_names_every_strategy_that_carries_rows(strategy) -> None:
    """An unnamed strategy leaves the answering model guessing how far to trust its rows."""
    assert f'"{strategy}"' in get_config().prompts.final_answer
