"""Whether to answer is decided on retrieval evidence, before the answer is written.

Review feedback on PR #57: the abstain/answer decision was happening in generation. It now
happens in a grader node between retrieve and answer — rows recovered by a widened search are
candidates, and a candidate that does not address the question is dropped before it can become
a confident wrong answer.
"""

from typing import Any

import pytest
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

from src.config.config import get_config
from src.mcp_server.tools.knowledge_graph.rag import RAG

QUESTION = "Kiedy jest pierwszy dzień wolny w semestrze zimowym?"
ROWS = [
    {"title": "2 XI 2026 r.", "context": "dzien wolny od zajec"},
    {"title": "Przerwa miedzysemestralna", "context": "23 lutego 2027"},
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
    rag.enable_debug = False
    rag.context_grader_template = PromptTemplate(
        input_variables=["user_question", "candidates"],
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


def test_a_primary_query_is_trusted_and_never_graded() -> None:
    """The query expressed the question's own structure, so its rows need no second opinion."""
    rag, llm = _grader_stub('{"relevant": []}')

    result = rag.grade_context(_state(retrieval_strategy="primary"))

    assert llm.prompts == []
    assert result == {"context_graded": False}


@pytest.mark.parametrize("strategy", ["repaired_literals", "label_agnostic_phrases"])
def test_every_rescue_path_is_graded(strategy) -> None:
    rag, llm = _grader_stub('{"relevant": [2]}')

    result = rag.grade_context(_state(retrieval_strategy=strategy))

    assert len(llm.prompts) == 1
    assert result["context"] == [ROWS[1]]


def test_an_empty_retrieval_skips_the_grader() -> None:
    rag, llm = _grader_stub('{"relevant": []}')

    result = rag.grade_context(_state(context=[], retrieval_strategy="empty"))

    assert llm.prompts == []
    assert result == {"context_graded": False}


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

    assert result == {"context_graded": False}


@pytest.mark.parametrize(
    "reply",
    ["not json at all", "", '{"nope": [1]}', '{"relevant": "1"}'],
)
def test_an_unusable_grader_reply_keeps_the_rows(reply) -> None:
    rag, _ = _grader_stub(reply)

    result = rag.grade_context(_state())

    assert result == {"context_graded": False}


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
