"""The replay benchmark for issue #108: same rows, N grader verdicts, and what each run kept."""

import json
from typing import Any

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

from benchmarks.run_grader_stability import (
    DEFAULT_CASES,
    kept_indices,
    load_cases,
    replay,
    summarize_case,
)
from src.config.config import get_config
from src.mcp_server.tools.knowledge_graph.rag import RAG

ROWS = [
    {"title": "Dzialalnosc dydaktyczna", "context": "Kategoria kryteriow"},
    {"title": "Odbyte szkolenia", "context": "Szkolenia odbyte w okresie oceny"},
    {"title": "prowadzenie zajec", "context": "Kryterium w kategorii Dzialalnosc dydaktyczna"},
]
CASE = {
    "id": "teaching",
    "question": "Jakie kryteria oceniają działalność dydaktyczną?",
    "retrieval_strategy": "label_agnostic_phrases",
    "rows": ROWS,
}


def _scripted_grader(replies: list[str | Exception]) -> RAG:
    """A RAG whose fast model answers from a script, one reply per grading call."""
    rag = object.__new__(RAG)
    rag.context_grader_template = PromptTemplate(
        input_variables=["user_question", "retrieval", "candidates"],
        template=get_config().prompts.context_grader,
    )
    script = iter(replies)

    def _invoke(prompt_value: Any) -> str:
        reply = next(script)
        if isinstance(reply, Exception):
            raise reply
        return reply

    rag.fast_llm = RunnableLambda(_invoke)
    rag._get_invoke_config = lambda **kwargs: {}
    return rag


def _reply(relevant: list[int], anchor: str | None = "Dzialalnosc dydaktyczna") -> str:
    return json.dumps({"entity": "działalność dydaktyczna", "anchor": anchor, "relevant": relevant})


def test_each_run_records_the_grader_list_next_to_what_was_kept() -> None:
    rag = _scripted_grader([_reply([1, 3]), _reply([1]), "not json", RuntimeError("down")])

    runs = replay(rag, CASE, 4)

    assert [run["grader_kept"] for run in runs] == [[0, 2], [0], None, None]
    # The second run is the #108 shape: the anchor keeps the criterion the list dropped. The
    # last two fail open and keep every row.
    assert [run["final_kept"] for run in runs] == [[0, 2], [0, 2], [0, 1, 2], [0, 1, 2]]
    assert runs[0]["anchor"] == "Dzialalnosc dydaktyczna"


def test_replay_puts_the_real_model_back() -> None:
    rag = _scripted_grader([_reply([1])])
    model = rag.fast_llm

    replay(rag, CASE, 1)

    assert rag.fast_llm is model


def test_the_summary_counts_how_far_the_verdict_moved() -> None:
    runs = [
        {"grader_kept": [0, 2], "final_kept": [0, 2]},
        {"grader_kept": [0], "final_kept": [0, 2]},
        {"grader_kept": [], "final_kept": []},
        {"grader_kept": None, "final_kept": [0, 1, 2]},
    ]

    summary = summarize_case(3, runs, expected=[2, 0])

    assert summary["grader_kept_counts"] == [2, 1, 0, None]
    assert summary["final_kept_counts"] == [2, 2, 0, 3]
    assert summary["no_verdict_runs"] == 1
    assert summary["distinct_final_sets"] == 3
    assert summary["final_empty_runs"] == 1
    assert summary["final_matches_expected"] == 2
    assert summary["row_keep_rate"][0] == {"grader": 0.5, "final": 0.75}
    assert summary["row_keep_rate"][1] == {"grader": 0.0, "final": 0.25}


def test_rows_kept_as_retrieved_count_as_every_row() -> None:
    assert kept_indices(ROWS, {"context_graded": False, "next_node": "end"}) == [0, 1, 2]
    assert kept_indices(ROWS, {"context": [ROWS[2]]}) == [2]


def test_the_checked_in_cases_name_rows_they_hold() -> None:
    payload = load_cases(DEFAULT_CASES)

    assert payload["cases"]
    for case in payload["cases"]:
        assert case["rows"], case["id"]
        assert all(0 <= index < len(case["rows"]) for index in case.get("expected", []))
