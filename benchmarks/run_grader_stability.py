"""Replay the context grader over fixed rows and count how often its verdict changes (#108).

The full-text search returns the same rows for the same question. What survives grading is one
fast-model call, and at temperature 0 that call still varies from run to run. This runs
``RAG.grade_context`` N times over one saved set of rows and reports, for every run, the rows
the grader listed next to the rows the run actually kept, so a change to the grading rules can
be measured on the same input before and after.

``--capture`` replaces each case's rows with what the full-text search returns from the Neo4j
in ``.env``, which is how the reconstructed rows in the checked-in cases get swapped for real
ones. Replaying needs only an LLM key.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = REPO_ROOT / "benchmarks" / "grader_stability_cases.json"


def load_cases(path: Path) -> dict[str, Any]:
    """Read a case file: a question, how its rows were found, and the rows themselves."""
    return json.loads(path.read_text(encoding="utf-8"))


def kept_indices(rows: list[Any], result: dict[str, Any]) -> list[int]:
    """
    Map what ``grade_context`` returned back to positions in ``rows``.

    A result without ``context`` left the rows as retrieved: a failing grader, or a primary
    list kept whole by its anchor.
    """
    if "context" not in result:
        return list(range(len(rows)))
    kept = {id(row) for row in result["context"]}
    return [index for index, row in enumerate(rows) if id(row) in kept]


def summarize_case(
    row_count: int, runs: list[dict[str, Any]], expected: list[int] | None = None
) -> dict[str, Any]:
    """
    Say how much one case's verdict moved across runs.

    Args:
        row_count: How many rows the grader was shown
        runs: One entry per run, with ``grader_kept`` (None when the reply was unusable) and
            ``final_kept``
        expected: The rows a correct verdict keeps, when the case names them

    Returns:
        Kept counts per run, how many runs got no usable verdict, how many distinct final row
        sets there were, how many runs ended with nothing, and how often each row was kept by
        the grader and by the run
    """
    total = len(runs)
    grader_counts: Counter[int] = Counter()
    final_counts: Counter[int] = Counter()
    for run in runs:
        grader_counts.update(run["grader_kept"] or [])
        final_counts.update(run["final_kept"])

    summary: dict[str, Any] = {
        "runs": total,
        "grader_kept_counts": [
            None if run["grader_kept"] is None else len(run["grader_kept"]) for run in runs
        ],
        "final_kept_counts": [len(run["final_kept"]) for run in runs],
        # A failed call or an unreadable reply keeps every row, so a run where the model was
        # unreachable looks perfectly stable. This says how many runs measured nothing.
        "no_verdict_runs": sum(run["grader_kept"] is None for run in runs),
        "distinct_final_sets": len({tuple(run["final_kept"]) for run in runs}),
        "final_empty_runs": sum(not run["final_kept"] for run in runs),
        "row_keep_rate": {
            index: {
                "grader": round(grader_counts[index] / total, 2) if total else 0.0,
                "final": round(final_counts[index] / total, 2) if total else 0.0,
            }
            for index in range(row_count)
        },
    }
    if expected is not None:
        summary["final_matches_expected"] = sum(
            run["final_kept"] == sorted(expected) for run in runs
        )
    return summary


def _recording(model: Any) -> tuple[Any, dict[str, str | None]]:
    """Wrap the grader model so every raw reply is kept next to what the run made of it."""
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnableLambda

    last: dict[str, str | None] = {"reply": None}
    to_text = StrOutputParser()

    def _invoke(prompt_value: Any, config: dict[str, Any] | None = None) -> Any:
        message = model.invoke(prompt_value, config=config)
        last["reply"] = to_text.invoke(message)
        return message

    return RunnableLambda(_invoke), last


def replay(rag: Any, case: dict[str, Any], runs: int) -> list[dict[str, Any]]:
    """
    Grade one case's rows ``runs`` times through the real ``grade_context``.

    Args:
        rag: A RAG whose ``fast_llm`` is the model to measure
        case: The question, its retrieval strategy, the query behind the rows and the rows
        runs: How many verdicts to collect

    Returns:
        One entry per run: the grader's raw reply, what it listed, and what the run kept
    """
    rows = case["rows"]
    model = rag.fast_llm
    rag.fast_llm, last = _recording(model)
    results = []
    try:
        for _ in range(runs):
            last["reply"] = None
            result = rag.grade_context(
                {
                    "user_question": case["question"],
                    "context": list(rows),
                    "retrieval_strategy": case.get("retrieval_strategy", "label_agnostic_phrases"),
                    "generated_cypher": case.get("generated_cypher") or "",
                    "rows_truncated": False,
                }
            )
            reply = last["reply"]
            verdict = rag._parse_grader_output(reply, len(rows)) if reply is not None else None
            results.append(
                {
                    "reply": reply,
                    "entity": verdict.entity if verdict else None,
                    "anchor": verdict.anchor if verdict else None,
                    "grader_kept": sorted(verdict.kept) if verdict else None,
                    "final_kept": kept_indices(rows, result),
                    "final_strategy": result.get(
                        "retrieval_strategy", case.get("retrieval_strategy")
                    ),
                }
            )
    finally:
        rag.fast_llm = model
    return results


def build_grader() -> Any:
    """
    Build a RAG that can grade and nothing else.

    ``RAG.__init__`` opens Neo4j and reads the schema, and replaying saved rows needs neither,
    so this sets up only what ``grade_context`` reads, the way the unit tests do.
    """
    from src.config.config import get_config
    from src.config.timeouts import get_llm_timeout_seconds
    from src.mcp_server.tools.knowledge_graph.rag import RAG

    rag = object.__new__(RAG)
    rag.config = get_config()
    rag.api_key = (
        os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or os.getenv("GOOGLE_API_KEY")
    )
    rag.llm_timeout_sec = get_llm_timeout_seconds()
    rag._initialize_prompt_templates()
    rag.fast_llm = rag._build_llm_with_fallback(use_accurate=False)
    return rag


def capture(cases_path: Path) -> None:
    """Replace every case's rows with what the full-text search returns from the live graph."""
    from src.mcp_server.tools.knowledge_graph.rag import RAG, RetrievalStrategy

    payload = load_cases(cases_path)
    rag = RAG(
        api_key=os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or "",
        neo4j_url=os.environ["NEO4J_URI"],
        neo4j_username=os.environ["NEO4J_USER"],
        neo4j_password=os.environ["NEO4J_PASSWORD"],
    )
    try:
        for case in payload["cases"]:
            found = rag._search_every_label(case["question"])
            case["rows"] = found["context"] if found else []
            case["retrieval_strategy"] = RetrievalStrategy.LABEL_AGNOSTIC_PHRASES.value
            case["generated_cypher"] = None
            case["source"] = f"captured {date.today().isoformat()} from {os.environ['NEO4J_URI']}"
            # The indices named rows that are no longer there.
            case.pop("expected", None)
            print(f"{case['id']}: {len(case['rows'])} row(s)")
    finally:
        rag.close()

    cases_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--runs", type=int, default=8)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--only", action="append", help="case id to run; repeatable")
    parser.add_argument("--capture", action="store_true", help="refresh rows from Neo4j")
    parser.add_argument("--env-file", type=Path, default=REPO_ROOT / ".env")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(args.env_file, override=False)
    sys.path.insert(0, str(REPO_ROOT))

    if args.capture:
        capture(args.cases)
        return

    payload = load_cases(args.cases)
    rag = build_grader()
    report = {"model": rag.config.llm.fast_model.name, "runs": args.runs, "cases": []}
    for case in payload["cases"]:
        if args.only and case["id"] not in args.only:
            continue
        runs = replay(rag, case, args.runs)
        summary = summarize_case(len(case["rows"]), runs, case.get("expected"))
        report["cases"].append({"id": case["id"], "question": case["question"], **summary})
        report["cases"][-1]["detail"] = runs
        print(
            f"{case['id']}: grader kept {summary['grader_kept_counts']}, "
            f"run kept {summary['final_kept_counts']}, "
            f"{summary['distinct_final_sets']} distinct set(s), "
            f"{summary['final_empty_runs']} empty"
            + (
                f", {summary['final_matches_expected']}/{summary['runs']} as expected"
                if "final_matches_expected" in summary
                else ""
            )
            + (
                f", {summary['no_verdict_runs']} without a verdict"
                if summary["no_verdict_runs"]
                else ""
            )
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    if report["cases"] and all(case["no_verdict_runs"] == case["runs"] for case in report["cases"]):
        sys.exit("No run got a verdict from the grader, so nothing was measured.")


if __name__ == "__main__":
    main()
