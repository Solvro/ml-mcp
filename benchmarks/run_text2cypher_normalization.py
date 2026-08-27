"""Run the Text2Cypher normalization benchmark against one code revision."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

POLISH_DIACRITIC_TRANSLATION = str.maketrans({"ł": "l", "Ł": "L"})
VARIANT_BUILDERS = {
    "canonical": lambda entity: entity["stored"],
    "diacritics": lambda entity: entity["original"],
    "case": lambda entity: entity["stored"].lower(),
    "case_and_diacritics": lambda entity: entity["original"].lower(),
}


def normalize_for_scoring(value: str) -> str:
    """Return a lowercase, diacritic-folded representation for result scoring."""
    translated = value.translate(POLISH_DIACRITIC_TRANSLATION)
    decomposed = unicodedata.normalize("NFKD", translated)
    folded = "".join(char for char in decomposed if not unicodedata.combining(char))
    return folded.casefold()


def load_cases(path: Path) -> tuple[str, list[dict[str, str]]]:
    """Expand entity definitions into the four requested variation buckets."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    template = payload["question_template"]
    cases: list[dict[str, str]] = []

    for entity in payload["entities"]:
        for category, builder in VARIANT_BUILDERS.items():
            rendered_entity = builder(entity)
            cases.append(
                {
                    "id": f"{entity['id']}:{category}",
                    "entity_id": entity["id"],
                    "category": category,
                    "question": template.format(entity=rendered_entity),
                    "search_value": rendered_entity,
                    "expected": entity["stored"],
                }
            )

    return payload["source"], cases


def summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    """Aggregate hit, non-empty, and toLower compliance metrics by category."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["category"]].append(row)
        grouped["overall"].append(row)

    summary: dict[str, dict[str, float | int]] = {}
    for category, category_rows in grouped.items():
        total = len(category_rows)
        summary[category] = {
            "total": total,
            "hits": sum(bool(row["hit"]) for row in category_rows),
            "hit_rate": round(sum(bool(row["hit"]) for row in category_rows) / total, 4),
            "non_empty": sum(bool(row["non_empty"]) for row in category_rows),
            "non_empty_rate": round(
                sum(bool(row["non_empty"]) for row in category_rows) / total,
                4,
            ),
            "tolower_compliant": sum(bool(row["tolower_compliant"]) for row in category_rows),
            "tolower_compliance_rate": round(
                sum(bool(row["tolower_compliant"]) for row in category_rows) / total,
                4,
            ),
        }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("baseline", "prompt-only", "full"), required=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--neo4j-uri", default="bolt://localhost:17687")
    parser.add_argument("--provider", choices=("openai", "clarin"), default="openai")
    return parser.parse_args()


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Generate and execute Cypher for every case using the selected revision."""
    repo_root = args.repo_root.resolve()
    load_dotenv(args.env_file.resolve(), override=False)
    os.environ["NEO4J_URI"] = args.neo4j_uri

    sys.path.insert(0, str(repo_root))
    os.chdir(repo_root)

    from src.config.config import get_config
    from src.mcp_server.tools.knowledge_graph.cypher_guardrails import (
        ensure_limit,
        strip_code_fences,
        validate_read_only,
    )
    from src.mcp_server.tools.knowledge_graph.rag import RAG

    config = get_config()
    api_key = (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("DEEPSEEK_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or ""
    )
    rag = RAG(
        api_key=api_key,
        neo4j_url=args.neo4j_uri,
        neo4j_username=os.environ["NEO4J_USER"],
        neo4j_password=os.environ["NEO4J_PASSWORD"],
        max_results=10,
    )
    model_name = config.llm.accurate_model.name
    if args.provider == "clarin":
        from langchain_openai import ChatOpenAI

        model_name = config.llm.clarin.name
        rag.cypher_llm = ChatOpenAI(
            model_name=model_name,
            base_url=config.llm.clarin.base_url,
            api_key=os.environ["CLARIN_API_KEY"],
            temperature=0,
        )

    source, cases = load_cases(args.cases)
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index:02d}/{len(cases)}] {args.mode} {case['id']}", flush=True)
        try:
            generated = rag.generate_cypher({"user_question": case["question"]})["generated_cypher"]
            if args.mode == "full":
                retrieval = rag.retrieve(
                    {"generated_cypher": generated, "user_question": case["question"]}
                )
                executed_query = retrieval["generated_cypher"]
                context = retrieval["context"]
            else:
                executed_query = ensure_limit(strip_code_fences(generated), rag.max_results)
                validate_read_only(executed_query)
                context = rag.database.query(executed_query)

            context_text = normalize_for_scoring(
                json.dumps(context, ensure_ascii=False, default=str)
            )
            expected_text = normalize_for_scoring(case["expected"])
            tolower_count = len(re.findall(r"\btoLower\s*\(", executed_query, re.IGNORECASE))
            rows.append(
                {
                    **case,
                    "generated_cypher": generated,
                    "executed_cypher": executed_query,
                    "context": context,
                    "hit": expected_text in context_text,
                    "non_empty": bool(context),
                    "tolower_compliant": tolower_count >= 2,
                    "error": None,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    **case,
                    "generated_cypher": None,
                    "executed_cypher": None,
                    "context": [],
                    "hit": False,
                    "non_empty": False,
                    "tolower_compliant": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    result = {
        "mode": args.mode,
        "revision": str(repo_root),
        "source": source,
        "provider": args.provider,
        "model": model_name,
        "case_count": len(rows),
        "summary": summarize(rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    args = parse_args()
    result = run_benchmark(args)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
