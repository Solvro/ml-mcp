"""Measure deterministic retrieval normalization against a real Neo4j instance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase

from benchmarks.run_text2cypher_normalization import load_cases, normalize_for_scoring, summarize
from src.text_normalization import (
    ensure_case_insensitive_fuzzy_matching,
    fold_diacritics,
    normalize_cypher_string_literals,
)

BENCHMARK_LABEL = "Text2CypherNormalizationBenchmark"


def escape_cypher_literal(value: str) -> str:
    """Escape a value for the controlled benchmark's single-quoted Cypher literal."""
    return value.replace("\\", "\\\\").replace("'", "\\'")


def build_query(search_value: str, mode: str) -> str:
    """Build and normalize the deliberately case-sensitive generated query."""
    query = (
        f"MATCH (n:{BENCHMARK_LABEL}) "
        f"WHERE n.title CONTAINS '{escape_cypher_literal(search_value)}' "
        "RETURN n.title AS title"
    )
    query = normalize_cypher_string_literals(query, normalizer=fold_diacritics)
    if mode == "full":
        query = ensure_case_insensitive_fuzzy_matching(query)
    return f"{query} LIMIT 10"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("diacritic-only", "full"), required=True)
    parser.add_argument("--neo4j-uri", default="bolt://localhost:17687")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-password", required=True)
    return parser.parse_args()


def run_matrix(args: argparse.Namespace) -> dict[str, Any]:
    """Seed seven entities and execute all 28 query variants on the same graph."""
    payload = json.loads(args.cases.resolve().read_text(encoding="utf-8"))
    source, cases = load_cases(args.cases.resolve())
    entities = payload["entities"]

    rows: list[dict[str, Any]] = []
    with GraphDatabase.driver(
        args.neo4j_uri,
        auth=(args.neo4j_user, args.neo4j_password),
    ) as driver:
        driver.execute_query(
            f"MATCH (n:{BENCHMARK_LABEL}) DETACH DELETE n",
            database_="neo4j",
        )
        driver.execute_query(
            f"UNWIND $entities AS entity CREATE (n:{BENCHMARK_LABEL}:Faculty {{"
            "title: entity.stored, context: entity.original})",
            entities=entities,
            database_="neo4j",
        )

        for case in cases:
            query = build_query(case["search_value"], args.mode)
            records, _, _ = driver.execute_query(query, database_="neo4j")
            context = [record.data() for record in records]
            context_text = normalize_for_scoring(json.dumps(context, ensure_ascii=False))
            expected_text = normalize_for_scoring(case["expected"])
            rows.append(
                {
                    **case,
                    "executed_cypher": query,
                    "context": context,
                    "hit": expected_text in context_text,
                    "non_empty": bool(context),
                    "tolower_compliant": query.lower().count("tolower(") >= 2,
                    "error": None,
                }
            )

    result = {
        "mode": args.mode,
        "query_source": "deterministic template (LLM-independent normalization isolation)",
        "source": source,
        "case_count": len(rows),
        "summary": summarize(rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    result = run_matrix(parse_args())
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
