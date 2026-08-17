"""Seed the normalization benchmark graph through the repository ingestion prompt."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def clean_generated_cypher(raw: str) -> str:
    """Turn the ingestion prompt's pipe-separated output into one Cypher query."""
    value = raw.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines.pop()
        value = "\n".join(lines).strip()

    parts = [part.strip() for part in value.split("|") if part.strip()]
    if not parts or any(not part.upper().startswith("MERGE") for part in parts):
        raise ValueError("Ingestion model returned non-MERGE Cypher output")
    return "\n".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--neo4j-uri", default="bolt://localhost:17687")
    parser.add_argument("--provider", choices=("openai", "clarin"), default="openai")
    return parser.parse_args()


def seed_graph(args: argparse.Namespace) -> dict[str, object]:
    """Generate graph writes with the selected revision's ingestion configuration."""
    repo_root = args.repo_root.resolve()
    load_dotenv(args.env_file.resolve(), override=False)
    os.environ["NEO4J_URI"] = args.neo4j_uri
    sys.path.insert(0, str(repo_root))
    os.chdir(repo_root)

    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import PromptTemplate
    from langchain_openai import ChatOpenAI
    from langchain_openai.chat_models.base import BaseChatOpenAI
    from neo4j import GraphDatabase
    from pydantic import SecretStr

    from src.config.config import get_config

    config = get_config()
    if args.provider == "clarin":
        model_name = config.llm.clarin.name
        model = ChatOpenAI(
            model_name=model_name,
            base_url=config.llm.clarin.base_url,
            api_key=os.environ["CLARIN_API_KEY"],
            temperature=0,
        )
    else:
        model_name = config.llm.accurate_model.name
        model = BaseChatOpenAI(
            model=model_name,
            api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
            temperature=config.llm.accurate_model.temperature,
        )
    prompt = PromptTemplate(
        input_variables=["context", "schema_context"],
        template=config.prompts.cypher_insert,
    )
    fixture = args.fixture.resolve().read_text(encoding="utf-8")
    raw = (prompt | model | StrOutputParser()).invoke(
        {"context": fixture, "schema_context": "(empty — first pass)"}
    )
    cypher = clean_generated_cypher(raw)

    username = os.environ["NEO4J_USER"]
    password = os.environ["NEO4J_PASSWORD"]
    with GraphDatabase.driver(args.neo4j_uri, auth=(username, password)) as driver:
        driver.execute_query(cypher, database_="neo4j")
        records, _, _ = driver.execute_query(
            "MATCH (n) RETURN labels(n) AS labels, n.title AS title, "
            "n.context AS context ORDER BY n.title",
            database_="neo4j",
        )

    result = {
        "revision": str(repo_root),
        "provider": args.provider,
        "model": model_name,
        "temperature": 0,
        "generated_cypher": cypher,
        "nodes": [record.data() for record in records],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    result = seed_graph(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
