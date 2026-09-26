from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

from prefect.logging import disable_run_logger

from src.data_pipeline.flows.llm_cypher_generation import generate_cypher_queries

LABEL_RE = re.compile(r"\(\s*[A-Za-z_]\w*\s*:\s*(?P<label>[A-Za-z_]\w*)")
REL_TYPE_RE = re.compile(r"\[:(?P<relationship_type>[A-Z_]+)\]")


def shape_signature(cypher: str) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    labels = Counter(match.group("label") for match in LABEL_RE.finditer(cypher))
    types = Counter(match.group("relationship_type") for match in REL_TYPE_RE.finditer(cypher))
    return sorted(labels.items()), sorted(types.items())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("page_file", type=Path)
    parser.add_argument("--runs", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.runs < 1:
        raise ValueError("--runs must be >= 1")

    page_text = args.page_file.read_text(encoding="utf-8")
    signatures: list[tuple[list[tuple[str, int]], list[tuple[str, int]]]] = []
    for _ in range(args.runs):
        with disable_run_logger():
            cypher = generate_cypher_queries.fn(page_text)
        signatures.append(shape_signature(cypher))

    print("run\tlabels\trelationships")
    for index, (labels, relationships) in enumerate(signatures, start=1):
        print(f"{index}\t{labels}\t{relationships}")

    unique = {repr(signature) for signature in signatures}
    if len(unique) != 1:
        print("reproducibility check failed: signatures differ across runs")
        return 1
    print("reproducibility check passed: all signatures match")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
