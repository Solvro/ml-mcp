"""Regression test for deterministic graph shape on repeated extraction runs (#104)."""

import re
from collections import Counter
from typing import NamedTuple
from unittest.mock import MagicMock

from src.data_pipeline.flows import llm_cypher_generation as cypher_module
from tests.data_pipeline.test_completeness import COMPETENCY_PAGE

LABEL_RE = re.compile(r"\(\s*[A-Za-z_]\w*\s*:\s*(?P<label>[A-Za-z_]\w*)")
REL_TYPE_RE = re.compile(r"\[:(?P<relationship_type>[A-Z_]+)\]")


class GraphShapeSignature(NamedTuple):
    labels: tuple[tuple[str, int], ...]
    relationship_types: tuple[tuple[str, int], ...]


def _shape_signature(cypher: str) -> GraphShapeSignature:
    labels = Counter(match.group("label") for match in LABEL_RE.finditer(cypher))
    relationship_types = Counter(
        match.group("relationship_type") for match in REL_TYPE_RE.finditer(cypher)
    )
    return GraphShapeSignature(
        labels=tuple(sorted(labels.items())),
        relationship_types=tuple(sorted(relationship_types.items())),
    )


def _run_pipeline(monkeypatch, parts: list[str]) -> str:
    class FakePipe:
        def run(self, context: str, schema_context: str = "") -> list[str]:
            return list(parts)

        def run_missing_rows(self, context: str, rows: list[str]) -> list[str]:
            return []

    monkeypatch.setattr(cypher_module, "LLMPipe", FakePipe)
    monkeypatch.setattr(cypher_module, "get_run_logger", MagicMock)
    return cypher_module.generate_cypher_queries.fn(COMPETENCY_PAGE)


def test_same_page_yields_same_shape_across_drifted_model_outputs(monkeypatch) -> None:
    common_nodes = [
        "MERGE (cat:CompetencyCategory {title: 'R1 - Naukowiec poczatkujacy', context: 'R1'})",
        "MERGE (n2:Competency {title: 'Prowadzi badania naukowe pod nadzorem opiekuna "
        "naukowego', context: 'w'})",
        "MERGE (n3:Competency {title: 'Zna metody badawcze stosowane w swojej dyscyplinie', "
        "context: 'w'})",
        "MERGE (n4:Competency {title: 'Publikuje wyniki swoich badan w czasopismach "
        "naukowych', context: 'w'})",
        "MERGE (n5:Competency {title: 'Publikacje o zasiegu krajowym', context: 'w'})",
        "MERGE (n6:Competency {title: 'Publikacje o zasiegu miedzynarodowym', context: 'w'})",
        "MERGE (n7:Competency {title: 'W grupie pracownikow dydaktycznych prowadzi zajecia "
        "pod opieka nauczyciela akademickiego', context: 'w'})",
    ]
    variants = [
        common_nodes
        + [
            "MERGE (cat)-[:HAS_CRITERION]->(n2)",
            "MERGE (cat)-[:HAS_CRITERION]->(n3)",
            "MERGE (cat)-[:RECOMMENDS]->(n4)",
            "MERGE (cat)-[:RECOMMENDS]->(n5)",
            "MERGE (cat)-[:RECOMMENDS]->(n6)",
            "MERGE (cat)-[:RECOMMENDS]->(n7)",
        ],
        common_nodes
        + [
            "MERGE (n2)-[:REQUIRES]->(cat)",
            "MERGE (n3)-[:RELATED_TO]->(cat)",
            "MERGE (cat)-[:HAS_CRITERION]->(n4)",
            "MERGE (cat)-[:HAS_CRITERION]->(n5)",
            "MERGE (cat)-[:HAS_CRITERION]->(n6)",
            "MERGE (cat)-[:HAS_CRITERION]->(n7)",
        ],
        common_nodes
        + [
            "MERGE (cat)-[:REQUIRES]->(n2)",
            "MERGE (cat)-[:REQUIRES]->(n3)",
            "MERGE (cat)-[:HAS_CRITERION]->(n4)",
            "MERGE (n5)-[:HAS_CRITERION]->(cat)",
            "MERGE (cat)-[:RELATED_TO]->(n6)",
            "MERGE (cat)-[:HAS_CRITERION]->(n7)",
        ],
    ]

    signatures = [_shape_signature(_run_pipeline(monkeypatch, parts)) for parts in variants]

    assert len(set(signatures)) == 1
    assert signatures[0].labels == (("Competency", 6), ("CompetencyCategory", 1))
    assert signatures[0].relationship_types == (("RECOMMENDS", 6),)
