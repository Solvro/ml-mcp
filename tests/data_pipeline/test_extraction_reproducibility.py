import re
from collections import Counter
from unittest.mock import MagicMock

from src.data_pipeline.flows import llm_cypher_generation as cypher_module
from tests.data_pipeline.test_completeness import COMPETENCY_PAGE

LABEL_RE = re.compile(r"\(\s*[A-Za-z_]\w*\s*:\s*(?P<label>[A-Za-z_]\w*)")
REL_TYPE_RE = re.compile(r"\[:(?P<relationship_type>[A-Z_]+)\]")


def node(variable: str, label: str, title: str, context: str = "w") -> str:
    return f"MERGE ({variable}:{label} {{title: '{title}', context: '{context}'}})"


def edge(source: str, relationship_type: str, target: str) -> str:
    return f"MERGE ({source})-[:{relationship_type}]->({target})"


ITEMS = [
    "Prowadzi badania naukowe pod nadzorem opiekuna naukowego",
    "Zna metody badawcze stosowane w swojej dyscyplinie",
    "Publikuje wyniki swoich badan w czasopismach naukowych",
    "Publikacje o zasiegu krajowym",
    "Publikacje o zasiegu miedzynarodowym",
    "W grupie pracownikow dydaktycznych prowadzi zajecia pod opieka nauczyciela akademickiego",
]
NODES = [
    node("cat", "CompetencyCategory", "R1 - Naukowiec poczatkujacy", "R1"),
    *[node(f"n{index}", "Competency", title) for index, title in enumerate(ITEMS, start=2)],
]


def _shape_signature(cypher: str) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    labels = Counter(match.group("label") for match in LABEL_RE.finditer(cypher))
    types = Counter(match.group("relationship_type") for match in REL_TYPE_RE.finditer(cypher))
    return sorted(labels.items()), sorted(types.items())


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
    variants = [
        [
            edge("cat", "HAS_CRITERION", "n2"),
            edge("cat", "HAS_CRITERION", "n3"),
            *[edge("cat", "RECOMMENDS", f"n{index}") for index in (4, 5, 6, 7)],
        ],
        [
            edge("n2", "REQUIRES", "cat"),
            edge("n3", "RELATED_TO", "cat"),
            *[edge("cat", "HAS_CRITERION", f"n{index}") for index in (4, 5, 6, 7)],
        ],
        [
            edge("cat", "REQUIRES", "n2"),
            edge("cat", "REQUIRES", "n3"),
            edge("cat", "HAS_CRITERION", "n4"),
            edge("n5", "HAS_CRITERION", "cat"),
            edge("cat", "RELATED_TO", "n6"),
            edge("cat", "HAS_CRITERION", "n7"),
        ],
    ]

    signatures = [_shape_signature(_run_pipeline(monkeypatch, NODES + edges)) for edges in variants]
    expected = ([("Competency", 6), ("CompetencyCategory", 1)], [("RECOMMENDS", 6)])

    assert signatures == [expected] * len(variants)
