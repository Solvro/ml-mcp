from types import SimpleNamespace

import pytest

from src.data_pipeline import deterministic_topology as topology_module
from src.data_pipeline.deterministic_topology import (
    TopologyRewriteReport,
    enforce_deterministic_category_item_topology,
    relationship_type_for_heading,
)


def node(variable: str, label: str, title: str, context: str = "w") -> str:
    return f"MERGE ({variable}:{label} {{title: '{title}', context: '{context}'}})"


def edge(source: str, relationship_type: str, target: str) -> str:
    return f"MERGE ({source})-[:{relationship_type}]->({target})"


def _run(page: str, statements: list[str]) -> tuple[str, TopologyRewriteReport]:
    rewritten, report = enforce_deterministic_category_item_topology(page, statements)
    return "|".join(rewritten), report


def _stage_page(*, max_stage: int, with_paragraphs: bool = False) -> str:
    lines = ["Kompetencje pożądane dla naukowców na kolejnych etapach kariery (R1-R4)"]
    for stage in range(1, max_stage + 1):
        lines.extend([f"R{stage} - Naukowiec etap {stage}", f"1) R{stage} kompetencja"])
        if with_paragraphs and stage < max_stage:
            lines.extend(["", "Ocena kompetencji odbywa się raz w roku.", ""])
    return "\n".join(lines)


def _stage_statements(*, max_stage: int, category_label: str, item_label: str) -> list[str]:
    stages = range(1, max_stage + 1)
    return [
        *[node(f"c{s}", category_label, f"R{s} - Naukowiec etap {s}", f"R{s}") for s in stages],
        *[node(f"i{s}", item_label, f"R{s} kompetencja") for s in stages],
        *[edge(f"c{s}", "HAS_CRITERION", f"i{s}") for s in stages],
    ]


@pytest.mark.parametrize(
    ("heading", "expected"),
    [
        ("Kompetencje pożądanych naukowców R2", "RECOMMENDS"),
        ("Kompetencje wymaganych naukowców R2", "REQUIRES"),
        ("Kompetencje niezbędnych naukowców R2", "REQUIRES"),
        ("Kompetencje niepożądane dla naukowca R2", None),
        ("Kompetencje nie pożądane dla naukowca R2", None),
        ("Kryteria dla naukowca R2", None),
    ],
)
def test_heading_qualifier_picks_relationship_type_from_tokens(
    heading: str, expected: str | None
) -> None:
    assert relationship_type_for_heading(heading) == expected


def test_heading_qualifier_uses_rules_from_config(monkeypatch) -> None:
    rule = SimpleNamespace(
        relationship_type="RECOMMENDS", token_stems=["autorskie"], words=["autorskie"]
    )
    schema = SimpleNamespace(relationship_qualifier_rules=[rule])
    monkeypatch.setattr(topology_module, "get_config", lambda: SimpleNamespace(graph_schema=schema))

    assert relationship_type_for_heading("Kompetencje autorskie dla naukowca R2") == "RECOMMENDS"
    assert relationship_type_for_heading("Kompetencje pożądane dla naukowca R2") is None


def test_section_rewrites_every_controlled_edge_to_recommends() -> None:
    words = ("one", "two", "three", "four", "five", "six")
    page = "Kompetencje pożądane dla naukowca R1:\n" + "".join(
        f"{index}) Item {word}\n" for index, word in enumerate(words, start=1)
    )
    statements = [
        node("cat", "CompetencyCategory", "Kompetencje pozadane dla naukowca R1", "R1"),
        *[node(f"i{index}", "Competency", f"Item {word}") for index, word in enumerate(words, 1)],
        *[edge("cat", "RECOMMENDS", f"i{index}") for index in (4, 5, 6)],
        *[edge("cat", "HAS_CRITERION", f"i{index}") for index in (1, 2, 3)],
    ]

    merged, report = _run(page, statements)

    assert "[:HAS_CRITERION]" not in merged
    assert merged.count("[:RECOMMENDS]") == 6
    assert (report.matched_sections, report.rewritten_relationships) == (1, 6)
    assert report.added_relationships == 0


def test_relationship_statement_that_creates_item_node_is_rewritten_in_place() -> None:
    page = "Kompetencje pożądane dla naukowca R2:\n1) Item one\n"
    statements = [
        node("cat", "CompetencyCategory", "Kompetencje pozadane dla naukowca R2", "R2"),
        "MERGE (cat)-[:HAS_CRITERION]->(i1:Competency {title: 'Item one', context: 'Wiersz'})",
    ]

    merged, report = _run(page, statements)

    assert "MERGE (cat)-[:RECOMMENDS]->(i1:Competency" in merged
    assert (report.rewritten_relationships, report.added_relationships) == (1, 0)


@pytest.mark.parametrize(
    ("page", "statements"),
    [
        (
            "Kompetencje pożądane dla naukowca R2:\n1) Item one\n2) Item two\n",
            [
                node("c3", "CompetencyCategory", "Kompetencje pozadane dla naukowca R3", "R3"),
                node("i1", "Competency", "Item one"),
                node("i2", "Competency", "Item two"),
                edge("c3", "HAS_CRITERION", "i1"),
            ],
        ),
        (
            "Kryteria:\n1) Item one\n",
            [
                node("cat", "CriterionCategory", "Kryteria", "Sekcja"),
                node("topic", "Topic", "Item one", "Inna sekcja"),
                edge("cat", "HAS_CRITERION", "topic"),
            ],
        ),
        (
            "Kalendarz:\n1) 1 XI 2026 r. - dzień wolny\n",
            [
                node("cat", "CompetencyCategory", "Kalendarz", "Sekcja"),
                node("row", "DayOff", "1 XI 2026 r.", "dzien wolny"),
                edge("cat", "HAS_DAY_OFF", "row"),
            ],
        ),
    ],
    ids=[
        "r2_rows_never_join_the_r3_category",
        "topic_node_is_never_an_item",
        "no_qualifier_and_not_a_criterion_category",
    ],
)
def test_untouched_when_section_does_not_qualify(page: str, statements: list[str]) -> None:
    rewritten, report = enforce_deterministic_category_item_topology(page, statements)

    assert rewritten == statements
    assert report.changed is False


def test_parenthesized_titles_in_nodes_are_rewritten_without_duplicate_edges() -> None:
    page = "Kryteria:\n1) Granty krajowe (NCN)\n"
    statements = [
        node("c", "CriterionCategory", "Kryteria", "Sekcja"),
        "MERGE (c)<-[:REQUIRES]-(g:Criterion {title: 'Granty krajowe (NCN)', context: 'Wiersz'})",
    ]

    merged, report = _run(page, statements)

    assert "[:REQUIRES]" not in merged
    assert merged.count("[:HAS_CRITERION]") == 1
    assert (report.rewritten_relationships, report.added_relationships) == (1, 0)


@pytest.mark.parametrize(
    ("max_stage", "with_paragraphs", "category_label", "item_label"),
    [
        (4, False, "CompetencyCategory", "Competency"),
        (3, True, "CriterionCategory", "Criterion"),
    ],
    ids=["r4_keeps_the_page_title_qualifier", "paragraphs_between_stages_keep_it_too"],
)
def test_stage_pages_inherit_recommends_from_parent_qualifier(
    max_stage: int, with_paragraphs: bool, category_label: str, item_label: str
) -> None:
    page = _stage_page(max_stage=max_stage, with_paragraphs=with_paragraphs)
    statements = _stage_statements(
        max_stage=max_stage, category_label=category_label, item_label=item_label
    )

    merged, _ = _run(page, statements)

    assert "[:HAS_CRITERION]" not in merged
    assert merged.count("[:RECOMMENDS]") == max_stage
