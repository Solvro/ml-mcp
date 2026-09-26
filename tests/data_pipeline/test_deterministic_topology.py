import pytest

from src.data_pipeline.deterministic_topology import (
    page_relationship_type,
    stabilize_category_item_edges,
)


def node(variable: str, label: str, title: str, context: str = "w") -> str:
    return f"MERGE ({variable}:{label} {{title: '{title}', context: '{context}'}})"


def edge(source: str, relationship_type: str, target: str) -> str:
    return f"MERGE ({source})-[:{relationship_type}]->({target})"


RECOMMENDS_PAGE = "Kompetencje pożądane dla naukowców (R1-R4)\n• prowadzi badania\n"
CATEGORY = node("cat", "CompetencyCategory", "R1 - Naukowiec poczatkujacy", "R1")
ITEM = node("item", "Competency", "Prowadzi badania")


@pytest.mark.parametrize(
    ("page", "expected"),
    [
        ("Kompetencje pożądane dla naukowców", "RECOMMENDS"),
        ("Wykaz kompetencji wymaganych", "REQUIRES"),
        ("Kompetencje niezbędne", "REQUIRES"),
        ("Działania niepożądane", None),
        ("Działania nie pożądane", None),
        ("Kompetencje pożądane i kompetencje niezbędne", None),
        ("Kryteria oceny", None),
    ],
    ids=[
        "recommends",
        "requires",
        "requires_other_word",
        "negated",
        "negated_apart",
        "both",
        "none",
    ],
)
def test_page_qualifier_names_one_type_or_none(page: str, expected: str | None) -> None:
    assert page_relationship_type(page) == expected


@pytest.mark.parametrize(
    ("drifted", "expected"),
    [
        (edge("cat", "HAS_CRITERION", "item"), edge("cat", "RECOMMENDS", "item")),
        (edge("item", "REQUIRES", "cat"), "MERGE (item)<-[:RECOMMENDS]-(cat)"),
        ("MERGE (cat)<-[:RELATED_TO]-(item)", edge("cat", "RECOMMENDS", "item")),
    ],
    ids=["wrong_type", "wrong_type_and_direction", "fallback_type_incoming"],
)
def test_category_item_edge_gets_the_page_type_from_the_category(
    drifted: str, expected: str
) -> None:
    """The pattern keeps its node order; only the arrow and the type change."""
    stabilized, rewrites = stabilize_category_item_edges(RECOMMENDS_PAGE, [CATEGORY, ITEM, drifted])

    assert stabilized[2] == expected
    assert len(rewrites) == 1


def test_an_edge_already_in_shape_is_not_reported() -> None:
    statements = [CATEGORY, ITEM, edge("cat", "RECOMMENDS", "item")]

    assert stabilize_category_item_edges(RECOMMENDS_PAGE, statements) == (statements, [])


def test_a_statement_that_binds_the_item_keeps_it_and_its_literals() -> None:
    statement = (
        "MERGE (cat)<-[r:REQUIRES {source: 'str. (2)'}]-"
        "(item:Competency {title: 'Granty (NCN)', context: 'R1'})"
    )

    stabilized, _ = stabilize_category_item_edges(RECOMMENDS_PAGE, [CATEGORY, statement])

    assert stabilized[1] == (
        "MERGE (cat)-[r:RECOMMENDS {source: 'str. (2)'}]->"
        "(item:Competency {title: 'Granty (NCN)', context: 'R1'})"
    )


def test_mixed_page_uses_nearest_qualifier_before_each_item_title() -> None:
    page = (
        "Kompetencje pożądane dla naukowca R2:\n"
        "1) Item one\n\n"
        "Kompetencje wymagane dla naukowca R2:\n"
        "1) Item two\n"
    )
    statements = [
        node("cat", "CompetencyCategory", "Kompetencje dla naukowca R2", "R2"),
        node("i1", "Competency", "Item one"),
        node("i2", "Competency", "Item two"),
        edge("cat", "HAS_CRITERION", "i1"),
        edge("cat", "HAS_CRITERION", "i2"),
    ]

    stabilized, _ = stabilize_category_item_edges(page, statements)

    assert stabilized[3] == edge("cat", "RECOMMENDS", "i1")
    assert stabilized[4] == edge("cat", "REQUIRES", "i2")


def test_topic_category_is_retyped_from_item_pair_and_rewritten() -> None:
    statements = [
        node("cat", "Topic", "R1 - Naukowiec poczatkujacy", "R1"),
        node("item", "Competency", "Prowadzi badania"),
        edge("item", "HAS_CRITERION", "cat"),
    ]

    stabilized, _ = stabilize_category_item_edges(RECOMMENDS_PAGE, statements)

    assert stabilized[0].startswith("MERGE (cat:CompetencyCategory")
    assert stabilized[2] == "MERGE (item)<-[:RECOMMENDS]-(cat)"


def test_topic_item_under_criterion_category_is_retyped_to_criterion() -> None:
    statements = [
        node("cat", "CriterionCategory", "Kryteria oceny"),
        node("item", "Topic", "Publikacje"),
        edge("cat", "REQUIRES", "item"),
    ]

    stabilized, _ = stabilize_category_item_edges("Kryteria oceny:\n1) Publikacje\n", statements)

    assert stabilized[1].startswith("MERGE (item:Criterion")
    assert stabilized[2] == edge("cat", "HAS_CRITERION", "item")


def test_topic_with_conflicting_pair_hints_stays_unchanged() -> None:
    statements = [
        node("topic", "Topic", "Wspolny wezel"),
        node("c", "Competency", "A"),
        node("k", "Criterion", "B"),
        edge("topic", "REQUIRES", "c"),
        edge("topic", "REQUIRES", "k"),
    ]

    assert stabilize_category_item_edges("Strona testowa\n", statements) == (statements, [])


def test_unlinked_item_is_attached_to_nearest_category_above() -> None:
    page = (
        "Kompetencje pożądane dla naukowca R1\nR1 - Naukowiec poczatkujacy\n1) Prowadzi badania\n"
    )
    statements = [
        node("r1", "CompetencyCategory", "R1 - Naukowiec poczatkujacy", "R1"),
        node("item", "Competency", "Prowadzi badania"),
    ]

    stabilized, rewrites = stabilize_category_item_edges(page, statements)

    assert stabilized[-1] == edge("r1", "RECOMMENDS", "item")
    assert "added MERGE (r1)-[:RECOMMENDS]->(item)" in rewrites


def test_unlinked_item_whose_title_is_absent_is_left_without_edge() -> None:
    page = (
        "Kompetencje pożądane dla naukowca R1\nR1 - Naukowiec poczatkujacy\n1) Prowadzi badania\n"
    )
    statements = [
        node("r1", "CompetencyCategory", "R1 - Naukowiec poczatkujacy", "R1"),
        node("item", "Competency", "Publikacje o zasiegu krajowym"),
    ]

    assert stabilize_category_item_edges(page, statements) == (statements, [])


def test_every_hop_of_a_chain_is_read() -> None:
    chain = "MERGE (doc)-[:DEFINED_IN]->(cat)-[:HAS_CRITERION]->(item)"
    statements = [node("doc", "Document", "PRK"), CATEGORY, ITEM, chain]

    stabilized, _ = stabilize_category_item_edges(RECOMMENDS_PAGE, statements)

    assert stabilized[3] == "MERGE (doc)-[:DEFINED_IN]->(cat)-[:RECOMMENDS]->(item)"


def test_a_criterion_category_falls_back_to_has_criterion_without_a_qualifier() -> None:
    statements = [
        node("cat", "CriterionCategory", "Kryteria oceny"),
        node("item", "Criterion", "Publikacje"),
        edge("item", "REQUIRES", "cat"),
    ]

    stabilized, _ = stabilize_category_item_edges("Kryteria oceny:\n1) Publikacje\n", statements)

    assert stabilized[2] == "MERGE (item)<-[:HAS_CRITERION]-(cat)"


@pytest.mark.parametrize(
    ("page", "statements"),
    [
        ("Kompetencje R1\n", [CATEGORY, ITEM, edge("cat", "REQUIRES", "item")]),
        (
            "Kompetencje pożądane i niezbędne\n",
            [CATEGORY, ITEM, edge("cat", "HAS_CRITERION", "item")],
        ),
        (RECOMMENDS_PAGE, [CATEGORY, ITEM, edge("cat", "DEFINED_IN", "item")]),
        (
            RECOMMENDS_PAGE,
            [CATEGORY, node("sub", "CompetencyCategory", "R2"), edge("cat", "REQUIRES", "sub")],
        ),
    ],
    ids=[
        "no_qualifier_and_not_a_criterion_category",
        "page_states_both_qualifiers",
        "type_outside_the_controlled_set",
        "category_to_category",
    ],
)
def test_edges_the_pass_cannot_decide_are_left_alone(page: str, statements: list[str]) -> None:
    assert stabilize_category_item_edges(page, statements) == (statements, [])
