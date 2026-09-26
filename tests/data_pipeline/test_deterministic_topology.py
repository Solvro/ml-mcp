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
        edge("cat", "HAS_CRITERION", "item"),
    ]

    stabilized, _ = stabilize_category_item_edges(RECOMMENDS_PAGE, statements)

    assert stabilized[0].startswith("MERGE (cat:CompetencyCategory")
    assert stabilized[2] == edge("cat", "RECOMMENDS", "item")


def test_topic_pointing_at_a_category_is_a_group_heading_and_stays() -> None:
    page = "W grupie pracownikow badawczych:\nDorobek naukowy:\n1) Publikacje w czasopismach\n"
    statements = [
        node("g", "Topic", "W grupie pracownikow badawczych"),
        node("cc", "CriterionCategory", "Dorobek naukowy"),
        node("c", "Criterion", "Publikacje w czasopismach"),
        edge("g", "HAS_CRITERION", "cc"),
        edge("cc", "HAS_CRITERION", "c"),
    ]

    assert stabilize_category_item_edges(page, statements) == (statements, [])


def test_prose_qualifiers_never_retype_a_criterion_edge() -> None:
    page = (
        "Kandydat składa wymagane dokumenty w terminie wskazanym w ogłoszeniu.\n\n"
        "Kryteria oceny:\n1) Publikacje w czasopismach\n"
    )
    statements = [
        node("cc", "CriterionCategory", "Kryteria oceny"),
        node("found", "Criterion", "Publikacje w czasopismach"),
        node("absent", "Criterion", "Granty zagraniczne"),
        edge("cc", "HAS_CRITERION", "found"),
        edge("cc", "HAS_CRITERION", "absent"),
    ]

    assert stabilize_category_item_edges(page, statements) == (statements, [])


def test_a_qualifier_in_a_sentence_does_not_override_the_heading_above() -> None:
    page = (
        "Kompetencje pożądane dla naukowców (R1-R4)\n\n"
        "Kandydat musi spełniać wymagane kryteria formalne.\n\n"
        "• prowadzi badania\n"
    )
    statements = [CATEGORY, ITEM, edge("cat", "HAS_CRITERION", "item")]

    stabilized, _ = stabilize_category_item_edges(page, statements)

    assert stabilized[2] == edge("cat", "RECOMMENDS", "item")


def test_wrapped_prose_qualifier_line_is_not_treated_as_heading() -> None:
    page = (
        "Kandydat sklada wymagane\n"
        "dokumenty w terminie.\n\n"
        "R1 - Naukowiec poczatkujacy\n"
        "1) Prowadzi badania\n"
    )
    statements = [CATEGORY, ITEM, edge("cat", "HAS_CRITERION", "item")]

    assert stabilize_category_item_edges(page, statements) == (statements, [])


def test_an_edge_of_any_type_between_a_pair_is_rewritten_not_doubled() -> None:
    statements = [CATEGORY, ITEM, edge("cat", "HAS_SUBCOMPETENCY", "item")]

    stabilized, rewrites = stabilize_category_item_edges(RECOMMENDS_PAGE, statements)

    assert stabilized == [CATEGORY, ITEM, edge("cat", "RECOMMENDS", "item")]
    assert len(rewrites) == 1


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
        (
            RECOMMENDS_PAGE,
            [CATEGORY, node("sub", "CompetencyCategory", "R2"), edge("cat", "REQUIRES", "sub")],
        ),
        (
            RECOMMENDS_PAGE,
            [CATEGORY, node("item", "Topic", "Prowadzi badania"), edge("item", "REQUIRES", "cat")],
        ),
    ],
    ids=[
        "no_qualifier_and_not_a_criterion_category",
        "page_states_both_qualifiers",
        "category_to_category",
        "topic_pointing_at_a_category",
    ],
)
def test_edges_the_pass_cannot_decide_are_left_alone(page: str, statements: list[str]) -> None:
    assert stabilize_category_item_edges(page, statements) == (statements, [])
