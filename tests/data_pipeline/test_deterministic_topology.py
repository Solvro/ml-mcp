from types import SimpleNamespace

from src.data_pipeline import deterministic_topology as topology_module
from src.data_pipeline.deterministic_topology import (
    enforce_deterministic_category_item_topology,
    relationship_type_for_heading,
)


def test_heading_qualifier_picks_relationship_type_from_tokens() -> None:
    assert relationship_type_for_heading("Kompetencje pozadanych naukowcow R2") == "RECOMMENDS"
    assert relationship_type_for_heading("Kompetencje wymaganych naukowcow R2") == "REQUIRES"
    assert relationship_type_for_heading("Kompetencje niezbednych naukowcow R2") == "REQUIRES"
    assert relationship_type_for_heading("Kompetencje niepozadane dla naukowca R2") is None
    assert relationship_type_for_heading("Kompetencje nie pozadane dla naukowca R2") is None
    assert relationship_type_for_heading("Kryteria dla naukowca R2") is None


def test_heading_qualifier_uses_rules_from_config(monkeypatch) -> None:
    custom_schema = SimpleNamespace(
        relationship_qualifier_rules=[
            SimpleNamespace(
                relationship_type="RECOMMENDS",
                token_stems=["autorskie"],
                words=["autorskie"],
            )
        ]
    )
    monkeypatch.setattr(
        topology_module, "get_config", lambda: SimpleNamespace(graph_schema=custom_schema)
    )

    assert relationship_type_for_heading("Kompetencje autorskie dla naukowca R2") == "RECOMMENDS"
    assert relationship_type_for_heading("Kompetencje pozadane dla naukowca R2") is None


def test_section_rewrites_every_controlled_edge_to_recommends() -> None:
    page = """Kompetencje pozadane dla naukowca R1:
1) Item one
2) Item two
3) Item three
4) Item four
5) Item five
6) Item six
"""
    statements = [
        "MERGE (cat:CompetencyCategory {title: 'Kompetencje pozadane dla naukowca R1', "
        "context: 'R1'})",
        "MERGE (i1:Competency {title: 'Item one', context: 'w'})",
        "MERGE (i2:Competency {title: 'Item two', context: 'w'})",
        "MERGE (i3:Competency {title: 'Item three', context: 'w'})",
        "MERGE (i4:Competency {title: 'Item four', context: 'w'})",
        "MERGE (i5:Competency {title: 'Item five', context: 'w'})",
        "MERGE (i6:Competency {title: 'Item six', context: 'w'})",
        "MERGE (cat)-[:RECOMMENDS]->(i4)",
        "MERGE (cat)-[:RECOMMENDS]->(i5)",
        "MERGE (cat)-[:RECOMMENDS]->(i6)",
        "MERGE (cat)-[:HAS_CRITERION]->(i1)",
        "MERGE (cat)-[:HAS_CRITERION]->(i2)",
        "MERGE (cat)-[:HAS_CRITERION]->(i3)",
    ]

    rewritten, report = enforce_deterministic_category_item_topology(page, statements)
    merged = "|".join(rewritten)

    assert "[:HAS_CRITERION]" not in merged
    assert merged.count("[:RECOMMENDS]") == 6
    assert report.matched_sections == 1
    assert report.rewritten_relationships == 6
    assert report.added_relationships == 0


def test_relationship_statement_that_creates_item_node_is_rewritten_in_place() -> None:
    page = """Kompetencje pozadane dla naukowca R2:
1) Item one
"""
    statements = [
        "MERGE (cat:CompetencyCategory {title: 'Kompetencje pozadane dla naukowca R2', "
        "context: 'R2'})",
        "MERGE (cat)-[:HAS_CRITERION]->(i1:Competency {title: 'Item one', context: 'Wiersz'})",
    ]

    rewritten, report = enforce_deterministic_category_item_topology(page, statements)

    assert rewritten[1].startswith("MERGE (cat)-[:RECOMMENDS]->(i1:Competency")
    assert report.rewritten_relationships == 1
    assert report.added_relationships == 0


def test_r2_section_is_not_matched_to_r3_category_when_r2_is_missing() -> None:
    page = """Kompetencje pozadane dla naukowca R2:
1) Item one
2) Item two
"""
    statements = [
        "MERGE (c3:CompetencyCategory {title: 'Kompetencje pozadane dla naukowca R3', "
        "context: 'R3'})",
        "MERGE (i1:Competency {title: 'Item one', context: 'w'})",
        "MERGE (i2:Competency {title: 'Item two', context: 'w'})",
        "MERGE (c3)-[:HAS_CRITERION]->(i1)",
    ]

    rewritten, report = enforce_deterministic_category_item_topology(page, statements)

    assert rewritten == statements
    assert report.changed is False


def test_topic_items_are_not_selected_as_section_rows() -> None:
    page = """Criteria:
1) Item one
"""
    statements = [
        "MERGE (cat:CriterionCategory {title: 'Criteria', context: 'Section'})",
        "MERGE (topic:Topic {title: 'Item one', context: 'Other section'})",
        "MERGE (cat)-[:HAS_CRITERION]->(topic)",
    ]

    rewritten, report = enforce_deterministic_category_item_topology(page, statements)

    assert rewritten == statements
    assert report.changed is False


def test_non_criterion_category_without_qualifier_is_left_untouched() -> None:
    page = """Calendar:
1) 1 XI 2026 r. - Day off
"""
    statements = [
        "MERGE (cat:CompetencyCategory {title: 'Calendar', context: 'Section'})",
        "MERGE (row:DayOff {title: '1 XI 2026 r.', context: 'Day off'})",
        "MERGE (cat)-[:HAS_DAY_OFF]->(row)",
    ]

    rewritten, report = enforce_deterministic_category_item_topology(page, statements)

    assert rewritten == statements
    assert report.changed is False


def test_parenthesized_titles_in_nodes_are_rewritten_without_duplicate_edges() -> None:
    page = """Criteria:
1) National grants (NCN)
"""
    statements = [
        "MERGE (c:CriterionCategory {title: 'Criteria', context: 'Section'})",
        "MERGE (c)<-[:REQUIRES]-(g:Criterion {title: 'National grants (NCN)', context: 'Row'})",
    ]

    rewritten, report = enforce_deterministic_category_item_topology(page, statements)
    merged = "|".join(rewritten)

    assert "[:REQUIRES]" not in merged
    assert merged.count("[:HAS_CRITERION]") == 1
    assert report.rewritten_relationships == 1
    assert report.added_relationships == 0


def test_page_level_qualifier_is_inherited_by_all_r1_to_r4_stages() -> None:
    page = """Competency framework
Kompetencje pozadane dla naukowcow na kolejnych etapach kariery (R1-R4)
R1 - Naukowiec poczatkujacy
1) R1 kompetencja
R2 - Naukowiec uznany
1) R2 kompetencja
R3 - Doswiadczony naukowiec
1) R3 kompetencja
R4 - Wiodacy naukowiec
1) R4 kompetencja
"""
    statements = [
        "MERGE (c1:CompetencyCategory {title: 'R1 - Naukowiec poczatkujacy', context: 'R1'})",
        "MERGE (c2:CompetencyCategory {title: 'R2 - Naukowiec uznany', context: 'R2'})",
        "MERGE (c3:CompetencyCategory {title: 'R3 - Doswiadczony naukowiec', context: 'R3'})",
        "MERGE (c4:CompetencyCategory {title: 'R4 - Wiodacy naukowiec', context: 'R4'})",
        "MERGE (i1:Competency {title: 'R1 kompetencja', context: 'w'})",
        "MERGE (i2:Competency {title: 'R2 kompetencja', context: 'w'})",
        "MERGE (i3:Competency {title: 'R3 kompetencja', context: 'w'})",
        "MERGE (i4:Competency {title: 'R4 kompetencja', context: 'w'})",
        "MERGE (c1)-[:HAS_CRITERION]->(i1)",
        "MERGE (c2)-[:HAS_CRITERION]->(i2)",
        "MERGE (c3)-[:HAS_CRITERION]->(i3)",
        "MERGE (c4)-[:HAS_CRITERION]->(i4)",
    ]

    rewritten, _ = enforce_deterministic_category_item_topology(page, statements)
    merged = "|".join(rewritten)

    assert "[:HAS_CRITERION]" not in merged
    assert merged.count("[:RECOMMENDS]") == 4


def test_paragraphs_between_stages_keep_recommends_for_all_stages() -> None:
    page = """Kompetencje pozadane dla naukowcow na kolejnych etapach kariery (R1-R3)
R1 - Naukowiec poczatkujacy
1) R1 kompetencja

Annual review happens once a year.

R2 - Naukowiec uznany
1) R2 kompetencja

Annual review happens once a year.

R3 - Doswiadczony naukowiec
1) R3 kompetencja
"""
    statements = [
        "MERGE (c1:CriterionCategory {title: 'R1 - Naukowiec poczatkujacy', context: 'R1'})",
        "MERGE (c2:CriterionCategory {title: 'R2 - Naukowiec uznany', context: 'R2'})",
        "MERGE (c3:CriterionCategory {title: 'R3 - Doswiadczony naukowiec', context: 'R3'})",
        "MERGE (i1:Criterion {title: 'R1 kompetencja', context: 'w'})",
        "MERGE (i2:Criterion {title: 'R2 kompetencja', context: 'w'})",
        "MERGE (i3:Criterion {title: 'R3 kompetencja', context: 'w'})",
        "MERGE (c1)-[:HAS_CRITERION]->(i1)",
        "MERGE (c2)-[:HAS_CRITERION]->(i2)",
        "MERGE (c3)-[:HAS_CRITERION]->(i3)",
    ]

    rewritten, _ = enforce_deterministic_category_item_topology(page, statements)
    merged = "|".join(rewritten)

    assert "[:HAS_CRITERION]" not in merged
    assert merged.count("[:RECOMMENDS]") == 3
