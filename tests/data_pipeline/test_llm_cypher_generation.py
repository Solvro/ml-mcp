from unittest.mock import MagicMock

from src.data_pipeline.flows import llm_cypher_generation as cypher_module


def test_generated_write_literals_are_diacritic_folded(monkeypatch) -> None:
    class FakePipe:
        def run(self, context: str, schema_context: str = "") -> list[str]:
            return [
                "MERGE (n:Faculty {title: 'Wydział Łączności', "
                "context: 'Znajduje się we Wrocławiu'})"
            ]

    monkeypatch.setattr(cypher_module, "LLMPipe", FakePipe)
    monkeypatch.setattr(cypher_module, "get_run_logger", MagicMock)

    result = cypher_module.generate_cypher_queries.fn("source text")

    # Since issue #53 the node also merges on a canonical key rather than on title + context.
    assert result.startswith("MERGE (n:Faculty {key: 'wydzial lacznosci'})")
    assert "n.title = 'Wydzial Lacznosci'" in result
    assert "n.context = 'Znajduje sie we Wroclawiu'" in result
    assert "ł" not in result
    assert "ę" not in result


# Review of PR #84: one page of seven failed with "Variable `node13` already declared", from a
# part that was a bare MERGE (node13) next to the node13 the page had already bound. A page's
# statements run as one query, so that one part cost every row on the page.
def test_a_statement_re_declaring_a_bound_variable_is_dropped(monkeypatch) -> None:
    class FakePipe:
        def run(self, context: str, schema_context: str = "") -> list[str]:
            return [
                "MERGE (node13:CriterionCategory {title: 'Inne wazne osiagniecia', "
                "context: 'Kategoria'})",
                "MERGE (node13)",
                "MERGE (node14:Criterion {title: 'Patenty i wdrozenia', context: 'Rodzaj'})",
            ]

        def run_missing_rows(self, context: str, rows: list[str]) -> list[str]:
            return []

    monkeypatch.setattr(cypher_module, "LLMPipe", FakePipe)
    monkeypatch.setattr(cypher_module, "get_run_logger", MagicMock)

    result = cypher_module.generate_cypher_queries.fn("Kryteria oceny.\n")

    assert result.count("MERGE") == 2
    assert "'inne wazne osiagniecia'" in result
    assert "'patenty i wdrozenia'" in result


def test_a_bare_merge_nothing_else_binds_is_left_alone() -> None:
    """Not this pass's call. With no label and no properties it matches every node in the graph,
    so the ingestion guardrail refuses it and fails the page (#93)."""
    parts = ["MERGE (node13)", "MERGE (node13)-[:HAS_CRITERION]->(node14)"]

    assert cypher_module._drop_redeclarations(parts, MagicMock()) == parts
