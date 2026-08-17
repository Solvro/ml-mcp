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

    assert result == (
        "MERGE (n:Faculty {title: 'Wydzial Lacznosci', context: 'Znajduje sie we Wroclawiu'})"
    )
