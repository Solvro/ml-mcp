from unittest.mock import MagicMock

from src.mcp_server.tools.knowledge_graph.rag import RAG


def _retrieval_stub() -> RAG:
    rag = object.__new__(RAG)
    rag.max_results = 5
    rag.enable_debug = False
    rag.database = MagicMock()
    return rag


def test_cypher_prompt_payload_keeps_original_and_adds_normalized_question() -> None:
    payload = RAG._build_cypher_prompt_payload(
        "Gdzie jest Wydział Informatyki we Wrocławiu?",
        "(:Department {title: STRING})",
    )

    assert payload == {
        "user_question": "Gdzie jest Wydział Informatyki we Wrocławiu?",
        "normalized_question": "gdzie jest wydzial informatyki we wroclawiu?",
        "schema": "(:Department {title: STRING})",
    }


def test_retrieve_normalizes_only_string_values_before_database_query() -> None:
    rag = _retrieval_stub()
    rag.database.query.return_value = [{"title": "Wydzial Informatyki"}]
    state = {
        "generated_cypher": (
            "MATCH (wydział:Wydział) "
            "WHERE toLower(wydział.tytuł) CONTAINS toLower('WYDZIAŁ INFORMATYKI') "
            "RETURN wydział.tytuł"
        )
    }

    result = rag.retrieve(state)

    executed_query = (
        "MATCH (wydział:Wydział) "
        "WHERE toLower(wydział.tytuł) CONTAINS toLower('WYDZIAL INFORMATYKI') "
        "RETURN wydział.tytuł LIMIT 5"
    )
    rag.database.query.assert_called_once_with(executed_query)
    assert result == {
        "context": [{"title": "Wydzial Informatyki"}],
        "generated_cypher": executed_query,
    }


def test_retrieve_preserves_dynamic_property_key_case() -> None:
    rag = _retrieval_stub()
    rag.database.query.return_value = [{"title": "Wroclaw"}]

    result = rag.retrieve(
        {
            "generated_cypher": (
                "MATCH (n:Faculty) "
                "WHERE toLower(n['ExactTitle']) = toLower('WROCŁAW') "
                "RETURN n['ExactTitle']"
            )
        }
    )

    executed_query = (
        "MATCH (n:Faculty) "
        "WHERE toLower(n['ExactTitle']) = toLower('WROCLAW') "
        "RETURN n['ExactTitle'] LIMIT 5"
    )
    rag.database.query.assert_called_once_with(executed_query)
    assert result["generated_cypher"] == executed_query


def test_retrieve_preserves_case_for_stable_ids() -> None:
    rag = _retrieval_stub()
    rag.database.query.return_value = [{"id": "AbC-123"}]

    result = rag.retrieve(
        {"generated_cypher": ("MATCH (n:Faculty) WHERE n.id = 'AbC-123' RETURN n.id")}
    )

    executed_query = "MATCH (n:Faculty) WHERE n.id = 'AbC-123' RETURN n.id LIMIT 5"
    rag.database.query.assert_called_once_with(executed_query)
    assert result["generated_cypher"] == executed_query


def test_retrieve_enforces_case_insensitive_fuzzy_matching() -> None:
    rag = _retrieval_stub()
    rag.database.query.return_value = [{"title": "Wydzial Zarzadzania"}]

    result = rag.retrieve(
        {
            "generated_cypher": (
                "MATCH (n:Faculty) WHERE n.title CONTAINS 'WYDZIAŁ ZARZĄDZANIA' RETURN n.title"
            )
        }
    )

    executed_query = (
        "MATCH (n:Faculty) WHERE toLower(n.title) CONTAINS "
        "toLower('WYDZIAL ZARZADZANIA') RETURN n.title LIMIT 5"
    )
    rag.database.query.assert_called_once_with(executed_query)
    assert result["generated_cypher"] == executed_query


def test_retrieve_still_blocks_disallowed_call_after_normalization() -> None:
    rag = _retrieval_stub()

    result = rag.retrieve(
        {
            "generated_cypher": (
                "CALL db.index.fulltext.queryNodes('Wydziały', 'WROCŁAW') YIELD node RETURN node"
            )
        }
    )

    rag.database.query.assert_not_called()
    assert result["context"] == []
    assert result["generated_cypher"].startswith("Blocked unsafe Cypher:")
