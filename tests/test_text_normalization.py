import pytest

from src.text_normalization import (
    fold_diacritics,
    normalize_cypher_string_literals,
    normalize_search_text,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Zażółć gęślą jaźń", "Zazolc gesla jazn"),
        ("WROCŁAW", "WROCLAW"),
        ("Łódź", "Lodz"),
        ("ĆĘŁŃÓŚŹŻ", "CELNOSZZ"),
        ("plain ASCII", "plain ASCII"),
    ],
)
def test_fold_diacritics_preserves_case(raw: str, expected: str) -> None:
    assert fold_diacritics(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Wrocław", "wroclaw"),
        ("WROCLAW", "wroclaw"),
        ("wroclaw", "wroclaw"),
        ("Wydział Informatyki", "wydzial informatyki"),
        ("ŁÓDŹ", "lodz"),
    ],
)
def test_normalize_search_text_is_case_and_diacritic_insensitive(
    raw: str,
    expected: str,
) -> None:
    assert normalize_search_text(raw) == expected


def test_normalize_cypher_string_literals_does_not_change_identifiers() -> None:
    query = (
        "MATCH (wydział:Wydział) "
        "WHERE wydział.tytuł CONTAINS 'Wydział Informatyki' "
        'AND wydział.miasto = "WROCŁAW" '
        "RETURN wydział.tytuł"
    )

    normalized = normalize_cypher_string_literals(query, normalizer=normalize_search_text)

    assert normalized == (
        "MATCH (wydział:Wydział) "
        "WHERE wydział.tytuł CONTAINS 'wydzial informatyki' "
        'AND wydział.miasto = "wroclaw" '
        "RETURN wydział.tytuł"
    )


def test_normalize_cypher_string_literals_preserves_dynamic_property_keys() -> None:
    query = (
        "MATCH (n:Faculty) "
        "WHERE toLower(n['ExactTitle']) CONTAINS toLower('WROCŁAW') "
        "RETURN n['ExactTitle']"
    )

    assert normalize_cypher_string_literals(query, normalizer=normalize_search_text) == (
        "MATCH (n:Faculty) "
        "WHERE toLower(n['ExactTitle']) CONTAINS toLower('wroclaw') "
        "RETURN n['ExactTitle']"
    )


def test_normalize_cypher_string_literals_normalizes_values_in_lists() -> None:
    query = "MATCH (n) WHERE n.city IN ['WROCŁAW'] RETURN n.city"

    assert normalize_cypher_string_literals(query, normalizer=normalize_search_text) == (
        "MATCH (n) WHERE n.city IN ['wroclaw'] RETURN n.city"
    )


def test_normalize_cypher_string_literals_preserves_escaping() -> None:
    query = r"MATCH (n) WHERE n.title = 'Wydział\' Informatyki' RETURN n.title"

    assert normalize_cypher_string_literals(query, normalizer=normalize_search_text) == (
        r"MATCH (n) WHERE n.title = 'wydzial\' informatyki' RETURN n.title"
    )


def test_normalize_cypher_string_literals_preserves_unquoted_query() -> None:
    query = "MATCH (n:Wydział) RETURN n.tytuł LIMIT 10"
    assert normalize_cypher_string_literals(query, normalizer=normalize_search_text) == query
