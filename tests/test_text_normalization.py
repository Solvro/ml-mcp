import pytest

from src.text_normalization import (
    ensure_case_insensitive_fuzzy_matching,
    fold_diacritics,
    join_orphaned_list_markers,
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


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        (
            "MATCH (n) WHERE n.title CONTAINS 'Wydzial' RETURN n.title",
            "MATCH (n) WHERE toLower(n.title) CONTAINS toLower('Wydzial') RETURN n.title",
        ),
        (
            "MATCH (n) WHERE toLower(n.title) STARTS WITH 'Wydzial' RETURN n.title",
            "MATCH (n) WHERE toLower(n.title) STARTS WITH toLower('Wydzial') RETURN n.title",
        ),
        (
            "MATCH (n) WHERE n.title ENDS WITH toLower('Wodnego') RETURN n.title",
            "MATCH (n) WHERE toLower(n.title) ENDS WITH toLower('Wodnego') RETURN n.title",
        ),
        (
            "MATCH (n) WHERE n['ExactTitle'] CONTAINS 'Wydzial' RETURN n['ExactTitle']",
            "MATCH (n) WHERE toLower(n['ExactTitle']) CONTAINS toLower('Wydzial') "
            "RETURN n['ExactTitle']",
        ),
        (
            'MATCH (n) WHERE n.`display title` CONTAINS "Wydzial" RETURN n',
            'MATCH (n) WHERE toLower(n.`display title`) CONTAINS toLower("Wydzial") RETURN n',
        ),
    ],
)
def test_ensure_case_insensitive_fuzzy_matching_wraps_both_sides(
    query: str,
    expected: str,
) -> None:
    assert ensure_case_insensitive_fuzzy_matching(query) == expected


def test_ensure_case_insensitive_fuzzy_matching_is_idempotent() -> None:
    query = "MATCH (n) WHERE toLower(n.title) CONTAINS toLower('wydzial') RETURN n.title"
    assert ensure_case_insensitive_fuzzy_matching(query) == query


def test_ensure_case_insensitive_fuzzy_matching_preserves_stable_id_equality() -> None:
    query = "MATCH (n) WHERE n.external_id = 'Faculty-ABC' RETURN n.title"
    assert ensure_case_insensitive_fuzzy_matching(query) == query


# Issue #78: a PDF text layer writes a bullet and its text as two lines, so a page of 168
# bullets contained no line that reads as a list row - for a reader, for the extraction model,
# or for the completeness check that has to verify every row became a node.
PDF_BULLET_PAGE = """R1 - Naukowiec początkujący
•
prowadzi badania naukowe pod nadzorem opiekuna naukowego
•
publikuje wyniki swoich badań w czasopismach:
a)
o zasięgu krajowym,
b)
o zasięgu międzynarodowym.
"""


def test_a_marker_on_its_own_line_is_rejoined_with_its_text() -> None:
    rejoined = join_orphaned_list_markers(PDF_BULLET_PAGE).splitlines()

    assert "• prowadzi badania naukowe pod nadzorem opiekuna naukowego" in rejoined
    assert "a) o zasięgu krajowym," in rejoined
    assert "b) o zasięgu międzynarodowym." in rejoined


def test_the_lines_a_row_wraps_onto_are_folded_into_it() -> None:
    page = (
        "•\nW grupie pracowników dydaktycznych (których podstawowym\n"
        "obowiązkiem jest kształcenie studentów) prowadzi zajęcia."
    )

    assert join_orphaned_list_markers(page) == (
        "• W grupie pracowników dydaktycznych (których podstawowym "
        "obowiązkiem jest kształcenie studentów) prowadzi zajęcia."
    )


def test_a_row_stops_at_the_next_marker() -> None:
    page = "•\npierwsza pozycja listy\n•\ndruga pozycja listy"

    assert join_orphaned_list_markers(page) == "• pierwsza pozycja listy\n• druga pozycja listy"


def test_a_finished_row_does_not_swallow_the_paragraph_after_it() -> None:
    page = "•\nostatnia pozycja listy.\nOcena kompetencji odbywa się raz w roku."

    assert join_orphaned_list_markers(page) == (
        "• ostatnia pozycja listy.\nOcena kompetencji odbywa się raz w roku."
    )


def test_a_list_that_already_reads_as_one_is_left_alone() -> None:
    page = "- 1 XI 2026 r. - Wszystkich Świętych\n- 2 XI 2026 r. - dzień wolny od zajęć"

    assert join_orphaned_list_markers(page) == page


def test_prose_is_left_alone() -> None:
    page = "Zajęcia odbywają się zgodnie z planem.\n\nDni wolne ogłasza rektor."

    assert join_orphaned_list_markers(page) == page


def test_rejoining_is_idempotent() -> None:
    once = join_orphaned_list_markers(PDF_BULLET_PAGE)

    assert join_orphaned_list_markers(once) == once


def test_a_marker_with_nothing_after_it_is_harmless() -> None:
    assert join_orphaned_list_markers("•") == "•"
    assert join_orphaned_list_markers("tekst\n•").splitlines() == ["tekst", "•"]
