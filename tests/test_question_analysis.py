"""Tests for the deterministic question analysis behind Text2Cypher retrieval repair."""

import pytest

from src.mcp_server.tools.knowledge_graph.question_analysis import (
    extract_search_phrases,
    is_question_like_literal,
    strip_question_literal_filters,
    tokenize_search_text,
)

# The question from issue #52 whose whole text was copied into a CONTAINS literal.
CRITERIA_QUESTION = "Jakie są kryteria doboru kandydatki lub kandydata?"
# The question whose answer sits under a label the model did not pick.
CONFERENCE_QUESTION = "Co obejmuje udział w konferencjach?"


def test_tokenize_folds_case_diacritics_and_punctuation() -> None:
    assert tokenize_search_text("Udział w Konferencjach?") == ["udzial", "w", "konferencjach"]


def test_tokenize_keeps_year_numbers_as_separate_tokens() -> None:
    assert tokenize_search_text("semestr zimowy 2026/2027") == [
        "semestr",
        "zimowy",
        "2026",
        "2027",
    ]


@pytest.mark.parametrize(
    "literal",
    [
        "jakie sa kryteria doboru kandydatki lub kandydata",
        "co obejmuje udzial w konferencjach",
        "na czym polega praca zespolowa",
        "kiedy zaczyna sie semestr zimowy",
    ],
)
def test_question_text_in_a_literal_is_detected(literal) -> None:
    assert is_question_like_literal(literal, CRITERIA_QUESTION) is True


@pytest.mark.parametrize(
    "literal",
    [
        "udzial w konferencjach",
        "praca zespolowa",
        "wydzial informatyki i telekomunikacji",
        "analiza matematyczna",
        "umiejetnosc pozyskiwania funduszy",
        "transfer wiedzy i mobilnosc",
    ],
)
def test_entity_names_are_not_mistaken_for_question_text(literal) -> None:
    """Short noun phrases are exactly what the prompt asks for; they must survive untouched."""
    assert is_question_like_literal(literal, CONFERENCE_QUESTION) is False


def test_long_verbatim_span_of_the_question_counts_as_question_text() -> None:
    """A copied span is question text even with no interrogative left in it."""
    question = "Podaj zasady przyznawania stypendium rektora dla najlepszych studentow"

    assert (
        is_question_like_literal("zasady przyznawania stypendium rektora dla najlepszych", question)
        is True
    )


def test_short_span_of_the_question_is_left_alone() -> None:
    question = "Gdzie jest wydzial informatyki?"

    assert is_question_like_literal("wydzial informatyki", question) is False


def test_empty_literal_is_not_question_text() -> None:
    assert is_question_like_literal("", CRITERIA_QUESTION) is False


def test_strip_replaces_the_question_predicate_with_true() -> None:
    cypher = (
        "MATCH (g:Guideline)-[:RECOMMENDS]->(c:Committee)-[:CONSIDERS]->(comp:Competency) "
        "WHERE toLower(g.title) CONTAINS "
        "toLower('jakie sa kryteria doboru kandydatki lub kandydata') "
        "RETURN g.title, c.title, comp.title"
    )

    repaired, dropped = strip_question_literal_filters(cypher, CRITERIA_QUESTION)

    assert dropped == ["jakie sa kryteria doboru kandydatki lub kandydata"]
    assert repaired == (
        "MATCH (g:Guideline)-[:RECOMMENDS]->(c:Committee)-[:CONSIDERS]->(comp:Competency) "
        "WHERE true "
        "RETURN g.title, c.title, comp.title"
    )


def test_strip_keeps_the_surrounding_boolean_clause_valid() -> None:
    """Neutralising instead of deleting means AND/OR structure needs no re-parsing."""
    cypher = (
        "MATCH (n:Competency) "
        "WHERE toLower(n.title) CONTAINS toLower('co obejmuje udzial w konferencjach') "
        "AND n.year = 2026 RETURN n.title"
    )

    repaired, dropped = strip_question_literal_filters(cypher, CONFERENCE_QUESTION)

    assert dropped == ["co obejmuje udzial w konferencjach"]
    assert "WHERE true AND n.year = 2026" in repaired


def test_strip_leaves_entity_filters_in_place() -> None:
    cypher = (
        "MATCH (c:Course)<-[:TEACHES]-(p:Person) "
        "WHERE toLower(c.title) CONTAINS toLower('analiza matematyczna') RETURN p.title"
    )

    repaired, dropped = strip_question_literal_filters(cypher, "Kto wyklada analize matematyczna?")

    assert dropped == []
    assert repaired == cypher


def test_strip_leaves_exact_equality_untouched() -> None:
    """Equality is reserved for stable IDs, which are never question text."""
    cypher = "MATCH (n:Faculty) WHERE n.id = 'jakie-sa-kryteria' RETURN n.id"

    repaired, dropped = strip_question_literal_filters(cypher, CRITERIA_QUESTION)

    assert dropped == []
    assert repaired == cypher


def test_phrases_recover_the_stored_title_without_truncating_it() -> None:
    phrases = extract_search_phrases(CONFERENCE_QUESTION)

    assert "udzial w konferencjach" in phrases
    assert "udzial w" not in phrases
    assert "w konferencjach" not in phrases


def test_phrases_never_start_with_a_question_word() -> None:
    phrases = extract_search_phrases(CRITERIA_QUESTION)

    assert phrases
    assert not any(phrase.startswith("jakie") for phrase in phrases)
    assert "kryteria doboru" in phrases


def test_phrases_are_ordered_from_most_to_least_specific() -> None:
    phrases = extract_search_phrases(CONFERENCE_QUESTION)
    lengths = [len(phrase.split()) for phrase in phrases]

    assert lengths == sorted(lengths, reverse=True)


def test_phrases_drop_short_single_words() -> None:
    phrases = extract_search_phrases("Co to jest ECTS?")

    assert "to" not in phrases
    assert "jest" not in phrases


def test_phrases_are_deduplicated_and_capped() -> None:
    question = "Jakie kryteria doboru obowiazuja przy ocenie wniosku o stypendium rektora?"

    phrases = extract_search_phrases(question, max_phrases=5)

    assert len(phrases) == 5
    assert len(set(phrases)) == 5


def test_question_with_only_function_words_yields_no_phrases() -> None:
    assert extract_search_phrases("Co to jest?") == []


def test_repeated_phrase_is_emitted_once() -> None:
    question = "Kryteria doboru i kryteria oceny - jakie kryteria doboru obowiazuja?"

    phrases = extract_search_phrases(question)

    assert phrases.count("kryteria doboru") == 1


def test_literal_longer_than_the_question_cannot_be_a_copied_span() -> None:
    """Guards the span check against a literal the question could not have contributed."""
    assert (
        is_question_like_literal(
            "zasady przyznawania stypendium rektora dla najlepszych studentow",
            "Stypendium rektora",
        )
        is False
    )
