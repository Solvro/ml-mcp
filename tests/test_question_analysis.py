"""Tests for the deterministic question analysis behind Text2Cypher retrieval repair."""

import pytest

from src.mcp_server.tools.knowledge_graph.question_analysis import (
    build_lucene_query,
    expand_inflected_token,
    extract_search_phrases,
    is_code_token,
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


# Issue #59: the index has no Polish analyzer, so an oblique-case question never reaches the
# nominative title. Long tokens are expanded by prefix and edit distance to close that gap.
INFLECTED_QUESTION = "Co się dzieje w semestrze zimowym?"


@pytest.mark.parametrize(
    ("token", "expected_prefix"),
    [
        ("semestrze", "semestr*"),
        ("zimowym", "zimow*"),
        ("konferencjach", "konferencja*"),
        ("kandydatki", "kandydat*"),
    ],
)
def test_a_long_token_is_searched_by_prefix(token, expected_prefix) -> None:
    clauses = expand_inflected_token(token)

    assert any(clause.startswith(expected_prefix) for clause in clauses)


@pytest.mark.parametrize("token", ["semestrze", "zimowym", "konferencjach"])
def test_a_long_token_is_also_searched_by_edit_distance(token) -> None:
    clauses = expand_inflected_token(token)

    assert any(clause.startswith(f"{token}~1") for clause in clauses)


@pytest.mark.parametrize("token", ["w", "we", "rok", "roku"])
def test_a_short_token_is_never_expanded(token) -> None:
    """Short tokens match half the graph by prefix, so they stay exact."""
    assert expand_inflected_token(token) == []


def test_the_prefix_never_shrinks_below_the_floor() -> None:
    clauses = expand_inflected_token("kurso")

    assert all(not clause.startswith("kur*") for clause in clauses)


def test_expansion_clauses_are_boosted_below_one() -> None:
    """A true nominative hit has to keep ranking above an inflected or fuzzy one."""
    for clause in expand_inflected_token("semestrze"):
        boost = float(clause.rsplit("^", 1)[1])
        assert boost < 1


def test_the_inflected_question_reaches_the_nominative_title() -> None:
    query = build_lucene_query(extract_search_phrases(INFLECTED_QUESTION))

    assert "semestr*" in query
    assert "zimow*" in query


def test_the_exact_phrase_still_outranks_every_expansion() -> None:
    query = build_lucene_query(["semestr zimowy"])

    assert query.startswith('"semestr zimowy"^2')
    exact_boost = 2.0
    for clause in query.split(" OR ")[1:]:
        assert float(clause.rsplit("^", 1)[1]) < exact_boost


def test_a_token_repeated_across_phrases_is_expanded_once() -> None:
    query = build_lucene_query(["semestr zimowy", "semestr", "zimowy"])

    assert query.count("semes*") == 1
    assert query.count("semestr~1") == 1


def test_expansion_still_drops_phrases_with_metacharacters() -> None:
    query = build_lucene_query(['title:"x" OR *', "semestrze"])

    assert "title" not in query
    assert "semestr*" in query


def test_no_phrases_still_produces_no_query() -> None:
    assert build_lucene_query([]) == ""


# Issue #106: "R2" was two characters, so it never became a phrase of its own, and inside the
# longer phrases it weighed nothing next to "kompetencje". The search came back with R3 and R4
# rows and the answer listed 4 of R2's 14 competencies as if that were all of them.
R2_QUESTION = "Jakie są kompetencje pożądane dla naukowca R2?"


@pytest.mark.parametrize("token", ["r2", "w4", "a1", "3d", "inz002152w"])
def test_a_token_mixing_letters_and_digits_is_a_code(token) -> None:
    assert is_code_token(token)


@pytest.mark.parametrize("token", ["2026", "27", "r", "ects", "kompetencje"])
def test_words_and_plain_numbers_are_not_codes(token) -> None:
    """A year is written too many ways for the question's spelling to be required of a node."""
    assert not is_code_token(token)


def test_a_short_code_is_kept_as_a_search_phrase() -> None:
    assert "r2" in extract_search_phrases(R2_QUESTION)
    assert extract_search_phrases("Co to jest R2?") == ["r2"]


def test_the_code_is_required_and_the_other_words_only_rank() -> None:
    query = build_lucene_query(extract_search_phrases(R2_QUESTION))

    assert query.startswith("+r2 (")
    assert query.endswith(")")
    ranking = query[len("+r2 (") : -1]
    assert '"kompetencje pozadane dla naukowca"^4' in ranking
    assert '"naukowca r2"^2' in ranking
    # Already required, so it is not repeated as an optional term.
    assert '"r2"' not in ranking


def test_several_codes_require_any_one_of_them() -> None:
    query = build_lucene_query(extract_search_phrases("Czym różni się R1 od R4?"))

    assert query.startswith("+(r1 OR r4) (")


def test_a_question_that_is_only_a_code_searches_for_the_code() -> None:
    assert build_lucene_query(["r2"]) == "r2"


@pytest.mark.parametrize("token", ["r2", "inz002152w"])
def test_a_code_is_never_expanded(token) -> None:
    """One edit away from a code is another code: "r2~1" would match "r3"."""
    assert expand_inflected_token(token) == []


def test_no_expansion_of_the_code_reaches_the_query() -> None:
    query = build_lucene_query(extract_search_phrases("Jakie kursy ma kod INZ002152W?"))

    assert query.startswith("+inz002152w (")
    assert "inz002152w~" not in query
    assert "inz0021*" not in query


def test_a_year_stays_optional() -> None:
    query = build_lucene_query(extract_search_phrases("Kiedy zaczyna się semestr zimowy 2026?"))

    assert "+" not in query
    assert '"semestr zimowy 2026"^3' in query
