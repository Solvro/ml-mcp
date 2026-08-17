from benchmarks.run_text2cypher_normalization import load_cases, summarize
from benchmarks.seed_text2cypher_normalization import clean_generated_cypher


def test_load_cases_builds_all_four_variants(tmp_path) -> None:
    path = tmp_path / "entities.json"
    path.write_text(
        """{
          "source": "https://example.test",
          "question_template": "Znajdź {entity}",
          "entities": [{
            "id": "faculty",
            "original": "Wydział Zarządzania",
            "stored": "Wydzial Zarzadzania"
          }]
        }""",
        encoding="utf-8",
    )

    source, cases = load_cases(path)

    assert source == "https://example.test"
    assert [case["category"] for case in cases] == [
        "canonical",
        "diacritics",
        "case",
        "case_and_diacritics",
    ]
    assert cases[1]["question"] == "Znajdź Wydział Zarządzania"
    assert cases[2]["question"] == "Znajdź wydzial zarzadzania"
    assert cases[3]["search_value"] == "wydział zarządzania"


def test_summarize_reports_rates_by_category_and_overall() -> None:
    rows = [
        {
            "category": "canonical",
            "hit": True,
            "non_empty": True,
            "tolower_compliant": True,
        },
        {
            "category": "case",
            "hit": False,
            "non_empty": True,
            "tolower_compliant": False,
        },
    ]

    summary = summarize(rows)

    assert summary["canonical"]["hit_rate"] == 1.0
    assert summary["case"]["hit_rate"] == 0.0
    assert summary["overall"] == {
        "total": 2,
        "hits": 1,
        "hit_rate": 0.5,
        "non_empty": 2,
        "non_empty_rate": 1.0,
        "tolower_compliant": 1,
        "tolower_compliance_rate": 0.5,
    }


def test_clean_generated_cypher_joins_ingestion_parts() -> None:
    raw = "```cypher\nMERGE (a:Faculty {title: 'A'})|MERGE (b:Faculty {title: 'B'})\n```"

    assert clean_generated_cypher(raw) == (
        "MERGE (a:Faculty {title: 'A'})\nMERGE (b:Faculty {title: 'B'})"
    )
