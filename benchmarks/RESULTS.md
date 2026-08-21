# Text2Cypher normalization benchmark results

Run date: 2026-08-17

## Deterministic retrieval isolation

The test graph contains seven real faculty names from the official PWr faculty list. The stored
`title` values follow the ingestion contract and contain ASCII text without Polish diacritics. Each
entity is queried in four forms, producing 28 total cases on the same Neo4j 5.18 instance.

The generated test query intentionally omits `toLower(...)`. This isolates whether retrieval is
deterministically safe when the LLM does not follow the prompt's case-insensitive matching rule.

| Query bucket | Cases | Diacritic folding only | Full deterministic rewrite |
|---|---:|---:|---:|
| Canonical stored spelling | 7 | 7/7 (100%) | 7/7 (100%) |
| Polish diacritics | 7 | 7/7 (100%) | 7/7 (100%) |
| Lowercase ASCII | 7 | 0/7 (0%) | 7/7 (100%) |
| Lowercase with Polish diacritics | 7 | 0/7 (0%) | 7/7 (100%) |
| **Overall correct-entity hit rate** | **28** | **14/28 (50%)** | **28/28 (100%)** |
| **Overall relevant non-empty rate** | **28** | **14/28 (50%)** | **28/28 (100%)** |
| **Queries with `toLower` on both sides** | **28** | **0/28 (0%)** | **28/28 (100%)** |

The full rewrite folds Polish diacritics in comparison literals and guarantees case-insensitive
matching for `CONTAINS`, `STARTS WITH`, and `ENDS WITH`. Exact equality is not rewritten because the
retrieval prompt reserves it for stable IDs, whose case may be significant.

Raw rows and executed Cypher are recorded in `results/diacritic-only.json` and `results/full.json`.

## LLM-in-the-loop benchmark

`run_text2cypher_normalization.py` and `seed_text2cypher_normalization.py` provide the requested
main vs prompt-only vs full benchmark using the repository's real ingestion and Text2Cypher
prompts. Running it still requires valid OpenAI or CLARIN credentials and, for production-level
evidence, the team's graph dump or Neo4j access. The deterministic result above does not claim to
measure model prompt compliance.

## Display-value follow-up

The existing ingestion contract stores ASCII-folded values in place, so returned values are also
ASCII-folded. This PR keeps that contract and focuses on retrieval correctness. Preserving Polish
display text should be a separate schema/data migration: retain the original value for responses,
write a parallel normalized property such as `title_norm` for matching, re-ingest existing data,
and update retrieval to search the normalized property while returning the original one.
