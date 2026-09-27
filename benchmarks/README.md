# Text2Cypher normalization benchmark

This benchmark measures retrieval behavior for 28 Polish questions built from seven real PWr
faculty names published at <https://pwr.edu.pl/wydzialy>. Each entity is queried in four buckets:

- `canonical`: stored ASCII spelling and casing
- `diacritics`: original Polish diacritics with stored casing
- `case`: stored ASCII spelling in lowercase
- `case_and_diacritics`: lowercase original Polish spelling

Run all modes against the same Neo4j graph and the same deterministic accurate model:

- `baseline`: `main` prompt and retrieval behavior
- `prompt-only`: PR prompt, without deterministic query-literal folding
- `full`: complete PR behavior

The runner records correct-entity hit rate, relevant non-empty rate, and whether generated Cypher
uses `toLower(...)` at least twice. The fixture is intentionally small and controlled; production
graph validation still requires the team's graph dump or access to its Neo4j instance.

The controlled graph can be reproduced with `seed_text2cypher_normalization.py`. The seeder sends
the fixture through the selected revision's real `cypher_insert` prompt, converts its documented
pipe-separated output into one executable Cypher query, and records the generated nodes and query.

Both scripts accept `--provider openai` (the default production path) or `--provider clarin` (the
repository's Polish-language fallback). Use one provider consistently for seeding and every run,
and record the choice in the result report.

`run_retrieval_normalization_matrix.py` is a provider-independent isolation test. It deliberately
creates the kind of case-sensitive fuzzy query called out in review and executes 28 variants on a
real Neo4j instance, first with diacritic folding only and then with the complete deterministic
rewrite. This proves the normalization layer independently of prompt compliance; it does not
replace the LLM-in-the-loop run above.

Run the isolation matrix twice against the same local Neo4j instance:

```powershell
uv run python -m benchmarks.run_retrieval_normalization_matrix `
  --cases benchmarks/text2cypher_normalization_entities.json `
  --output benchmarks/results/diacritic-only.json `
  --mode diacritic-only `
  --neo4j-password '<password>'

uv run python -m benchmarks.run_retrieval_normalization_matrix `
  --cases benchmarks/text2cypher_normalization_entities.json `
  --output benchmarks/results/full.json `
  --mode full `
  --neo4j-password '<password>'
```

See `RESULTS.md` for the checked-in comparison table and limitations.

Example:

```powershell
uv run python benchmarks/run_text2cypher_normalization.py `
  --repo-root C:\path\to\ml-mcp `
  --cases benchmarks/text2cypher_normalization_entities.json `
  --output benchmarks/results/full.json `
  --mode full
```

# Grader stability replay

`run_grader_stability.py` measures how much the context grader's verdict moves when nothing
else does (issue #108). It runs `RAG.grade_context` N times over one saved set of rows and
reports, for every run, the rows the grader listed next to the rows the run kept. That way a
change to the grading rules can be judged on the same input before and after.

```powershell
uv run python -m benchmarks.run_grader_stability --runs 16 `
  --output benchmarks/results/grader-stability.json
```

The rows in `grader_stability_cases.json` are rebuilt from the shapes described in #102, #106
and #108, not captured from the production graph. With `NEO4J_URI`, `NEO4J_USER` and
`NEO4J_PASSWORD` pointing at a real graph, `--capture` replaces each case's rows with what the
full-text search returns and drops `expected`, whose indices named the old rows. Replaying
needs only an LLM key; `--only <case id>` limits a run to one case.
