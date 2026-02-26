# Git Workflow — SOLVRO MCP

## Commit Message Format

Follow **Conventional Commits**:

```
<type>: <short description>

[optional body]
```

**Types:**
- `feat:` — new feature or capability
- `fix:` — bug fix
- `docs:` — documentation only
- `refactor:` — code restructuring without behavior change
- `test:` — adding or updating tests
- `chore:` — dependency updates, build config, tooling

**Examples from this repo:**
```
feat: Add new RAG node for entity extraction
fix: Resolve Neo4j connection timeout issue
docs: Update API reference for knowledge_graph_tool
refactor: Simplify Cypher generation logic
test: Add unit tests for guardrails node
chore: Update dependencies in pyproject.toml
```

## Branching Convention

**Observed pattern:**
- `main` — stable, deployable
- `feature/<description>` — feature branches (e.g., `feature/data_pipeline`)
- PRs target `main`

**Inferred convention** (from PR #13 merge):
- One feature per branch
- Branch names use hyphens and are descriptive (`docker-stack`, `data_pipeline`)

## PR Workflow

1. Create branch from `main`: `git checkout -b feature/<name>`
2. Implement changes
3. Run `just ci` (lint + test) before pushing
4. Push and open PR against `main`
5. PRs are merged (not rebased based on merge commit history)

## Pre-Commit Checks

No pre-commit hooks detected in the repo. Before committing, manually run:

```bash
just lint    # ruff format + ruff check
just test    # pytest with coverage
```

Or together:
```bash
just ci
```

## CI/CD

No `.github/workflows/` detected. CI is local via `just ci`. Docker builds are manual.

## Current Branch State

The active branch is `feature/data_pipeline` with substantial uncommitted changes:
- Modified: pipeline flows, config models, Docker files, pyproject.toml
- Deleted: old `scripts/data_pipeline/` (migrated to `src/data_pipeline/flows/`)
- Added: `prefect-entrypoint.sh`, `text_extraction.py`

Stage and commit these as: `feat: Refactor data pipeline into Prefect flows with Azure Blob support`
