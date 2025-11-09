mcp-server:
    uv run server 

mcp-client QUERY:
    uv run kg {{QUERY}}

lint:
    uv run ruff format src
    uv run ruff check src

pipeline:
    uv run pipeline 

pipeline-clear:
    uv run pipeline --clear-db

test:
    uv run pytest --cov=src --cov-report=term tests/

test-verbose:
    uv run pytest -v --cov=src --cov-report=term --cov-report=html tests/

build:
    uv build

ci: lint test build
    @echo "✅ All CI checks passed!"

setup:
    uv sync
    @echo "✅ Development environment ready!"
    @echo "Copy .env.example to .env and configure your environment variables"

clean:
    rm -rf dist/
    rm -rf .pytest_cache/
    rm -rf htmlcov/
    rm -rf .coverage
    find . -type d -name __pycache__ -exec rm -rf {} +
    find . -type f -name "*.pyc" -delete

help:
    @just --list