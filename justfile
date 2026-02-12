# ============================================================================
# SOLVRO MCP - Project Commands
# ============================================================================
# Usage: just <recipe>
# Run `just help` to see all available commands
# ============================================================================

set dotenv-load := true

# Default recipe - show help
default: help

# ============================================================================
# 🚀 LOCAL DEVELOPMENT
# ============================================================================

# Start MCP server locally
[group('dev')]
mcp-server:
    uv run mcp-server

# Query knowledge graph via CLI
[group('dev')]
kg QUERY:
    uv run kg "{{QUERY}}"

# Start FastAPI backend locally
[group('dev')]
api:
    uv run topwr-api

# ============================================================================
# 🐳 DOCKER STACK (Neo4j + MCP Server + API)
# ============================================================================

# Start all services
[group('docker')]
up:
    docker compose --env-file .env -f docker/compose.stack.yml up -d --build

# Stop all services
[group('docker')]
down:
    docker compose --env-file .env -f docker/compose.stack.yml down

# Restart all services
[group('docker')]
restart:
    docker compose --env-file .env -f docker/compose.stack.yml restart

# Show service status
[group('docker')]
ps:
    docker compose --env-file .env -f docker/compose.stack.yml ps

# View all logs
[group('docker')]
logs:
    docker compose --env-file .env -f docker/compose.stack.yml logs -f

# View MCP server logs
[group('docker')]
logs-mcp:
    docker compose --env-file .env -f docker/compose.stack.yml logs -f mcp-server

# View API logs
[group('docker')]
logs-api:
    docker compose --env-file .env -f docker/compose.stack.yml logs -f topwr-api

# View Neo4j logs
[group('docker')]
logs-neo4j:
    docker compose --env-file .env -f docker/compose.stack.yml logs -f neo4j

# Remove all containers and volumes
[group('docker')]
nuke:
    docker compose --env-file .env -f docker/compose.stack.yml down -v --remove-orphans

# ============================================================================
# 📊 PREFECT DATA PIPELINE (separate service)
# ============================================================================

# Start Prefect server
[group('prefect')]
prefect-up:
    docker compose --env-file .env -f docker/compose.prefect.yml up -d --build

# Stop Prefect server
[group('prefect')]
prefect-down:
    docker compose --env-file .env -f docker/compose.prefect.yml down

# View Prefect logs
[group('prefect')]
prefect-logs:
    docker compose --env-file .env -f docker/compose.prefect.yml logs -f

# Run data pipeline locally
[group('prefect')]
pipeline:
    uv run prefect-pipeline

# ============================================================================
# 🧪 QUALITY & TESTING
# ============================================================================

# Format and lint code
[group('quality')]
lint:
    uv run ruff format src
    uv run ruff check src --fix

# Run tests with coverage
[group('quality')]
test:
    uv run pytest --cov=src --cov-report=term tests/

# Run tests with verbose output and HTML report
[group('quality')]
test-verbose:
    uv run pytest -v --cov=src --cov-report=term --cov-report=html tests/

# Run full CI pipeline
[group('quality')]
ci: lint test
    @echo "✅ All CI checks passed!"

# ============================================================================
# 🔧 SETUP & MAINTENANCE
# ============================================================================

# Initial project setup
[group('setup')]
setup:
    uv sync
    just generate-models
    @echo ""
    @echo "✅ Development environment ready!"
    @echo ""
    @echo "Next steps:"
    @echo "  1. cp .env.example .env"
    @echo "  2. Edit .env with your API keys"
    @echo "  3. just up"

# Generate Pydantic models from config
[group('setup')]
generate-models:
    uv run python src/scripts/config/generate_models.py

# Build package
[group('setup')]
build:
    uv build

# Clean build artifacts
[group('setup')]
clean:
    rm -rf dist/ .pytest_cache/ htmlcov/ .coverage .ruff_cache/
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    @echo "✨ Cleaned!"

# ============================================================================
# 📖 HELP
# ============================================================================

# Show all available commands
[group('help')]
help:
    @echo "SOLVRO MCP - Knowledge Graph RAG System"
    @echo ""
    @echo "Quick Start:"
    @echo "  just setup    # Initial setup"
    @echo "  just up       # Start Docker stack"
    @echo "  just logs     # View logs"
    @echo "  just down     # Stop services"
    @echo ""
    @just --list --unsorted
