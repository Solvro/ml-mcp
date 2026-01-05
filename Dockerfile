FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

ENV UV_NO_CACHE=1

COPY pyproject.toml uv.lock ./

RUN uv sync --locked --no-dev --compile-bytecode

FROM python:3.12-slim-bookworm

WORKDIR /app

RUN groupadd -r mcpuser && useradd -r -g mcpuser mcpuser

COPY --from=builder /app/.venv /app/.venv
COPY ./src/mcp_server ./src/mcp_server

ENV PATH="/app/.venv/bin:$PATH"

USER mcpuser

EXPOSE 8005

CMD ["python3", "-m", "src.mcp_server.server", "--host", "0.0.0.0", "--port", "8005"]
