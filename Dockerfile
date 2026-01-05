FROM astral/uv:python3.12-bookworm-slim

WORKDIR /app

COPY ./pyproject.toml .
COPY ./uv.lock .
COPY ./src/mcp_server ./src/mcp_server

RUN uv sync --locked --no-dev

EXPOSE 8005


CMD ["uv", "run", "server"]
