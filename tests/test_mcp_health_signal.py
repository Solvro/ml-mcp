"""The health signal has to mean "I can serve", not "my port is open".

Issue #64: the compose healthcheck opened a TCP socket, so the container went green the instant
uvicorn bound 8005 - with Neo4j gone, with the driver dead, either way. Anything wired to it
with `depends_on: service_healthy` was being handed a server that could not answer.

The same issue covers the other half of that shallow interface: `knowledge_graph_tool` reported
"Error: RAG not initialized" as an ordinary string, in the channel answers travel through, so
the backend could not tell a broken graph from a real reply.

Every case builds its own ASGI app: the streamable-http session manager refuses to run twice on
one instance.
"""

import asyncio

import pytest
from fastmcp.exceptions import ToolError
from starlette.testclient import TestClient

import src.mcp_server.server as server


class FakeRag:
    """Stands in for a RAG that owns a Neo4j driver."""

    def __init__(self, *, ping_error: Exception | None = None, ping_delay: float = 0.0):
        self.ping_error = ping_error
        self.ping_delay = ping_delay
        self.close_calls = 0

    def ping_database(self) -> None:
        if self.ping_delay:
            # Blocking on purpose: ping_database runs in a worker thread.
            import time

            time.sleep(self.ping_delay)
        if self.ping_error:
            raise self.ping_error

    def close(self) -> None:
        self.close_calls += 1


@pytest.fixture
def restore_rag():
    """The module holds RAG in a global; put back whatever was there."""
    original = server.rag
    yield
    server.rag = original


def get_health(rag) -> tuple[int, dict]:
    """Drive one GET /health through a freshly built ASGI app."""
    server.rag = rag
    with TestClient(server.mcp.http_app()) as client:
        response = client.get("/health")
        return response.status_code, response.json()


def test_health_is_served_next_to_the_mcp_endpoint(restore_rag) -> None:
    """The route has to sit at the root; the probe curls it, it is not an MCP call."""
    server.rag = FakeRag()
    paths = {getattr(route, "path", None) for route in server.mcp.http_app().routes}

    assert "/health" in paths
    assert "/mcp" in paths, "the MCP endpoint must keep its path"


def test_healthy_when_the_graph_answers(restore_rag) -> None:
    status, payload = get_health(FakeRag())

    assert status == 200
    assert payload["status"] == "healthy"


def test_unhealthy_when_rag_was_never_initialized(restore_rag) -> None:
    status, payload = get_health(None)

    assert status == 503
    assert payload["reason"] == "rag_not_initialized"


def test_unhealthy_when_neo4j_is_unreachable(restore_rag) -> None:
    """The port is open in this case. That is the whole point of the issue."""
    status, payload = get_health(FakeRag(ping_error=RuntimeError("Unable to connect to bolt://x")))

    assert status == 503
    assert payload["reason"] == "neo4j_unreachable"
    assert "Unable to connect" in payload["detail"], "the reason has to survive to the operator"


def test_unhealthy_when_the_graph_stops_answering(restore_rag, monkeypatch) -> None:
    """A stalled graph must answer the probe, not hold it until docker kills it."""
    monkeypatch.setattr(server, "HEALTH_PING_TIMEOUT_SECONDS", 0.05)

    status, payload = get_health(FakeRag(ping_delay=0.5))

    assert status == 503
    assert payload["reason"] == "neo4j_timeout"


def test_lifespan_closes_the_graph_driver(restore_rag) -> None:
    """RAG opens a driver in its constructor; a restart without this leaks one."""
    rag = FakeRag()
    server.rag = rag

    with TestClient(server.mcp.http_app()):
        pass

    assert rag.close_calls == 1
    assert server.rag is None, "a closed driver must not stay reachable through the global"


def test_close_rag_is_safe_to_repeat(restore_rag) -> None:
    rag = FakeRag()
    server.rag = rag

    server.close_rag()
    server.close_rag()

    assert rag.close_calls == 1


def test_close_rag_survives_a_driver_that_fails_to_close(restore_rag) -> None:
    """Shutdown must not be derailed by the cleanup it is doing."""

    class ExplodingRag(FakeRag):
        def close(self) -> None:
            super().close()
            raise RuntimeError("driver already gone")

    rag = ExplodingRag()
    server.rag = rag

    server.close_rag()

    assert rag.close_calls == 1
    assert server.rag is None


def test_tool_raises_instead_of_answering_when_rag_is_missing(restore_rag) -> None:
    """A failure has to arrive as a failed call, not as text that looks like content."""
    server.rag = None

    with pytest.raises(ToolError):
        asyncio.run(server.knowledge_graph_tool.fn("Kto wyklada analize?"))


def test_tool_raises_on_pipeline_timeout(restore_rag) -> None:
    class TimingOutRag:
        async def ainvoke(self, **kwargs):
            raise TimeoutError

    server.rag = TimingOutRag()

    with pytest.raises(ToolError):
        asyncio.run(server.knowledge_graph_tool.fn("Kto wyklada analize?"))


def test_tool_returns_the_answer_when_the_pipeline_works(restore_rag) -> None:
    """The abstention answer is content, not a failure, and must come back normally."""

    class AnsweringRag:
        async def ainvoke(self, **kwargs):
            return {"answer": "Brak danych w grafie wiedzy dla tego pytania.", "metadata": {}}

    server.rag = AnsweringRag()

    answer = asyncio.run(server.knowledge_graph_tool.fn("Kto wyklada analize?"))

    assert answer == "Brak danych w grafie wiedzy dla tego pytania."
