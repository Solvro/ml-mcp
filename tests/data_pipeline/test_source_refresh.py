import httpx
import pytest

from src.data_pipeline import staging
from src.data_pipeline.flows import source_refresh
from src.data_pipeline.flows.source_refresh import (
    DiscoveredDoc,
    WebConnector,
    build_connector,
    refresh_sources_flow,
)


def make_client(routes: dict[str, httpx.Response]) -> httpx.Client:
    """HTTP client with canned responses; unknown URLs get 404."""

    def handler(request: httpx.Request) -> httpx.Response:
        return routes.get(str(request.url), httpx.Response(404))

    return httpx.Client(transport=httpx.MockTransport(handler))


HOME = "https://pwr.edu.pl/"
SUBPAGE = "https://pwr.edu.pl/studenci"
PDF = "https://pwr.edu.pl/files/regulamin.pdf"
EXTERNAL = "https://inna-domena.pl/strona"


def test_discover_collects_seed_page_linked_pdf_and_subpage():
    routes = {
        HOME: httpx.Response(
            200,
            text=(
                f'<html><body><a href="{PDF}">regulamin</a>'
                f'<a href="/studenci">studenci</a>'
                f'<a href="{EXTERNAL}">obce</a></body></html>'
            ),
            headers={"content-type": "text/html"},
        ),
        SUBPAGE: httpx.Response(200, text="<html><body>studenci</body></html>"),
        PDF: httpx.Response(200, content=b"%PDF-1.4"),
    }
    connector = WebConnector([HOME], max_depth=1, client=make_client(routes))
    docs = connector.discover()
    by_url = {d.origin_url: d for d in docs}
    assert by_url[HOME].kind == "html"
    assert by_url[SUBPAGE].kind == "html"
    assert by_url[PDF].kind == "pdf"
    assert EXTERNAL not in by_url


def test_discover_respects_max_depth_zero():
    routes = {
        HOME: httpx.Response(200, text='<a href="/studenci">s</a>'),
    }
    connector = WebConnector([HOME], max_depth=0, client=make_client(routes))
    docs = connector.discover()
    assert [d.origin_url for d in docs] == [HOME]


def test_discover_skips_failing_pages_and_deduplicates():
    routes = {
        HOME: httpx.Response(200, text='<a href="/broken">x</a><a href="/broken">x</a>'),
    }
    connector = WebConnector([HOME], max_depth=1, client=make_client(routes))
    docs = connector.discover()
    assert [d.origin_url for d in docs] == [HOME]


def test_fetch_html_strips_tags_to_text():
    routes = {
        SUBPAGE: httpx.Response(
            200,
            text="<html><body><h1>Studenci</h1><p>Informacje</p></body></html>",
            headers={"ETag": 'W/"v1"'},
        ),
    }
    connector = WebConnector([HOME], client=make_client(routes))
    result = connector.fetch(DiscoveredDoc(origin_url=SUBPAGE, kind="html"))
    assert result.status == "fetched"
    assert b"<h1>" not in result.content
    assert "Studenci".encode() in result.content
    assert result.etag == 'W/"v1"'


def test_fetch_pdf_returns_raw_bytes():
    routes = {PDF: httpx.Response(200, content=b"%PDF-1.4 raw")}
    connector = WebConnector([HOME], client=make_client(routes))
    result = connector.fetch(DiscoveredDoc(origin_url=PDF, kind="pdf"))
    assert result.status == "fetched"
    assert result.content == b"%PDF-1.4 raw"


def test_fetch_sends_conditional_headers_and_handles_304():
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(dict(request.headers))
        return httpx.Response(304)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    connector = WebConnector([HOME], client=client)
    result = connector.fetch(
        DiscoveredDoc(origin_url=PDF, kind="pdf"),
        etag='W/"v1"',
        last_modified="Wed, 01 Jan 2026 00:00:00 GMT",
    )
    assert result.status == "unchanged"
    assert captured["if-none-match"] == 'W/"v1"'
    assert captured["if-modified-since"] == "Wed, 01 Jan 2026 00:00:00 GMT"


def test_fetch_error_and_non_200_return_failed():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    connector = WebConnector([HOME], client=client)
    assert connector.fetch(DiscoveredDoc(origin_url=PDF, kind="pdf")).status == "failed"

    connector_404 = WebConnector([HOME], client=make_client({}))
    assert connector_404.fetch(DiscoveredDoc(origin_url=PDF, kind="pdf")).status == "failed"


@pytest.fixture
def staging_env(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_PIPELINE_STAGING_DIR", str(tmp_path))
    return tmp_path


def routes_v1() -> dict[str, httpx.Response]:
    return {
        HOME: httpx.Response(
            200, text=f'<html><body>Strona główna <a href="{PDF}">pdf</a></body></html>'
        ),
        PDF: httpx.Response(200, content=b"%PDF-1.4 v1"),
    }


def test_refresh_stages_documents_and_writes_manifest(staging_env):
    connector = WebConnector([HOME], max_depth=1, client=make_client(routes_v1()))
    stats = refresh_sources_flow(trigger_downstream=False, connector=connector)
    assert stats["fetched"] == 2
    staged_pdf = staging_env / "pwr.edu.pl" / "files" / "regulamin.pdf"
    assert staged_pdf.read_bytes() == b"%PDF-1.4 v1"
    manifest = staging.load_manifest(staging_env)
    sid = staging.source_id_for("pwr.edu.pl/files/regulamin.pdf")
    assert manifest[sid]["origin"] == PDF
    assert manifest[sid]["sha256"]


def test_refresh_second_run_is_idempotent(staging_env):
    connector = WebConnector([HOME], max_depth=1, client=make_client(routes_v1()))
    refresh_sources_flow(trigger_downstream=False, connector=connector)
    connector2 = WebConnector([HOME], max_depth=1, client=make_client(routes_v1()))
    stats = refresh_sources_flow(trigger_downstream=False, connector=connector2)
    assert stats["fetched"] == 0
    assert stats["unchanged"] == 2


def test_refresh_partial_failure_stages_successes(staging_env):
    routes = routes_v1()
    routes[PDF] = httpx.Response(500)
    connector = WebConnector([HOME], max_depth=1, client=make_client(routes))
    stats = refresh_sources_flow(trigger_downstream=False, connector=connector)
    assert stats["fetched"] == 1  # the HTML page
    assert stats["failed"] == 1  # the PDF; retried next run
    sid = staging.source_id_for("pwr.edu.pl/files/regulamin.pdf")
    assert sid not in staging.load_manifest(staging_env)


def test_refresh_raises_when_everything_fails(staging_env):
    routes = {HOME: httpx.Response(200, text=f'<a href="{PDF}">x</a>')}
    connector = WebConnector([HOME], max_depth=1, client=make_client(routes))

    def failing_fetch(doc, *, etag=None, last_modified=None):
        return source_refresh.FetchResult(status="failed")

    connector.fetch = failing_fetch
    with pytest.raises(RuntimeError):
        refresh_sources_flow(trigger_downstream=False, connector=connector)


def test_refresh_triggers_downstream_only_on_change(staging_env, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(source_refresh, "data_pipeline_flow", lambda: calls.append("run"))

    connector = WebConnector([HOME], max_depth=1, client=make_client(routes_v1()))
    refresh_sources_flow(trigger_downstream=True, connector=connector)
    assert calls == ["run"]

    connector2 = WebConnector([HOME], max_depth=1, client=make_client(routes_v1()))
    refresh_sources_flow(trigger_downstream=True, connector=connector2)
    assert calls == ["run"]  # unchanged content -> no second trigger


def test_build_connector_requires_seeds(monkeypatch):
    monkeypatch.delenv("DATA_PIPELINE_SOURCE_URLS", raising=False)
    with pytest.raises(ValueError):
        build_connector()


def test_build_connector_reads_env(monkeypatch):
    monkeypatch.setenv("DATA_PIPELINE_SOURCE_URLS", f"{HOME}, https://wit.pwr.edu.pl/")
    monkeypatch.setenv("DATA_PIPELINE_CRAWL_DEPTH", "2")
    connector = build_connector()
    assert connector.seed_urls == [HOME, "https://wit.pwr.edu.pl/"]
    assert connector.max_depth == 2


def test_serve_refresh_uses_cron_from_env(monkeypatch):
    captured: dict[str, object] = {}

    def fake_serve(self, **kwargs):
        captured.update(kwargs)

    monkeypatch.setenv("DATA_PIPELINE_REFRESH_CRON", "0 5 * * 1")
    monkeypatch.setattr(type(refresh_sources_flow), "serve", fake_serve)
    source_refresh.serve_refresh()
    assert captured["cron"] == "0 5 * * 1"
    assert captured["name"] == "source-refresh"
