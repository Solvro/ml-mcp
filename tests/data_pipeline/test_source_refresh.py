import time

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


def test_refresh_write_failure_counts_as_failed_and_continues(staging_env, monkeypatch):
    def failing_write(target, data):
        if target.name.endswith(".pdf"):
            raise OSError(63, "File name too long")
        staging.atomic_write_bytes(target, data)

    monkeypatch.setattr(source_refresh, "atomic_write_bytes", failing_write)
    connector = WebConnector([HOME], max_depth=1, client=make_client(routes_v1()))
    stats = refresh_sources_flow(trigger_downstream=False, connector=connector)
    assert stats["fetched"] == 1  # HTML page staged
    assert stats["failed"] == 1  # PDF write failed, run survived
    sid = staging.source_id_for("pwr.edu.pl/files/regulamin.pdf")
    assert sid not in staging.load_manifest(staging_env)


def test_fetch_sends_identifying_user_agent():
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(dict(request.headers))
        return httpx.Response(200, content=b"ok")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    connector = WebConnector([HOME], client=client)
    connector.fetch(DiscoveredDoc(origin_url=PDF, kind="pdf"))
    assert "solvro" in captured["user-agent"].lower()


def test_refresh_keeps_staged_file_when_content_shrinks_drastically(staging_env):
    routes = routes_v1()
    routes[PDF] = httpx.Response(200, content=b"%PDF-1.4 " + b"real document body " * 20)
    connector = WebConnector([HOME], max_depth=1, client=make_client(routes))
    refresh_sources_flow(trigger_downstream=False, connector=connector)

    staged_pdf = staging_env / "pwr.edu.pl" / "files" / "regulamin.pdf"
    original = staged_pdf.read_bytes()
    manifest = staging.load_manifest(staging_env)
    sid = staging.source_id_for("pwr.edu.pl/files/regulamin.pdf")
    assert manifest[sid]["size"] == len(original)

    # Server now answers with a tiny bot-check page instead of the document.
    routes[PDF] = httpx.Response(200, content=b"robot check")
    connector2 = WebConnector([HOME], max_depth=1, client=make_client(routes))
    stats = refresh_sources_flow(trigger_downstream=False, connector=connector2)

    assert stats["failed"] == 1
    assert staged_pdf.read_bytes() == original
    assert staging.load_manifest(staging_env)[sid]["sha256"] == manifest[sid]["sha256"]


def test_build_connector_reads_request_delay(monkeypatch):
    monkeypatch.setenv("DATA_PIPELINE_SOURCE_URLS", HOME)
    monkeypatch.setenv("DATA_PIPELINE_REQUEST_DELAY", "2.5")
    assert build_connector().request_delay == 2.5

    monkeypatch.delenv("DATA_PIPELINE_REQUEST_DELAY", raising=False)
    assert build_connector().request_delay == 1.0


def test_connector_paces_requests():
    calls: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(time.monotonic())
        return httpx.Response(200, content=b"ok")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    connector = WebConnector([HOME], client=client, request_delay=0.05)
    doc = DiscoveredDoc(origin_url=PDF, kind="pdf")
    connector.fetch(doc)
    connector.fetch(doc)
    assert calls[1] - calls[0] >= 0.05


ROBOTS = "https://pwr.edu.pl/robots.txt"
SITEMAP = "https://pwr.edu.pl/sitemap.xml"


def _sitemap_xml(*urls: str) -> str:
    locs = "".join(f"<url><loc>{u}</loc></url>" for u in urls)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        f"{locs}</urlset>"
    )


def _sitemap_response(*urls: str) -> httpx.Response:
    return httpx.Response(
        200, text=_sitemap_xml(*urls), headers={"content-type": "application/xml"}
    )


def test_discover_prefers_sitemap_over_crawling():
    routes = {
        ROBOTS: httpx.Response(200, text=f"User-agent: *\nSitemap: {SITEMAP}\n"),
        SITEMAP: _sitemap_response(HOME, SUBPAGE),
        HOME: httpx.Response(200, text="<html><body>home</body></html>"),
        SUBPAGE: httpx.Response(200, text="<html><body>studenci</body></html>"),
    }
    connector = WebConnector([HOME], max_depth=1, client=make_client(routes))
    urls = sorted(d.origin_url for d in connector.discover())
    assert urls == sorted([HOME, SUBPAGE])


def test_discover_falls_back_to_crawl_without_sitemap():
    routes = {
        ROBOTS: httpx.Response(404),
        SITEMAP: httpx.Response(404),
        HOME: httpx.Response(200, text=f'<a href="{SUBPAGE}">s</a>'),
        SUBPAGE: httpx.Response(200, text="<html>studenci</html>"),
    }
    connector = WebConnector([HOME], max_depth=1, client=make_client(routes))
    urls = sorted(d.origin_url for d in connector.discover())
    assert urls == sorted([HOME, SUBPAGE])


def test_discover_respects_robots_disallow():
    routes = {
        ROBOTS: httpx.Response(200, text="User-agent: *\nDisallow: /files/\n"),
        SITEMAP: httpx.Response(404),
        HOME: httpx.Response(200, text=f'<a href="{PDF}">pdf</a><a href="{SUBPAGE}">s</a>'),
        SUBPAGE: httpx.Response(200, text="<html>studenci</html>"),
        PDF: httpx.Response(200, content=b"%PDF"),
    }
    connector = WebConnector([HOME], max_depth=1, client=make_client(routes))
    urls = [d.origin_url for d in connector.discover()]
    assert PDF not in urls  # /files/ is disallowed by robots.txt
    assert SUBPAGE in urls


def test_discover_skips_excluded_patterns():
    tracker = "https://pwr.edu.pl/addtrack/abc123"
    routes = {
        ROBOTS: httpx.Response(404),
        SITEMAP: httpx.Response(404),
        HOME: httpx.Response(200, text=f'<a href="{tracker}">t</a><a href="{SUBPAGE}">s</a>'),
        SUBPAGE: httpx.Response(200, text="<html>studenci</html>"),
        tracker: httpx.Response(200, text="<html>redirect</html>"),
    }
    connector = WebConnector(
        [HOME], max_depth=1, client=make_client(routes), exclude_patterns=["/addtrack/"]
    )
    urls = [d.origin_url for d in connector.discover()]
    assert tracker not in urls
    assert SUBPAGE in urls


def test_discover_follows_sitemap_index():
    child = "https://pwr.edu.pl/sitemap-pages.xml"
    index = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        f"<sitemap><loc>{child}</loc></sitemap></sitemapindex>"
    )
    routes = {
        ROBOTS: httpx.Response(200, text=f"Sitemap: {SITEMAP}\n"),
        SITEMAP: httpx.Response(200, text=index, headers={"content-type": "application/xml"}),
        child: _sitemap_response(SUBPAGE),
        SUBPAGE: httpx.Response(200, text="<html>studenci</html>"),
    }
    connector = WebConnector([HOME], client=make_client(routes))
    assert [d.origin_url for d in connector.discover()] == [SUBPAGE]


def test_discover_honours_max_documents():
    routes = {
        ROBOTS: httpx.Response(200, text=f"Sitemap: {SITEMAP}\n"),
        SITEMAP: _sitemap_response(HOME, SUBPAGE, "https://pwr.edu.pl/trzecia"),
        HOME: httpx.Response(200, text="<html>a</html>"),
        SUBPAGE: httpx.Response(200, text="<html>b</html>"),
        "https://pwr.edu.pl/trzecia": httpx.Response(200, text="<html>c</html>"),
    }
    connector = WebConnector([HOME], client=make_client(routes), max_documents=2)
    assert len(connector.discover()) == 2


def test_build_connector_reads_exclude_and_limit(monkeypatch):
    monkeypatch.setenv("DATA_PIPELINE_SOURCE_URLS", HOME)
    monkeypatch.setenv("DATA_PIPELINE_EXCLUDE_PATTERNS", "/addtrack/, /kalendarz/")
    monkeypatch.setenv("DATA_PIPELINE_MAX_DOCUMENTS", "50")
    c = build_connector()
    assert c.exclude_patterns == ["/addtrack/", "/kalendarz/"]
    assert c.max_documents == 50


def test_sitemap_is_scoped_to_seed_path():
    kursy = "https://pwr.edu.pl/studenci/kursy"
    inne = "https://pwr.edu.pl/pracownicy"
    routes = {
        ROBOTS: httpx.Response(200, text=f"Sitemap: {SITEMAP}\n"),
        SITEMAP: _sitemap_response(HOME, SUBPAGE, kursy, inne),
        SUBPAGE: httpx.Response(200, text="<html>studenci</html>"),
        kursy: httpx.Response(200, text="<html>kursy</html>"),
    }
    connector = WebConnector([SUBPAGE], client=make_client(routes))
    urls = sorted(d.origin_url for d in connector.discover())
    assert urls == sorted([SUBPAGE, kursy])  # seed path scopes the sitemap


def test_sitemap_root_seed_takes_whole_host():
    inne = "https://pwr.edu.pl/pracownicy"
    routes = {
        ROBOTS: httpx.Response(200, text=f"Sitemap: {SITEMAP}\n"),
        SITEMAP: _sitemap_response(SUBPAGE, inne),
        SUBPAGE: httpx.Response(200, text="<html>a</html>"),
        inne: httpx.Response(200, text="<html>b</html>"),
    }
    connector = WebConnector([HOME], client=make_client(routes))
    assert len(connector.discover()) == 2
