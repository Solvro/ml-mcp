import httpx

from src.data_pipeline.flows.source_refresh import DiscoveredDoc, WebConnector


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
