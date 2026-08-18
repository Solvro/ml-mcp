import httpx

from src.data_pipeline.flows.source_refresh import WebConnector


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
