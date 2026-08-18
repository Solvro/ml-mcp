import logging
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

module_logger = logging.getLogger(__name__)

DOWNLOADABLE_EXTENSIONS = {".pdf", ".txt", ".md"}
_HTML_EXTENSIONS = {"", ".html", ".htm", ".php", ".aspx"}
_REQUEST_TIMEOUT_SECONDS = 30.0


@dataclass
class DiscoveredDoc:
    """One document found during discovery."""

    origin_url: str
    kind: str  # "html" | "pdf" | "txt" | "md"


class WebConnector:
    """Discovers and fetches documents from university web pages.

    Crawls seed URLs breadth-first up to ``max_depth`` link hops, staying on
    the seed domains. HTML pages become documents themselves; links ending in
    a downloadable extension become file documents.
    """

    def __init__(
        self,
        seed_urls: list[str],
        max_depth: int = 1,
        client: httpx.Client | None = None,
    ):
        self.seed_urls = [u for u in (s.strip() for s in seed_urls) if u]
        self.max_depth = max(0, max_depth)
        self._client = client or httpx.Client(
            timeout=_REQUEST_TIMEOUT_SECONDS, follow_redirects=True
        )
        self._allowed_hosts = {urlparse(u).netloc for u in self.seed_urls}

    def _same_domain(self, url: str) -> bool:
        return urlparse(url).netloc in self._allowed_hosts

    def discover(self) -> list[DiscoveredDoc]:
        """Crawl seed pages and return every unique discovered document."""
        seen_pages: set[str] = set()
        docs: dict[str, DiscoveredDoc] = {}
        frontier: list[tuple[str, int]] = [(url, 0) for url in self.seed_urls]

        while frontier:
            url, depth = frontier.pop(0)
            if url in seen_pages:
                continue
            seen_pages.add(url)
            try:
                response = self._client.get(url, follow_redirects=True)
            except httpx.HTTPError as exc:
                module_logger.warning("Discovery failed for %s: %s", url, exc)
                continue
            if response.status_code != 200:
                module_logger.warning("Discovery got HTTP %d for %s", response.status_code, url)
                continue

            docs[url] = DiscoveredDoc(origin_url=url, kind="html")
            if depth >= self.max_depth:
                continue

            soup = BeautifulSoup(response.text, "html.parser")
            for anchor in soup.find_all("a", href=True):
                link = urljoin(url, anchor["href"]).split("#")[0].rstrip("/") or url
                if not self._same_domain(link):
                    continue
                extension = Path(urlparse(link).path).suffix.lower()
                if extension in DOWNLOADABLE_EXTENSIONS:
                    kind = extension.lstrip(".")
                    docs.setdefault(link, DiscoveredDoc(origin_url=link, kind=kind))
                elif extension in _HTML_EXTENSIONS:
                    frontier.append((link, depth + 1))
        return list(docs.values())
