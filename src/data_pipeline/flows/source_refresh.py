import logging
import os
import time
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from prefect import flow, get_run_logger
from prefect.exceptions import MissingContextError

from src.config.logging_config import configure_logging
from src.data_pipeline.pipeline import data_pipeline_flow
from src.data_pipeline.staging import (
    atomic_write_bytes,
    get_staging_dir,
    load_manifest,
    save_manifest,
    source_id_for,
    url_to_relative_path,
)

module_logger = logging.getLogger(__name__)

DOWNLOADABLE_EXTENSIONS = {".pdf", ".txt", ".md"}
_HTML_EXTENSIONS = {"", ".html", ".htm", ".php", ".aspx"}
_REQUEST_TIMEOUT_SECONDS = 30.0
_USER_AGENT = (
    "SolvroMCP-SourceRefresh/1.0 (knowledge graph ingestion; contact: kn.solvro@pwr.edu.pl)"
)

# Servers answer bot checks and maintenance pages with HTTP 200, so a document
# collapsing to a fraction of its known size is treated as a failed fetch
# instead of overwriting good staged content.
_SUSPICIOUS_SHRINK_RATIO = 0.3

# Sitemap indexes point at further sitemaps; bound how deep that nesting is followed.
_MAX_SITEMAP_DEPTH = 2


@dataclass
class DiscoveredDoc:
    """One document found during discovery."""

    origin_url: str
    kind: str  # "html" | "pdf" | "txt" | "md"


@dataclass
class FetchResult:
    """Outcome of fetching one document."""

    status: str  # "fetched" | "unchanged" | "failed"
    content: bytes | None = None
    etag: str | None = None
    last_modified: str | None = None


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
        request_delay: float = 0.0,
        exclude_patterns: list[str] | None = None,
        max_documents: int = 0,
    ):
        self.seed_urls = [u for u in (s.strip() for s in seed_urls) if u]
        self.max_depth = max(0, max_depth)
        self.request_delay = max(0.0, request_delay)
        self.exclude_patterns = [p for p in (exclude_patterns or []) if p]
        self.max_documents = max(0, max_documents)
        self._client = client or httpx.Client(
            timeout=_REQUEST_TIMEOUT_SECONDS, follow_redirects=True
        )
        self._allowed_hosts = {urlparse(u).netloc for u in self.seed_urls}
        self._last_request_at: float | None = None
        self._robots: dict[str, RobotFileParser | None] = {}

    def _same_domain(self, url: str) -> bool:
        return urlparse(url).netloc in self._allowed_hosts

    def _get(self, url: str, headers: dict[str, str] | None = None) -> httpx.Response:
        """Issue one paced GET that identifies the crawler."""
        if self.request_delay and self._last_request_at is not None:
            elapsed = time.monotonic() - self._last_request_at
            if elapsed < self.request_delay:
                time.sleep(self.request_delay - elapsed)
        self._last_request_at = time.monotonic()
        return self._client.get(
            url, headers={"User-Agent": _USER_AGENT, **(headers or {})}, follow_redirects=True
        )

    def _robots_for(self, url: str) -> RobotFileParser | None:
        """Fetch and cache the robots.txt rules for the host serving ``url``."""
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        if origin in self._robots:
            return self._robots[origin]

        rules: RobotFileParser | None = None
        try:
            response = self._get(f"{origin}/robots.txt")
            if response.status_code == 200:
                rules = RobotFileParser()
                rules.parse(response.text.splitlines())
        except httpx.HTTPError as exc:
            module_logger.warning("robots.txt unavailable for %s: %s", origin, exc)
        self._robots[origin] = rules
        return rules

    def _is_allowed(self, url: str) -> bool:
        """True when the URL passes the exclude list and the host's robots.txt."""
        if any(pattern in url for pattern in self.exclude_patterns):
            return False
        rules = self._robots_for(url)
        return True if rules is None else rules.can_fetch(_USER_AGENT, url)

    def _sitemap_urls(self, sitemap_url: str, depth: int = 0) -> list[str]:
        """Read page URLs out of a sitemap, following sitemap indexes."""
        if depth > _MAX_SITEMAP_DEPTH:
            return []
        try:
            response = self._get(sitemap_url)
        except httpx.HTTPError as exc:
            module_logger.warning("Sitemap fetch failed for %s: %s", sitemap_url, exc)
            return []
        if response.status_code != 200:
            return []
        try:
            root = ElementTree.fromstring(response.text)
        except ElementTree.ParseError as exc:
            module_logger.warning("Sitemap %s is not valid XML: %s", sitemap_url, exc)
            return []

        locations = [
            element.text.strip()
            for element in root.findall(".//{*}loc")
            if element.text and element.text.strip()
        ]
        if not root.tag.lower().endswith("sitemapindex"):
            return locations

        nested: list[str] = []
        for location in locations:
            nested.extend(self._sitemap_urls(location, depth + 1))
        return nested

    def _discover_from_sitemaps(self) -> list[DiscoveredDoc]:
        """Collect documents from the sitemaps the seed hosts advertise."""
        docs: dict[str, DiscoveredDoc] = {}
        for seed in self.seed_urls:
            rules = self._robots_for(seed)
            parsed = urlparse(seed)
            # A seed pointing at a subpath scopes the run to that subtree; only a
            # host root pulls in the whole sitemap.
            scope = parsed.path.rstrip("/")
            declared = list(rules.site_maps() or []) if rules else []
            for sitemap_url in declared or [f"{parsed.scheme}://{parsed.netloc}/sitemap.xml"]:
                for url in self._sitemap_urls(sitemap_url):
                    if url in docs or not self._same_domain(url) or not self._is_allowed(url):
                        continue
                    candidate = urlparse(url).path.rstrip("/")
                    if scope and candidate != scope and not candidate.startswith(f"{scope}/"):
                        continue
                    docs[url] = DiscoveredDoc(origin_url=url, kind=self._kind_for(url))
                    if self.max_documents and len(docs) >= self.max_documents:
                        return list(docs.values())
        return list(docs.values())

    @staticmethod
    def _kind_for(url: str) -> str:
        extension = Path(urlparse(url).path).suffix.lower()
        return extension.lstrip(".") if extension in DOWNLOADABLE_EXTENSIONS else "html"

    def discover(self) -> list[DiscoveredDoc]:
        """Return every discoverable document, preferring sitemaps over crawling.

        A sitemap lists the site's pages directly, so it replaces hundreds of
        crawl requests with a couple. Hosts without one fall back to the
        breadth-first crawl. Both paths honour robots.txt and the exclude list.
        """
        from_sitemaps = self._discover_from_sitemaps()
        if from_sitemaps:
            module_logger.info("Discovered %d documents from sitemaps", len(from_sitemaps))
            return from_sitemaps
        return self._crawl()

    def _crawl(self) -> list[DiscoveredDoc]:
        """Crawl seed pages and return every unique discovered document."""
        seen_pages: set[str] = set()
        docs: dict[str, DiscoveredDoc] = {}
        frontier: list[tuple[str, int]] = [(url, 0) for url in self.seed_urls]

        while frontier:
            if self.max_documents and len(docs) >= self.max_documents:
                break
            url, depth = frontier.pop(0)
            if url in seen_pages or not self._is_allowed(url):
                continue
            seen_pages.add(url)
            try:
                response = self._get(url)
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
                if not self._same_domain(link) or not self._is_allowed(link):
                    continue
                extension = Path(urlparse(link).path).suffix.lower()
                if extension in DOWNLOADABLE_EXTENSIONS:
                    kind = extension.lstrip(".")
                    docs.setdefault(link, DiscoveredDoc(origin_url=link, kind=kind))
                elif extension in _HTML_EXTENSIONS:
                    frontier.append((link, depth + 1))
        return list(docs.values())

    def fetch(
        self,
        doc: DiscoveredDoc,
        *,
        etag: str | None = None,
        last_modified: str | None = None,
    ) -> FetchResult:
        """Fetch one document, using conditional GET when validators are known.

        HTML pages are reduced to plain text (bytes, UTF-8) so the staging
        directory only ever contains text-extractable content.
        """
        headers: dict[str, str] = {}
        if etag:
            headers["If-None-Match"] = etag
        if last_modified:
            headers["If-Modified-Since"] = last_modified

        try:
            response = self._get(doc.origin_url, headers)
        except httpx.HTTPError as exc:
            module_logger.warning("Fetch failed for %s: %s", doc.origin_url, exc)
            return FetchResult(status="failed")

        if response.status_code == 304:
            return FetchResult(status="unchanged")
        if response.status_code != 200:
            module_logger.warning("Fetch got HTTP %d for %s", response.status_code, doc.origin_url)
            return FetchResult(status="failed")

        if doc.kind == "html":
            text = BeautifulSoup(response.text, "html.parser").get_text(separator="\n", strip=True)
            content = text.encode("utf-8")
        else:
            content = response.content

        return FetchResult(
            status="fetched",
            content=content,
            etag=response.headers.get("ETag"),
            last_modified=response.headers.get("Last-Modified"),
        )


def _get_logger() -> logging.Logger:
    """Return Prefect run logger when available, otherwise the module logger."""
    try:
        return get_run_logger()
    except MissingContextError:
        return module_logger


def build_connector() -> WebConnector:
    """Build a WebConnector from environment configuration.

    Raises:
        ValueError: when DATA_PIPELINE_SOURCE_URLS is empty.
    """
    raw_seeds = os.getenv("DATA_PIPELINE_SOURCE_URLS", "")
    seeds = [s.strip() for s in raw_seeds.split(",") if s.strip()]
    if not seeds:
        raise ValueError("DATA_PIPELINE_SOURCE_URLS must list at least one seed URL")
    try:
        depth = int(os.getenv("DATA_PIPELINE_CRAWL_DEPTH", "1").strip())
    except ValueError:
        depth = 1
    try:
        delay = float(os.getenv("DATA_PIPELINE_REQUEST_DELAY", "1.0").strip())
    except ValueError:
        delay = 1.0
    excluded = [p.strip() for p in os.getenv("DATA_PIPELINE_EXCLUDE_PATTERNS", "").split(",")]
    try:
        max_documents = int(os.getenv("DATA_PIPELINE_MAX_DOCUMENTS", "0").strip())
    except ValueError:
        max_documents = 0
    return WebConnector(
        seeds,
        max_depth=depth,
        request_delay=delay,
        exclude_patterns=[p for p in excluded if p],
        max_documents=max_documents,
    )


@flow(log_prints=True)
def refresh_sources_flow(
    trigger_downstream: bool = True,
    connector: WebConnector | None = None,
) -> dict[str, int]:
    """Discover, fetch and stage new/changed source documents.

    Successfully fetched documents are written atomically into the staging
    directory and recorded in ``manifest.json``. Failed fetches are skipped
    (retried on the next scheduled run). When at least one document changed,
    the downstream ``data_pipeline_flow`` runs as a subflow.

    Args:
        trigger_downstream: When True, run the extraction pipeline on changes.
        connector: Override for tests; defaults to env-configured WebConnector.

    Returns:
        Stats dict: discovered / fetched / unchanged / failed counts.

    Raises:
        RuntimeError: when documents were discovered but none could be fetched.
    """
    load_dotenv()
    logger = _get_logger()
    connector = connector or build_connector()

    staging_dir = get_staging_dir()
    staging_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(staging_dir)

    documents = connector.discover()
    stats = {"discovered": len(documents), "fetched": 0, "unchanged": 0, "failed": 0}
    logger.info("Discovered %d documents", len(documents))

    changed_any = False
    for doc in documents:
        relative_path = url_to_relative_path(doc.origin_url, is_html=doc.kind == "html")
        sid = source_id_for(relative_path)
        entry = manifest.get(sid, {})
        result = connector.fetch(
            doc, etag=entry.get("etag"), last_modified=entry.get("last_modified")
        )

        if result.status == "failed":
            stats["failed"] += 1
            logger.warning("Skipping %s (fetch failed); will retry next run", doc.origin_url)
            continue
        if result.status == "unchanged":
            stats["unchanged"] += 1
            continue

        content = result.content or b""
        previous_size = entry.get("size")
        if previous_size and len(content) < previous_size * _SUSPICIOUS_SHRINK_RATIO:
            stats["failed"] += 1
            logger.warning(
                "Keeping staged %s: server returned %d bytes for a %d byte document "
                "(bot check or outage?); will retry next run",
                doc.origin_url,
                len(content),
                previous_size,
            )
            continue

        content_hash = sha256(content).hexdigest()
        if entry.get("sha256") == content_hash:
            # Server re-sent identical content (no/changed validators) — refresh
            # validators but do not rewrite or count as a change.
            manifest[sid] = {
                **entry,
                "etag": result.etag,
                "last_modified": result.last_modified,
            }
            stats["unchanged"] += 1
            continue

        try:
            atomic_write_bytes(staging_dir / relative_path, content)
        except OSError as exc:
            stats["failed"] += 1
            logger.warning("Staging write failed for %s: %s", doc.origin_url, exc)
            continue
        manifest[sid] = {
            "origin": doc.origin_url,
            "etag": result.etag,
            "last_modified": result.last_modified,
            "sha256": content_hash,
            "size": len(content),
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
        stats["fetched"] += 1
        changed_any = True

    save_manifest(staging_dir, manifest)
    logger.info(
        "Refresh summary: discovered=%d fetched=%d unchanged=%d failed=%d",
        stats["discovered"],
        stats["fetched"],
        stats["unchanged"],
        stats["failed"],
    )

    if documents and stats["fetched"] == 0 and stats["unchanged"] == 0:
        raise RuntimeError(f"Source refresh failed for all {len(documents)} documents")

    if changed_any and trigger_downstream:
        logger.info("Changes staged; triggering downstream data pipeline")
        data_pipeline_flow()

    return stats


def serve_refresh() -> None:
    """Serve ``refresh_sources_flow`` as a scheduled Prefect deployment.

    Cadence comes from ``DATA_PIPELINE_REFRESH_CRON`` (default: daily 03:00).
    Blocks forever; intended as a container/console entry point.
    """
    load_dotenv()
    configure_logging()
    cron = os.getenv("DATA_PIPELINE_REFRESH_CRON", "0 3 * * *").strip() or "0 3 * * *"
    refresh_sources_flow.serve(name="source-refresh", cron=cron)


if __name__ == "__main__":
    refresh_sources_flow()
