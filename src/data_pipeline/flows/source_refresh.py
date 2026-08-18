import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from prefect import flow, get_run_logger
from prefect.exceptions import MissingContextError

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
            response = self._client.get(doc.origin_url, headers=headers, follow_redirects=True)
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
    return WebConnector(seeds, max_depth=depth)


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

        content_hash = sha256(result.content or b"").hexdigest()
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

        atomic_write_bytes(staging_dir / relative_path, result.content or b"")
        manifest[sid] = {
            "origin": doc.origin_url,
            "etag": result.etag,
            "last_modified": result.last_modified,
            "sha256": content_hash,
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
