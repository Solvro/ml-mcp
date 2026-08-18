import json
import os
import re
from hashlib import sha256
from pathlib import Path
from urllib.parse import unquote, urlparse

MANIFEST_FILENAME = "manifest.json"

_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._/-]")


def get_staging_dir() -> Path:
    """Resolve the staging root directory from DATA_PIPELINE_STAGING_DIR.

    Returns:
        Path to the staging root (default ``data/staging``).
    """
    raw = os.getenv("DATA_PIPELINE_STAGING_DIR", "").strip()
    return Path(raw) if raw else Path("data/staging")


def url_to_relative_path(url: str, *, is_html: bool) -> str:
    """Map a source URL to a deterministic staging-relative POSIX path.

    Same URL always yields the same path, so re-fetching a changed document
    overwrites it in place. Query strings are folded into a short hash suffix
    to keep distinct URLs distinct.

    Args:
        url: Absolute source URL.
        is_html: True when the document is an HTML page (stored as ``.md``).

    Returns:
        Relative path like ``pwr.edu.pl/studenci/aktualnosci.md``.
    """
    parsed = urlparse(url)
    path = unquote(parsed.path).strip("/") or "index"
    path = _UNSAFE_CHARS.sub("_", path)
    if parsed.query:
        query_hash = sha256(parsed.query.encode("utf-8")).hexdigest()[:8]
        path = f"{path}_{query_hash}"
    relative = f"{parsed.netloc}/{path}"
    if is_html and not relative.endswith(".md"):
        relative = f"{relative}.md"
    return relative


def source_id_for(relative_path: str) -> str:
    """Build the pipeline source id for a staged file (``file://{relative_path}``)."""
    return f"file://{relative_path}"


def atomic_write_bytes(target: Path, data: bytes) -> None:
    """Write bytes atomically: write ``<name>.part`` first, then rename over target."""
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    part.write_bytes(data)
    os.replace(part, target)


def load_manifest(staging_dir: Path) -> dict[str, dict]:
    """Load ``manifest.json`` from the staging root; empty dict when missing/corrupt."""
    path = staging_dir / MANIFEST_FILENAME
    if not path.is_file():
        return {}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def save_manifest(staging_dir: Path, manifest: dict[str, dict]) -> None:
    """Persist the manifest atomically into the staging root."""
    payload = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
    atomic_write_bytes(staging_dir / MANIFEST_FILENAME, payload)
