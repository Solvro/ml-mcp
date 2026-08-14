import logging
import os
from pathlib import Path
from typing import Any

from prefect import get_run_logger, task

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}


def _get_logger() -> Any:
    """Return Prefect run logger when available, otherwise stdlib logger."""
    try:
        return get_run_logger()
    except Exception:
        return logging.getLogger(__name__)


def _build_source_id(staging_dir: Path, file_path: Path) -> str:
    """Build stable source id from path relative to staging root."""
    relative_path = file_path.relative_to(staging_dir).as_posix()
    return f"file://{relative_path}"


@task
def acquire_data() -> list[dict[str, str]]:
    """Load staged document references for downstream OCR extraction."""
    logger = _get_logger()

    staging_dir_raw = os.getenv("DATA_PIPELINE_STAGING_DIR", "").strip()
    if not staging_dir_raw:
        logger.warning("DATA_PIPELINE_STAGING_DIR is not set; acquisition skipped")
        return []

    staging_dir = Path(staging_dir_raw).expanduser().resolve()
    if not staging_dir.exists() or not staging_dir.is_dir():
        logger.warning(
            "staging directory does not exist or is not a directory: %s; acquisition skipped",
            staging_dir,
        )
        return []

    documents: list[dict[str, str]] = []
    for file_path in sorted(staging_dir.rglob("*")):
        if not file_path.is_file():
            continue

        suffix = file_path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            logger.debug("skipping unsupported extension: %s", file_path)
            continue

        documents.append(
            {
                "source_id": _build_source_id(staging_dir, file_path),
                "path": str(file_path),
            }
        )

    if not documents:
        logger.info("staging scan completed: 0 supported documents found in %s", staging_dir)
        return []

    logger.info("staging scan completed: %d documents found in %s", len(documents), staging_dir)
    return documents
