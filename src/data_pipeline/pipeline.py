import os
from hashlib import sha256

from dotenv import load_dotenv
from prefect import flow, get_run_logger
from prefect.futures import as_completed

from src.data_pipeline.flows.data_acquisition import acquire_data
from src.data_pipeline.flows.graph_populating import (
    GraphPopulator,
    claim_document_for_processing,
    populate_graph,
)
from src.data_pipeline.flows.llm_cypher_generation import generate_cypher_queries
from src.data_pipeline.flows.ocr_extraction import ocr_extraction
from src.data_pipeline.flows.schema_reflection import reflect_on_schema
from src.data_pipeline.graph_dump import (
    ensure_host_dump_dir,
    export_graph_to_cypher,
    host_dump_path,
    import_graph_from_cypher_dump,
)


def _get_max_concurrency() -> int:
    """Read max concurrency from env with a safe fallback."""
    raw_value = os.getenv("DATA_PIPELINE_MAX_CONCURRENCY", "4").strip()
    try:
        parsed_value = int(raw_value)
    except ValueError:
        return 4
    return max(1, parsed_value)


def _compute_page_hash(page_content: str) -> str:
    """Create a stable idempotency hash for one page of content."""
    normalized_content = page_content.strip().encode("utf-8")
    return sha256(normalized_content).hexdigest()


def _normalize_source_pages(extracted: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Normalize OCR output to non-empty (source_id, content) pairs."""
    if not isinstance(extracted, list):
        return []
    out: list[tuple[str, str]] = []
    for item in extracted:
        if not isinstance(item, tuple) or len(item) != 2:
            continue
        source_id, content = item
        source_id_str = str(source_id).strip()
        content_str = str(content).strip()
        if source_id_str and content_str:
            out.append((source_id_str, content_str))
    return out


def _safe_reflect_schema(phase: str, logger) -> str:
    """Reflect schema once and degrade safely on transient errors."""
    try:
        schema_summary = reflect_on_schema()
    except Exception as exc:
        logger.warning("Schema reflection failed during %s: %s", phase, exc)
        return ""

    if not schema_summary:
        logger.info("Schema reflection during %s returned empty summary", phase)
        return ""

    logger.info(
        "Schema reflection during %s produced %d chars",
        phase,
        len(schema_summary),
    )
    return schema_summary


@flow(log_prints=True)
def data_pipeline_flow():
    """Agentic graph extraction loop with batched parallel processing.

    Each page is processed through the same generate -> populate chain.
    Pages are submitted in batches so concurrency is bounded by
    DATA_PIPELINE_MAX_CONCURRENCY.

    Once all batches finish, schema reflection runs once to summarize the final
    graph state for observability.
    """
    load_dotenv()
    logger = get_run_logger()
    populator = GraphPopulator()

    # A dump is a bootstrap for a fresh database only; once the graph has any
    # data, scheduled runs must keep extracting instead of restoring.
    if host_dump_path().is_file() and not populator.graph_has_data():
        try:
            import_graph_from_cypher_dump()
            populator.record_restore_run()
            logger.info("Loaded graph from dump; skipped LLM extraction.")
            return
        except Exception as exc:
            logger.warning("Dump restore failed (%s); continuing with extraction", exc)

    acquired = acquire_data()
    extracted = ocr_extraction(acquired)
    source_pages = _normalize_source_pages(extracted)
    if not source_pages:
        logger.warning("No non-empty pages found; stopping pipeline early")
        return

    last_source_hashes = populator.get_latest_pipeline_source_hashes()
    full_source_hashes = {sid: _compute_page_hash(text) for sid, text in source_pages}
    work_items: list[tuple[str, str]] = [
        (sid, text)
        for sid, text in source_pages
        if last_source_hashes.get(sid) != full_source_hashes[sid]
    ]
    if not work_items:
        logger.info("No new or changed source files since last pipeline run")
        return

    pages = [text for _, text in work_items]

    stats = {
        "total_pages": len(pages),
        "claimed_pages": 0,
        "submitted_pages": 0,
        "skipped_duplicates": 0,
        "successful_pages": 0,
        "failed_pages": 0,
        "claim_errors": 0,
        "schema_snapshot_chars": 0,
        "schema_final_chars": 0,
    }

    max_concurrency = _get_max_concurrency()
    logger.info(
        "Submitting %d pages (incremental from %d sources) with max concurrency %d",
        len(pages),
        len(source_pages),
        max_concurrency,
    )

    # Race-safe schema strategy: capture once before fan-out and reuse for all pages.
    schema_context = _safe_reflect_schema(phase="before_parallel_batches", logger=logger)
    stats["schema_snapshot_chars"] = len(schema_context)

    for batch_start in range(0, len(pages), max_concurrency):
        batch_end = min(batch_start + max_concurrency, len(pages))
        logger.info("Processing batch pages %d-%d", batch_start + 1, batch_end)

        claim_futures = []
        claim_lookup = {}
        batch_futures = []

        for page_index, page_content in enumerate(
            pages[batch_start:batch_end],
            start=batch_start + 1,
        ):
            page_hash = _compute_page_hash(page_content)
            claim_future = claim_document_for_processing.submit(page_hash)
            claim_futures.append(claim_future)
            claim_lookup[id(claim_future)] = (page_index, page_content, page_hash)

        for claim_future in as_completed(claim_futures):
            page_index, page_content, page_hash = claim_lookup[id(claim_future)]
            try:
                is_claimed = claim_future.result()
            except Exception as exc:
                stats["failed_pages"] += 1
                stats["claim_errors"] += 1
                logger.error(
                    "Claim failed for page %d / %d (hash=%s): %s",
                    page_index,
                    len(pages),
                    page_hash,
                    exc,
                )
                continue

            if not is_claimed:
                stats["skipped_duplicates"] += 1
                logger.info(
                    "Skipping page %d / %d (already processed hash=%s)",
                    page_index,
                    len(pages),
                    page_hash,
                )
                continue

            stats["claimed_pages"] += 1
            logger.info("Submitting page %d / %d", page_index, len(pages))
            cypher_future = generate_cypher_queries.submit(page_content, schema_context)
            populate_future = populate_graph.submit(cypher_future, page_hash)
            stats["submitted_pages"] += 1
            batch_futures.append((page_index, page_hash, populate_future))

        for page_index, page_hash, future in batch_futures:
            try:
                future.result()
                stats["successful_pages"] += 1
            except Exception as exc:
                stats["failed_pages"] += 1
                logger.error(
                    "Processing failed for page %d / %d (hash=%s): %s",
                    page_index,
                    len(pages),
                    page_hash,
                    exc,
                )

    # Run one final reflection only after all futures have settled.
    final_schema = _safe_reflect_schema(phase="after_parallel_batches", logger=logger)
    stats["schema_final_chars"] = len(final_schema)
    logger.info("Pipeline complete. Final schema summary length: %d", stats["schema_final_chars"])
    logger.info(
        "Pipeline summary: total=%d claimed=%d submitted=%d success=%d "
        "skipped_duplicates=%d failed=%d claim_errors=%d "
        "schema_snapshot_chars=%d schema_final_chars=%d",
        stats["total_pages"],
        stats["claimed_pages"],
        stats["submitted_pages"],
        stats["successful_pages"],
        stats["skipped_duplicates"],
        stats["failed_pages"],
        stats["claim_errors"],
        stats["schema_snapshot_chars"],
        stats["schema_final_chars"],
    )

    if stats["failed_pages"] > 0:
        logger.warning("Pipeline finished with %d failed pages", stats["failed_pages"])

    if stats["failed_pages"] == 0 and stats["claim_errors"] == 0:
        mode = "incremental" if len(work_items) < len(source_pages) else "full"
        populator.record_pipeline_run(full_source_hashes, mode=mode)

    # Export even on partial failure so the host dump reflects the latest good graph.
    ensure_host_dump_dir()
    export_graph_to_cypher()

    logger.info("Pipeline complete. Graph is ready for querying.")


if __name__ == "__main__":
    data_pipeline_flow()
