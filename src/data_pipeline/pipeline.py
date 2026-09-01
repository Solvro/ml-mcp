import os
from hashlib import sha256
from typing import NamedTuple

from dotenv import load_dotenv
from prefect import flow, get_run_logger
from prefect.futures import as_completed

from src.data_pipeline.flows.data_acquisition import acquire_data
from src.data_pipeline.flows.graph_populating import (
    GraphPopulator,
    claim_document_for_processing,
    populate_graph,
)
from src.data_pipeline.flows.llm_cypher_generation import (
    generate_cypher_queries,
    missed_row_passes_used,
    reset_missed_row_passes,
)
from src.data_pipeline.flows.ocr_extraction import ocr_extraction
from src.data_pipeline.flows.schema_reflection import reflect_on_schema
from src.data_pipeline.graph_dump import (
    ensure_host_dump_dir,
    export_graph_to_cypher,
    host_dump_path,
    import_graph_from_cypher_dump,
)
from src.data_pipeline.staging import relative_path_from_source_id, source_id_for


class PipelineOutcome(NamedTuple):
    """What one pipeline run confirmed.

    Attributes:
        processed: Document-level source ids confirmed to be in the graph.
        deleted: Document-level source ids whose Source nodes were detached and
            whose orphaned entities were removed.
    """

    processed: set[str]
    deleted: set[str]


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


def _document_id(source_id: str) -> str:
    """Strip the page fragment so a page id maps back to its document."""
    return source_id.split("#", 1)[0]


def _filter_acquired_by_changed(
    acquired: list[dict[str, str]],
    changed: list[str],
) -> tuple[list[dict[str, str]], set[str]]:
    """Keep only acquired docs whose relative path is present in ``changed``."""
    changed_set = {str(path).strip().replace("\\", "/") for path in changed if str(path).strip()}

    if not changed_set:
        return [], set()

    selected: list[dict[str, str]] = []
    matched: set[str] = set()

    for item in acquired:
        if not isinstance(item, dict):
            continue
        source_id = str(item.get("source_id", "")).strip()
        relative = relative_path_from_source_id(source_id)
        if not relative:
            continue
        if relative in changed_set:
            selected.append(item)
            matched.add(relative)

    return selected, changed_set - matched


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


def _normalize_deleted_source_ids(deleted: list[str] | None) -> set[str]:
    if not deleted:
        return set()
    out: set[str] = set()
    for path in deleted:
        rel = str(path).strip().replace("\\", "/")
        if rel:
            out.add(source_id_for(rel))
    return out


def _prune_source_hashes_for_deleted(
    source_hashes: dict[str, str],
    deleted_document_source_ids: set[str],
) -> dict[str, str]:
    if not deleted_document_source_ids:
        return dict(source_hashes)
    return {
        sid: h
        for sid, h in source_hashes.items()
        if _document_id(sid) not in deleted_document_source_ids
    }


@flow(log_prints=True)
def data_pipeline_flow(
    changed: list[str] | None = None,
    deleted: list[str] | None = None,
) -> PipelineOutcome:
    """Agentic graph extraction loop with batched parallel processing.

    Each page is processed through the same generate -> populate chain.
    Pages are submitted in batches so concurrency is bounded by
    DATA_PIPELINE_MAX_CONCURRENCY.

    Once all batches finish, schema reflection runs once to summarize the final
    graph state for observability.

    Args:
        changed: List of staging-relative POSIX paths for new/changed documents.
            None means full scan (legacy/manual behavior).
        deleted: List of staging-relative POSIX paths removed upstream.

    Returns:
        Outcome of this run. ``processed`` holds document-level source ids
        (``file://relative/path``, no page fragment) confirmed to be in the
        graph; a document whose extraction produced nothing, or any of whose
        pages failed, is omitted so the caller retries it. ``deleted`` holds
        ids whose removal completed; an id missing from it keeps its manifest
        entry, so the deletion is reported again on the next run.
    """
    load_dotenv()
    logger = get_run_logger()
    changed_summary = "full_scan" if changed is None else f"{len(changed)} paths"
    deleted_count = 0 if deleted is None else len(deleted)
    logger.info(
        "Pipeline trigger payload: changed=%s deleted=%d",
        changed_summary,
        deleted_count,
    )
    populator = GraphPopulator()
    populator.ensure_entity_key_indexes()

    requested_deleted_source_ids = _normalize_deleted_source_ids(deleted)
    confirmed_deleted_source_ids: set[str] = set()
    restored_from_dump = False

    # A dump is a bootstrap for a fresh database only; once the graph has any
    # data, scheduled runs must keep extracting instead of restoring.
    if host_dump_path().is_file() and not populator.graph_has_data():
        try:
            import_graph_from_cypher_dump()
            populator.record_restore_run()
            restored_from_dump = True
            logger.info(
                "Loaded graph from dump; applying requested deletions before finishing run."
            )
        except Exception as exc:
            logger.warning("Dump restore failed (%s); continuing with extraction", exc)

    if requested_deleted_source_ids:
        confirmed_deleted_source_ids = populator.delete_sources_for_documents(
            sorted(requested_deleted_source_ids)
        )

    if restored_from_dump:
        return PipelineOutcome(processed=set(), deleted=confirmed_deleted_source_ids)

    acquired = acquire_data()

    last_source_hashes = populator.get_latest_pipeline_source_hashes()
    pruned_last_source_hashes = _prune_source_hashes_for_deleted(
        last_source_hashes,
        requested_deleted_source_ids,
    )
    if changed is not None:
        filtered_acquired, missing_paths = _filter_acquired_by_changed(acquired, changed)

        for path in sorted(missing_paths):
            logger.info("Changed path not present in staging scan, skipping: %s", path)

        if not filtered_acquired:
            logger.info("Trigger reported no matching changed documents; skipping extraction")
            if requested_deleted_source_ids:
                populator.record_pipeline_run(pruned_last_source_hashes, mode="incremental")
                ensure_host_dump_dir()
                export_graph_to_cypher()
            return PipelineOutcome(processed=set(), deleted=confirmed_deleted_source_ids)

        logger.info(
            "Changed-path filter selected %d/%d acquired documents",
            len(filtered_acquired),
            len(acquired),
        )
        acquired = filtered_acquired

    attempted_documents = {
        _document_id(str(item["source_id"]))
        for item in acquired
        if isinstance(item, dict) and item.get("source_id")
    }
    extracted = ocr_extraction(acquired)
    source_pages = _normalize_source_pages(extracted)
    if not source_pages:
        logger.warning("No non-empty pages found; stopping pipeline early")
        return PipelineOutcome(processed=set(), deleted=confirmed_deleted_source_ids)

    all_documents = {_document_id(sid) for sid, _ in source_pages}
    full_source_hashes = {sid: _compute_page_hash(text) for sid, text in source_pages}
    work_items: list[tuple[str, str]] = [
        (sid, text)
        for sid, text in source_pages
        if pruned_last_source_hashes.get(sid) != full_source_hashes[sid]
    ]
    if not work_items:
        logger.info("No new or changed source files since last pipeline run")
        if requested_deleted_source_ids:
            populator.record_pipeline_run(pruned_last_source_hashes, mode="incremental")
            ensure_host_dump_dir()
            export_graph_to_cypher()
        return PipelineOutcome(processed=all_documents, deleted=confirmed_deleted_source_ids)

    written_keys: set[str] = set()
    reset_missed_row_passes()
    stats = {
        "total_pages": len(work_items),
        "claimed_pages": 0,
        "submitted_pages": 0,
        "skipped_duplicates": 0,
        "successful_pages": 0,
        "failed_pages": 0,
        "claim_errors": 0,
        "schema_snapshot_chars": 0,
        "schema_final_chars": 0,
        "missed_row_passes": 0,
    }

    failed_documents: set[str] = set()
    max_concurrency = _get_max_concurrency()
    logger.info(
        "Submitting %d pages (incremental from %d sources) with max concurrency %d",
        len(work_items),
        len(source_pages),
        max_concurrency,
    )

    # Race-safe schema strategy: capture once before fan-out and reuse for all pages.
    schema_context = _safe_reflect_schema(phase="before_parallel_batches", logger=logger)
    stats["schema_snapshot_chars"] = len(schema_context)

    for batch_start in range(0, len(work_items), max_concurrency):
        batch_end = min(batch_start + max_concurrency, len(work_items))
        logger.info("Processing batch pages %d-%d", batch_start + 1, batch_end)

        claim_futures = []
        claim_lookup = {}
        batch_futures = []

        for page_index, (source_id, page_content) in enumerate(
            work_items[batch_start:batch_end],
            start=batch_start + 1,
        ):
            page_hash = _compute_page_hash(page_content)
            claim_future = claim_document_for_processing.submit(page_hash)
            claim_futures.append(claim_future)
            claim_lookup[id(claim_future)] = (page_index, source_id, page_content, page_hash)

        for claim_future in as_completed(claim_futures):
            page_index, source_id, page_content, page_hash = claim_lookup[id(claim_future)]
            try:
                is_claimed = claim_future.result()
            except Exception as exc:
                stats["failed_pages"] += 1
                stats["claim_errors"] += 1
                logger.error(
                    "Claim failed for page %d / %d (hash=%s): %s",
                    page_index,
                    len(work_items),
                    page_hash,
                    exc,
                )
                failed_documents.add(_document_id(source_id))
                continue

            if not is_claimed:
                stats["skipped_duplicates"] += 1
                logger.info(
                    "Skipping page %d / %d (already processed hash=%s)",
                    page_index,
                    len(work_items),
                    page_hash,
                )
                try:
                    populator.link_processed_document_to_source(page_hash, source_id)
                except Exception as exc:
                    logger.warning(
                        "Failed to link skipped hash %s to source %s: %s",
                        page_hash,
                        source_id,
                        exc,
                    )
                continue

            stats["claimed_pages"] += 1
            logger.info("Submitting page %d / %d", page_index, len(work_items))
            cypher_future = generate_cypher_queries.submit(page_content, schema_context)
            populate_future = populate_graph.submit(cypher_future, page_hash, source_id)
            stats["submitted_pages"] += 1
            batch_futures.append((page_index, source_id, page_hash, populate_future))

        for page_index, source_id, page_hash, future in batch_futures:
            try:
                written_keys.update(future.result() or [])
                stats["successful_pages"] += 1
            except Exception as exc:
                stats["failed_pages"] += 1
                logger.error(
                    "Processing failed for page %d / %d (hash=%s): %s",
                    page_index,
                    len(work_items),
                    page_hash,
                    exc,
                )
                failed_documents.add(_document_id(source_id))

    stats["missed_row_passes"] = missed_row_passes_used()

    # Repair before reflecting, so the summary describes the deduplicated graph. Only the keys
    # this run wrote are examined, so the cost tracks what changed rather than the graph size;
    # `uv run dedup-graph` runs the full repair for nodes written before these rules existed.
    dedup_stats = populator.deduplicate_entities(sorted(written_keys))
    logger.info(
        "Deduplication over %d key(s) written this run: groups_merged=%d",
        len(written_keys),
        dedup_stats["groups_merged"],
    )

    # Run one final reflection only after all futures have settled.
    final_schema = _safe_reflect_schema(phase="after_parallel_batches", logger=logger)
    stats["schema_final_chars"] = len(final_schema)
    logger.info("Pipeline complete. Final schema summary length: %d", stats["schema_final_chars"])
    logger.info(
        "Pipeline summary: total=%d claimed=%d submitted=%d success=%d "
        "skipped_duplicates=%d failed=%d claim_errors=%d "
        "schema_snapshot_chars=%d schema_final_chars=%d missed_row_passes=%d",
        stats["total_pages"],
        stats["claimed_pages"],
        stats["submitted_pages"],
        stats["successful_pages"],
        stats["skipped_duplicates"],
        stats["failed_pages"],
        stats["claim_errors"],
        stats["schema_snapshot_chars"],
        stats["schema_final_chars"],
        stats["missed_row_passes"],
    )

    if stats["failed_pages"] > 0:
        logger.warning("Pipeline finished with %d failed pages", stats["failed_pages"])

    if stats["failed_pages"] == 0 and stats["claim_errors"] == 0:
        if changed is not None:
            mode = "incremental"
        else:
            mode = "incremental" if len(work_items) < len(source_pages) else "full"
        merged_source_hashes = {
            sid: page_hash
            for sid, page_hash in pruned_last_source_hashes.items()
            if _document_id(sid) not in attempted_documents
        }
        merged_source_hashes.update(full_source_hashes)
        populator.record_pipeline_run(merged_source_hashes, mode=mode)

    # Export even on partial failure so the host dump reflects the latest good graph.
    ensure_host_dump_dir()
    export_graph_to_cypher()

    processed_documents = all_documents - failed_documents
    logger.info(
        "Pipeline complete. Graph is ready for querying. Confirmed %d / %d documents",
        len(processed_documents),
        len(all_documents),
    )
    return PipelineOutcome(processed=processed_documents, deleted=confirmed_deleted_source_ids)


if __name__ == "__main__":
    data_pipeline_flow()
