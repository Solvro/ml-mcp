"""Graph labels owned by pipeline internals, not user-facing entities."""

# These labels are infrastructure/provenance bookkeeping and must stay outside:
# - ingestion label-vocabulary relabeling,
# - duplicate-merge key backfill/merge passes,
# - full-text retrieval index label sets.
SYSTEM_LABELS = frozenset({"ProcessedDocument", "PipelineRun", "Source"})
