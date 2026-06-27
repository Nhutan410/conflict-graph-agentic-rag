"""Re-exports from pipeline for backward compatibility with adapter.* imports."""

from query_generation_loop.pipeline import (  # noqa: F401
    build_conflict_records,
    has_regions_for_query,
    infer_query_id_from_claim_id,
    is_contradiction_edge,
    load_claims_by_id,
    load_edges,
    load_jsonl,
    load_queries,
    load_region_conflicts,
)
