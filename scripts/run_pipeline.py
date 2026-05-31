"""
scripts/run_pipeline.py
CLI entry point for the Conflict-Aware RAG Pipeline.

Usage:
    python scripts/run_pipeline.py --query_id q_000 --limit 5 --output outputs/result.json

Environment:
    OPENAI_API_KEY: Required for LLM calls (Phase 1 extraction + Phase 7 generation).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add project root to path so "src" is importable
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env nếu có (trước mọi import dùng env vars)
_env_file = PROJECT_ROOT / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

from data.loaders import load_claims, load_queries, load_documents_for_query
from src.pipeline import ConflictAwarePipeline, PipelineConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Conflict-Aware RAG Pipeline on RAMDocs data."
    )
    parser.add_argument(
        "--query_id",
        type=str,
        default=None,
        help="Query ID to run (e.g. 'q_000'). If not set, runs first query.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of claims to use (for faster testing).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save JSON result. If not set, prints to stdout.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file. Defaults to configs/default.yaml.",
    )
    parser.add_argument(
        "--use_documents",
        action="store_true",
        help="If set, run from raw documents (Phase 1 extraction). Otherwise use preprocessed claims.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Configure logging
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Check OpenAI API key
    if not os.environ.get("OPENAI_API_KEY"):
        logger.warning(
            "OPENAI_API_KEY not set. LLM calls (Phase 1, Phase 7) will fail. "
            "Use --use_documents=False to skip Phase 1 when using preprocessed claims."
        )

    # Load config
    config_path = args.config or str(PROJECT_ROOT / "configs" / "default.yaml")
    if Path(config_path).exists():
        logger.info("Loading config from: %s", config_path)
        pipeline = ConflictAwarePipeline.from_config(config_path)
    else:
        logger.info("Config not found at %s, using defaults", config_path)
        pipeline = ConflictAwarePipeline(PipelineConfig())

    # Load queries to determine which query to run
    queries = load_queries()
    if not queries:
        logger.error("No queries found in preprocessed data.")
        sys.exit(1)

    if args.query_id:
        query_rec = next((q for q in queries if q["query_id"] == args.query_id), None)
        if query_rec is None:
            logger.error("Query ID '%s' not found. Available: %s",
                         args.query_id, [q["query_id"] for q in queries[:5]])
            sys.exit(1)
    else:
        query_rec = queries[0]
        logger.info("No query_id specified, using first query: %s", query_rec["query_id"])

    query_id = query_rec["query_id"]
    query_text = query_rec["user_query"]
    logger.info("Running pipeline for query_id=%s: '%s'", query_id, query_text[:100])

    # Load data
    if args.use_documents:
        # Phase 1: extract from raw documents
        documents = load_documents_for_query(query_id)
        if not documents:
            logger.error("No documents found for query_id=%s", query_id)
            sys.exit(1)
        logger.info("Loaded %d documents for query %s", len(documents), query_id)
        result = pipeline.run(query=query_text, query_id=query_id, documents=documents)
    else:
        # Use preprocessed claims (skip Phase 1)
        claims = load_claims(query_id)
        if not claims:
            logger.error("No claims found for query_id=%s", query_id)
            sys.exit(1)
        if args.limit:
            claims = claims[:args.limit]
            logger.info("Limited to %d claims", len(claims))
        else:
            logger.info("Loaded %d claims for query %s", len(claims), query_id)
        result = pipeline.run(query=query_text, query_id=query_id, claims=claims)

    # Format output
    output_data = {
        "query_id": result.query_id,
        "query": query_text,
        "resolved": result.resolved,
        "iterations_run": result.iterations_run,
        "final_answer": result.final_answer,
        "n_validated_claims": len(result.validated_claims),
        "n_conflict_localizations": len(result.conflict_localizations),
        "conflict_localizations": [
            {
                "claim_i_id": loc.claim_i_id,
                "claim_j_id": loc.claim_j_id,
                "slot": loc.slot,
                "value_i": loc.value_i,
                "value_j": loc.value_j,
                "conflict_intensity": loc.conflict_intensity,
                "credibility_i": loc.credibility_i,
                "credibility_j": loc.credibility_j,
            }
            for loc in result.conflict_localizations
        ],
        "validated_claims": [
            {
                "claim_id": c.claim_id,
                "doc_id": c.doc_id,
                "text": c.text,
                "retrieval_relevance": c.retrieval_relevance,
                "claim_confidence": c.claim_confidence,
            }
            for c in result.validated_claims
        ],
    }

    # Save or print output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        logger.info("Result saved to: %s", output_path)
    else:
        print(json.dumps(output_data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
