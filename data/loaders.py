"""
data/loaders.py
Adapter: RAMDocs preprocessed JSONL → Pydantic models (Claim, Document).
"""

import json
from pathlib import Path

from src.schema import Claim, Document

PREPROCESSED_DIR = Path(__file__).parent / "preprocessed"


def load_claims(query_id: str, preprocessed_dir=None) -> list[Claim]:
    """Load claims cho 1 query từ preprocessed/claims.jsonl.

    Args:
        query_id: Query identifier, e.g. "q_000".
        preprocessed_dir: Optional override path to preprocessed directory.

    Returns:
        List of Claim objects for the given query_id.
    """
    d = Path(preprocessed_dir or PREPROCESSED_DIR)
    # claim_id format: c_q000_d00_s00
    idx = query_id.split("_")[-1]  # "000"
    claims = []
    with open(d / "claims.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            if rec["claim_id"].startswith(f"c_q{idx}"):
                claims.append(_claim_from_record(rec))
    return claims


def load_all_claims(preprocessed_dir=None) -> list[Claim]:
    """Load all representative claims từ preprocessed/claims.jsonl.

    Args:
        preprocessed_dir: Optional override path to preprocessed directory.

    Returns:
        List of all representative Claim objects.
    """
    d = Path(preprocessed_dir or PREPROCESSED_DIR)
    claims = []
    with open(d / "claims.jsonl") as f:
        for line in f:
            rec = json.loads(line.strip())
            if rec.get("is_representative", True):
                claims.append(_claim_from_record(rec))
    return claims


def load_queries(preprocessed_dir=None) -> list[dict]:
    """Load all queries từ preprocessed/queries.jsonl.

    Args:
        preprocessed_dir: Optional override path to preprocessed directory.

    Returns:
        List of query dicts with keys "query_id" and "user_query".
    """
    d = Path(preprocessed_dir or PREPROCESSED_DIR)
    queries = []
    with open(d / "queries.jsonl") as f:
        for line in f:
            queries.append(json.loads(line.strip()))
    return queries


def load_documents_for_query(query_id: str, preprocessed_dir=None) -> list[Document]:
    """Load documents cho 1 query từ preprocessed/documents.jsonl.

    Args:
        query_id: Query identifier, e.g. "q_000".
        preprocessed_dir: Optional override path to preprocessed directory.

    Returns:
        List of Document objects for the given query_id.
    """
    d = Path(preprocessed_dir or PREPROCESSED_DIR)
    with open(d / "documents.jsonl") as f:
        for line in f:
            rec = json.loads(line.strip())
            if rec["query_id"] == query_id:
                return [_doc_from_record(doc) for doc in rec["documents"]]
    return []


def load_metadata(preprocessed_dir=None) -> dict:
    """Load ground truth metadata. CHI dung cho evaluation.

    Args:
        preprocessed_dir: Optional override path to preprocessed directory.

    Returns:
        Dict mapping query_id → metadata record.
    """
    d = Path(preprocessed_dir or PREPROCESSED_DIR)
    meta = {}
    with open(d / "metadata.jsonl") as f:
        for line in f:
            rec = json.loads(line.strip())
            meta[rec["query_id"]] = rec
    return meta


def _claim_from_record(rec: dict) -> Claim:
    """Convert a raw JSONL record to a Claim Pydantic model."""
    return Claim(
        claim_id=rec["claim_id"],
        doc_id=rec["doc_id"],
        text=rec["claim_text"],                        # claim_text → text
        embedding=rec.get("claim_embedding"),          # claim_embedding → embedding
        retrieval_relevance=rec.get("retrieval_relevance", -1.0),
        claim_confidence=rec.get("claim_confidence", -1.0),
        source_credibility=-1.0,
    )


def _doc_from_record(rec: dict) -> Document:
    """Convert a raw JSONL record to a Document Pydantic model."""
    return Document(
        doc_id=rec["doc_id"],
        source=rec.get("source_id", "unknown"),
        text=rec["text"],
        url=rec.get("metadata", {}).get("url"),
        credibility_score=-1.0,
    )
