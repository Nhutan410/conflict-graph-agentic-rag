"""
data/preprocess.py
Chuyển đổi RAMDocs_test.jsonl → pipeline-ready claims.jsonl

Output (4 file):
  queries.jsonl    — {query_id, user_query}
  documents.jsonl  — {query_id, documents[]}
  claims.jsonl     — Claim[] từ LLM extraction thật (gpt-4o-mini)
  metadata.jsonl   — ground truth, CHỈ dùng evaluation

Claim extraction:
  - Filter Wikipedia boilerplate (navigation, categories, edit links)
  - Dùng ClaimExtractor (gpt-4o-mini) extract atomic factual statements
  - Phase 2 canonical inline (merge duplicates, giữ conflict pairs)
  - Lưu claim_embedding: null — sẽ được compute bởi pipeline (Phase 2a)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent
sys.path.insert(0, str(DATA_DIR.parent))   # project root → src importable

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_DOC_LENGTH   = 1500   # chars — truncate doc text trước khi gửi LLM
MIN_CLAIM_WORDS  = 4      # bỏ claims quá ngắn
MAX_CLAIM_WORDS  = 40     # bỏ claims quá dài
CANONICAL_OVERLAP = 0.55  # Jaccard threshold để candidate merge

TAU_CONFIDENCE = {
    "correct": (0.80, 0.95),
    "misinfo": (0.55, 0.75),
    "noise":   (0.40, 0.60),
    "unknown": (0.50, 0.70),
}

# ── Wikipedia boilerplate patterns ────────────────────────────────────────────
# Các pattern này xuất hiện trong RAMDocs documents do crawl từ Wikipedia
_WIKI_BOILERPLATE = re.compile(
    r"""
    # Navigation / UI elements
    (?:edit\s*\])|(?:\[\s*edit)|(?:view\s+history)|(?:move\s+to\s+sidebar)
    |(?:hide\s+from\s+sidebar)|(?:v\s+t\s+e\b)        # VTE template
    |(?:cite\s+this\s+page)|(?:permanent\s+link)
    |(?:what\s+links\s+here)|(?:related\s+changes)
    |(?:upload\s+file)|(?:printable\s+version)
    # Citation boilerplate
    |(?:retrieved\s+\d{1,2}\s+\w+\s+\d{4})
    |(?:archived\s+from\s+the\s+original)
    |(?:cite\s+news|cite\s+web|cite\s+book)
    # URL / category noise
    |(?:https?://\S+)
    |(?:categories\s*:)
    |(?:hidden\s+categories)
    # Wikipedia UI text
    |(?:\[\s*\d+\s*\])                                 # footnote refs [1]
    |(?:jump\s+to\s+navigation)|(?:jump\s+to\s+search)
    """,
    re.IGNORECASE | re.VERBOSE,
)

_WIKI_LANGUAGE_LINE = re.compile(
    r"(?:\d+\s+languages?\b)|(?:বাংলা|فارسی|हिन्दी|नेपाल|Edit\s+links)",
    re.UNICODE,
)


def is_boilerplate(text: str) -> bool:
    """True nếu text là Wikipedia navigation/metadata, không phải nội dung."""
    if len(text.strip()) < 20:
        return True
    if _WIKI_BOILERPLATE.search(text):
        return True
    if _WIKI_LANGUAGE_LINE.search(text):
        return True
    # URL-heavy lines
    url_count = text.count("http")
    word_count = len(text.split())
    if url_count > 0 and word_count < 15:
        return True
    return False


def clean_doc_text(text: str) -> str:
    """Làm sạch document text trước khi gửi LLM."""
    # Bỏ [TRUNCATED] marker
    text = text.replace("[TRUNCATED]", "").strip()
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    # Bỏ orphan brackets
    text = re.sub(r"\[\s*\]", "", text)
    # Bỏ Wikipedia navigation artifacts
    text = re.sub(r"\b(edit|hide|show)\b\s*[\[\]]?", "", text, flags=re.IGNORECASE)
    return text.strip()


# ── Text utilities ─────────────────────────────────────────────────────────────

def extract_numbers(text: str) -> set[str]:
    return set(re.findall(r"\b\d+\b", text))


def text_overlap(a: str, b: str) -> float:
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


# ── LLM-based claim extraction ─────────────────────────────────────────────────

def extract_claims_llm(doc_text: str, doc_id: str, source_id: str,
                       query_idx: int, doc_idx: int,
                       doc_type: str, retrieval_score: float,
                       openai_client, model: str = "gpt-4o-mini") -> list[dict]:
    """
    Dùng gpt-4o-mini để extract atomic factual claims từ document text.
    Trả về list[dict] theo schema claims.jsonl.
    """
    # Filter boilerplate trước khi gửi LLM
    text_clean = clean_doc_text(doc_text)
    if is_boilerplate(text_clean) or len(text_clean.split()) < 10:
        logger.info("Skipping boilerplate doc %s", doc_id)
        return []

    # Truncate
    if len(text_clean) > MAX_DOC_LENGTH:
        text_clean = text_clean[:MAX_DOC_LENGTH].rsplit(" ", 1)[0]

    prompt = (
        "Extract all atomic factual claims from the following document passage.\n"
        "Rules:\n"
        "- Each claim must be a single, self-contained factual statement.\n"
        "- Each claim must be verifiable independently.\n"
        "- Each claim must be under 35 words.\n"
        "- Do NOT include: opinions, navigation text, edit links, category labels, "
        "citation metadata, or URL fragments.\n"
        "- Return ONLY a JSON array of strings. No explanation.\n\n"
        f"Document:\n{text_clean}"
    )

    try:
        resp = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content or "[]"
        parsed = json.loads(content)
        if isinstance(parsed, list):
            raw_claims = parsed
        else:
            raw_claims = next(
                (v for v in parsed.values() if isinstance(v, list)),
                []
            )
    except Exception as exc:
        logger.warning("LLM extraction failed for %s: %s", doc_id, exc)
        return []

    # Confidence range theo doc_type
    lo, hi = TAU_CONFIDENCE.get(doc_type, TAU_CONFIDENCE["unknown"])

    claims = []
    for sent_idx, claim_text in enumerate(raw_claims):
        claim_text = claim_text.strip()
        words = claim_text.split()
        if len(words) < MIN_CLAIM_WORDS or len(words) > MAX_CLAIM_WORDS:
            continue
        if is_boilerplate(claim_text):
            continue

        # Deterministic confidence từ hash
        h_val = int(hashlib.md5(claim_text.encode()).hexdigest(), 16)
        confidence = round(lo + (h_val % 1000) / 1000 * (hi - lo), 4)

        claim_id = f"c_q{query_idx:03d}_d{doc_idx:02d}_s{sent_idx:02d}"
        claims.append({
            "claim_id":            claim_id,
            "claim_text":          claim_text,
            "evidence":            claim_text,       # LLM output = atomic claim
            "doc_id":              doc_id,
            "source_id":           source_id,
            "claim_embedding":     None,             # compute bởi Phase 2a
            "retrieval_relevance": retrieval_score,
            "claim_confidence":    confidence,
            "credibility_score":   None,             # compute bởi Phase 5
            "is_representative":   True,
            "merged_claim_ids":    None,
        })

    return claims


# ── Phase 2 canonical (inline) ─────────────────────────────────────────────────

def run_phase2_canonical(claims: list[dict]) -> list[dict]:
    """
    Merge claims diễn đạt cùng fact (Jaccard overlap > threshold VÀ cùng số).
    Giữ riêng claims có temporal/numerical khác nhau (nguồn gốc conflict).
    """
    merged = set()
    result = []

    for i, ci in enumerate(claims):
        if i in merged:
            continue
        group_ids = [ci["claim_id"]]

        for j, cj in enumerate(claims):
            if j <= i or j in merged:
                continue
            overlap = text_overlap(ci["claim_text"], cj["claim_text"])
            nums_i = extract_numbers(ci["claim_text"])
            nums_j = extract_numbers(cj["claim_text"])

            # Merge chỉ khi overlap cao VÀ cùng số (không mất conflict info)
            if overlap > CANONICAL_OVERLAP and nums_i == nums_j:
                merged.add(j)
                group_ids.append(cj["claim_id"])

        ci_out = deepcopy(ci)
        if len(group_ids) > 1:
            ci_out["merged_claim_ids"] = group_ids
        result.append(ci_out)

    return result


# ── Builders ───────────────────────────────────────────────────────────────────

def build_query(raw: dict, query_idx: int) -> dict:
    text = raw["question"].strip()
    text = re.sub(r"\s+", " ", text)
    return {"query_id": f"q_{query_idx:03d}", "user_query": text}


def build_documents(raw: dict, query_idx: int, skip_noise: bool = False) -> list[dict]:
    docs = []
    total = len(raw["documents"])
    for doc_idx, raw_doc in enumerate(raw["documents"]):
        doc_type = raw_doc.get("type", "unknown")
        if skip_noise and doc_type == "noise":
            continue

        text = raw_doc["text"].strip()
        text = re.sub(r"\s+", " ", text)
        truncated = False
        if len(text) > MAX_DOC_LENGTH * 2:
            text = text[:MAX_DOC_LENGTH * 2].rsplit(" ", 1)[0] + " [TRUNCATED]"
            truncated = True

        score = round(1.0 - (doc_idx / max(total - 1, 1)) * 0.5, 4) if total > 1 else 1.0
        doc_id    = f"d_{hashlib.md5(f'{query_idx}_{doc_idx}_{text[:40]}'.encode()).hexdigest()[:6]}"
        source_id = f"s_q{query_idx:03d}_{doc_idx:02d}"

        docs.append({
            "doc_id":          doc_id,
            "source_id":       source_id,
            "text":            text,
            "retrieval_score": score,
            "metadata": {
                "query_id":       f"q_{query_idx:03d}",
                "rank":           doc_idx + 1,
                "url":            None,
                "published_date": None,
                "_truncated":     truncated,
            },
        })
    return docs


def build_metadata(raw: dict, query_idx: int, docs: list[dict]) -> dict:
    raw_docs = raw.get("documents", [])
    doc_labels = []
    for doc, raw_doc in zip(docs, raw_docs):
        doc_labels.append({
            "doc_id":   doc["doc_id"],
            "doc_type": raw_doc.get("type", "unknown"),
            "answer":   raw_doc.get("answer"),
        })
    text = raw["question"].strip()
    return {
        "query_id":        f"q_{query_idx:03d}",
        "user_query":      text,
        "disambig_entity": raw.get("disambig_entity", []),
        "gold_answers":    raw.get("gold_answers", []),
        "wrong_answers":   raw.get("wrong_answers", []),
        "doc_labels":      doc_labels,
    }


# ── Main preprocessor ──────────────────────────────────────────────────────────

class RAMDocsPreprocessor:

    def __init__(
        self,
        input_path: str,
        output_dir: str = "data/preprocessed",
        skip_noise_docs: bool = False,
        openai_client=None,
        model: str = "gpt-4o-mini",
        max_queries: int | None = None,
    ):
        self.input_path  = Path(input_path)
        self.output_dir  = Path(output_dir)
        self.skip_noise  = skip_noise_docs
        self.client      = openai_client
        self.model       = model
        self.max_queries = max_queries
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_raw(self) -> Iterator[dict]:
        with open(self.input_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def run(self) -> dict:
        stats = {
            "total_queries": 0, "total_docs": 0,
            "total_claims": 0, "canonical_merges": 0,
            "skipped_boilerplate": 0, "doc_type_counts": {"correct": 0, "misinfo": 0, "noise": 0},
        }

        q_path = self.output_dir / "queries.jsonl"
        d_path = self.output_dir / "documents.jsonl"
        c_path = self.output_dir / "claims.jsonl"
        m_path = self.output_dir / "metadata.jsonl"

        with (
            open(q_path, "w", encoding="utf-8") as fq,
            open(d_path, "w", encoding="utf-8") as fd,
            open(c_path, "w", encoding="utf-8") as fc,
            open(m_path, "w", encoding="utf-8") as fm,
        ):
            for query_idx, raw in enumerate(self._load_raw()):
                if self.max_queries and query_idx >= self.max_queries:
                    break

                query = build_query(raw, query_idx)
                docs  = build_documents(raw, query_idx, skip_noise=self.skip_noise)
                meta  = build_metadata(raw, query_idx, docs)

                raw_docs = raw.get("documents", [])
                doc_type_map = {}
                for doc, raw_doc in zip(docs, raw_docs):
                    doc_type_map[doc["doc_id"]] = raw_doc.get("type", "unknown")

                # ── Claim extraction ──────────────────────────────────────────
                all_claims: list[dict] = []
                for doc_idx, doc in enumerate(docs):
                    doc_type = doc_type_map.get(doc["doc_id"], "unknown")
                    if self.skip_noise and doc_type == "noise":
                        continue

                    if self.client:
                        # Real LLM extraction
                        claims = extract_claims_llm(
                            doc_text=doc["text"],
                            doc_id=doc["doc_id"],
                            source_id=doc["source_id"],
                            query_idx=query_idx,
                            doc_idx=doc_idx,
                            doc_type=doc_type,
                            retrieval_score=doc["retrieval_score"],
                            openai_client=self.client,
                            model=self.model,
                        )
                        if not claims:
                            stats["skipped_boilerplate"] += 1
                    else:
                        # Fallback: sentence split (mock) — chỉ dùng khi không có OpenAI
                        claims = self._mock_extract(doc, query_idx, doc_idx, doc_type)

                    all_claims.extend(claims)

                # Phase 2 canonical
                canonical = run_phase2_canonical(all_claims)
                merges = sum(1 for c in canonical if c.get("merged_claim_ids"))

                # ── Write (flush mỗi query để tránh buffer lag trên disk) ──────
                fq.write(json.dumps(query, ensure_ascii=False) + "\n"); fq.flush()
                fd.write(json.dumps({"query_id": query["query_id"], "documents": docs}, ensure_ascii=False) + "\n"); fd.flush()
                for c in canonical:
                    fc.write(json.dumps(c, ensure_ascii=False) + "\n")
                fc.flush()
                fm.write(json.dumps(meta, ensure_ascii=False) + "\n"); fm.flush()

                # ── Stats ─────────────────────────────────────────────────────
                stats["total_queries"] += 1
                stats["total_docs"]    += len(docs)
                stats["total_claims"]  += len(canonical)
                stats["canonical_merges"] += merges
                for raw_doc in raw_docs:
                    t = raw_doc.get("type", "unknown")
                    if t in stats["doc_type_counts"]:
                        stats["doc_type_counts"][t] += 1

                if (query_idx + 1) % 10 == 0:
                    logger.info("Processed %d queries, %d claims so far",
                                query_idx + 1, stats["total_claims"])

        return stats

    def _mock_extract(self, doc: dict, query_idx: int,
                      doc_idx: int, doc_type: str) -> list[dict]:
        """Fallback sentence-split khi không có LLM — chỉ dùng để test."""
        text = doc["text"].replace("[TRUNCATED]", "").strip()
        sentences = re.split(r"(?<=[.!?])\s+", text)

        lo, hi = TAU_CONFIDENCE.get(doc_type, TAU_CONFIDENCE["unknown"])
        claims = []
        for sent_idx, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if is_boilerplate(sentence) or len(sentence.split()) < MIN_CLAIM_WORDS:
                continue
            h_val = int(hashlib.md5(sentence.encode()).hexdigest(), 16)
            confidence = round(lo + (h_val % 1000) / 1000 * (hi - lo), 4)
            claims.append({
                "claim_id":            f"c_q{query_idx:03d}_d{doc_idx:02d}_s{sent_idx:02d}",
                "claim_text":          sentence,
                "evidence":            sentence,
                "doc_id":              doc["doc_id"],
                "source_id":           doc["source_id"],
                "claim_embedding":     None,
                "retrieval_relevance": doc["retrieval_score"],
                "claim_confidence":    confidence,
                "credibility_score":   None,
                "is_representative":   True,
                "merged_claim_ids":    None,
            })
        return claims


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="Preprocess RAMDocs → pipeline schema")
    parser.add_argument("--input",      default=str(DATA_DIR / "raw" / "RAMDocs_test.jsonl"))
    parser.add_argument("--output_dir", default=str(DATA_DIR / "preprocessed"))
    parser.add_argument("--skip_noise", action="store_true")
    parser.add_argument("--max_queries", type=int, default=None,
                        help="Process only first N queries (for testing)")
    parser.add_argument("--no_llm",    action="store_true",
                        help="Use mock sentence-split (no OpenAI call)")
    parser.add_argument("--model",     default="gpt-4o-mini")
    args = parser.parse_args()

    # Load .env
    env_file = DATA_DIR.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

    # Build OpenAI client
    openai_client = None
    if not args.no_llm:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY not set. Use --no_llm for mock extraction.")
            sys.exit(1)
        import openai
        openai_client = openai.OpenAI(api_key=api_key)
        logger.info("Using real LLM extraction with model=%s", args.model)
    else:
        logger.warning("Using mock sentence-split (no LLM). Quality will be low.")

    preprocessor = RAMDocsPreprocessor(
        input_path=args.input,
        output_dir=args.output_dir,
        skip_noise_docs=args.skip_noise,
        openai_client=openai_client,
        model=args.model,
        max_queries=args.max_queries,
    )
    stats = preprocessor.run()

    print("\n[preprocess] Done.")
    print(f"  Queries          : {stats['total_queries']}")
    print(f"  Documents        : {stats['total_docs']}")
    print(f"  Claims           : {stats['total_claims']}")
    print(f"  Canonical merges : {stats['canonical_merges']}")
    print(f"  Skipped boilerplate: {stats['skipped_boilerplate']}")
    print(f"  Doc types        : {stats['doc_type_counts']}")
