"""
data/preprocess_revised.py
Chuyển đổi RAMDocs_test.jsonl → pipeline-ready JSONL files.

Pipeline: Conflict-State-Guided Retrieval for Conflict-Aware RAG
Modules covered ở bước preprocess:
  User Query → Initial Evidence Retrieval → Atomic Claim Extraction (Module 2)
  → Claim Feature Initialization (Module 4, partial: retrieval_relevance,
    claim_confidence, context_completeness ước lượng từ text; claim_evidence_coverage=0.0)

Output (4 file):
  queries.jsonl    — {query_id, user_query}
  documents.jsonl  — {query_id, documents[]} (không có Pydantic model, dùng để reference)
  claims.jsonl     — Claim[] theo schema_revised.Claim
  metadata.jsonl   — ground truth, CHỈ dùng evaluation

Claim schema (claims.jsonl):
  claim_id, claim_text,
  canonical_claim_id  ← tự reference nếu canonical; None nếu chưa resolve
  duplicate_of        ← None (set bởi iterative loop khi claim mới trùng claim cũ)
  factoid_features    ← None (populated bởi Module 3)
  claim_features:
    retrieval_relevance      ← từ rank-based score
    claim_confidence         ← hash-based trong range theo doc_type
    claim_evidence_coverage  ← 0.0 (updated bởi Module 4 sau retrieval)
    context_completeness     ← ước lượng từ factoid slot presence
  doc_id, source_id          ← pipeline tracing (ngoài Pydantic schema)

Đã loại bỏ: Document model, credibility_score, Retrieval Decision Agent,
  ActionLabel, evidence field, is_representative, merged_claim_ids.
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
sys.path.insert(0, str(DATA_DIR.parent))

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_DOC_LENGTH    = 1500   # chars — truncate trước khi gửi LLM
MIN_CLAIM_WORDS   = 4
MAX_CLAIM_WORDS   = 40
CANONICAL_OVERLAP = 0.55   # Jaccard threshold để merge duplicate claims
_FACTOID_SLOTS    = 5      # Number, Entity, Temporal, Negation, Verb

# Confidence range theo doc_type — claim_confidence initialization (Module 4)
# credibility_score đã bị loại (Source Credibility Agent removed)
TAU_CONFIDENCE = {
    "correct": (0.80, 0.95),
    "misinfo": (0.55, 0.75),
    "noise":   (0.40, 0.60),
    "unknown": (0.50, 0.70),
}

# ── Wikipedia boilerplate patterns ────────────────────────────────────────────
_WIKI_BOILERPLATE = re.compile(
    r"""
    (?:edit\s*\])|(?:\[\s*edit)|(?:view\s+history)|(?:move\s+to\s+sidebar)
    |(?:hide\s+from\s+sidebar)|(?:v\s+t\s+e\b)
    |(?:cite\s+this\s+page)|(?:permanent\s+link)
    |(?:what\s+links\s+here)|(?:related\s+changes)
    |(?:upload\s+file)|(?:printable\s+version)
    |(?:retrieved\s+\d{1,2}\s+\w+\s+\d{4})
    |(?:archived\s+from\s+the\s+original)
    |(?:cite\s+news|cite\s+web|cite\s+book)
    |(?:https?://\S+)
    |(?:categories\s*:)|(?:hidden\s+categories)
    |(?:\[\s*\d+\s*\])
    |(?:jump\s+to\s+navigation)|(?:jump\s+to\s+search)
    """,
    re.IGNORECASE | re.VERBOSE,
)
_WIKI_LANGUAGE_LINE = re.compile(
    r"(?:\d+\s+languages?\b)|(?:বাংলা|فارسی|हिन्दी|नेपाल|Edit\s+links)",
    re.UNICODE,
)

# ── Factoid presence patterns — dùng cho estimate_context_completeness ─────────
_RE_NUMBER = re.compile(r'\b\d+[\.,]?\d*\b')
_RE_TEMPORAL = re.compile(
    r'\b(\d{4}|january|february|march|april|may|june|july|august|september'
    r'|october|november|december|monday|tuesday|wednesday|thursday|friday'
    r'|saturday|sunday|today|yesterday|recently|currently|since|until'
    r'|before|after|during|in\s+\d{4})\b',
    re.IGNORECASE,
)
_RE_NEGATION = re.compile(
    r"\b(not|no|never|neither|nor|without|cannot|can't|don't"
    r"|doesn't|isn't|aren't|wasn't|weren't)\b",
    re.IGNORECASE,
)
_RE_PROPER_NOUN = re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b')
_RE_VERB = re.compile(
    r'\b(is|are|was|were|has|have|had|will|would|can|could|should'
    r'|may|might|must|shall|do|does|did|be|been|being)\b',
    re.IGNORECASE,
)


# ── Boilerplate helpers ────────────────────────────────────────────────────────

def is_boilerplate(text: str) -> bool:
    if len(text.strip()) < 20:
        return True
    if _WIKI_BOILERPLATE.search(text):
        return True
    if _WIKI_LANGUAGE_LINE.search(text):
        return True
    if text.count("http") > 0 and len(text.split()) < 15:
        return True
    return False


def clean_doc_text(text: str) -> str:
    text = text.replace("[TRUNCATED]", "").strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\[\s*\]", "", text)
    text = re.sub(r"\b(edit|hide|show)\b\s*[\[\]]?", "", text, flags=re.IGNORECASE)
    return text.strip()


# ── Feature helpers ────────────────────────────────────────────────────────────

def extract_numbers(text: str) -> set[str]:
    return set(re.findall(r"\b\d+\b", text))


def text_overlap(a: str, b: str) -> float:
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def estimate_context_completeness(claim_text: str) -> float:
    """
    Module 4, mục 6.2 — ước lượng context_completeness từ text.
    = số factoid slot hiện diện / 5 (Number, Entity, Temporal, Negation, Verb).
    Sẽ được overwrite chính xác bởi Module 3 (FactoidFeatures extraction).
    """
    slots = sum([
        bool(_RE_NUMBER.search(claim_text)),
        bool(_RE_TEMPORAL.search(claim_text)),
        bool(_RE_NEGATION.search(claim_text)),
        bool(_RE_PROPER_NOUN.search(claim_text)),
        bool(_RE_VERB.search(claim_text)),
    ])
    return round(slots / _FACTOID_SLOTS, 2)


def _make_claim_dict(
    claim_id: str,
    claim_text: str,
    evidence: str,
    doc_id: str,
    source_id: str,
    retrieval_score: float,
    claim_confidence: float,
) -> dict:
    """Tạo claim dict conform schema_revised.Claim."""
    return {
        # --- schema_revised.Claim ---
        "claim_id":           claim_id,
        "claim_text":         claim_text,
        "canonical_claim_id": None,   # set bởi run_phase2_canonical
        "duplicate_of":       None,   # set bởi iterative loop khi claim mới trùng
        "factoid_features":   None,   # populated bởi Module 3
        "claim_features": {
            "retrieval_relevance":     round(retrieval_score, 4),
            "claim_confidence":        round(claim_confidence, 4),
            "claim_evidence_coverage": 0.0,   # updated bởi Module 4 sau retrieval
            "context_completeness":    estimate_context_completeness(claim_text),
        },
        # --- pipeline tracing (không trong Pydantic schema) ---
        "evidence":  evidence,   # raw excerpt từ document
        "doc_id":    doc_id,
        "source_id": source_id,
    }


# ── LLM-based claim extraction (Module 2) ─────────────────────────────────────

_EXTRACTION_PROMPT = (
    "Extract all atomic factual claims from the following document passage.\n"
    "For each claim return TWO fields:\n"
    "  - \"evidence\": the exact sentence or phrase copied verbatim from the document.\n"
    "  - \"claim_text\": reformulate into ONE self-contained factual statement (max 35 words).\n"
    "Rules:\n"
    "- claim_text must be verifiable independently.\n"
    "- Do NOT include: opinions, navigation text, edit links, category labels, "
    "citation metadata, or URL fragments.\n"
    "- Return ONLY a JSON object: {\"claims\": [{\"evidence\": \"...\", \"claim_text\": \"...\"}]}\n\n"
    "Document:\n{text}"
)


def _parse_llm_json(content: str) -> list[dict]:
    """Parse JSON response → list[dict] với keys evidence + claim_text."""
    parsed = json.loads(content)
    if isinstance(parsed, list):
        items = parsed
    elif isinstance(parsed, dict):
        items = parsed.get("claims") or next(
            (v for v in parsed.values() if isinstance(v, list)), []
        )
    else:
        return []
    # Normalise: nếu LLM vẫn trả về plain string thì dùng làm cả hai field
    return [
        {"evidence": v, "claim_text": v} if isinstance(v, str) else v
        for v in items
    ]


def _call_openai(prompt: str, client, model: str) -> list[dict]:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return _parse_llm_json(resp.choices[0].message.content or "[]")


def _call_gemini(prompt: str, client, model: str) -> list[dict]:
    from google.genai import types as genai_types
    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0,
        ),
    )
    return _parse_llm_json(resp.text or "[]")


def extract_claims_llm(
    doc_text: str,
    doc_id: str,
    source_id: str,
    query_idx: int,
    doc_idx: int,
    doc_type: str,
    retrieval_score: float,
    client,
    provider: str,   # "openai" | "gemini"
    model: str,
) -> list[dict]:
    """Module 2 — Atomic Claim Extraction via LLM."""
    text_clean = clean_doc_text(doc_text)
    if is_boilerplate(text_clean) or len(text_clean.split()) < 10:
        logger.info("Skipping boilerplate doc %s", doc_id)
        return []

    if len(text_clean) > MAX_DOC_LENGTH:
        text_clean = text_clean[:MAX_DOC_LENGTH].rsplit(" ", 1)[0]

    prompt = _EXTRACTION_PROMPT.format(text=text_clean)
    try:
        if provider == "openai":
            raw_claims = _call_openai(prompt, client, model)
        else:
            raw_claims = _call_gemini(prompt, client, model)
    except Exception as exc:
        logger.warning("LLM extraction failed for %s (%s): %s", doc_id, provider, exc)
        return []

    lo, hi = TAU_CONFIDENCE.get(doc_type, TAU_CONFIDENCE["unknown"])
    return _build_claim_list(raw_claims, doc_id, source_id, query_idx, doc_idx, lo, hi, retrieval_score)


def _build_claim_list(
    raw_claims: list[dict],
    doc_id: str,
    source_id: str,
    query_idx: int,
    doc_idx: int,
    conf_lo: float,
    conf_hi: float,
    retrieval_score: float,
) -> list[dict]:
    claims = []
    for sent_idx, item in enumerate(raw_claims):
        claim_text = item.get("claim_text", "").strip()
        evidence   = item.get("evidence", claim_text).strip()
        words = claim_text.split()
        if len(words) < MIN_CLAIM_WORDS or len(words) > MAX_CLAIM_WORDS:
            continue
        if is_boilerplate(claim_text):
            continue
        h_val = int(hashlib.md5(claim_text.encode()).hexdigest(), 16)
        confidence = conf_lo + (h_val % 1000) / 1000 * (conf_hi - conf_lo)
        claim_id = f"c_q{query_idx:03d}_d{doc_idx:02d}_s{sent_idx:02d}"
        claims.append(_make_claim_dict(
            claim_id, claim_text, evidence, doc_id, source_id, retrieval_score, confidence
        ))
    return claims


# ── Phase 2: Canonical merge ───────────────────────────────────────────────────

def run_phase2_canonical(claims: list[dict]) -> list[dict]:
    """
    Merge claims diễn đạt cùng fact (Jaccard > threshold VÀ cùng số).
    Giữ riêng claims có temporal/numerical khác nhau — nguồn gốc conflict.

    Schema mới:
      - Claim canonical: canonical_claim_id = claim_id (self-reference)
      - Duplicates: bị drop khỏi output (không xuất hiện trong ConflictGraph)
      - duplicate_of: được set bởi iterative loop khi claim MỚI đến trùng claim cũ
    """
    merged: set[int] = set()
    result: list[dict] = []

    for i, ci in enumerate(claims):
        if i in merged:
            continue
        for j, cj in enumerate(claims):
            if j <= i or j in merged:
                continue
            overlap = text_overlap(ci["claim_text"], cj["claim_text"])
            if overlap > CANONICAL_OVERLAP and extract_numbers(ci["claim_text"]) == extract_numbers(cj["claim_text"]):
                merged.add(j)

        ci_out = deepcopy(ci)
        ci_out["canonical_claim_id"] = ci["claim_id"]  # self-reference = canonical
        result.append(ci_out)

    return result


# ── Builders ───────────────────────────────────────────────────────────────────

def build_query(raw: dict, query_idx: int) -> dict:
    return {
        "query_id":   f"q_{query_idx:03d}",
        "user_query": re.sub(r"\s+", " ", raw["question"].strip()),
    }


def build_documents(raw: dict, query_idx: int, skip_noise: bool = False) -> list[dict]:
    """
    Document không có Pydantic model trong schema mới.
    Lưu dạng flat dict để pipeline reference và claim tracing.
    """
    docs = []
    total = len(raw["documents"])
    for doc_idx, raw_doc in enumerate(raw["documents"]):
        doc_type = raw_doc.get("type", "unknown")
        if skip_noise and doc_type == "noise":
            continue

        text = re.sub(r"\s+", " ", raw_doc["text"].strip())
        truncated = False
        if len(text) > MAX_DOC_LENGTH * 2:
            text = text[:MAX_DOC_LENGTH * 2].rsplit(" ", 1)[0] + " [TRUNCATED]"
            truncated = True

        retrieval_score = (
            round(1.0 - (doc_idx / max(total - 1, 1)) * 0.5, 4) if total > 1 else 1.0
        )
        doc_id = f"d_{hashlib.md5(f'{query_idx}_{doc_idx}_{text[:40]}'.encode()).hexdigest()[:6]}"

        docs.append({
            "doc_id":          doc_id,
            "source_id":       f"s_q{query_idx:03d}_{doc_idx:02d}",
            "query_id":        f"q_{query_idx:03d}",
            "text":            text,
            "retrieval_score": retrieval_score,
            "doc_type":        doc_type,
            "rank":            doc_idx + 1,
            "url":             raw_doc.get("url"),
            "published_date":  raw_doc.get("published_date"),
            "_truncated":      truncated,
        })
    return docs


def build_metadata(raw: dict, query_idx: int, docs: list[dict]) -> dict:
    raw_docs = raw.get("documents", [])
    return {
        "query_id":        f"q_{query_idx:03d}",
        "user_query":      raw["question"].strip(),
        "disambig_entity": raw.get("disambig_entity", []),
        "gold_answers":    raw.get("gold_answers", []),
        "wrong_answers":   raw.get("wrong_answers", []),
        "doc_labels": [
            {
                "doc_id":   doc["doc_id"],
                "doc_type": raw_doc.get("type", "unknown"),
                "answer":   raw_doc.get("answer"),
            }
            for doc, raw_doc in zip(docs, raw_docs)
        ],
    }


# ── Main preprocessor ──────────────────────────────────────────────────────────

class RAMDocsPreprocessor:

    def __init__(
        self,
        input_path: str,
        output_dir: str = "data/preprocessed",
        skip_noise_docs: bool = False,
        client=None,
        provider: str = "openai",   # "openai" | "gemini"
        model: str = "gpt-4o-mini",
        max_queries: int | None = None,
    ):
        self.input_path  = Path(input_path)
        self.output_dir  = Path(output_dir)
        self.skip_noise  = skip_noise_docs
        self.client      = client
        self.provider    = provider
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
            "total_queries":       0,
            "total_docs":          0,
            "total_claims":        0,
            "canonical_claims":    0,
            "dropped_duplicates":  0,
            "skipped_boilerplate": 0,
            "doc_type_counts":     {"correct": 0, "misinfo": 0, "noise": 0},
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
                doc_type_map = {doc["doc_id"]: doc["doc_type"] for doc in docs}

                # ── Claim extraction (Module 2) ───────────────────────────────
                all_claims: list[dict] = []
                for doc_idx, doc in enumerate(docs):
                    doc_type = doc_type_map.get(doc["doc_id"], "unknown")
                    if self.skip_noise and doc_type == "noise":
                        continue

                    if self.client:
                        claims = extract_claims_llm(
                            doc_text=doc["text"],
                            doc_id=doc["doc_id"],
                            source_id=doc["source_id"],
                            query_idx=query_idx,
                            doc_idx=doc_idx,
                            doc_type=doc_type,
                            retrieval_score=doc["retrieval_score"],
                            client=self.client,
                            provider=self.provider,
                            model=self.model,
                        )
                        if not claims:
                            stats["skipped_boilerplate"] += 1
                    else:
                        claims = self._mock_extract(doc, query_idx, doc_idx, doc_type)

                    all_claims.extend(claims)

                # ── Phase 2: canonical merge ──────────────────────────────────
                raw_count  = len(all_claims)
                canonical  = run_phase2_canonical(all_claims)
                n_dropped  = raw_count - len(canonical)

                # ── Write (flush per query) ───────────────────────────────────
                fq.write(json.dumps(query, ensure_ascii=False) + "\n"); fq.flush()
                fd.write(json.dumps(
                    {"query_id": query["query_id"], "documents": docs},
                    ensure_ascii=False,
                ) + "\n"); fd.flush()
                for c in canonical:
                    fc.write(json.dumps(c, ensure_ascii=False) + "\n")
                fc.flush()
                fm.write(json.dumps(meta, ensure_ascii=False) + "\n"); fm.flush()

                # ── Stats ─────────────────────────────────────────────────────
                stats["total_queries"]      += 1
                stats["total_docs"]         += len(docs)
                stats["total_claims"]       += len(canonical)
                stats["canonical_claims"]   += len(canonical)
                stats["dropped_duplicates"] += n_dropped
                for raw_doc in raw.get("documents", []):
                    t = raw_doc.get("type", "unknown")
                    if t in stats["doc_type_counts"]:
                        stats["doc_type_counts"][t] += 1

                if (query_idx + 1) % 10 == 0:
                    logger.info(
                        "Processed %d queries — %d claims, %d duplicates dropped",
                        query_idx + 1, stats["total_claims"], stats["dropped_duplicates"],
                    )

        return stats

    def _mock_extract(
        self, doc: dict, query_idx: int, doc_idx: int, doc_type: str
    ) -> list[dict]:
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
            confidence = lo + (h_val % 1000) / 1000 * (hi - lo)
            claim_id = f"c_q{query_idx:03d}_d{doc_idx:02d}_s{sent_idx:02d}"
            claims.append(_make_claim_dict(
                claim_id, sentence, sentence,   # mock: evidence = claim_text = raw sentence
                doc["doc_id"], doc["source_id"],
                doc["retrieval_score"], confidence,
            ))
        return claims


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Preprocess RAMDocs → pipeline schema (Conflict-State-Guided RAG)"
    )
    parser.add_argument("--input",       default=str(DATA_DIR / "raw" / "RAMDocs_test.jsonl"))
    parser.add_argument("--output_dir",  default=str(DATA_DIR / "preprocessed"))
    parser.add_argument("--skip_noise",  action="store_true")
    parser.add_argument("--max_queries", type=int, default=None,
                        help="Chỉ xử lý N query đầu (để test)")
    parser.add_argument("--no_llm",     action="store_true",
                        help="Dùng mock sentence-split, bỏ qua mọi LLM")
    parser.add_argument("--model",      default=None,
                        help="Tên model (mặc định: gpt-4o-mini nếu OpenAI, gemini-2.5-flash nếu Gemini)")
    args = parser.parse_args()

    env_file = DATA_DIR.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

    llm_client = None
    provider   = "openai"
    model      = args.model

    if not args.no_llm:
        openai_key = os.environ.get("OPENAI_API_KEY")
        gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

        if openai_key:
            import openai
            llm_client = openai.OpenAI(api_key=openai_key)
            provider   = "openai"
            model      = model or "gpt-4o-mini"
            logger.info("Using OpenAI: model=%s", model)
        elif gemini_key:
            from google import genai as google_genai
            llm_client = google_genai.Client(api_key=gemini_key)
            provider   = "gemini"
            model      = model or "gemini-2.5-flash"
            logger.info("OpenAI key not found — falling back to Gemini: model=%s", model)
        else:
            logger.warning("No OPENAI_API_KEY or GEMINI_API_KEY found — using mock sentence-split.")
    else:
        logger.warning("--no_llm: using mock sentence-split. Quality will be low.")

    preprocessor = RAMDocsPreprocessor(
        input_path=args.input,
        output_dir=args.output_dir,
        skip_noise_docs=args.skip_noise,
        client=llm_client,
        provider=provider,
        model=model or "gpt-4o-mini",
        max_queries=args.max_queries,
    )
    stats = preprocessor.run()

    print("\n[preprocess] Done.")
    print(f"  Queries            : {stats['total_queries']}")
    print(f"  Documents          : {stats['total_docs']}")
    print(f"  Canonical claims   : {stats['canonical_claims']}")
    print(f"  Dropped duplicates : {stats['dropped_duplicates']}")
    print(f"  Skipped boilerplate: {stats['skipped_boilerplate']}")
    print(f"  Doc types          : {stats['doc_type_counts']}")