"""
data/preprocess_factoids_revised.py
Module 3 — Factoid Claim Extraction (LLM-based).

Input:  data/preprocessed/claims.jsonl   (output của preprocess_revised.py)
Output: data/preprocessed/factoid_claims.jsonl   (overwrite in-place; backup tự động → .jsonl.bak)

Với mỗi claim_text → LLM extract FactoidFeatures (schema_revised.FactoidFeatures):
  number[]    ← FactoidNumberValue   {value, unit}
  entity[]    ← FactoidEntityMention {raw_mention, canonical_entity, attribute}
  temporal    ← FactoidTemporal      {raw_time, start, end, granularity}
  negation    ← FactoidNegation      {polarity 0|1}
  verb        ← FactoidVerb          {lemma, tense}

Sau khi extract:
  - Validate qua Pydantic (FactoidFeatures.model_validate)
  - Cập nhật claim_features.context_completeness = slots_present / 5 (chính xác hơn heuristic)

Provider: GPT-4o (primary) → Gemini 2.5 Flash (fallback nếu không có OPENAI_API_KEY)
Batching: BATCH_SIZE claims/call để giảm API latency.
Resume:   --skip_existing (mặc định bật) — bỏ qua claims đã có factoid_features != null.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent
sys.path.insert(0, str(DATA_DIR.parent))

from src.schema_revised import (
    FactoidEntityMention,
    FactoidFeatures,
    FactoidNegation,
    FactoidNumberValue,
    FactoidTemporal,
    FactoidVerb,
)

# ── Constants ─────────────────────────────────────────────────────────────────
BATCH_SIZE  = 10    # claims per LLM call
RETRY_MAX   = 2
RETRY_DELAY = 2.0   # seconds

# ── Batch prompt ──────────────────────────────────────────────────────────────

_PROMPT_TEMPLATE = """\
You are a factoid extraction system for a fact-checking NLP pipeline.
For EACH claim in the numbered list, extract structured factoid features.

Return ONLY a JSON object with key "results" — an array of exactly {n} factoid objects (same order as claims).

Each factoid object schema:
{{
  "number":   [ {{"value": <float>, "unit": <string|null>}} ],
  "entity":   [ {{"raw_mention": <string>, "canonical_entity": <string>, "attribute": <string|null>}} ],
  "temporal": {{"raw_time": <string|null>, "start": <"YYYY-MM-DD"|null>, "end": <"YYYY-MM-DD"|null>,
                "granularity": "day"|"month"|"year"|"range"|"none"}}
              or null if no time reference exists,
  "negation": {{"polarity": 0}}   (affirmative)  OR  {{"polarity": 1}}   (negated),
  "verb":     {{"lemma": <string>, "tense": <"past"|"present"|"future"|null>}}
}}

Field rules:
- number: extract every numeric quantity mentioned.
- entity.canonical_entity: resolve abbreviations (e.g. "US" → "United States").
- entity.attribute: the aspect asserted about the entity (e.g. "approval_status", "dosage").
- temporal.start/end: full ISO 8601 date (YYYY-MM-DD) inferred from context; null if not determinable.
- temporal.granularity: "none" if a time is mentioned but cannot be resolved.
- negation.polarity: 1 if the main assertion is negated; 0 otherwise.
- verb: the main predicate lemma of the claim.

Claims:
{claims}

Return ONLY the JSON. No explanation."""


def _make_batch_prompt(claim_texts: list[str]) -> str:
    numbered = "\n".join(f'{i + 1}. "{t}"' for i, t in enumerate(claim_texts))
    return _PROMPT_TEMPLATE.format(n=len(claim_texts), claims=numbered)


# ── context_completeness ──────────────────────────────────────────────────────

def compute_context_completeness(f: FactoidFeatures) -> float:
    """
    Module 4, mục 6.2 — tính chính xác sau khi có FactoidFeatures.
    Slot hiện diện nếu có giá trị có nghĩa (meaningful):
      Number:   có ít nhất 1 giá trị số
      Entity:   có ít nhất 1 entity
      Temporal: có start date được resolve
      Negation: polarity = 1 (có phủ định)
      Verb:     có main verb
    """
    slots = [
        bool(f.number),
        bool(f.entity),
        f.temporal is not None and f.temporal.start is not None,
        f.negation is not None and f.negation.polarity == 1,
        f.verb is not None,
    ]
    return round(sum(slots) / 5, 2)


# ── LLM response parsing ──────────────────────────────────────────────────────

def _parse_batch_response(content: str, expected: int) -> list[dict | None]:
    """Parse JSON response → list[dict|None] với đúng expected length."""
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            results = parsed.get("results")
            if not isinstance(results, list):
                results = next(
                    (v for v in parsed.values() if isinstance(v, list)), None
                )
        elif isinstance(parsed, list):
            results = parsed
        else:
            results = None

        if results is None or len(results) != expected:
            logger.warning(
                "Batch response length mismatch: expected %d, got %s",
                expected, len(results) if results else "None",
            )
            return [None] * expected

        return [r if isinstance(r, dict) else None for r in results]
    except json.JSONDecodeError as exc:
        logger.warning("JSON parse error: %s", exc)
        return [None] * expected


def _normalize_raw_factoid(raw: dict) -> dict:
    """Normalize trước khi Pydantic validate: fix bool polarity, null temporal."""
    if "negation" in raw and isinstance(raw["negation"], dict):
        p = raw["negation"].get("polarity")
        if isinstance(p, bool):
            raw["negation"]["polarity"] = 1 if p else 0
    if raw.get("temporal") == {}:
        raw["temporal"] = None
    return raw


# ── LLM call helpers ──────────────────────────────────────────────────────────

def _call_openai_batch(claim_texts: list[str], client, model: str) -> list[dict | None]:
    prompt = _make_batch_prompt(claim_texts)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return _parse_batch_response(resp.choices[0].message.content or "{}", len(claim_texts))


def _call_gemini_batch(claim_texts: list[str], client, model: str) -> list[dict | None]:
    from google.genai import types as genai_types
    prompt = _make_batch_prompt(claim_texts)
    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0,
        ),
    )
    return _parse_batch_response(resp.text or "{}", len(claim_texts))


def extract_factoids_batch(
    claim_texts: list[str],
    client,
    provider: str,
    model: str,
) -> list[dict | None]:
    """Gọi LLM với retry — trả về list[dict|None] aligned với claim_texts."""
    for attempt in range(RETRY_MAX + 1):
        try:
            if provider == "openai":
                return _call_openai_batch(claim_texts, client, model)
            else:
                return _call_gemini_batch(claim_texts, client, model)
        except Exception as exc:
            if attempt < RETRY_MAX:
                logger.warning(
                    "Batch attempt %d/%d failed: %s — retrying in %.1fs",
                    attempt + 1, RETRY_MAX + 1, exc, RETRY_DELAY,
                )
                time.sleep(RETRY_DELAY)
            else:
                logger.error("All retries exhausted for batch: %s", exc)
    return [None] * len(claim_texts)


# ── Validation ────────────────────────────────────────────────────────────────

def validate_factoid(raw: dict) -> FactoidFeatures | None:
    """Validate raw dict → FactoidFeatures. Returns None nếu không hợp lệ."""
    try:
        raw = _normalize_raw_factoid(raw)
        return FactoidFeatures.model_validate(raw)
    except Exception as exc:
        logger.debug("Validation failed: %s | raw=%s", exc, raw)
        return None


# ── Mock extraction (no LLM) ──────────────────────────────────────────────────

_RE_NUMBER   = re.compile(r'\b(\d+(?:[.,]\d+)?)\s*([a-zA-Z%]+)?\b')
_RE_TEMPORAL = re.compile(
    r'\b(\d{4}(?:-\d{2}(?:-\d{2})?)?'
    r'|(?:january|february|march|april|may|june|july|august|september'
    r'|october|november|december)\s+\d{4}'
    r'|(?:in|since|until|before|after)\s+\d{4})\b',
    re.IGNORECASE,
)
_RE_NEGATION = re.compile(
    r"\b(not|no|never|neither|nor|without|cannot|can't|don't|doesn't"
    r"|isn't|aren't|wasn't|weren't)\b",
    re.IGNORECASE,
)
_RE_PROPER   = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b')
_RE_VERB     = re.compile(
    r'\b(is|are|was|were|has|have|had|will|would|can|could|should'
    r'|may|might|must|shall|approved|banned|rejected|confirmed|found'
    r'|showed|reported|stated|claimed)\b',
    re.IGNORECASE,
)


def mock_extract_factoid(claim_text: str) -> FactoidFeatures:
    """Regex-based fallback khi không có LLM — chất lượng thấp, chỉ dùng để test."""
    numbers: list[FactoidNumberValue] = []
    for m in _RE_NUMBER.finditer(claim_text):
        try:
            val = float(m.group(1).replace(",", "."))
            unit = m.group(2) or None
            numbers.append(FactoidNumberValue(value=val, unit=unit))
        except ValueError:
            pass

    entities: list[FactoidEntityMention] = []
    for m in _RE_PROPER.finditer(claim_text):
        name = m.group(1)
        if name.lower() not in {"the", "a", "an", "this", "that", "it", "he", "she", "they"}:
            entities.append(FactoidEntityMention(canonical_entity=name, raw_mention=name))

    temporal = None
    t_match = _RE_TEMPORAL.search(claim_text)
    if t_match:
        temporal = FactoidTemporal(raw_time=t_match.group(0))

    negation = FactoidNegation(polarity=1 if _RE_NEGATION.search(claim_text) else 0)

    verb = None
    v_match = _RE_VERB.search(claim_text)
    if v_match:
        verb = FactoidVerb(lemma=v_match.group(1).lower())

    return FactoidFeatures(
        number=numbers,
        entity=entities,
        temporal=temporal,
        negation=negation,
        verb=verb,
    )


# ── Main processor ────────────────────────────────────────────────────────────

class FactoidPreprocessor:

    def __init__(
        self,
        claims_path: str,
        output_path: str,
        client=None,
        provider: str = "openai",
        model: str = "gpt-4o",
        batch_size: int = BATCH_SIZE,
        skip_existing: bool = True,
        use_mock: bool = False,
    ):
        self.claims_path   = Path(claims_path)
        self.output_path   = Path(output_path)
        self.client        = client
        self.provider      = provider
        self.model         = model
        self.batch_size    = batch_size
        self.skip_existing = skip_existing
        self.use_mock      = use_mock

    def _load_claims(self) -> list[dict]:
        with open(self.claims_path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def _process_batch(self, batch: list[dict]) -> tuple[list[dict], int, int, int]:
        """
        Returns (updated_batch, n_extracted, n_skipped, n_failed).
        """
        to_process: list[int] = []   # indices into batch
        for i, claim in enumerate(batch):
            if self.skip_existing and claim.get("factoid_features") is not None:
                continue
            to_process.append(i)

        n_skipped = len(batch) - len(to_process)
        if not to_process:
            return batch, 0, n_skipped, 0

        claim_texts = [batch[i]["claim_text"] for i in to_process]

        if self.use_mock:
            raw_results: list[dict | None] = [
                mock_extract_factoid(t).model_dump() for t in claim_texts
            ]
        else:
            raw_results = extract_factoids_batch(
                claim_texts, self.client, self.provider, self.model
            )

        n_extracted = 0
        n_failed = 0
        for idx, raw in zip(to_process, raw_results):
            if raw is None:
                n_failed += 1
                continue
            factoid = validate_factoid(raw)
            if factoid is None:
                n_failed += 1
                continue
            batch[idx]["factoid_features"] = factoid.model_dump()
            batch[idx]["claim_features"]["context_completeness"] = (
                compute_context_completeness(factoid)
            )
            n_extracted += 1

        return batch, n_extracted, n_skipped, n_failed

    def run(self) -> dict:
        stats = {
            "total_claims":     0,
            "extracted":        0,
            "skipped_existing": 0,
            "failed":           0,
        }

        claims = self._load_claims()
        stats["total_claims"] = len(claims)

        if not claims:
            logger.warning("No claims found in %s", self.claims_path)
            return stats

        # Backup nếu output == input
        if self.output_path.resolve() == self.claims_path.resolve():
            backup = self.claims_path.with_suffix(".jsonl.bak")
            shutil.copy2(self.claims_path, backup)
            logger.info("Backup → %s", backup)

        updated: list[dict] = []
        batches = [
            claims[i: i + self.batch_size]
            for i in range(0, len(claims), self.batch_size)
        ]

        for b_idx, batch in enumerate(batches):
            processed, n_ext, n_skip, n_fail = self._process_batch(batch)
            stats["extracted"]        += n_ext
            stats["skipped_existing"] += n_skip
            stats["failed"]           += n_fail
            updated.extend(processed)

            done = len(updated)
            if (b_idx + 1) % 5 == 0 or (b_idx + 1) == len(batches):
                logger.info(
                    "Batch %d/%d — extracted %d | skipped %d | failed %d (total %d/%d)",
                    b_idx + 1, len(batches),
                    stats["extracted"], stats["skipped_existing"],
                    stats["failed"], done, len(claims),
                )

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            for c in updated:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

        logger.info("Written %d claims → %s", len(updated), self.output_path)
        return stats


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Module 3 — Factoid extraction for Conflict-State-Guided RAG"
    )
    parser.add_argument(
        "--claims",
        default=str(DATA_DIR / "preprocessed" / "claims.jsonl"),
        help="Input claims.jsonl (output của preprocess_revised.py)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path (mặc định: overwrite input file)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=BATCH_SIZE,
        help=f"Số claims mỗi LLM call (mặc định: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--no_skip", action="store_true",
        help="Re-extract ngay cả claims đã có factoid_features",
    )
    parser.add_argument(
        "--no_llm", action="store_true",
        help="Dùng regex mock (không gọi LLM) — chỉ để test",
    )
    parser.add_argument(
        "--model", default=None,
        help="Tên model (mặc định: gpt-4o nếu OpenAI, gemini-2.5-flash nếu Gemini)",
    )
    args = parser.parse_args()

    # Load .env
    env_file = DATA_DIR.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

    # Build LLM client
    llm_client = None
    provider   = "openai"
    model      = args.model

    if not args.no_llm:
        openai_key = os.environ.get("OPENAI_API_KEY")
        gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

        if openai_key:
            import openai as _openai
            llm_client = _openai.OpenAI(api_key=openai_key)
            provider   = "openai"
            model      = model or "gpt-4o"
            logger.info("Using OpenAI: model=%s", model)
        elif gemini_key:
            from google import genai as _genai
            llm_client = _genai.Client(api_key=gemini_key)
            provider   = "gemini"
            model      = model or "gemini-2.5-flash"
            logger.info("OpenAI key not found — falling back to Gemini: model=%s", model)
        else:
            logger.warning("No API key found — using regex mock extraction.")
            args.no_llm = True

    output_path = args.output or str(DATA_DIR / "preprocessed" / "factoid_claims.jsonl")

    preprocessor = FactoidPreprocessor(
        claims_path=args.claims,
        output_path=output_path,
        client=llm_client,
        provider=provider,
        model=model or "gpt-4o",
        batch_size=args.batch_size,
        skip_existing=not args.no_skip,
        use_mock=args.no_llm,
    )
    stats = preprocessor.run()

    print("\n[preprocess_factoids] Done.")
    print(f"  Total claims       : {stats['total_claims']}")
    print(f"  Extracted          : {stats['extracted']}")
    print(f"  Skipped (existing) : {stats['skipped_existing']}")
    print(f"  Failed             : {stats['failed']}")