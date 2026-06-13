"""
src/edge_features.py
Module 5 — Edge Feature Extraction using NLI.

Input:  data/preprocessed/factoid_claims_revised.jsonl
Output: data/preprocessed/edges.json

For each pair of claims in the same query group (same q??? prefix) but from
different source documents, run NLI to classify the relation and build all
ClaimPairFeatures from factoid fields.

Usage:
    python src/edge_features.py
    python src/edge_features.py --max_pairs_per_query 200 --batch_size 64
    python src/edge_features.py --all_pairs  # include same-doc pairs
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.schema_revised import (
    AggregateEdgeFeatures,
    AttributePairFeatures,
    ClaimPairFeatures,
    Edge,
    EntityPairFeatures,
    NegationPairFeatures,
    NLIEdgeFeatures,
    NumberPairFeatures,
    RelationType,
    TemporalPairFeatures,
)

# ── Constants ──────────────────────────────────────────────────────────────────
NLI_MODEL  = "cross-encoder/nli-deberta-v3-small"
BATCH_SIZE = 32
MAX_LEN    = 512

INPUT_PATH  = PROJECT_ROOT / "data" / "preprocessed" / "factoid_claims_revised.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "data" / "preprocessed" / "edges.json"


# ── NLI helpers ────────────────────────────────────────────────────────────────

def load_nli_model(model_name: str = NLI_MODEL):
    """Load tokenizer + model; return with label-index mapping from config."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    logger.info("NLI model '%s' loaded on %s", model_name, device)

    id2label = {k: v.lower() for k, v in model.config.id2label.items()}
    label2idx = {v: k for k, v in id2label.items()}
    contra_idx  = label2idx.get("contradiction", 0)
    entail_idx  = label2idx.get("entailment",    1)
    neutral_idx = label2idx.get("neutral",        2)
    return tokenizer, model, device, contra_idx, entail_idx, neutral_idx


def _run_nli_batch(
    pairs: list[tuple[str, str]],
    tokenizer,
    model,
    device,
    contra_idx: int,
    entail_idx: int,
    neutral_idx: int,
    batch_size: int = BATCH_SIZE,
) -> list[tuple[float, float, float]]:
    """Raw NLI inference. Returns (contradiction, entailment, neutral) per pair."""
    results: list[tuple[float, float, float]] = []
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        premises   = [p for p, _ in batch]
        hypotheses = [h for _, h in batch]
        enc = tokenizer(
            premises, hypotheses,
            padding=True, truncation=True,
            max_length=MAX_LEN, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        probs = F.softmax(logits, dim=-1).cpu().tolist()
        for row in probs:
            results.append((row[contra_idx], row[entail_idx], row[neutral_idx]))
    return results


def batch_nli_bidirectional(
    pairs: list[tuple[str, str]],
    tokenizer,
    model,
    device,
    contra_idx: int,
    entail_idx: int,
    neutral_idx: int,
    batch_size: int = BATCH_SIZE,
) -> list[dict[str, float]]:
    """
    Run NLI in both directions (A→B and B→A) and derive 4-way normalized
    probabilities for the four RelationType values.

    Strategy:
      - contradiction_prob = avg(contra_AB, contra_BA)
      - support_prob       = min(entail_AB, entail_BA)   ← mutual entailment
      - entailment_prob    = max(entail_AB, entail_BA) - support_prob  ← asymmetric
      - neutral_prob       = avg(neutral_AB, neutral_BA)

    All four are renormalized to sum to 1.0.
    Returns a dict with keys: entailment_prob, support_prob,
                              contradiction_prob, neutral_prob.
    """
    forward  = pairs
    backward = [(h, p) for p, h in pairs]
    all_raw  = _run_nli_batch(
        forward + backward,
        tokenizer, model, device,
        contra_idx, entail_idx, neutral_idx,
        batch_size,
    )
    n   = len(pairs)
    fwd = all_raw[:n]
    bwd = all_raw[n:]

    combined: list[dict[str, float]] = []
    for (c_ab, e_ab, n_ab), (c_ba, e_ba, n_ba) in zip(fwd, bwd):
        p_contra  = (c_ab + c_ba) / 2.0
        p_support = min(e_ab, e_ba)              # both directions entail → mutual support
        p_entail  = max(e_ab, e_ba) - p_support  # asymmetric entailment only
        p_neutral = (n_ab + n_ba) / 2.0

        total = p_contra + p_support + p_entail + p_neutral
        if total > 0:
            p_contra  /= total
            p_support /= total
            p_entail  /= total
            p_neutral /= total

        combined.append({
            "entailment_prob":    round(p_entail,  4),
            "support_prob":       round(p_support, 4),
            "contradiction_prob": round(p_contra,  4),
            "neutral_prob":       round(p_neutral, 4),
            # raw directional scores kept for downstream use
            "_entail_ab": e_ab,
            "_entail_ba": e_ba,
            "_contra_ab": c_ab,
        })
    return combined


def map_relation_type(
    entail_p: float,
    support_p: float,
    contra_p: float,
    neutral_p: float,
) -> RelationType:
    """Argmax over the four normalized probabilities."""
    scores: dict[str, float] = {
        "entailment":    entail_p,
        "support":       support_p,
        "contradiction": contra_p,
        "neutral":       neutral_p,
    }
    best = max(scores, key=scores.__getitem__)
    return best  # type: ignore[return-value]  # keys match RelationType literals exactly


# ── Claim-pair feature builders ────────────────────────────────────────────────

def _entity_features(ff_i: dict, ff_j: dict) -> EntityPairFeatures:
    ents_i = {e["canonical_entity"] for e in (ff_i.get("entity") or [])}
    ents_j = {e["canonical_entity"] for e in (ff_j.get("entity") or [])}
    pres_i = 1 if ents_i else 0
    pres_j = 1 if ents_j else 0
    if ents_i and ents_j:
        union     = ents_i | ents_j
        intersect = ents_i & ents_j
        jaccard   = len(intersect) / len(union) if union else 0.0
        mismatch  = 0 if intersect else 1
    elif not ents_i and not ents_j:
        jaccard  = 1.0
        mismatch = 0
    else:
        jaccard  = 0.0
        mismatch = 0  # absence on one side is not a conflict
    return EntityPairFeatures(
        entity_match=round(jaccard, 4),
        entity_presence_i=pres_i,
        entity_presence_j=pres_j,
        entity_mismatch=mismatch,
    )


def _number_features(ff_i: dict, ff_j: dict) -> Optional[NumberPairFeatures]:
    nums_i = ff_i.get("number") or []
    nums_j = ff_j.get("number") or []
    pres_i = 1 if nums_i else 0
    pres_j = 1 if nums_j else 0
    if not pres_i and not pres_j:
        return None

    vals_i = {round(n["value"], 2) for n in nums_i}
    vals_j = {round(n["value"], 2) for n in nums_j}
    number_match: Optional[int] = None
    diff_ratio:   Optional[float] = None
    unit_mismatch: Optional[int] = None
    number_mismatch = 0

    if pres_i and pres_j:
        shared = vals_i & vals_j
        number_match = 1 if shared else 0
        number_mismatch = 0 if shared else 1

        if not shared and vals_i and vals_j:
            diffs = []
            for v in vals_i:
                closest = min(vals_j, key=lambda x: abs(x - v))
                denom   = max(abs(v), abs(closest), 1e-9)
                diffs.append(abs(v - closest) / denom)
            diff_ratio = round(sum(diffs) / len(diffs), 4)

        # unit mismatch: same numeric value but different units
        tuples_i = {(round(n["value"], 2), n.get("unit")) for n in nums_i}
        tuples_j = {(round(n["value"], 2), n.get("unit")) for n in nums_j}
        shared_vals = {v for v, _ in tuples_i} & {v for v, _ in tuples_j}
        unit_mismatch_flag = 0
        for v in shared_vals:
            units_i = {u for vv, u in tuples_i if vv == v}
            units_j = {u for vv, u in tuples_j if vv == v}
            if units_i != units_j:
                unit_mismatch_flag = 1
                break
        unit_mismatch = unit_mismatch_flag

    return NumberPairFeatures(
        number_presence_i=pres_i,
        number_presence_j=pres_j,
        number_match=number_match,
        number_diff_ratio=diff_ratio,
        unit_mismatch=unit_mismatch,
        number_mismatch=number_mismatch,
    )


def _temporal_features(ff_i: dict, ff_j: dict) -> Optional[TemporalPairFeatures]:
    temp_i = ff_i.get("temporal")
    temp_j = ff_j.get("temporal")
    pres_i = 1 if temp_i else 0
    pres_j = 1 if temp_j else 0
    if not pres_i and not pres_j:
        return None

    temporal_relation     = None
    gran_i                = None
    gran_j                = None
    gran_match: Optional[int] = None
    order_mismatch: Optional[int] = None
    temporal_mismatch     = 0

    if pres_i and pres_j:
        raw_i = (temp_i.get("raw_time") or "").strip().lower()
        raw_j = (temp_j.get("raw_time") or "").strip().lower()
        gran_i = temp_i.get("granularity")
        gran_j = temp_j.get("granularity")
        gran_match = 1 if gran_i == gran_j else 0
        if raw_i and raw_j:
            if raw_i == raw_j:
                temporal_relation = "equal"
                temporal_mismatch = 0
            else:
                temporal_relation = "disjoint"
                temporal_mismatch = 1
                order_mismatch    = 1

    return TemporalPairFeatures(
        temporal_presence_i=pres_i,
        temporal_presence_j=pres_j,
        temporal_relation=temporal_relation,
        temporal_granularity_i=gran_i,
        temporal_granularity_j=gran_j,
        temporal_granularity_match=gran_match,
        temporal_order_mismatch=order_mismatch,
        temporal_mismatch=temporal_mismatch,
    )


def _negation_features(
    ff_i: dict,
    ff_j: dict,
    contra_p: float,
) -> NegationPairFeatures:
    neg_i = int((ff_i.get("negation") or {}).get("polarity", 0))
    neg_j = int((ff_j.get("negation") or {}).get("polarity", 0))
    mismatch = 1 if neg_i != neg_j else 0
    conflict_score = round(contra_p, 4) if mismatch else 0.0
    return NegationPairFeatures(
        negation_i=neg_i,
        negation_j=neg_j,
        negation_mismatch=mismatch,
        negation_conflict_score=conflict_score,
    )


def _attribute_features(
    ff_i: dict,
    ff_j: dict,
    contra_p: float,
) -> Optional[AttributePairFeatures]:
    attrs_i = {e["attribute"] for e in (ff_i.get("entity") or []) if e.get("attribute")}
    attrs_j = {e["attribute"] for e in (ff_j.get("entity") or []) if e.get("attribute")}
    pres_i = 1 if attrs_i else 0
    pres_j = 1 if attrs_j else 0
    if not pres_i and not pres_j:
        return None

    if attrs_i and attrs_j:
        union     = attrs_i | attrs_j
        intersect = attrs_i & attrs_j
        match     = len(intersect) / len(union) if union else 0.0
        rel_mismatch      = 0 if intersect else 1
        implied_missing   = None
        excl_conflict     = round(contra_p, 4) if rel_mismatch else 0.0
    else:
        match           = 0.0
        rel_mismatch    = 0
        implied_missing = 1
        excl_conflict   = None

    return AttributePairFeatures(
        attribute_match=round(match, 4),
        attribute_presence_i=pres_i,
        attribute_presence_j=pres_j,
        relation_mismatch=rel_mismatch,
        implied_attribute_missing=implied_missing,
        attribute_exclusivity_conflict=excl_conflict,
    )


def _aggregate_features(
    entity_f:    EntityPairFeatures,
    number_f:    Optional[NumberPairFeatures],
    temporal_f:  Optional[TemporalPairFeatures],
    negation_f:  NegationPairFeatures,
    attribute_f: Optional[AttributePairFeatures],
    contra_p:    float,
) -> AggregateEdgeFeatures:
    e_mis  = entity_f.entity_mismatch
    n_mis  = number_f.number_mismatch if number_f else 0
    t_mis  = temporal_f.temporal_mismatch if temporal_f else 0
    neg_mis = negation_f.negation_mismatch
    r_mis  = attribute_f.relation_mismatch if attribute_f else 0
    attr_excl = (
        1 if (attribute_f and attribute_f.attribute_exclusivity_conflict
              and attribute_f.attribute_exclusivity_conflict > 0.5)
        else 0
    )

    diagnostic_vector = [e_mis, n_mis, t_mis, neg_mis, r_mis]

    _priority = [
        ("negation_mismatch",              neg_mis),
        ("number_mismatch",                n_mis),
        ("entity_mismatch",                e_mis),
        ("temporal_mismatch",              t_mis),
        ("attribute_exclusivity_conflict", attr_excl),
        ("relation_mismatch",              r_mis),
    ]
    dominant_fail = next(
        (name for name, flag in _priority if flag == 1),
        "none",
    )

    ea_alignment = max(
        0.0,
        min(
            1.0,
            entity_f.entity_match
            * (1.0 - 0.2 * n_mis - 0.2 * t_mis - 0.2 * neg_mis),
        ),
    )
    conflict_intensity = min(
        1.0,
        contra_p * 0.7 + negation_f.negation_conflict_score * 0.3,
    )
    entity_coverage = (entity_f.entity_presence_i + entity_f.entity_presence_j) / 2.0

    return AggregateEdgeFeatures(
        ea_alignment_score=round(ea_alignment, 4),
        conflict_intensity_score=round(conflict_intensity, 4),
        entity_presence_coverage=round(entity_coverage, 4),
        diagnostic_vector=diagnostic_vector,
        dominant_fail_type=dominant_fail,
    )


def build_claim_pair_features(
    ff_i: dict,
    ff_j: dict,
    contra_p: float,
    entail_p: float,
) -> ClaimPairFeatures:
    entity_f    = _entity_features(ff_i, ff_j)
    number_f    = _number_features(ff_i, ff_j)
    temporal_f  = _temporal_features(ff_i, ff_j)
    negation_f  = _negation_features(ff_i, ff_j, contra_p)
    attribute_f = _attribute_features(ff_i, ff_j, contra_p)
    aggregate_f = _aggregate_features(
        entity_f, number_f, temporal_f, negation_f, attribute_f, contra_p,
    )
    return ClaimPairFeatures(
        entity=entity_f,
        attribute=attribute_f,
        number=number_f,
        temporal=temporal_f,
        negation=negation_f,
        aggregate=aggregate_f,
    )


# ── ID helpers ─────────────────────────────────────────────────────────────────

def _query_key(claim_id: str) -> str:
    """c_q000_d00_s01 → q000"""
    parts = claim_id.split("_")
    return parts[1] if len(parts) >= 2 else claim_id


def _doc_key(claim_id: str) -> str:
    """c_q000_d00_s01 → d00"""
    parts = claim_id.split("_")
    return parts[2] if len(parts) >= 3 else ""


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run(
    input_path: Path = INPUT_PATH,
    output_path: Path = OUTPUT_PATH,
    model_name: str = NLI_MODEL,
    batch_size: int = BATCH_SIZE,
    max_pairs_per_query: int = 500,
    cross_doc_only: bool = True,
) -> list[Edge]:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    logger.info("Loading claims from %s", input_path)
    claims_by_query: dict[str, list[dict]] = defaultdict(list)
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            claims_by_query[_query_key(c["claim_id"])].append(c)
    logger.info("%d query groups, %d total claims",
                len(claims_by_query),
                sum(len(v) for v in claims_by_query.values()))

    tokenizer, model, device, contra_idx, entail_idx, neutral_idx = load_nli_model(model_name)

    # Enumerate all valid pairs
    all_pairs: list[tuple[dict, dict]] = []
    for group in claims_by_query.values():
        pairs = [
            (ci, cj)
            for ci, cj in combinations(group, 2)
            if not cross_doc_only or _doc_key(ci["claim_id"]) != _doc_key(cj["claim_id"])
        ]
        if len(pairs) > max_pairs_per_query:
            pairs = pairs[:max_pairs_per_query]
        all_pairs.extend(pairs)
    logger.info("%d pairs to process", len(all_pairs))

    # NLI inference (bidirectional to distinguish support from entailment)
    nli_inputs = [(ci["claim_text"], cj["claim_text"]) for ci, cj in all_pairs]
    logger.info("Running bidirectional NLI (model=%s, batch_size=%d)", model_name, batch_size)
    nli_results = batch_nli_bidirectional(
        nli_inputs, tokenizer, model, device,
        contra_idx, entail_idx, neutral_idx, batch_size,
    )

    # Build Edge objects
    edges: list[Edge] = []
    for (ci, cj), nli in zip(all_pairs, nli_results):
        entail_p  = nli["entailment_prob"]
        support_p = nli["support_prob"]
        contra_p  = nli["contradiction_prob"]
        neutral_p = nli["neutral_prob"]

        nli_feat = NLIEdgeFeatures(
            entailment_prob=entail_p,
            support_prob=support_p,
            contradiction_prob=contra_p,
            neutral_prob=neutral_p,
        )
        relation   = map_relation_type(entail_p, support_p, contra_p, neutral_p)
        confidence = round(max(entail_p, support_p, contra_p, neutral_p), 4)

        ff_i = ci.get("factoid_features") or {}
        ff_j = cj.get("factoid_features") or {}
        pair_feat = build_claim_pair_features(ff_i, ff_j, contra_p, entail_p)

        edges.append(Edge(
            edge_id=f"e_{ci['claim_id']}_{cj['claim_id']}",
            source_claim_id=ci["claim_id"],
            target_claim_id=cj["claim_id"],
            relation_type=relation,
            relation_confidence=confidence,
            nli_features=nli_feat,
            claim_pair_features=pair_feat,
        ))

    logger.info("Writing %d edges to %s", len(edges), output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([e.model_dump() for e in edges], f, ensure_ascii=False, indent=2)
    logger.info("Done.")
    return edges


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Edge feature extraction via NLI.")
    parser.add_argument("--input",  default=str(INPUT_PATH),  help="Input JSONL path")
    parser.add_argument("--output", default=str(OUTPUT_PATH), help="Output JSON path")
    parser.add_argument("--model",  default=NLI_MODEL,        help="HuggingFace NLI model name")
    parser.add_argument("--batch_size",          type=int, default=BATCH_SIZE)
    parser.add_argument("--max_pairs_per_query", type=int, default=500,
                        help="Cap pairs per query group to avoid O(n²) blowup")
    parser.add_argument("--all_pairs", action="store_true",
                        help="Include same-document pairs (default: cross-doc only)")
    args = parser.parse_args()

    run(
        input_path=Path(args.input),
        output_path=Path(args.output),
        model_name=args.model,
        batch_size=args.batch_size,
        max_pairs_per_query=args.max_pairs_per_query,
        cross_doc_only=not args.all_pairs,
    )