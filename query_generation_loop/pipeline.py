"""Conflict-state-guided iterative retrieval loop for conflict-aware RAG.

Sections:
  1. CONFIG & TYPES — constants, TypedDict schemas
  2. INFERENCE — answer type, relevance, slot inference, state inference
  3. QUERY PLANNING & RETRIEVAL — candidates, scoring, local retrieval
  4. FINAL ANSWER & EVIDENCE — evidence selection, answer composition
  5. ITERATIVE LOOP — multi-turn loop, re-analysis
  6. EXPERIMENT RUNNER — CLI entry point (run_experiment, main)

Usage:  python -m query_generation_loop.pipeline --query_id q_002
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import string
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Literal, Optional, TypedDict

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None


# =============================================================================
# SECTION 1: CONFIG & TYPES
# =============================================================================

# -- Config constants ---------------------------------------------------------

METHODOLOGY_BASELINE = "methodology_baseline"
ENHANCED_QUERY_ALIGNED = "enhanced_query_aligned"
MODES = (METHODOLOGY_BASELINE, ENHANCED_QUERY_ALIGNED)

PRIORITY_WEIGHTS = {"conflict_intensity": 0.4, "slot_importance": 0.25, "query_specificity": 0.2, "state_probability": 0.15}
QUERY_SPECIFICITY = {"claim_verification": 0.82, "comparison": 0.78, "context_disambiguation": 0.76, "evidence_coverage_expansion": 0.68}
SLOT_IMPORTANCE = {"temporal": 0.95, "numerical": 0.90, "entity": 0.85, "location": 0.80, "scope": 0.75, "condition": 0.75, "negation": 0.70, "unknown": 0.50}
STATE_INFERENCE_THRESHOLDS = {
    "base_score": 0.2, "small_confidence_gap": 0.08, "large_confidence_gap": 0.20,
    "low_context_completeness": 0.5, "low_evidence_coverage": 0.4,
    "moderate_relation_confidence_min": 0.4, "moderate_relation_confidence_max": 0.75,
    "high_relation_confidence": 0.80, "high_contradiction_prob": 0.70,
    "strong_claim_confidence": 0.85, "strong_claim_gap": 0.15,
}
ITERATIVE_DEFAULTS = {"top_k": 5, "top_n_candidates": 2, "max_iterations": 10, "min_coverage_gain": 0.1, "state_stable_patience": 2}
ENDPOINT_RELEVANCE = {
    "baseline_threshold": 0.0, "enhanced_threshold": 0.15, "pair_max_weight": 0.7, "pair_min_weight": 0.3,
    "director_boost": 0.35, "population_boost": 0.25, "sport_boost": 0.25, "director_downweight_multiplier": 0.6,
}
CONFLICT_QUALITY_WEIGHTS = {"min_endpoint_query_relevance": 0.30, "answer_type_alignment_score": 0.25, "pair_query_relevance": 0.20, "conflict_intensity": 0.15, "relation_confidence": 0.10}
CONFLICT_QUALITY_THRESHOLDS = {"good": 0.65, "weak": 0.40}

# -- TypedDict schemas --------------------------------------------------------

class ClaimFeatures(TypedDict, total=False):
    retrieval_relevance: float; claim_confidence: float; claim_evidence_coverage: float; context_completeness: float

class ClaimRecord(TypedDict, total=False):
    claim_id: str; claim_text: str; canonical_claim_id: str; duplicate_of: str | None
    factoid_features: dict[str, Any] | None; claim_features: ClaimFeatures; evidence: str; doc_id: str; source_id: str

class FactoidClaimRecord(ClaimRecord, total=False):
    factoid_features: dict[str, Any]

class EdgeRecord(TypedDict, total=False):
    edge_id: str; source_claim_id: str; target_claim_id: str; relation_type: str; relation_confidence: float
    nli_features: dict[str, float]; claim_pair_features: dict[str, Any]

class QueryRecord(TypedDict):
    query_id: str; user_query: str

class ConflictRecord(TypedDict, total=False):
    query_id: str; edge_id: str; source_claim_id: str; target_claim_id: str
    claim_i_id: str; claim_j_id: str; claim_i_text: str; claim_j_text: str
    claim_i_confidence: float; claim_j_confidence: float
    claim_i_context_completeness: float; claim_j_context_completeness: float
    claim_i_evidence_coverage: float; claim_j_evidence_coverage: float
    relation_type: str; relation_confidence: float; nli_features: dict[str, float]; claim_pair_features: dict[str, Any]
    slot: str; value_i: Any; value_j: Any
    claim_i_factoid_features: dict[str, Any]; claim_j_factoid_features: dict[str, Any]
    claim_i_predicate: str | None; claim_j_predicate: str | None; conflict_intensity: float
    query_relevance_score: float; claim_i_query_relevance: float; claim_j_query_relevance: float
    min_endpoint_query_relevance: float; max_endpoint_query_relevance: float; pair_query_relevance: float
    answer_type: str; answer_type_alignment_score: float
    conflict_quality_score: float; conflict_quality_label: str; recommended_action: str; quality_signals: dict[str, float]
    region_predicted_state: str; region_id: str

class ConflictFeatures(TypedDict, total=False):
    slot: str; slot_importance: float; claim_i_confidence: float; claim_j_confidence: float; confidence_gap: float
    avg_confidence: float; relation_confidence: float; contradiction_prob: float; conflict_intensity: float
    query_relevance_score: float; claim_i_query_relevance: float; claim_j_query_relevance: float
    min_endpoint_query_relevance: float; max_endpoint_query_relevance: float; pair_query_relevance: float
    answer_type: str; answer_type_alignment_score: float
    claim_i_context_completeness: float; claim_j_context_completeness: float; avg_context_completeness: float
    claim_i_evidence_coverage: float; claim_j_evidence_coverage: float; avg_evidence_coverage: float
    has_temporal_mismatch: bool; has_number_mismatch: bool; has_entity_mismatch: bool; has_negation_mismatch: bool
    dominant_fail_type: str | None; same_predicate: bool | None

StateName = Literal["Contextual", "Underdetermined", "Resolvable"]
QueryType = Literal["context_disambiguation", "evidence_coverage_expansion", "resolution_verification"]

class StateInferenceOutput(TypedDict):
    predicted_state: StateName; state_probabilities: dict[str, float]; state_rationale: dict[str, Any]

class QueryCandidate(TypedDict, total=False):
    candidate_name: str; query_text: str; query_type: QueryType; target_claim_ids: list[str]
    missing_slots_targeted: list[str]; expected_evidence_type: str; candidate_reason: str; query_specificity: float
    state_probability_for_candidate_type: float; priority_score: float; priority: int
    conflict_quality_score: float; conflict_quality_label: str; recommended_action: str; quality_signals: dict[str, float]

class TargetedQuery(QueryCandidate, total=False):
    targeted_query_id: str; query_id: str; edge_id: str; iteration: int
    predicted_state: StateName; state_probabilities: dict[str, float]; state_rationale: dict[str, Any]

class IterationLog(TypedDict, total=False):
    iteration: int; query_id: str; edge_id: str; conflict_slot: str; conflict_intensity: float
    relation_confidence: float; contradiction_prob: float; predicted_state: StateName
    state_probabilities: dict[str, float]; conflict_quality_score: float; conflict_quality_label: str
    recommended_action: str; quality_signals: dict[str, float]; selected_queries: list[TargetedQuery]
    retrieved_claim_ids: list[str]; new_claim_ids: list[str]; coverage_gain: float
    seen_claims_before: int; seen_claims_after: int; stop_decision: dict[str, Any]

class ExperimentSummary(TypedDict, total=False):
    query_id: str; user_query: str; mode: str; num_claims: int; num_edges: int; num_contradiction_edges: int
    num_conflicts_processed: int; num_real_conflicts_processed: int; num_synthetic_corrective_queries: int
    num_generated_queries: int; state_distribution: dict[str, int]; query_type_distribution: dict[str, int]
    slot_distribution: dict[str, int]; avg_priority_score: float; avg_coverage_gain: float
    avg_new_claims_per_iteration: float; avg_conflict_quality_score: float
    conflict_quality_distribution: dict[str, int]; corrective_action_distribution: dict[str, int]
    corrective_fallback_used: bool; stopping_reason_distribution: dict[str, int]
    final_loop_decision: dict[str, Any]; final_evidence_selection: dict[str, Any]; final_answer: dict[str, Any]
    final_resolution_status: str; final_answer_policy: str; final_answer_confidence: float
    final_answer_should_be_generated: bool; final_response_type: str; answer_quality: dict[str, Any]
    answer_quality_score: float; response_quality_score: float; factual_answer_quality_score: float | None
    evaluated_response_type: str; answer_quality_gate_decision: str
    top_examples: list[dict[str, Any]]; risks_or_warnings: list[str]


# =============================================================================
# SECTION 2: INFERENCE — answer type, relevance, slot inference, state
# =============================================================================

_STOPWORDS = {"a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "it",
              "of", "on", "or", "the", "to", "was", "were", "what", "when", "where", "which",
              "who", "whom", "whose", "with", "this", "that"}
_INTENT_BOOSTS = [
    ({"director", "directors", "directed"}, {"director", "directors", "directed", "filmmaker"}, ENDPOINT_RELEVANCE["director_boost"]),
    ({"population"}, {"population", "people", "census"}, ENDPOINT_RELEVANCE["population_boost"]),
    ({"sport"}, {"sport", "player", "team"}, ENDPOINT_RELEVANCE["sport_boost"]),
]
_DIRECTOR_TERMS = {"director", "directors", "directed", "filmmaker"}
_DIRECTOR_DOWNWEIGHT_TERMS = {"released", "music", "lyrics", "song", "actor", "plot"}
_RELEASE_TERMS = {"released", "release", "year", "date"}


def _tokens(text: str) -> set[str]:
    t = str.maketrans(string.punctuation, " " * len(string.punctuation))
    return {token for token in re.findall(r"\w+", (text or "").lower().translate(t)) if token not in _STOPWORDS}


def _float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def infer_query_answer_type(user_query: str) -> str:
    tokens = _tokens(user_query)
    normalized = " ".join((user_query or "").lower().split())
    if tokens & {"director", "directors", "directed", "filmmaker"}: return "director"
    if tokens & {"population", "people", "residents", "census"}: return "population"
    if normalized.startswith("when") or tokens & {"year", "date", "released"}: return "date_or_time"
    if normalized.startswith("where") or tokens & {"located", "location"}: return "location"
    if normalized.startswith("who"): return "person"
    if "how many" in normalized or "how much" in normalized or "number" in tokens: return "numeric"
    return "generic"


def compute_answer_type_alignment(answer_type: str, conflict_slot: str, claim_i_text: str, claim_j_text: str) -> float:
    ci, cj = _tokens(claim_i_text), _tokens(claim_j_text); both = ci | cj; slot = conflict_slot or "unknown"
    if answer_type == "director":
        di, dj = bool(ci & _DIRECTOR_TERMS), bool(cj & _DIRECTOR_TERMS)
        if slot in {"entity", "unknown"} and di and dj: return 1.0
        if slot in {"entity", "unknown"} and (di or dj): return 0.65
        if slot == "numerical": return 0.35 if di and dj else (0.12 if di or dj else (0.05 if both & _RELEASE_TERMS else 0.05))
        return 0.2 if both & _DIRECTOR_TERMS else 0.05
    if answer_type == "population":
        return min(1.0, (0.6 if slot == "numerical" else 0.25) + (0.3 if both & {"population", "people", "census", "residents"} else 0.0))
    if answer_type == "date_or_time": return 0.9 if slot == "temporal" else (0.75 if slot == "numerical" and both & _RELEASE_TERMS else 0.3)
    if answer_type == "location": return 0.85 if slot in {"entity", "location"} else 0.3
    if answer_type == "person": return 0.75 if slot in {"entity", "unknown"} else 0.35
    if answer_type == "numeric": return 0.8 if slot == "numerical" else 0.3
    return 0.5


def compute_endpoint_query_relevance(user_query: str, claim_text: str) -> float:
    qt, ct = _tokens(user_query), _tokens(claim_text)
    if not qt or not ct: return 0.0
    score = len(qt & ct) / len(qt)
    for q_terms, c_terms, boost in _INTENT_BOOSTS:
        if qt & q_terms and ct & c_terms: score += boost
    if qt & {"director", "directors", "directed"} and ct & _DIRECTOR_DOWNWEIGHT_TERMS and not (ct & {"director", "directors", "directed", "filmmaker"}):
        score *= ENDPOINT_RELEVANCE["director_downweight_multiplier"]
    return max(0.0, min(1.0, score))


def compute_query_relevance_metrics(user_query: str, claim_i_text: str, claim_j_text: str) -> dict[str, float]:
    ci, cj = compute_endpoint_query_relevance(user_query, claim_i_text), compute_endpoint_query_relevance(user_query, claim_j_text)
    mn, mx = min(ci, cj), max(ci, cj)
    return {"claim_i_query_relevance": ci, "claim_j_query_relevance": cj, "min_endpoint_query_relevance": mn,
            "max_endpoint_query_relevance": mx, "pair_query_relevance": ENDPOINT_RELEVANCE["pair_max_weight"] * mx + ENDPOINT_RELEVANCE["pair_min_weight"] * mn}


def compute_query_relevance(user_query: str, claim_i_text: str, claim_j_text: str) -> float:
    return compute_query_relevance_metrics(user_query, claim_i_text, claim_j_text)["pair_query_relevance"]


# -- Conflict features --------------------------------------------------------

def _dominant_fail_type(conflict_record: ConflictRecord) -> str | None:
    pf = (conflict_record.get("claim_pair_features") or {}).get("aggregate") or {}
    d = pf.get("dominant_fail_type")
    return str(d) if d is not None else None


def extract_conflict_features(conflict_record: ConflictRecord) -> ConflictFeatures:
    ci_c, cj_c = _float(conflict_record.get("claim_i_confidence")), _float(conflict_record.get("claim_j_confidence"))
    cgap = abs(ci_c - cj_c)
    ci_ctx, cj_ctx = _float(conflict_record.get("claim_i_context_completeness")), _float(conflict_record.get("claim_j_context_completeness"))
    ci_cov, cj_cov = _float(conflict_record.get("claim_i_evidence_coverage")), _float(conflict_record.get("claim_j_evidence_coverage"))
    slot = str(conflict_record.get("slot") or "unknown")
    nli = conflict_record.get("nli_features") or {}
    p_i, p_j = conflict_record.get("claim_i_predicate"), conflict_record.get("claim_j_predicate")
    same_p = (p_i == p_j) if p_i and p_j else None
    return {
        "slot": slot, "slot_importance": SLOT_IMPORTANCE.get(slot, SLOT_IMPORTANCE["unknown"]),
        "claim_i_confidence": ci_c, "claim_j_confidence": cj_c, "confidence_gap": cgap, "avg_confidence": (ci_c + cj_c) / 2.0,
        "relation_confidence": _float(conflict_record.get("relation_confidence")),
        "contradiction_prob": _float(nli.get("contradiction_prob")),
        "conflict_intensity": _float(conflict_record.get("conflict_intensity")),
        "query_relevance_score": _float(conflict_record.get("query_relevance_score")),
        "claim_i_query_relevance": _float(conflict_record.get("claim_i_query_relevance")),
        "claim_j_query_relevance": _float(conflict_record.get("claim_j_query_relevance")),
        "min_endpoint_query_relevance": _float(conflict_record.get("min_endpoint_query_relevance")),
        "max_endpoint_query_relevance": _float(conflict_record.get("max_endpoint_query_relevance")),
        "pair_query_relevance": _float(conflict_record.get("pair_query_relevance")),
        "answer_type": str(conflict_record.get("answer_type") or "generic"),
        "answer_type_alignment_score": _float(conflict_record.get("answer_type_alignment_score")),
        "claim_i_context_completeness": ci_ctx, "claim_j_context_completeness": cj_ctx, "avg_context_completeness": (ci_ctx + cj_ctx) / 2.0,
        "claim_i_evidence_coverage": ci_cov, "claim_j_evidence_coverage": cj_cov, "avg_evidence_coverage": (ci_cov + cj_cov) / 2.0,
        "has_temporal_mismatch": slot == "temporal", "has_number_mismatch": slot == "numerical",
        "has_entity_mismatch": slot == "entity", "has_negation_mismatch": slot == "negation",
        "dominant_fail_type": _dominant_fail_type(conflict_record), "same_predicate": same_p,
    }


# -- Slot inference -----------------------------------------------------------

def _is_one(value: Any) -> bool:
    return value == 1 or value is True or str(value).strip() == "1"


def infer_slot_from_pair_features(claim_pair_features: dict[str, Any]) -> str:
    n = (claim_pair_features.get("negation") or {})
    if _is_one(n.get("negation_mismatch")): return "negation"
    nb = (claim_pair_features.get("number") or {})
    if _is_one(nb.get("number_mismatch")): return "numerical"
    t = (claim_pair_features.get("temporal") or {})
    if _is_one(t.get("temporal_mismatch")) or _is_one(t.get("temporal_order_mismatch")): return "temporal"
    e = (claim_pair_features.get("entity") or {})
    if _is_one(e.get("entity_mismatch")): return "entity"
    dom = str(((claim_pair_features.get("aggregate") or {}).get("dominant_fail_type") or "")).lower()
    if dom in {"negation"}: return "negation"
    if dom in {"number", "numerical"}: return "numerical"
    if dom in {"temporal", "time"}: return "temporal"
    if dom in {"entity"}: return "entity"
    if dom in {"location", "scope", "condition"}: return dom
    return "unknown"


def _snippet(text: str, max_chars: int = 120) -> str:
    n = " ".join((text or "").split())
    return n if len(n) <= max_chars else n[:max_chars - 1].rstrip() + "..."


def _polarity_label(fd: dict[str, Any]) -> str:
    p = (fd.get("negation") or {}).get("polarity", 0) if isinstance(fd.get("negation"), dict) else 0
    try:
        return "negated" if int(p) == 1 else "affirmed"
    except (TypeError, ValueError):
        return "affirmed"


def _format_numbers(fd: dict[str, Any]) -> str | None:
    nums = fd.get("number") or []
    if not isinstance(nums, list) or not nums: return None
    vals: list[str] = []
    for item in nums:
        if not isinstance(item, dict): continue
        v = item.get("value")
        if v is None: continue
        t = str(v)
        if t.endswith(".0"):
            t = t[:-2]  # "10.0" -> "10"
        u = item.get("unit")
        vals.append(f"{t} {u}" if u else t)
    return "; ".join(vals) if vals else None


def _format_temporal(fd: dict[str, Any]) -> str | None:
    t = fd.get("temporal")
    if not isinstance(t, dict): return None
    if t.get("raw_time"): return str(t["raw_time"])
    s, e = t.get("start"), t.get("end")
    if s and e and s != e: return f"{s} to {e}"
    if s: return str(s)
    if e: return str(e)
    return str(t.get("granularity")) if t.get("granularity") else None


def _format_entities(fd: dict[str, Any]) -> str | None:
    ents = fd.get("entity") or []
    if not isinstance(ents, list) or not ents: return None
    vals: list[str] = []
    for item in ents:
        if not isinstance(item, dict): continue
        if item.get("canonical_entity"): vals.append(str(item["canonical_entity"]))
        elif item.get("raw_mention"): vals.append(str(item["raw_mention"]))
    return "; ".join(dict.fromkeys(vals)) if vals else None


def extract_slot_values(slot: str, fd_i: dict[str, Any], fd_j: dict[str, Any], text_i: str, text_j: str) -> tuple[str, str]:
    if slot == "negation": return _polarity_label(fd_i), _polarity_label(fd_j)
    if slot == "numerical": return _format_numbers(fd_i) or _snippet(text_i), _format_numbers(fd_j) or _snippet(text_j)
    if slot == "temporal": return _format_temporal(fd_i) or _snippet(text_i), _format_temporal(fd_j) or _snippet(text_j)
    if slot == "entity": return _format_entities(fd_i) or _snippet(text_i), _format_entities(fd_j) or _snippet(text_j)
    return _snippet(text_i), _snippet(text_j)


# -- State inference ----------------------------------------------------------

def _normalize(scores: dict[str, float]) -> dict[str, float]:
    safe = {k: max(0.0, float(v)) for k, v in scores.items()}
    t = sum(safe.values())
    return {k: v / t for k, v in safe.items()} if t > 0 else {k: 1.0 / len(safe) for k in safe}


def infer_conflict_state(features: ConflictFeatures) -> StateInferenceOutput:
    ctx, und, res = 0.2, 0.2, 0.2
    signals: list[str] = []
    slot = str(features.get("slot") or "unknown")
    fail = str(features.get("dominant_fail_type") or "none").lower()
    cgap = float(features.get("confidence_gap") or 0.0)
    cov = float(features.get("avg_evidence_coverage") or 0.0)
    ctx_c = float(features.get("avg_context_completeness") or 0.0)
    rc = float(features.get("relation_confidence") or 0.0)
    cp = float(features.get("contradiction_prob") or 0.0)
    c1 = float(features.get("claim_i_confidence") or 0.0)
    c2 = float(features.get("claim_j_confidence") or 0.0)
    sp = features.get("same_predicate")

    if slot in {"temporal", "entity", "location", "scope", "condition", "negation"}:
        ctx += 0.5; signals.append("contextual_slot")
    if fail not in {"", "none", "null"}:
        ctx += 0.25; signals.append("dominant_fail_type")
    if ctx_c < 0.5: ctx += 0.25; signals.append("low_context_completeness")
    if sp is False: ctx += 0.15; signals.append("predicate_mismatch")
    if cgap <= 0.08: und += 0.5; signals.append("small_confidence_gap")
    if cov < 0.4: und += 0.3; signals.append("low_evidence_coverage")
    if 0.4 <= rc <= 0.75: und += 0.2; signals.append("moderate_relation_confidence")
    if cgap >= 0.20: res += 0.45; signals.append("large_confidence_gap")
    if rc >= 0.80: res += 0.30; signals.append("high_relation_confidence")
    if cp >= 0.70: res += 0.30; signals.append("high_contradiction_prob")
    if max(c1, c2) >= 0.85 and cgap >= 0.15: res += 0.25; signals.append("one_claim_clearly_stronger")

    probs = _normalize({"Contextual": ctx, "Underdetermined": und, "Resolvable": res})
    state = max(probs, key=probs.get)
    return {"predicted_state": state, "state_probabilities": probs,
            "state_rationale": {"main_signal": signals[0] if signals else "balanced_default", "signals": signals,
                                "feature_snapshot": dict(features), "reason": f"Predicted {state} from heuristic scores."}}


# =============================================================================
# SECTION 3: QUERY PLANNING & RETRIEVAL
# =============================================================================

_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


def _overlap_score(qt: list[str], dt: list[str]) -> float:
    if not qt or not dt: return 0.0
    qc, dc = Counter(qt), Counter(dt)
    overlap = sum(min(qc[t], dc[t]) for t in qc)
    d = math.sqrt(sum(qc.values()) * sum(dc.values()))
    return overlap / d if d else 0.0


def retrieve_claims(query_text: str, candidate_claims: list[dict[str, Any]], top_k: int = 5, exclude_claim_ids: list[str] | None = None) -> list[dict[str, Any]]:
    excluded = set(exclude_claim_ids or [])
    filtered = [c for c in candidate_claims if c.get("claim_id") not in excluded]
    if top_k <= 0 or not filtered: return []
    qt = _tokenize(query_text)
    ct = [_tokenize(str(c.get("claim_text") or "")) for c in filtered]
    if BM25Okapi is not None and any(ct):
        scores = BM25Okapi(ct).get_scores(qt)
    else:
        scores = [_overlap_score(qt, dt) for dt in ct]
    ranked = sorted(zip(filtered, scores), key=lambda x: float(x[1]), reverse=True)
    return [{"claim_id": str(c.get("claim_id", "")), "claim_text": str(c.get("claim_text", "")), "score": float(s)} for c, s in ranked[:top_k]]


# -- Query planning -----------------------------------------------------------

def clean_text_for_template(text: str) -> str:
    return " ".join((text or "").split()).strip().rstrip(" .?!")


def _state_probability_for_query_type(query_type: str, state_output: StateInferenceOutput) -> float:
    p = state_output.get("state_probabilities", {})
    if query_type == "context_disambiguation": return float(p.get("Contextual", 0.0))
    if query_type == "evidence_coverage_expansion": return float(p.get("Underdetermined", 0.0))
    if query_type == "resolution_verification": return float(p.get("Resolvable", 0.0))
    return 0.0


def generate_query_candidates(user_query: str, conflict_record: ConflictRecord, state_output: StateInferenceOutput) -> list[QueryCandidate]:
    uq = clean_text_for_template(user_query)
    slot = str(conflict_record.get("slot") or "unknown")
    vi = clean_text_for_template(str(conflict_record.get("value_i") or "unknown"))
    vj = clean_text_for_template(str(conflict_record.get("value_j") or "unknown"))
    ti = clean_text_for_template(str(conflict_record.get("claim_i_text") or ""))
    tj = clean_text_for_template(str(conflict_record.get("claim_j_text") or ""))
    idi, idj = str(conflict_record.get("claim_i_id") or ""), str(conflict_record.get("claim_j_id") or "")
    qs = QUERY_SPECIFICITY
    return [
        {"candidate_name": "context_disambiguation_query", "query_type": "context_disambiguation",
         "query_text": f"Clarify the context for: {uq}. Conflicting {slot} values: {vi} vs {vj}. What context, time period, entity meaning, or condition explains the difference?",
         "target_claim_ids": [idi, idj], "missing_slots_targeted": [slot],
         "expected_evidence_type": "contextual disambiguating evidence", "candidate_reason": "Targets missing context behind the contradiction.",
         "query_specificity": qs["context_disambiguation"]},
        {"candidate_name": "evidence_coverage_expansion_query", "query_type": "evidence_coverage_expansion",
         "query_text": f"Find additional evidence for: {uq}. Current conflicting claims say: {ti} vs {tj}. Retrieve independent evidence covering the missing {slot} information.",
         "target_claim_ids": [idi, idj], "missing_slots_targeted": [slot],
         "expected_evidence_type": "broader corroborating evidence", "candidate_reason": "Expands evidence coverage before resolving the contradiction.",
         "query_specificity": qs["evidence_coverage_expansion"]},
        {"candidate_name": "claim_i_verification_query", "query_type": "resolution_verification",
         "query_text": f"Verify whether this claim is correct for: {uq}. Claim: {ti}. Find authoritative evidence about {slot}: {vi}.",
         "target_claim_ids": [idi], "missing_slots_targeted": [slot],
         "expected_evidence_type": "authoritative evidence for claim_i", "candidate_reason": "Directly verifies the source-side claim.",
         "query_specificity": qs["claim_verification"]},
        {"candidate_name": "claim_j_verification_query", "query_type": "resolution_verification",
         "query_text": f"Verify whether this claim is correct for: {uq}. Claim: {tj}. Find authoritative evidence about {slot}: {vj}.",
         "target_claim_ids": [idj], "missing_slots_targeted": [slot],
         "expected_evidence_type": "authoritative evidence for claim_j", "candidate_reason": "Directly verifies the target-side claim.",
         "query_specificity": qs["claim_verification"]},
        {"candidate_name": "comparison_query", "query_type": "resolution_verification",
         "query_text": f"Compare these conflicting claims for: {uq}. Claim A: {ti}. Claim B: {tj}. Which is better supported, and what evidence resolves the {slot} conflict?",
         "target_claim_ids": [idi, idj], "missing_slots_targeted": [slot],
         "expected_evidence_type": "comparative authoritative evidence", "candidate_reason": "Compares both claims in one retrieval request.",
         "query_specificity": qs["comparison"]},
    ]


def score_query_candidate(candidate: QueryCandidate, features: ConflictFeatures, state_output: StateInferenceOutput) -> float:
    sp = _state_probability_for_query_type(str(candidate.get("query_type") or ""), state_output)
    return (PRIORITY_WEIGHTS["conflict_intensity"] * float(features.get("conflict_intensity") or 0.0)
            + PRIORITY_WEIGHTS["slot_importance"] * float(features.get("slot_importance") or 0.0)
            + PRIORITY_WEIGHTS["query_specificity"] * float(candidate.get("query_specificity") or 0.0)
            + PRIORITY_WEIGHTS["state_probability"] * sp)


def select_top_queries(candidates: list[QueryCandidate], top_n: int = 2, *, query_id: str = "", edge_id: str = "", iteration: int = 0, state_output: StateInferenceOutput | None = None) -> list[TargetedQuery]:
    ranked = sorted(candidates, key=lambda x: float(x.get("priority_score") or 0.0), reverse=True)
    selected: list[TargetedQuery] = []
    for idx, cand in enumerate(ranked[:top_n], start=1):
        t: TargetedQuery = dict(cand)
        t.update({"targeted_query_id": f"tq_{query_id}_{edge_id}_{iteration}_{idx}", "priority": idx, "query_id": query_id, "edge_id": edge_id, "iteration": iteration})
        if state_output is not None:
            t["predicted_state"] = state_output["predicted_state"]
            t["state_probabilities"] = state_output["state_probabilities"]
            t["state_rationale"] = state_output["state_rationale"]
        selected.append(t)
    return selected


# =============================================================================
# SECTION 4: FINAL ANSWER & EVIDENCE
# =============================================================================

_STOP_REASONS_UNRESOLVED = {"state_stable", "max_iterations_reached", "no_new_claims", "no_new_relevant_claims", "coverage_gain_below_threshold"}


def _claim_confidence(claim: dict[str, Any]) -> float:
    cf = claim.get("claim_features") or {}
    return _float(cf.get("claim_confidence") if isinstance(cf, dict) else claim.get("claim_confidence"))


def _latest_state(iteration_logs: list[dict[str, Any]]) -> str:
    return str(iteration_logs[-1].get("predicted_state") or "") if iteration_logs else ""


def _dominant_state(summary: dict[str, Any], iteration_logs: list[dict[str, Any]]) -> str:
    lst = _latest_state(iteration_logs)
    if lst: return lst
    dist = summary.get("state_distribution") or {}
    return max(dist.items(), key=lambda x: int(x[1]))[0] if dist else ""


def _primary_stop_reason(summary: dict[str, Any], iteration_logs: list[dict[str, Any]]) -> str:
    if iteration_logs:
        r = (iteration_logs[-1].get("stop_decision") or {}).get("stopping_reason")
        if r: return str(r)
    dist = summary.get("stopping_reason_distribution") or {}
    return max(dist.items(), key=lambda x: int(x[1]))[0] if dist else "unknown"


def derive_final_loop_decision(summary: dict[str, Any], iteration_logs: list[dict[str, Any]], targeted_queries: list[dict[str, Any]]) -> dict[str, Any]:
    state = _dominant_state(summary, iteration_logs)
    stop = _primary_stop_reason(summary, iteration_logs)
    real_conflicts = int(summary.get("num_real_conflicts_processed") or 0)
    query_edges = int(summary.get("num_query_contradiction_edges") or 0)
    fallback = bool(summary.get("corrective_fallback_used", False))
    avg_q = _float(summary.get("avg_conflict_quality_score"))
    has_q = bool(summary.get("conflict_quality_distribution"))
    unres: list[str] = []; uncert: list[str] = []

    if query_edges == 0: unres.append("No contradiction edges were available for this query.")
    if real_conflicts == 0: unres.append("No real contradiction conflict was processed by the loop.")
    if fallback: unres.append("Corrective fallback requested broader evidence before answering.")

    if query_edges == 0 or real_conflicts == 0 or fallback:
        eq = [q for q in targeted_queries if q.get("query_type") == "evidence_coverage_expansion"]
        if eq: uncert.append(str(eq[0].get("query_text") or "More targeted evidence is needed."))
        return {"resolution_status": "insufficient_evidence", "answer_policy": "evidence_coverage_expansion_before_answer",
                "confidence_level": "low", "primary_stop_reason": stop, "why_not_resolved": unres,
                "recommended_next_action": "Run the generated evidence coverage expansion query before composing a final answer.",
                "remaining_uncertainties": uncert or ["More targeted evidence is needed before selecting a final answer."]}

    if state == "Resolvable" and (not has_q or avg_q >= 0.65):
        if has_q and avg_q >= 0.65:
            return {"resolution_status": "resolved", "answer_policy": "answer_with_selected_claim", "confidence_level": "high",
                    "primary_stop_reason": stop, "why_not_resolved": [],
                    "recommended_next_action": "Use the selected supporting claim as the final answer evidence.", "remaining_uncertainties": []}
        return {"resolution_status": "likely_resolved_with_caution", "answer_policy": "answer_with_caution", "confidence_level": "medium",
                "primary_stop_reason": stop, "why_not_resolved": ["The baseline mode does not provide conflict-quality evaluator scores."],
                "recommended_next_action": "Use the selected claim, but report medium confidence.",
                "remaining_uncertainties": ["Conflict quality was not evaluated in baseline mode."]}

    if state == "Resolvable" and avg_q >= 0.40:
        return {"resolution_status": "likely_resolved_with_caution", "answer_policy": "answer_with_caution", "confidence_level": "medium",
                "primary_stop_reason": stop, "why_not_resolved": ["Average conflict quality is below the high-confidence threshold."],
                "recommended_next_action": "Use the selected claim with a caution note.", "remaining_uncertainties": ["Conflict quality is only weak or moderate."]}

    if stop in _STOP_REASONS_UNRESOLVED: unres.append(f"The loop stopped with {stop} before resolving the conflict.")
    else: unres.append("The loop did not reach a resolvable final state.")
    return {"resolution_status": "unresolved", "answer_policy": "present_competing_claims_with_uncertainty", "confidence_level": "low",
            "primary_stop_reason": stop, "why_not_resolved": unres,
            "recommended_next_action": "Present competing claims and gather additional targeted evidence.",
            "remaining_uncertainties": ["Available claims remain conflicting or underdetermined."]}


def _target_claim_counts(targeted_queries: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for idx, q in enumerate(targeted_queries):
        if q.get("edge_id") == "corrective_fallback": continue
        w = max(1, len(targeted_queries) - idx)
        for cid in q.get("target_claim_ids") or []:
            if cid: counts[str(cid)] += w
    return counts


def _answer_values(text: str, answer_type: str) -> set[str]:
    if answer_type in {"population", "numeric"}: return set(re.findall(r"\d[\d,]*(?:\.\d+)?", text))
    if answer_type == "date_or_time": return set(re.findall(r"\b(?:1[5-9]\d{2}|20\d{2}|21\d{2})\b", text))
    return set()


def _is_query_aligned(text_lower: str, answer_type: str, slot_terms: set[str]) -> bool:
    return (answer_type != "generic" and answer_type in text_lower) or any(t in text_lower for t in slot_terms if t and t != "unknown")


def select_final_evidence(claims: list[dict[str, Any]], targeted_queries: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    if summary.get("final_resolution_status") == "insufficient_evidence" or summary.get("corrective_fallback_used"):
        excluded = [{"claim_id": str(c.get("claim_id", "")), "claim_text": str(c.get("claim_text", "")), "reason": "insufficient_evidence"} for c in claims if c.get("claim_id")]
        return {"selected_answer_claims": [], "competing_claims": [], "excluded_claims": excluded, "supporting_claims": []}

    target_counts = _target_claim_counts(targeted_queries)
    candidates: list[dict[str, Any]] = []; excluded: list[dict[str, Any]] = []
    answer_type = str(summary.get("inferred_answer_type") or "generic")
    slot_terms = set(str(s).lower() for s in (summary.get("slot_distribution") or {}).keys())

    for claim in claims:
        cid = str(claim.get("claim_id") or "")
        if not cid: continue
        text = str(claim.get("claim_text") or "")
        if cid not in target_counts:
            excluded.append({"claim_id": cid, "claim_text": text, "reason": "not_targeted_by_top_queries"}); continue
        if not _is_query_aligned(text.lower(), answer_type, slot_terms):
            excluded.append({"claim_id": cid, "claim_text": text, "reason": "low_query_answer_alignment"}); continue
        conf = _claim_confidence(claim)
        score = 0.5 * min(1.0, target_counts[cid] / 2.0) + 0.35 * conf + 0.1
        reasons = ["targeted_by_top_query", "aligned_with_answer_type_or_slot"]
        vals = _answer_values(text, answer_type)
        if vals: reasons.append("contains_answer_value"); score += 0.05
        candidates.append({"claim_id": cid, "claim_text": text, "selection_score": min(1.0, score), "selection_reason": reasons, "answer_values": vals})

    candidates.sort(key=lambda x: x["selection_score"], reverse=True)
    if not candidates: return {"selected_answer_claims": [], "competing_claims": [], "excluded_claims": excluded, "supporting_claims": []}

    winner = candidates[0]
    selected = [{"claim_id": winner["claim_id"], "claim_text": winner["claim_text"], "selection_score": winner["selection_score"], "selection_reason": winner["selection_reason"]}]
    wv = winner.get("answer_values") or set()
    competing = []
    for cand in candidates[1:]:
        cv = cand.get("answer_values") or set()
        reason = ["lower_scoring_query_aligned_target_claim"]
        if wv and cv and cv != wv: reason.append("different_answer_value_for_same_target")
        competing.append({"claim_id": cand["claim_id"], "claim_text": cand["claim_text"], "selection_score": cand["selection_score"], "competition_reason": reason})
    return {"selected_answer_claims": selected, "competing_claims": competing, "excluded_claims": excluded, "supporting_claims": selected}


def compose_final_answer(user_query: str, final_decision: dict[str, Any], final_evidence: dict[str, Any]) -> dict[str, Any]:
    status = str(final_decision.get("resolution_status") or "unresolved")
    policy = str(final_decision.get("answer_policy") or "present_competing_claims_with_uncertainty")
    selected = final_evidence.get("selected_answer_claims") or final_evidence.get("selected_supporting_claims") or []
    competing = final_evidence.get("competing_claims") or []
    remaining = final_decision.get("remaining_uncertainties") or []
    conf_by_level = {"high": 0.9, "medium": 0.65, "low": 0.25}
    conf = conf_by_level.get(str(final_decision.get("confidence_level") or "low"), 0.25)

    if status == "resolved" and selected:
        ans = f"Best-supported answer: {selected[0]['claim_text']}"; rt = "factual_answer"
    elif status == "likely_resolved_with_caution" and selected:
        ans = f"Likely answer: {selected[0]['claim_text']} The evidence supports this answer, but not strongly enough for high confidence."
        if competing: ans += " Competing claims remain: " + "; ".join(c["claim_text"] for c in competing[:2])
        rt = "cautious_factual_answer"
    elif status == "insufficient_evidence":
        act = str(final_decision.get("recommended_next_action") or "More targeted evidence is needed before answering.")
        hint = f" Suggested evidence query: {remaining[0]}" if remaining else ""
        ans = f"Insufficient evidence to choose a final answer for: {user_query}. {act}{hint}"; conf = 0.0; rt = "insufficient_evidence_guidance"
    else:
        texts = "; ".join(c["claim_text"] for c in [*selected, *competing][:3]) if selected or competing else "no final answer claim was selected."
        ans = f"Available claims remain unresolved: {texts}"; rt = "uncertainty_answer"

    return {"answer": ans, "answer_confidence": conf, "selected_answer_claims": selected, "competing_claims": competing,
            "excluded_claims": final_evidence.get("excluded_claims", []), "supporting_claims": selected,
            "remaining_uncertainties": remaining, "answer_policy": policy, "resolution_status": status, "final_response_type": rt}


# =============================================================================
# SECTION 5: ITERATIVE LOOP
# =============================================================================

ReanalysisFn = Optional[Callable[[list[dict[str, Any]], list[dict[str, Any]], list[ConflictRecord]], dict[str, Any]]]


def _classify_new_claim_support(new_text: str, i_text: str, j_text: str, cache: dict[str, set[str]], new_cid: str, i_id: str, j_id: str) -> tuple[str, float]:
    new_t = cache.get(new_cid, set(_TOKEN_RE.findall((new_text or "").lower())))
    i_t = cache.get(i_id, set(_TOKEN_RE.findall((i_text or "").lower())))
    j_t = cache.get(j_id, set(_TOKEN_RE.findall((j_text or "").lower())))
    if not i_t or not j_t: return "neutral", 0.0
    oi, oj = len(new_t & i_t) / max(len(i_t), 1), len(new_t & j_t) / max(len(j_t), 1)
    if oi > oj * 1.5 and oi > 0.05: return "claim_i", oi
    if oj > oi * 1.5 and oj > 0.05: return "claim_j", oj
    return "neutral", max(oi, oj)


def _reanalyze_conflict(conflict: ConflictRecord, sup_i: list[str], sup_j: list[str], side: str) -> dict[str, Any]:
    orig = float(conflict.get("conflict_intensity") or 0.0)
    bal = len(sup_i) - len(sup_j)
    red = 0.0; resolved = False; reason = ""
    if side == "claim_i" and len(sup_i) >= 2:
        red = min(0.25, abs(bal) * 0.08); reason = f"New evidence supports claim_i (+{len(sup_i)} supporting claims)"
    elif side == "claim_j" and len(sup_j) >= 2:
        red = min(0.25, abs(bal) * 0.08); reason = f"New evidence supports claim_j (+{len(sup_j)} supporting claims)"
    elif side == "neutral": reason = "New evidence is neutral, no side favored"
    new_i = max(0.0, orig - red)
    if new_i < 0.15: resolved = True; reason += "; intensity dropped below threshold -> conflict resolved"
    return {"conflict_intensity": new_i, "resolved": resolved, "resolution_reason": reason.strip("; "), "total_support_i": len(sup_i), "total_support_j": len(sup_j)}


def should_stop_loop(state: dict[str, Any]) -> dict[str, Any]:
    signals = {"max_iterations_reached": bool(state.get("max_iterations_reached", False)),
               "all_conflicts_resolved": bool(state.get("all_conflicts_resolved", False)),
               "no_new_claims": bool(state.get("no_new_claims", False)),
               "no_new_relevant_claims": bool(state.get("no_new_relevant_claims", state.get("no_new_claims", False))),
               "coverage_gain_below_threshold": bool(state.get("coverage_gain_below_threshold", False)),
               "state_stable": bool(state.get("state_stable", False))}
    reason = "continue"
    for k, v in signals.items():
        if v: reason = k; break
    return {"should_stop": reason != "continue", "stopping_reason": reason, "signals": signals}


def run_offline_iteration(query_id: str, user_query: str, conflicts: list[ConflictRecord], claims_for_query: list[dict[str, Any]],
                          top_k: int = 5, top_n_candidates: int = 2, max_iterations: int = 10,
                          min_coverage_gain: float = 0.1, state_stable_patience: int = 2, reanalysis_fn: ReanalysisFn = None) -> dict[str, Any]:
    """Multi-turn iterative retrieval loop with re-analysis per round."""
    pool: dict[str, dict[str, Any]] = {}
    for c in conflicts:
        eid = str(c.get("edge_id") or f"{c.get('claim_i_id')}__{c.get('claim_j_id')}")
        pool[eid] = {"conflict": c, "state": None, "resolved": False,
                      "claim_i_id": str(c.get("claim_i_id") or ""), "claim_j_id": str(c.get("claim_j_id") or ""),
                      "supporting_ids_i": [], "supporting_ids_j": [], "evidence_coverage_i": 0.0, "evidence_coverage_j": 0.0}

    token_cache: dict[str, set[str]] = {c.get("claim_id", ""): set(_TOKEN_RE.findall((c.get("claim_text", "") or "").lower())) for c in claims_for_query}
    seen: set[str] = set(); state_hist: list[str] = []; queries_out: list[TargetedQuery] = []
    logs: list[IterationLog] = []; state_dist: Counter[str] = Counter()
    qtype_dist: Counter[str] = Counter(); slot_dist: Counter[str] = Counter()
    stop_dist: Counter[str] = Counter(); all_priorities: list[float] = []
    gains: list[float] = []; claim_counts: list[float] = []
    all_pair_rel: list[float] = []; all_min_ep: list[float] = []; all_align: list[float] = []

    for iteration in range(1, max_iterations + 1):
        unresolved = [(eid, p) for eid, p in pool.items() if not p["resolved"]]
        if not unresolved: break
        for _, p in unresolved:
            p["state"] = infer_conflict_state(extract_conflict_features(p["conflict"]))
        unresolved.sort(key=lambda x: float(x[1]["conflict"].get("conflict_intensity") or 0.0), reverse=True)

        iter_new_ids: set[str] = set()
        processed = []

        for eid, cp in unresolved[:top_n_candidates]:
            c = cp["conflict"]; so = cp["state"]; slot = str(c.get("slot") or "")
            sn = so["predicted_state"]; state_hist.append(sn)
            state_dist[sn] += 1; slot_dist[slot] += 1
            pair_rel = float(c.get("pair_query_relevance") or c.get("query_relevance_score") or 0.0)
            min_ep = float(c.get("min_endpoint_query_relevance") or 0.0)
            align = float(c.get("answer_type_alignment_score") or 0.0)
            all_pair_rel.append(pair_rel); all_min_ep.append(min_ep); all_align.append(align)

            cands = generate_query_candidates(user_query, c, so)
            scored = []
            for cand in cands:
                s = dict(cand)
                s["state_probability_for_candidate_type"] = _state_probability_for_query_type(str(s.get("query_type") or ""), so)
                s["query_relevance_score"] = pair_rel; s["claim_i_query_relevance"] = float(c.get("claim_i_query_relevance") or 0.0)
                s["claim_j_query_relevance"] = float(c.get("claim_j_query_relevance") or 0.0)
                s["min_endpoint_query_relevance"] = min_ep; s["pair_query_relevance"] = pair_rel
                s["answer_type"] = str(c.get("answer_type") or "generic"); s["answer_type_alignment_score"] = align
                s["conflict_slot"] = slot
                for k in ["conflict_quality_score", "conflict_quality_label", "recommended_action", "quality_signals"]:
                    if k in c: s[k] = c[k]
                s["priority_score"] = score_query_candidate(s, extract_conflict_features(c), so)
                scored.append(s); all_priorities.append(s["priority_score"])

            selected = select_top_queries(scored, top_n=1, query_id=query_id, edge_id=eid, iteration=iteration, state_output=so)
            if not selected: continue
            qtype_dist[str(selected[0].get("query_type") or "unknown")] += 1
            queries_out.append(selected[0])

            retrieved = retrieve_claims(str(selected[0].get("query_text") or ""), claims_for_query, top_k=top_k,
                                        exclude_claim_ids=[cp["claim_i_id"], cp["claim_j_id"]])
            side: str | None = None
            i_txt = str(c.get("claim_i_text") or ""); j_txt = str(c.get("claim_j_text") or "")
            for item in retrieved:
                cid = str(item.get("claim_id") or "")
                if cid in seen: continue
                seen.add(cid); iter_new_ids.add(cid)
                s, _ = _classify_new_claim_support(str(item.get("claim_text", "")), i_txt, j_txt, token_cache, cid, cp["claim_i_id"], cp["claim_j_id"])
                if s == "claim_i": cp["supporting_ids_i"].append(cid)
                elif s == "claim_j": cp["supporting_ids_j"].append(cid)
                if side is None and s in ("claim_i", "claim_j"): side = s
            if side is None and iter_new_ids: side = "neutral"

            ra = _reanalyze_conflict(c, cp["supporting_ids_i"], cp["supporting_ids_j"], side or "neutral")
            cp["resolved"] = ra["resolved"]; cp["conflict"]["conflict_intensity"] = ra["conflict_intensity"]
            cp["evidence_coverage_i"] = min(1.0, len(cp["supporting_ids_i"]) / max(1, top_k))
            cp["evidence_coverage_j"] = min(1.0, len(cp["supporting_ids_j"]) / max(1, top_k))
            processed.append(eid)

        # Phase 3: batch reanalysis via external function (Hương's code)
        if reanalysis_fn is not None and iter_new_ids:
            new_list = [c for c in claims_for_query if c.get("claim_id") in iter_new_ids]
            old_list = [c for c in claims_for_query if c.get("claim_id") not in iter_new_ids]
            result = reanalysis_fn(new_list, old_list, conflicts)
            if result:
                for eid, cp in pool.items():
                    up = result.get(eid, {})
                    if up.get("resolved", False): cp["resolved"] = True
                    new_i = up.get("conflict_intensity")
                    if new_i is not None: cp["conflict"]["conflict_intensity"] = new_i
                    new_s = up.get("predicted_state")
                    if new_s and cp["state"]: cp["state"]["predicted_state"] = new_s

        gain = len(iter_new_ids) / max(1, top_k * max(1, len(unresolved[:top_n_candidates])))
        gains.append(gain); claim_counts.append(len(iter_new_ids))
        stable = len(set(state_hist[-state_stable_patience:])) == 1 if len(state_hist) >= state_stable_patience else False
        all_resolved = all(p["resolved"] for p in pool.values())
        stop = should_stop_loop({"max_iterations_reached": iteration >= max_iterations, "all_conflicts_resolved": all_resolved,
                                  "no_new_claims": len(iter_new_ids) == 0, "no_new_relevant_claims": len(iter_new_ids) == 0,
                                  "coverage_gain_below_threshold": gain < min_coverage_gain, "state_stable": stable})
        stop_dist[stop["stopping_reason"]] += 1

        logs.append({"iteration": iteration, "query_id": query_id, "processed_conflicts": processed,
                      "conflict_snapshot": [{"edge_id": eid, "resolved": p["resolved"],
                                             "conflict_intensity": p["conflict"].get("conflict_intensity", 0.0),
                                             "state": p["state"]["predicted_state"] if p["state"] else "unknown",
                                             "supporting_ids_i": len(p["supporting_ids_i"]), "supporting_ids_j": len(p["supporting_ids_j"])}
                                            for eid, p in sorted(pool.items())],
                      "new_claim_ids": sorted(iter_new_ids), "coverage_gain": gain, "seen_claims_total": len(seen), "stop_decision": stop})
        if stop["should_stop"]: break

    return {"iteration_logs": logs, "targeted_queries": queries_out, "state_distribution": dict(state_dist),
            "query_type_distribution": dict(qtype_dist), "slot_distribution": dict(slot_dist),
            "stopping_reason_distribution": dict(stop_dist),
            "avg_priority_score": sum(all_priorities) / len(all_priorities) if all_priorities else 0.0,
            "avg_coverage_gain": sum(gains) / len(gains) if gains else 0.0,
            "avg_new_claims_per_iteration": sum(claim_counts) / len(claim_counts) if claim_counts else 0.0,
            "avg_query_relevance_score": sum(all_pair_rel) / len(all_pair_rel) if all_pair_rel else 0.0,
            "avg_pair_query_relevance": sum(all_pair_rel) / len(all_pair_rel) if all_pair_rel else 0.0,
            "avg_min_endpoint_query_relevance": sum(all_min_ep) / len(all_min_ep) if all_min_ep else 0.0,
            "avg_answer_type_alignment_score": sum(all_align) / len(all_align) if all_align else 0.0,
            "avg_conflict_quality_score": 0.0, "conflict_quality_distribution": {}, "corrective_action_distribution": {},
            "corrective_fallback_used": False, "major_conflicts_resolved": "resolved" if all(p["resolved"] for p in pool.values()) else "not_available_offline",
            "confidence_delta_below_threshold": "not_available_without_graph_update", "no_new_relevant_claims_note": "offline alias of no_new_claims",
            "state_history": state_hist,
            "final_conflict_pool_snapshot": [{"edge_id": eid, "resolved": p["resolved"], "conflict_intensity": p["conflict"].get("conflict_intensity", 0.0),
                                               "claim_i_id": p["claim_i_id"], "claim_j_id": p["claim_j_id"],
                                               "state": p["state"]["predicted_state"] if p["state"] else "unknown",
                                               "n_supporting_i": len(p["supporting_ids_i"]), "n_supporting_j": len(p["supporting_ids_j"])}
                                              for eid, p in sorted(pool.items())]}


# -- Phase 3 reanalyzer factory -----------------------------------------------

def make_conflict_reanalyzer(claims_path: str, edges_path: str) -> ReanalysisFn:
    """Factory: wraps src.conflict_region_builder_fixed.run_pipeline for real NLI re-analysis."""
    claim_dict: dict[str, dict[str, Any]] = {}
    p = Path(claims_path)
    if p.exists():
        with p.open(encoding="utf-8") as f:
            for line in f:
                ln = line.strip()
                if ln:
                    try:
                        obj = json.loads(ln)
                        claim_dict[obj.get("claim_id", "")] = obj
                    except json.JSONDecodeError:
                        pass

    def _reanalyze(new_claims: list[dict[str, Any]], old_claims: list[dict[str, Any]], current_conflicts: list[ConflictRecord]) -> dict[str, Any]:
        conflict_by_edge = {str(c.get("edge_id", "")): c for c in current_conflicts if c.get("edge_id")}
        try:
            from src.conflict_region_builder_fixed import Claim, read_claims, read_edges, run_pipeline
            all_raw: list[dict[str, Any]] = []
            seen_ids: set[str] = set()
            for c in new_claims:
                cid = c.get("claim_id", "")
                if cid and cid not in seen_ids: all_raw.append(claim_dict.get(cid, c)); seen_ids.add(cid)
            for c in old_claims:
                cid = c.get("claim_id", "")
                if cid and cid not in seen_ids: all_raw.append(claim_dict.get(cid, c)); seen_ids.add(cid)
            with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as tmp:
                for c in all_raw: tmp.write(json.dumps(c, ensure_ascii=False) + "\n")
                tmp_path = tmp.name
            try:
                hc = read_claims(tmp_path); he = read_edges(edges_path)
                records = run_pipeline(claims=hc, edges=he, expansion_hops=1, min_conflict_confidence=0.50, use_heuristic_if_no_mlp=True)
                result: dict[str, Any] = {}
                for rec in records:
                    cr = rec.conflict_region
                    st = cr.predicted_state.value if cr.predicted_state else "underdetermined"
                    for eid in rec.edge_ids:
                        edge = he.get(eid)
                        if edge is None: continue
                        intensity = edge.claim_pair_features.aggregate.conflict_intensity_score if edge.claim_pair_features else 0.0
                        result[eid] = {"resolved": st in ("resolvable", "no_conflict"), "conflict_intensity": intensity, "predicted_state": st, "region_id": cr.conflict_region_id}
                return result
            finally:
                os.unlink(tmp_path)
        except ImportError:
            return {}
        except Exception:
            return {}

    return _reanalyze


# =============================================================================
# SECTION 6: EXPERIMENT RUNNER (CLI)
# =============================================================================

def _require_file(path: str | Path) -> Path:
    r = Path(path)
    if not r.exists(): raise FileNotFoundError(f"Required input file not found: {r}")
    if not r.is_file(): raise FileNotFoundError(f"Required input path is not a file: {r}")
    return r


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    r = _require_file(path)
    records: list[dict[str, Any]] = []
    with r.open(encoding="utf-8") as f:
        for n, line in enumerate(f, start=1):
            s = line.strip()
            if not s: continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at {r}:{n}: {e}") from e
            if not isinstance(obj, dict): raise ValueError(f"Expected object at {r}:{n}")
            records.append(obj)
    return records


def load_json(path: str | Path) -> Any:
    r = _require_file(path)
    try:
        return json.loads(r.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON file {r}: {e}") from e


def load_queries(path: str | Path) -> dict[str, str]:
    fallback_paths = [
        "data/preprocessed/queries.jsonl",
        "data/raw/RAMDocs_test.jsonl",
    ]
    result: dict[str, str] = {}
    _merge_queries_from(result, str(path))
    for fp in fallback_paths:
        if Path(fp).exists() and Path(fp) != Path(path):
            _merge_queries_from(result, fp)
    return result


def _merge_queries_from(dest: dict[str, str], path: str | Path) -> None:
    p = Path(path)
    try:
        if p.suffix == ".json":
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, list):
                for entry in data:
                    q = entry.get("query") or entry
                    qid = q.get("query_id", "")
                    qt = q.get("user_query", "")
                    if qid and qt and qid not in dest:
                        dest[qid] = qt
        elif p.suffix == ".jsonl":
            ramdocs = p.name.startswith("RAMDocs")
            all_rows = load_jsonl(str(p))
            if ramdocs:
                for idx, r in enumerate(all_rows):
                    qtext = r.get("question", "")
                    if isinstance(qtext, str) and qtext:
                        qid = f"q_{idx:03d}"
                        if qid not in dest:
                            dest[qid] = qtext
            else:
                for r in all_rows:
                    if isinstance(r.get("query_id"), str) and isinstance(r.get("user_query"), str) and r["query_id"] not in dest:
                        dest[r["query_id"]] = r["user_query"]
    except (json.JSONDecodeError, OSError):
        pass


def load_claims_by_id(path: str | Path) -> dict[str, ClaimRecord]:
    fallback_paths = [
        "data/preprocessed_gpt-4o/factoid_claims_revised.jsonl",
        "data/confict_rag/claims.jsonl",
    ]
    result: dict[str, ClaimRecord] = {}
    for r in load_jsonl(str(path)):
        if isinstance(r.get("claim_id"), str):
            result.setdefault(r["claim_id"], r)
    for fp in fallback_paths:
        if Path(fp).exists() and Path(fp) != Path(path):
            for r in load_jsonl(str(fp)):
                if isinstance(r.get("claim_id"), str) and r["claim_id"] not in result:
                    result[r["claim_id"]] = r
    return result


def load_edges(path: str | Path) -> list[EdgeRecord]:
    fallback_paths = [
        "data/confict_rag/claim_graph_edges.jsonl",
        "data/confict_rag/nli_edges.jsonl",
    ]
    result: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    def _extract(p: str | Path) -> None:
        nonlocal result, seen_ids
        pobj = Path(p)
        if not pobj.exists():
            return
        try:
            records: list[dict[str, Any]] = []
            if pobj.suffix == ".jsonl":
                with pobj.open(encoding="utf-8") as f:
                    for line in f:
                        s = line.strip()
                        if s:
                            records.append(json.loads(s))
            else:
                data = json.loads(pobj.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    records = data
                elif isinstance(data, dict) and "edges" in data:
                    records = data["edges"]
            for item in records:
                if not isinstance(item, dict):
                    continue
                eid = str(item.get("edge_id", ""))
                if eid not in seen_ids:
                    seen_ids.add(eid)
                    result.append(item)
        except (json.JSONDecodeError, OSError):
            pass

    _extract(path)
    for fp in fallback_paths:
        _extract(fp)

    return result


# -- Conflict building from edges ---------------------------------------------

_QUERY_ID_RE = re.compile(r"^c_q(?P<num>\d{3})_")


def infer_query_id_from_claim_id(claim_id: str) -> str | None:
    m = _QUERY_ID_RE.match(claim_id or "")
    return f"q_{m.group('num')}" if m else None


def is_contradiction_edge(edge: EdgeRecord) -> bool:
    return edge.get("relation_type") == "contradiction"


def _build_record(edge: EdgeRecord, ci: ClaimRecord, cj: ClaimRecord, fi: FactoidClaimRecord, fj: FactoidClaimRecord, qid: str, uq: str = "", at: str = "generic") -> ConflictRecord:
    ffi = fi.get("factoid_features") or {}; ffj = fj.get("factoid_features") or {}
    cpf = edge.get("claim_pair_features") or {}; nli = edge.get("nli_features") or {}
    slot = infer_slot_from_pair_features(cpf)
    ti = str(ci.get("claim_text") or fi.get("claim_text") or ""); tj = str(cj.get("claim_text") or fj.get("claim_text") or "")
    vi, vj = extract_slot_values(slot, ffi, ffj, ti, tj)
    src = str(edge.get("source_claim_id") or ""); tgt = str(edge.get("target_claim_id") or "")
    rel = compute_query_relevance_metrics(uq, ti, tj) if uq else {k: 0.0 for k in ["claim_i_query_relevance", "claim_j_query_relevance", "min_endpoint_query_relevance", "max_endpoint_query_relevance", "pair_query_relevance"]}
    alias = compute_answer_type_alignment(at, slot, ti, tj)

    def _safe_get(record: dict[str, Any], key: str) -> float:
        cf = record.get("claim_features") or {}
        return _float(cf.get(key) if isinstance(cf, dict) else record.get(key))

    def _pf(fd: dict) -> str | None:
        v = fd.get("verb")
        return str(v.get("lemma")).lower() if isinstance(v, dict) and v.get("lemma") else None

    def _ci(cpf: dict) -> float:
        return _float((cpf.get("aggregate") or {}).get("conflict_intensity_score"))

    return {"query_id": qid, "edge_id": str(edge.get("edge_id") or f"edge_{src}_{tgt}"),
            "source_claim_id": src, "target_claim_id": tgt, "claim_i_id": src, "claim_j_id": tgt,
            "claim_i_text": ti, "claim_j_text": tj, "claim_i_confidence": _safe_get(ci, "claim_confidence"),
            "claim_j_confidence": _safe_get(cj, "claim_confidence"),
            "claim_i_context_completeness": _safe_get(ci, "context_completeness"),
            "claim_j_context_completeness": _safe_get(cj, "context_completeness"),
            "claim_i_evidence_coverage": _safe_get(ci, "claim_evidence_coverage"),
            "claim_j_evidence_coverage": _safe_get(cj, "claim_evidence_coverage"),
            "relation_type": str(edge.get("relation_type") or ""), "relation_confidence": _float(edge.get("relation_confidence")),
            "nli_features": dict(nli) if isinstance(nli, dict) else {},
            "claim_pair_features": dict(cpf) if isinstance(cpf, dict) else {},
            "slot": slot, "value_i": vi, "value_j": vj,
            "claim_i_factoid_features": dict(ffi) if isinstance(ffi, dict) else {},
            "claim_j_factoid_features": dict(ffj) if isinstance(ffj, dict) else {},
            "claim_i_predicate": _pf(ffi), "claim_j_predicate": _pf(ffj),
            "conflict_intensity": _ci(cpf),
            "query_relevance_score": rel["pair_query_relevance"],
            "claim_i_query_relevance": rel["claim_i_query_relevance"], "claim_j_query_relevance": rel["claim_j_query_relevance"],
            "min_endpoint_query_relevance": rel["min_endpoint_query_relevance"], "max_endpoint_query_relevance": rel["max_endpoint_query_relevance"],
            "pair_query_relevance": rel["pair_query_relevance"], "answer_type": at, "answer_type_alignment_score": alias}


def build_conflict_records(claims_by_id: dict[str, ClaimRecord], factoid_claims_by_id: dict[str, FactoidClaimRecord],
                           edges: list[EdgeRecord], query_id: str | None = None, max_conflicts: int | None = None,
                           user_query: str = "", mode: str = METHODOLOGY_BASELINE,
                           min_endpoint_query_relevance_threshold: float = 0.15, allow_low_relevance_fallback: bool = True,
                           selection_metadata: dict[str, Any] | None = None) -> list[ConflictRecord]:
    at = infer_query_answer_type(user_query) if user_query else "generic"
    records: list[ConflictRecord] = []
    for edge in edges:
        if not is_contradiction_edge(edge): continue
        s, t = edge.get("source_claim_id"), edge.get("target_claim_id")
        if not isinstance(s, str) or not isinstance(t, str): continue
        sq = infer_query_id_from_claim_id(s); tq = infer_query_id_from_claim_id(t)
        if sq is None or tq is None or sq != tq: continue
        if query_id is not None and sq != query_id: continue
        ci, cj = claims_by_id.get(s), claims_by_id.get(t)
        fi, fj = factoid_claims_by_id.get(s), factoid_claims_by_id.get(t)
        if ci is None or cj is None or fi is None or fj is None: continue
        records.append(_build_record(edge, ci, cj, fi, fj, sq, user_query, at))

    if mode == ENHANCED_QUERY_ALIGNED:
        css = "answer_type_alignment,min_endpoint_query_relevance,pair_query_relevance,conflict_intensity,relation_confidence,contradiction_prob"
        records.sort(key=lambda x: (float(x.get("answer_type_alignment_score") or 0.0), float(x.get("min_endpoint_query_relevance") or 0.0),
                                    float(x.get("pair_query_relevance") or 0.0), float(x.get("conflict_intensity") or 0.0),
                                    float(x.get("relation_confidence") or 0.0), float((x.get("nli_features") or {}).get("contradiction_prob") or 0.0)), reverse=True)
        passing = [r for r in records if float(r.get("min_endpoint_query_relevance") or 0.0) >= min_endpoint_query_relevance_threshold]
        used_fb = False
        if passing: selected = passing
        elif allow_low_relevance_fallback: selected = records; used_fb = bool(records)
        else: selected = []
    else:
        css = "conflict_intensity,relation_confidence,contradiction_prob"
        records.sort(key=lambda x: (float(x.get("conflict_intensity") or 0.0), float(x.get("relation_confidence") or 0.0),
                                    float((x.get("nli_features") or {}).get("contradiction_prob") or 0.0)), reverse=True)
        passing = records; selected = records; used_fb = False

    if selection_metadata is not None:
        selection_metadata.update({"mode": mode, "inferred_answer_type": at, "conflict_sort_strategy": css,
                                    "num_conflicts_before_endpoint_filter": len(records),
                                    "num_conflicts_passing_endpoint_threshold": len(passing),
                                    "min_endpoint_query_relevance_threshold": min_endpoint_query_relevance_threshold,
                                    "used_low_relevance_fallback": used_fb, "candidate_conflicts": records if mode == ENHANCED_QUERY_ALIGNED else []})
    return selected[:max_conflicts] if max_conflicts is not None else selected


# -- Conflict building from regions (Phase 2) ---------------------------------

def _find_contradiction_edge_for_claims(edge_ids: list[str], graph_edges: list[dict[str, Any]], i_id: str, j_id: str) -> dict[str, Any] | None:
    for e in graph_edges:
        if str(e.get("edge_id", "")) not in edge_ids: continue
        if {str(e.get("source_claim_id", "")), str(e.get("target_claim_id", ""))} == {i_id, j_id}: return e
    return None


def _extract_conflict_pairs_from_region(region: dict[str, Any], claims_by_id: dict[str, Any], graph_edges: list[dict[str, Any]], uq: str, at: str, qid: str) -> list[ConflictRecord]:
    cr = region.get("conflict_region", region)
    cids = cr.get("claim_ids", []); eids = region.get("edge_ids", []); ps = cr.get("predicted_state", "underdetermined")
    records: list[ConflictRecord] = []
    for i in range(len(cids)):
        for j in range(i + 1, len(cids)):
            edge = _find_contradiction_edge_for_claims(eids, graph_edges, cids[i], cids[j])
            if edge is None: continue
            ci, cj = claims_by_id.get(cids[i]), claims_by_id.get(cids[j])
            if ci is None or cj is None: continue
            rec = _build_record(edge, ci, cj, ci, cj, qid, uq, at)
            rec["region_predicted_state"] = ps; rec["region_id"] = cr.get("conflict_region_id", "")
            records.append(rec)
    return records


def load_region_conflicts(region_path: str | Path, claims_by_id: dict[str, Any], claim_graph_edges: list[dict[str, Any]],
                          query_id: str | None = None, user_query: str = "", answer_type: str = "generic",
                          max_conflicts: int | None = None) -> list[ConflictRecord]:
    all_records: list[ConflictRecord] = []; seen_pairs: set[str] = set()
    for region in load_jsonl(region_path):
        cr = region.get("conflict_region", region)
        if query_id and not any(infer_query_id_from_claim_id(c) == query_id for c in cr.get("claim_ids", [])): continue
        for rec in _extract_conflict_pairs_from_region(region, claims_by_id, claim_graph_edges, user_query, answer_type, query_id or "unknown"):
            pk = f"{rec.get('claim_i_id')}__{rec.get('claim_j_id')}"
            if pk not in seen_pairs: seen_pairs.add(pk); all_records.append(rec)
    sp = {"resolvable": 3, "underdetermined": 2, "contextual": 1, "no_conflict": 0}
    all_records.sort(key=lambda r: (sp.get(str(r.get("region_predicted_state", "")).lower(), 0), float(r.get("conflict_intensity") or 0.0)), reverse=True)
    return all_records[:max_conflicts] if max_conflicts is not None else all_records


def has_regions_for_query(region_path: str | Path, query_id: str) -> bool:
    return any(infer_query_id_from_claim_id(c) == query_id for region in load_jsonl(region_path)
               for c in (region.get("conflict_region", region)).get("claim_ids", []))


# -- Configuration defaults ---------------------------------------------------

DEFAULT_FACTOID_CLAIMS_PATH = "data/confict_rag/factoid_claims.jsonl"
DEFAULT_EDGES_PATH = "data/preprocessed_gpt-4o/edges.json"
DEFAULT_QUERIES_PATH = "data/confict_rag/conflict_region_query_claims.json"
DEFAULT_REGION_PATH = "data/confict_rag/conflict_regions.jsonl"
DEFAULT_CLAIM_GRAPH_EDGES_PATH = "data/confict_rag/claim_graph_edges.jsonl"


# -- Answer quality evaluator (conflict + answer quality) ---------------------

def evaluate_conflict_quality(conflict_record: dict[str, Any]) -> dict[str, Any]:
    signals = {k: _float(conflict_record.get(k)) for k in ["min_endpoint_query_relevance", "answer_type_alignment_score", "pair_query_relevance", "conflict_intensity", "relation_confidence"]}
    score = sum(CONFLICT_QUALITY_WEIGHTS[k] * signals[k] for k in signals)
    if score >= CONFLICT_QUALITY_THRESHOLDS["good"]: lbl, act = "good", "use_conflict_for_state_guided_query"
    elif score >= CONFLICT_QUALITY_THRESHOLDS["weak"]: lbl, act = "weak", "use_with_warning"
    else: lbl, act = "poor", "generate_evidence_expansion_query"
    return {"conflict_quality_score": score, "conflict_quality_label": lbl, "recommended_action": act, "quality_signals": signals}


def _eval_tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"\w+", (text or "").lower()) if len(t) > 2}


def _eval_overlap(left: str, right: str) -> float:
    lt, rt = _eval_tokens(left), _eval_tokens(right)
    return len(lt & rt) / len(lt) if lt and rt else 0.0


def evaluate_answer_quality(user_query: str, final_answer: dict[str, Any], final_evidence_selection: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    ans = str(final_answer.get("answer") or "")
    policy = str(final_answer.get("answer_policy") or summary.get("final_answer_policy") or "")
    status = str(final_answer.get("resolution_status") or summary.get("final_resolution_status") or "")
    sel = (final_answer.get("selected_answer_claims") or final_answer.get("supporting_claims") or
           final_evidence_selection.get("selected_answer_claims") or final_evidence_selection.get("supporting_claims") or [])
    comp = final_answer.get("competing_claims") or final_evidence_selection.get("competing_claims") or []

    def groundedness():
        if status == "insufficient_evidence": return (1.0, []) if not sel and "insufficient evidence" in ans.lower() else (0.5, ["..."])
        if policy in {"answer_with_selected_claim", "answer_with_caution"}:
            if not sel: return 0.0, ["Factual answer has no selected answer claim."]
            bo = max(_eval_overlap(str(cl.get("claim_text", "")), ans) for cl in sel)
            return (1.0, []) if bo >= 0.5 else (0.5, ["Weak lexical overlap."])
        return (1.0, []) if sel else (0.5, ["No traceable claims."])

    def conflict_awareness():
        if not comp: return 1.0, []
        al = ans.lower()
        return (1.0, []) if any(t in al for t in ["competing", "conflict", "caution", "uncertain", "unresolved"]) else (0.0, ["Competing claims not mentioned."])

    def calibration():
        conf = float(final_answer.get("answer_confidence") or 0.0)
        gen = bool(summary.get("final_answer_should_be_generated", status != "insufficient_evidence"))
        if status == "insufficient_evidence": return (1.0, []) if conf == 0.0 and not gen else (0.25, ["Wrong confidence for insufficient-evidence."])
        if status == "likely_resolved_with_caution": return (1.0, []) if 0.4 <= conf <= 0.75 else (0.5, ["Confidence outside medium range."])
        if status == "resolved": return (1.0, []) if conf >= 0.7 else (0.5, ["Resolved confidence too low."])
        return (1.0, []) if conf <= 0.4 else (0.5, ["Unresolved confidence too high."])

    def traceability():
        if status == "insufficient_evidence": return (1.0, []) if not sel and "evidence" in ans.lower() else (0.0, ["Missing evidence trace."])
        if policy in {"answer_with_selected_claim", "answer_with_caution"}: return (1.0, []) if sel and all(cl.get("claim_id") for cl in sel) else (0.0, ["Missing claim IDs."])
        return 1.0, []

    def alignment():
        at = str(summary.get("inferred_answer_type") or "generic")
        combined = " ".join([user_query, ans, *(str(cl.get("claim_text", "")) for cl in sel)]).lower()
        if at in {"population", "numeric"}:
            return (1.0, []) if ("population" in combined or "people" in combined) and re.search(r"\d", combined) else (0.25, ["Missing population/number alignment."])
        if at == "director":
            return (1.0, []) if any(t in combined for t in ["director", "directed", "filmmaker"]) else (0.25, ["Missing director alignment."])
        ov = _eval_overlap(user_query, ans)
        return (1.0, []) if ov >= 0.2 else (0.5, ["Weak query alignment."])

    gs, gw = groundedness(); cs, cw = conflict_awareness(); us, uw = calibration()
    ts, tw = traceability(); als, aw = alignment()
    overall = 0.30 * gs + 0.20 * cs + 0.20 * us + 0.20 * ts + 0.10 * als

    if status == "insufficient_evidence": gate = "do_not_show_factual_answer"
    elif overall >= 0.85 and status == "resolved": gate = "safe_to_show"
    elif overall >= 0.60: gate = "show_with_caution"
    else: gate = "do_not_show_factual_answer"

    rt_map = {"insufficient_evidence": "insufficient_evidence_guidance", "likely_resolved_with_caution": "cautious_factual_answer", "resolved": "factual_answer"}
    rt = str(final_answer.get("final_response_type") or rt_map.get(status, "uncertainty_answer"))
    fi_qual = None if gate == "do_not_show_factual_answer" or status == "insufficient_evidence" else float(summary.get("final_answer_confidence", final_answer.get("answer_confidence", 0.0)))

    qint = {"insufficient_evidence_guidance": "The response is high quality as an insufficient-evidence guidance response; it is not a factual correctness score.",
            "cautious_factual_answer": "The response is a cautious factual answer; interpret with reported confidence.",
            "factual_answer": "The response is evaluated as a factual answer grounded in selected evidence."}.get(rt, "Uncertainty-aware answer.")

    return {"evaluated_response_type": rt, "groundedness_score": gs, "conflict_awareness_score": cs,
            "uncertainty_calibration_score": us, "evidence_traceability_score": ts, "answer_alignment_score": als,
            "response_quality_score": overall, "factual_answer_quality_score": fi_qual,
            "overall_answer_quality_score": overall, "quality_score_interpretation": qint,
            "quality_gate_decision": gate, "quality_warnings": gw + cw + uw + tw + aw}


# -- CLI entry point -----------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run conflict-state-guided iterative retrieval loop.")
    p.add_argument("--mode", choices=MODES, default=METHODOLOGY_BASELINE)
    p.add_argument("--query_id", required=True)
    p.add_argument("--claims_path", default=DEFAULT_FACTOID_CLAIMS_PATH)
    p.add_argument("--edges_path", default=DEFAULT_EDGES_PATH)
    p.add_argument("--queries_path", default=DEFAULT_QUERIES_PATH)
    p.add_argument("--region_path", default=None)
    p.add_argument("--claim_graph_edges_path", default=DEFAULT_CLAIM_GRAPH_EDGES_PATH)
    p.add_argument("--use_regions", action="store_true")
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--max_conflicts", type=int, default=10)
    p.add_argument("--min_endpoint_query_relevance_threshold", type=float, default=None)
    p.add_argument("--no_low_relevance_fallback", action="store_true")
    p.add_argument("--output", default=None)
    p.add_argument("--claims_final_output", default=None)
    return p.parse_args()


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records: f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _claims_for_query(claims_by_id: dict[str, dict[str, Any]], query_id: str) -> list[dict[str, Any]]:
    return [c for cid, c in claims_by_id.items() if infer_query_id_from_claim_id(cid) == query_id]


def _query_contradiction_edge_count(edges: list[dict[str, Any]], query_id: str) -> int:
    return sum(1 for e in edges if is_contradiction_edge(e) and infer_query_id_from_claim_id(str(e.get("source_claim_id", ""))) == query_id
               and infer_query_id_from_claim_id(str(e.get("target_claim_id", ""))) == query_id)


def _answer_focus(user_query: str) -> dict[str, str]:
    at = infer_query_answer_type(user_query)
    mapping = {"director": ("director", "authoritative source explicitly naming the director or directors"),
               "population": ("population", "authoritative population or census evidence"),
               "date_or_time": ("date_or_time", "authoritative temporal evidence"),
               "location": ("location", "authoritative location evidence"),
               "person": ("person", "authoritative source explicitly naming the person"),
               "numeric": ("numeric", "authoritative numerical evidence")}
    slot, ev = mapping.get(at, ("answer", "authoritative evidence for the missing answer"))
    return {"target_slot": slot, "expected_evidence_type": ev}


def _quoted_subject(user_query: str) -> str:
    cleaned = " ".join((user_query or "").split()).strip().rstrip(" ?.!;")
    lower = cleaned.lower()
    if "film " in lower:
        idx = lower.rfind("film ") + 5; subject = cleaned[idx:].strip().strip('"')
        return f'the film "{subject}"' if subject else cleaned
    return cleaned or "the original question"


def _build_evidence_expansion_result(query_id: str, user_query: str, quality: dict[str, Any] | None) -> dict[str, Any]:
    focus = _answer_focus(user_query); subject = _quoted_subject(user_query)
    qt = f"Find authoritative evidence for the {focus['target_slot']} of {subject}. Prefer sources that explicitly state the {focus['target_slot']}."
    q = {"candidate_name": "corrective_evidence_expansion_query", "targeted_query_id": f"tq_{query_id}_corrective_1_1",
         "query_id": query_id, "edge_id": "corrective_fallback", "iteration": 1, "query_text": qt,
         "query_type": "evidence_coverage_expansion", "target_claim_ids": [], "missing_slots_targeted": [focus["target_slot"]],
         "expected_evidence_type": focus["expected_evidence_type"],
         "candidate_reason": "Conflict quality evaluator or strict endpoint filtering judged available conflicts as insufficient; expanding evidence coverage for the missing answer slot.",
         "query_specificity": 0.68, "priority": 1, "priority_score": 0.85, "predicted_state": "Underdetermined",
         "state_probabilities": {"Underdetermined": 1.0, "Contextual": 0.0, "Resolvable": 0.0}, "state_rationale": {"corrective_action": "generate_evidence_expansion_query"}}
    if quality:
        for k in ["conflict_quality_score", "conflict_quality_label", "quality_signals"]:
            if k in quality: q[k] = quality[k]
    q["recommended_action"] = "generate_evidence_expansion_query"
    log = {"iteration": 1, "query_id": query_id, "edge_id": "corrective_fallback", "conflict_slot": focus["target_slot"],
           "conflict_intensity": 0.0, "query_relevance_score": 0.0, "claim_i_query_relevance": 0.0, "claim_j_query_relevance": 0.0,
           "min_endpoint_query_relevance": 0.0, "pair_query_relevance": 0.0, "answer_type": infer_query_answer_type(user_query),
           "answer_type_alignment_score": 0.0, "relation_confidence": 0.0, "contradiction_prob": 0.0,
           "predicted_state": "Underdetermined", "state_probabilities": q["state_probabilities"], "selected_queries": [q],
           "retrieved_claim_ids": [], "new_claim_ids": [], "coverage_gain": 0.0, "seen_claims_before": 0, "seen_claims_after": 0,
           "stop_decision": {"should_stop": True, "stopping_reason": "corrective_fallback", "signals": {"corrective_fallback": True}}}
    if quality:
        for k in ["conflict_quality_score", "conflict_quality_label", "quality_signals"]:
            if k in quality: log[k] = quality[k]
    log["recommended_action"] = "generate_evidence_expansion_query"
    s = _float(quality.get("conflict_quality_score")) if quality else 0.0
    lbl = str(quality.get("conflict_quality_label") or "poor") if quality else "poor"
    return {"iteration_logs": [log], "targeted_queries": [q], "state_distribution": {"Underdetermined": 1},
            "query_type_distribution": {"evidence_coverage_expansion": 1}, "slot_distribution": {focus["target_slot"]: 1},
            "stopping_reason_distribution": {"corrective_fallback": 1}, "avg_priority_score": 0.85, "avg_coverage_gain": 0.0,
            "avg_new_claims_per_iteration": 0.0, "avg_query_relevance_score": 0.0, "avg_pair_query_relevance": 0.0,
            "avg_min_endpoint_query_relevance": 0.0, "avg_answer_type_alignment_score": 0.0, "avg_conflict_quality_score": s,
            "conflict_quality_distribution": {lbl: 1}, "corrective_action_distribution": {"generate_evidence_expansion_query": 1},
            "corrective_fallback_used": True, "num_synthetic_corrective_queries": 1,
            "major_conflicts_resolved": "not_available_offline", "confidence_delta_below_threshold": "not_available_without_graph_update",
            "no_new_relevant_claims_note": "offline alias of no_new_claims", "state_history": ["Underdetermined"]}


def _auto_output_path(query_id: str, args: argparse.Namespace) -> str:
    tag = "region" if getattr(args, 'use_regions', False) else args.mode.replace("methodology_", "").replace("enhanced_", "")
    return f"outputs/query_generation_loop_{query_id}_{tag}.json"


def _auto_claims_final_path(query_id: str, args: argparse.Namespace) -> str:
    tag = "region" if getattr(args, 'use_regions', False) else args.mode.replace("methodology_", "").replace("enhanced_", "")
    return f"outputs/claims_final_{query_id}_{tag}.json"


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    qm = load_queries(args.queries_path)
    if args.query_id not in qm: raise SystemExit(f"Query ID not found: {args.query_id}")
    cid = load_claims_by_id(args.claims_path)
    edges = load_edges(args.edges_path)
    warnings: list[str] = []; uq = qm[args.query_id]
    cfq = _claims_for_query(cid, args.query_id)
    if not cfq: warnings.append(f"No claims found for query_id={args.query_id}")
    missing = sorted({inf for cid in cid for inf in [infer_query_id_from_claim_id(cid)] if inf and inf not in qm})
    if missing: warnings.append(f"Claims exist for query IDs missing from queries file: {', '.join(missing)}")
    threshold = args.min_endpoint_query_relevance_threshold or (ENDPOINT_RELEVANCE["enhanced_threshold"] if args.mode == ENHANCED_QUERY_ALIGNED else ENDPOINT_RELEVANCE["baseline_threshold"])
    sm: dict[str, Any] = {}

    # Conflict loading
    if getattr(args, 'use_regions', False):
        rp = getattr(args, 'region_path', None) or DEFAULT_REGION_PATH
        cge = getattr(args, 'claim_graph_edges_path', None) or DEFAULT_CLAIM_GRAPH_EDGES_PATH
        if not has_regions_for_query(rp, args.query_id):
            warnings.append(f"No regions for {args.query_id}. Falling back to edge-based.")
            conflicts = build_conflict_records(cid, cid, edges, args.query_id, args.max_conflicts, uq, args.mode, threshold, not args.no_low_relevance_fallback, sm)
        else:
            graph_edges = load_edges(cge)
            at = infer_query_answer_type(uq)
            conflicts = load_region_conflicts(rp, cid, graph_edges, args.query_id, uq, at, args.max_conflicts)
            sm.update({"mode": args.mode, "inferred_answer_type": at, "conflict_sort_strategy": "region_predicted_state,conflict_intensity",
                        "num_conflicts_before_endpoint_filter": len(conflicts), "num_conflicts_passing_endpoint_threshold": len(conflicts),
                        "min_endpoint_query_relevance_threshold": 0.0, "used_low_relevance_fallback": False, "candidate_conflicts": []})
    else:
        conflicts = build_conflict_records(cid, cid, edges, args.query_id, args.max_conflicts, uq, args.mode, threshold, not args.no_low_relevance_fallback, sm)

    # Phase 3 reanalysis setup
    rfn = None
    if getattr(args, 'use_regions', False):
        try:
            cgp = getattr(args, 'claim_graph_edges_path', None) or DEFAULT_CLAIM_GRAPH_EDGES_PATH
            if Path(cgp).exists() and Path(args.claims_path).exists():
                rfn = make_conflict_reanalyzer(claims_path=args.claims_path, edges_path=cgp)
        except Exception:
            pass

    # Run loop
    if args.mode == ENHANCED_QUERY_ALIGNED:
        evaled = []
        for c in sm.get("candidate_conflicts", []):
            d = dict(c); d.update(evaluate_conflict_quality(d)); evaled.append(d)
        ecid = {str(c.get("edge_id", "")): c for c in evaled}
        conflicts = [ecid.get(str(c.get("edge_id", "")), c) for c in conflicts]
        sm["candidate_conflicts"] = evaled
        acts = {str(c.get("recommended_action", "")) for c in conflicts}
        if not conflicts or acts == {"generate_evidence_expansion_query"}:
            result = _build_evidence_expansion_result(args.query_id, uq, evaled[0] if evaled else None)
        else:
            result = run_offline_iteration(args.query_id, uq, conflicts, cfq, args.top_k, max_iterations=args.max_conflicts, reanalysis_fn=rfn)
    else:
        result = run_offline_iteration(args.query_id, uq, conflicts, cfq, args.top_k, max_iterations=args.max_conflicts, reanalysis_fn=rfn)

    # Summary
    tq = result.get("targeted_queries", []); pl = result.get("iteration_logs", [])
    th = float(threshold)
    low_min = sum(1 for log in pl if float(log.get("min_endpoint_query_relevance") or 0.0) < th)
    if sm.get("used_low_relevance_fallback"): warnings.append("No conflicts passed endpoint threshold; using fallback.")
    if low_min: warnings.append(f"{low_min} selected conflicts have low min endpoint relevance (< {th}).")
    if pl and all(float(log.get("answer_type_alignment_score") or 0.0) < 0.3 for log in pl): warnings.append("All conflicts have low answer-type alignment.")
    if result.get("corrective_fallback_used"): warnings.append("Corrective fallback used.")
    if result.get("avg_conflict_quality_score", 0.0) and float(result.get("avg_conflict_quality_score") or 0.0) < 0.4: warnings.append("Average conflict quality is poor (< 0.4).")
    tect = sum(1 for e in edges if is_contradiction_edge(e))
    qect = _query_contradiction_edge_count(edges, args.query_id)
    syn = int(result.get("num_synthetic_corrective_queries") or 0)
    rcp = max(0, len(pl) - syn)

    examples = []
    for q in tq[:3]:
        examples.append({k: q.get(k) for k in ["candidate_name", "query_type", "predicted_state", "priority_score",
                        "query_relevance_score", "claim_i_query_relevance", "claim_j_query_relevance",
                        "min_endpoint_query_relevance", "pair_query_relevance", "answer_type_alignment_score",
                        "conflict_quality_score", "conflict_quality_label", "recommended_action", "quality_signals",
                        "conflict_slot", "candidate_reason", "query_text"]})

    summary = {"query_id": args.query_id, "user_query": uq, "mode": sm.get("mode", METHODOLOGY_BASELINE),
               "conflict_sort_strategy": sm.get("conflict_sort_strategy", "unknown"),
               "methodology_alignment": {"module_11": "aligned", "module_12": "offline_approximation", "graph_update": "not_implemented_in_isolated_module"},
               "num_claims": len(cfq), "num_edges": len(edges), "num_contradiction_edges": tect,
               "num_total_contradiction_edges": tect, "num_query_contradiction_edges": qect,
               "num_conflicts_processed": len(pl), "num_real_conflicts_processed": rcp,
               "num_synthetic_corrective_queries": syn, "num_generated_queries": len(tq),
               "inferred_answer_type": sm.get("inferred_answer_type", "generic"),
               "num_conflicts_passing_endpoint_threshold": sm.get("num_conflicts_passing_endpoint_threshold", 0),
               "used_low_relevance_fallback": bool(sm.get("used_low_relevance_fallback", False)),
               "state_distribution": result.get("state_distribution", {}),
               "query_type_distribution": result.get("query_type_distribution", {}),
               "slot_distribution": result.get("slot_distribution", {}),
               "avg_priority_score": result.get("avg_priority_score", 0.0),
               "avg_coverage_gain": result.get("avg_coverage_gain", 0.0),
               "avg_new_claims_per_iteration": result.get("avg_new_claims_per_iteration", 0.0),
               "avg_query_relevance_score": result.get("avg_query_relevance_score", 0.0),
               "avg_pair_query_relevance": result.get("avg_pair_query_relevance", 0.0),
               "avg_min_endpoint_query_relevance": result.get("avg_min_endpoint_query_relevance", 0.0),
               "avg_answer_type_alignment_score": result.get("avg_answer_type_alignment_score", 0.0),
               "avg_conflict_quality_score": result.get("avg_conflict_quality_score", 0.0),
               "conflict_quality_distribution": result.get("conflict_quality_distribution", {}),
               "corrective_action_distribution": result.get("corrective_action_distribution", {}),
               "corrective_fallback_used": bool(result.get("corrective_fallback_used", False)),
               "stopping_reason_distribution": result.get("stopping_reason_distribution", {}),
               "major_conflicts_resolved": result.get("major_conflicts_resolved", "not_available_offline"),
               "confidence_delta_below_threshold": "not_available_without_graph_update",
               "no_new_relevant_claims_note": "offline alias of no_new_claims",
               "top_examples": examples, "risks_or_warnings": warnings}

    fld = derive_final_loop_decision(summary, pl, tq)
    fe = select_final_evidence(cfq, tq, {**summary, "final_resolution_status": fld["resolution_status"]})
    fa = compose_final_answer(uq, fld, fe)
    fsf = {"final_loop_decision": fld, "final_evidence_selection": fe, "final_answer": fa,
           "final_resolution_status": fld["resolution_status"], "final_answer_policy": fld["answer_policy"],
           "final_answer_confidence": fa["answer_confidence"],
           "final_answer_should_be_generated": fld["resolution_status"] != "insufficient_evidence",
           "final_response_type": fa["final_response_type"]}
    aq = evaluate_answer_quality(uq, fa, fe, {**summary, **fsf})
    summary.update({"methodology_alignment": {**summary["methodology_alignment"], "module_13": "offline_template_composition",
                     "answer_quality_diagnostics": "deterministic_ragchecker_inspired"},
                    **fsf, "answer_quality": aq, "answer_quality_score": aq["overall_answer_quality_score"],
                    "response_quality_score": aq["response_quality_score"],
                    "factual_answer_quality_score": aq["factual_answer_quality_score"],
                    "evaluated_response_type": aq["evaluated_response_type"],
                    "answer_quality_gate_decision": aq["quality_gate_decision"]})

    # Write experiment JSON
    op = Path(getattr(args, 'output', None) or _auto_output_path(args.query_id, args))
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_jsonl(op.parent / "query_generation_loop_targeted_queries.jsonl", tq)
    _write_jsonl(op.parent / "query_generation_loop_iteration_logs.jsonl", pl)

    # Write claims_final for An
    cfp = Path(getattr(args, 'claims_final_output', None) or _auto_claims_final_path(args.query_id, args))
    cfp.parent.mkdir(parents=True, exist_ok=True)
    cf_out = {"query_id": args.query_id, "user_query": uq, "mode": args.mode,
              "source": "region_based" if getattr(args, 'use_regions', False) else "edge_based",
              "num_iterations_run": len(pl), "num_generated_queries": len(tq),
              "generated_queries": [{"query_text": q.get("query_text"), "query_type": q.get("query_type"),
                                      "predicted_state": q.get("predicted_state"), "priority_score": q.get("priority_score")} for q in tq],
              "selected_answer_claims": summary.get("final_evidence_selection", {}).get("selected_answer_claims", []),
              "competing_claims": summary.get("final_evidence_selection", {}).get("competing_claims", []),
              "excluded_claims": summary.get("final_evidence_selection", {}).get("excluded_claims", []),
              "answer_policy": summary.get("final_answer_policy", "present_competing_claims_with_uncertainty"),
              "confidence": summary.get("final_answer_confidence", 0.0),
              "remaining_uncertainties": fld.get("remaining_uncertainties", []),
              "final_answer": fa.get("answer", "")}
    cfp.write_text(json.dumps(cf_out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Claims final saved to: {cfp}")
    return summary


def main() -> None:
    summary = run_experiment(parse_args())
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
