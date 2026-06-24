# edge_feature_builder.py

from __future__ import annotations

import argparse
import json
from datetime import datetime
from difflib import SequenceMatcher
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Literal, Iterable, Any, Tuple

from pydantic import BaseModel, Field, ConfigDict, NonNegativeFloat
from typing_extensions import Annotated


# =============================================================================
# Common types
# =============================================================================

UnitInterval = Annotated[float, Field(ge=0.0, le=1.0)]
BinaryFlag = Annotated[int, Field(ge=0, le=1)]


# =============================================================================
# Enums
# =============================================================================

class RelationType(str, Enum):
    neutral = "neutral"
    entailment = "entailment"
    support = "support"
    contradiction = "contradiction"
    contextual_conflict = "contextual_conflict"


class TemporalRelation(str, Enum):
    equal = "equal"
    overlap = "overlap"
    contains = "contains"
    within = "within"
    before = "before"
    after = "after"
    disjoint = "disjoint"
    unknown = "unknown"


class TemporalGranularity(str, Enum):
    day = "day"
    month = "month"
    year = "year"
    decade = "decade"
    range = "range"
    none = "none"
    unknown = "unknown"


class DominantFailType(str, Enum):
    none = "none"
    entity = "entity"
    attribute = "attribute"
    number = "number"
    temporal = "temporal"
    negation = "negation"
    multiple = "multiple"


# =============================================================================
# Input claim schema
# =============================================================================

class FactoidEntityMention(BaseModel):
    raw_mention: Optional[str] = None
    canonical_entity: str
    attribute: Optional[str] = None


class FactoidNumberValue(BaseModel):
    value: float
    unit: Optional[str] = None


class FactoidVerb(BaseModel):
    lemma: str
    tense: Optional[str] = None


class FactoidTemporal(BaseModel):
    raw_time: Optional[str] = None
    start: Optional[str] = None
    end: Optional[str] = None

    # Dùng str để tránh lỗi khi dữ liệu extract tự động sinh ra:
    # range, none, period, season...
    granularity: Optional[str] = None


class FactoidNegation(BaseModel):
    polarity: BinaryFlag  # 0 = khẳng định, 1 = phủ định


class FactoidFeatures(BaseModel):
    number: List[FactoidNumberValue] = Field(default_factory=list)
    entity: List[FactoidEntityMention] = Field(default_factory=list)
    temporal: Optional[FactoidTemporal] = None
    negation: Optional[FactoidNegation] = None
    verb: Optional[FactoidVerb] = None


class ClaimFeatures(BaseModel):
    retrieval_relevance: UnitInterval
    claim_confidence: UnitInterval
    claim_evidence_coverage: UnitInterval
    context_completeness: UnitInterval


class Claim(BaseModel):
    model_config = ConfigDict(extra="allow")

    claim_id: str
    claim_text: str
    canonical_claim_id: Optional[str] = None
    duplicate_of: Optional[str] = None
    factoid_features: Optional[FactoidFeatures] = None
    claim_features: ClaimFeatures

    evidence: Optional[str] = None
    doc_id: Optional[str] = None
    source_id: Optional[str] = None


# =============================================================================
# Input NLI edge schema
# =============================================================================

class NLIEdgeFeatures(BaseModel):
    entailment_prob: UnitInterval
    neutral_prob: UnitInterval
    support_prob: UnitInterval
    contradiction_prob: UnitInterval


class RawNLIEdge(BaseModel):
    """
    Schema linh hoạt để đọc output từ nli_inferencer.py.

    Hỗ trợ cả:
    - nli_edge_features
    - nli_features
    """

    model_config = ConfigDict(extra="allow")

    edge_id: str
    source_claim_id: str
    target_claim_id: str
    relation_type: str
    relation_confidence: UnitInterval

    nli_edge_features: Optional[NLIEdgeFeatures] = None
    nli_features: Optional[NLIEdgeFeatures] = None

    def get_nli_features(self) -> Optional[NLIEdgeFeatures]:
        if self.nli_features is not None:
            return self.nli_features
        return self.nli_edge_features


# =============================================================================
# Output edge feature schema
# =============================================================================

class EntityPairFeatures(BaseModel):
    """Module 5, mục 7.4 — bảng Entity."""
    entity_match: UnitInterval
    entity_presence_i: BinaryFlag
    entity_presence_j: BinaryFlag
    entity_mismatch: BinaryFlag


class AttributePairFeatures(BaseModel):
    """Module 5, mục 7.4 — bảng Relation / attribute."""
    attribute_match: UnitInterval
    attribute_presence_i: BinaryFlag
    attribute_presence_j: BinaryFlag
    relation_mismatch: BinaryFlag
    implied_attribute_missing: Optional[BinaryFlag] = None
    attribute_exclusivity_conflict: Optional[UnitInterval] = None


class NumberPairFeatures(BaseModel):
    """Module 5, mục 7.4 — bảng Number."""
    number_presence_i: BinaryFlag
    number_presence_j: BinaryFlag
    number_match: Optional[BinaryFlag] = None
    number_diff_ratio: Optional[NonNegativeFloat] = None
    unit_mismatch: Optional[BinaryFlag] = None
    number_mismatch: BinaryFlag


class TemporalPairFeatures(BaseModel):
    """Module 5, mục 7.4 — bảng Temporal."""
    temporal_presence_i: BinaryFlag
    temporal_presence_j: BinaryFlag
    temporal_relation: Optional[TemporalRelation] = None
    temporal_granularity_i: Optional[str] = None
    temporal_granularity_j: Optional[str] = None
    temporal_granularity_match: Optional[BinaryFlag] = None
    temporal_order_mismatch: Optional[BinaryFlag] = None
    temporal_mismatch: BinaryFlag


class NegationPairFeatures(BaseModel):
    """Module 5, mục 7.4 — bảng Negation."""
    negation_i: BinaryFlag
    negation_j: BinaryFlag
    negation_mismatch: BinaryFlag
    negation_conflict_score: UnitInterval


class AggregateEdgeFeatures(BaseModel):
    """Module 5, mục 7.4 — tổng hợp toàn edge."""
    ea_alignment_score: UnitInterval
    conflict_intensity_score: UnitInterval
    entity_presence_coverage: Optional[UnitInterval] = None
    diagnostic_vector: Annotated[List[BinaryFlag], Field(min_length=5, max_length=5)]
    dominant_fail_type: DominantFailType


class ClaimPairFeatures(BaseModel):
    """Module 5, mục 7.4 — Edge Features by Claim-Pair."""
    entity: EntityPairFeatures
    attribute: Optional[AttributePairFeatures] = None
    number: Optional[NumberPairFeatures] = None
    temporal: Optional[TemporalPairFeatures] = None
    negation: Optional[NegationPairFeatures] = None
    aggregate: AggregateEdgeFeatures


class Edge(BaseModel):
    """Relation giữa hai claim — hợp nhất NLI features và claim-pair features."""
    edge_id: str
    source_claim_id: str
    target_claim_id: str
    relation_type: RelationType
    relation_confidence: UnitInterval
    nli_features: Optional[NLIEdgeFeatures] = None
    claim_pair_features: Optional[ClaimPairFeatures] = None


# =============================================================================
# IO helpers
# =============================================================================

def read_jsonl_dicts(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    records: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except Exception as e:
                raise ValueError(
                    f"Invalid JSONL at line {line_no}: {e}\n"
                    f"Line content: {line[:500]}"
                )

    return records


def read_claims(path: str | Path) -> Dict[str, Claim]:
    records = read_jsonl_dicts(path)
    claims: Dict[str, Claim] = {}

    for idx, obj in enumerate(records, start=1):
        try:
            claim = Claim.model_validate(obj)
        except Exception as e:
            raise ValueError(
                f"Invalid Claim at line {idx}: {e}\n"
                f"Line content: {json.dumps(obj, ensure_ascii=False)[:500]}"
            )

        claims[claim.claim_id] = claim

    return claims


def read_nli_edges(path: str | Path) -> List[RawNLIEdge]:
    records = read_jsonl_dicts(path)
    edges: List[RawNLIEdge] = []

    for idx, obj in enumerate(records, start=1):
        try:
            edge = RawNLIEdge.model_validate(obj)
            edges.append(edge)
        except Exception as e:
            raise ValueError(
                f"Invalid NLI edge at line {idx}: {e}\n"
                f"Line content: {json.dumps(obj, ensure_ascii=False)[:500]}"
            )

    return edges


def write_jsonl(records: Iterable[BaseModel | Dict[str, Any]], path: str | Path) -> None:
    path = Path(path)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            if isinstance(record, BaseModel):
                obj = record.model_dump(mode="json")
            else:
                obj = record

            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# =============================================================================
# Basic helpers
# =============================================================================

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def binary(x: bool) -> int:
    return 1 if x else 0


def norm_text(x: Optional[str]) -> str:
    if x is None:
        return ""
    return " ".join(str(x).lower().strip().split())


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def string_similarity(a: Optional[str], b: Optional[str]) -> float:
    a = norm_text(a)
    b = norm_text(b)

    if not a or not b:
        return 0.0

    if a == b:
        return 1.0

    return clamp01(SequenceMatcher(None, a, b).ratio())


def safe_relation_type(x: str) -> RelationType:
    x = norm_text(x)

    mapping = {
        "neutral": RelationType.neutral,
        "entailment": RelationType.entailment,
        "support": RelationType.support,
        "contradiction": RelationType.contradiction,
        "contextual_conflict": RelationType.contextual_conflict,
        "contextual conflict": RelationType.contextual_conflict,
    }

    if x not in mapping:
        raise ValueError(f"Unknown relation_type: {x}")

    return mapping[x]


# =============================================================================
# Factoid extract helpers
# =============================================================================

def get_factoid(claim: Claim) -> FactoidFeatures:
    return claim.factoid_features or FactoidFeatures()


def get_entity_set(claim: Claim) -> set[str]:
    f = get_factoid(claim)
    return {
        norm_text(e.canonical_entity)
        for e in f.entity
        if norm_text(e.canonical_entity)
    }


def get_entity_attribute_set(claim: Claim) -> set[str]:
    f = get_factoid(claim)

    attrs = {
        norm_text(e.attribute)
        for e in f.entity
        if norm_text(e.attribute)
    }

    if f.verb and norm_text(f.verb.lemma):
        attrs.add(f"verb:{norm_text(f.verb.lemma)}")

    return attrs


def get_main_verb(claim: Claim) -> Optional[str]:
    f = get_factoid(claim)
    if f.verb:
        return norm_text(f.verb.lemma)
    return None


def get_numbers(claim: Claim) -> List[FactoidNumberValue]:
    return get_factoid(claim).number


def get_temporal(claim: Claim) -> Optional[FactoidTemporal]:
    return get_factoid(claim).temporal


def get_negation(claim: Claim) -> int:
    f = get_factoid(claim)
    if f.negation is None:
        return 0
    return int(f.negation.polarity)


# =============================================================================
# Entity features
# =============================================================================

def compute_entity_pair_features(
    claim_i: Claim,
    claim_j: Claim,
    mismatch_threshold: float = 0.30,
) -> EntityPairFeatures:
    ent_i = get_entity_set(claim_i)
    ent_j = get_entity_set(claim_j)

    presence_i = binary(len(ent_i) > 0)
    presence_j = binary(len(ent_j) > 0)

    match = jaccard(ent_i, ent_j)

    mismatch = (
        presence_i == 1
        and presence_j == 1
        and match < mismatch_threshold
    )

    return EntityPairFeatures(
        entity_match=clamp01(match),
        entity_presence_i=presence_i,
        entity_presence_j=presence_j,
        entity_mismatch=binary(mismatch),
    )


# =============================================================================
# Attribute / relation features
# =============================================================================

EXCLUSIVE_ATTRIBUTE_PAIRS = {
    ("verb:born", "verb:died"),
    ("verb:win", "verb:lose"),
    ("verb:increase", "verb:decrease"),
    ("verb:open", "verb:close"),
    ("verb:start", "verb:end"),
    ("verb:include", "verb:exclude"),
    ("verb:allow", "verb:ban"),
    ("verb:approve", "verb:reject"),
}


def compute_attribute_exclusivity_conflict(
    attrs_i: set[str],
    attrs_j: set[str],
) -> float:
    for a in attrs_i:
        for b in attrs_j:
            pair = (a, b)
            pair_rev = (b, a)

            if pair in EXCLUSIVE_ATTRIBUTE_PAIRS or pair_rev in EXCLUSIVE_ATTRIBUTE_PAIRS:
                return 1.0

    return 0.0


def compute_attribute_pair_features(
    claim_i: Claim,
    claim_j: Claim,
    relation_type: RelationType,
    mismatch_threshold: float = 0.30,
) -> AttributePairFeatures:
    attrs_i = get_entity_attribute_set(claim_i)
    attrs_j = get_entity_attribute_set(claim_j)

    presence_i = binary(len(attrs_i) > 0)
    presence_j = binary(len(attrs_j) > 0)

    attr_match = jaccard(attrs_i, attrs_j)

    relation_mismatch = (
        presence_i == 1
        and presence_j == 1
        and attr_match < mismatch_threshold
    )

    implied_attribute_missing: Optional[int] = None

    if relation_type in {RelationType.entailment, RelationType.support}:
        implied_attribute_missing = binary(presence_i != presence_j)

    exclusivity = compute_attribute_exclusivity_conflict(attrs_i, attrs_j)

    return AttributePairFeatures(
        attribute_match=clamp01(attr_match),
        attribute_presence_i=presence_i,
        attribute_presence_j=presence_j,
        relation_mismatch=binary(relation_mismatch),
        implied_attribute_missing=implied_attribute_missing,
        attribute_exclusivity_conflict=clamp01(exclusivity),
    )


# =============================================================================
# Number features
# =============================================================================

def number_unit(x: FactoidNumberValue) -> str:
    return norm_text(x.unit)


def compute_number_pair_features(
    claim_i: Claim,
    claim_j: Claim,
    tolerance: float = 1e-6,
) -> NumberPairFeatures:
    nums_i = get_numbers(claim_i)
    nums_j = get_numbers(claim_j)

    presence_i = binary(len(nums_i) > 0)
    presence_j = binary(len(nums_j) > 0)

    if not nums_i or not nums_j:
        return NumberPairFeatures(
            number_presence_i=presence_i,
            number_presence_j=presence_j,
            number_match=None,
            number_diff_ratio=None,
            unit_mismatch=None,
            number_mismatch=0,
        )

    best_diff_ratio: Optional[float] = None
    any_match = False
    any_unit_mismatch = False
    comparable_pairs = 0
    any_comparable_mismatch = False

    for a in nums_i:
        for b in nums_j:
            unit_a = number_unit(a)
            unit_b = number_unit(b)

            if unit_a and unit_b and unit_a != unit_b:
                any_unit_mismatch = True
                continue

            comparable_pairs += 1

            va = float(a.value)
            vb = float(b.value)

            denom = max(abs(va), abs(vb), tolerance)
            diff_ratio = abs(va - vb) / denom

            if best_diff_ratio is None:
                best_diff_ratio = diff_ratio
            else:
                best_diff_ratio = min(best_diff_ratio, diff_ratio)

            if diff_ratio <= tolerance:
                any_match = True
            else:
                any_comparable_mismatch = True

    number_match = binary(any_match)

    number_mismatch = binary(
        comparable_pairs > 0
        and not any_match
        and any_comparable_mismatch
    )

    return NumberPairFeatures(
        number_presence_i=presence_i,
        number_presence_j=presence_j,
        number_match=number_match,
        number_diff_ratio=best_diff_ratio,
        unit_mismatch=binary(any_unit_mismatch),
        number_mismatch=number_mismatch,
    )


# =============================================================================
# Temporal features
# =============================================================================

def parse_date_safe(x: Optional[str]) -> Optional[datetime]:
    if not x:
        return None

    x = str(x).strip()

    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            return datetime.strptime(x, fmt)
        except ValueError:
            continue

    return None


def normalize_temporal_granularity(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None

    x = norm_text(x)

    if not x:
        return None

    return x


def temporal_bounds(t: Optional[FactoidTemporal]) -> Tuple[Optional[datetime], Optional[datetime]]:
    if t is None:
        return None, None

    start = parse_date_safe(t.start)
    end = parse_date_safe(t.end)

    if start is not None and end is None:
        end = start

    return start, end


def infer_temporal_relation(
    ti: Optional[FactoidTemporal],
    tj: Optional[FactoidTemporal],
) -> TemporalRelation:
    si, ei = temporal_bounds(ti)
    sj, ej = temporal_bounds(tj)

    if si is None or ei is None or sj is None or ej is None:
        return TemporalRelation.unknown

    if si == sj and ei == ej:
        return TemporalRelation.equal

    if ei < sj:
        return TemporalRelation.before

    if ej < si:
        return TemporalRelation.after

    if si <= sj and ei >= ej:
        return TemporalRelation.contains

    if sj <= si and ej >= ei:
        return TemporalRelation.within

    if max(si, sj) <= min(ei, ej):
        return TemporalRelation.overlap

    return TemporalRelation.disjoint


def compute_temporal_pair_features(
    claim_i: Claim,
    claim_j: Claim,
) -> TemporalPairFeatures:
    ti = get_temporal(claim_i)
    tj = get_temporal(claim_j)

    presence_i = binary(ti is not None and (ti.raw_time is not None or ti.start is not None or ti.end is not None))
    presence_j = binary(tj is not None and (tj.raw_time is not None or tj.start is not None or tj.end is not None))

    gi = normalize_temporal_granularity(ti.granularity if ti else None)
    gj = normalize_temporal_granularity(tj.granularity if tj else None)

    granularity_match: Optional[int] = None
    if gi is not None and gj is not None:
        granularity_match = binary(gi == gj)

    temporal_relation: Optional[TemporalRelation] = None

    if presence_i or presence_j:
        temporal_relation = infer_temporal_relation(ti, tj)

    temporal_order_mismatch: Optional[int] = None
    temporal_mismatch = 0

    if temporal_relation in {
        TemporalRelation.before,
        TemporalRelation.after,
        TemporalRelation.disjoint,
    }:
        temporal_order_mismatch = 1
        temporal_mismatch = 1
    elif temporal_relation in {
        TemporalRelation.equal,
        TemporalRelation.overlap,
        TemporalRelation.contains,
        TemporalRelation.within,
    }:
        temporal_order_mismatch = 0
        temporal_mismatch = 0
    else:
        temporal_order_mismatch = None
        temporal_mismatch = 0

    return TemporalPairFeatures(
        temporal_presence_i=presence_i,
        temporal_presence_j=presence_j,
        temporal_relation=temporal_relation,
        temporal_granularity_i=gi,
        temporal_granularity_j=gj,
        temporal_granularity_match=granularity_match,
        temporal_order_mismatch=temporal_order_mismatch,
        temporal_mismatch=binary(temporal_mismatch),
    )


# =============================================================================
# Negation features
# =============================================================================

def compute_negation_pair_features(
    claim_i: Claim,
    claim_j: Claim,
) -> NegationPairFeatures:
    ni = get_negation(claim_i)
    nj = get_negation(claim_j)

    mismatch = binary(ni != nj)

    return NegationPairFeatures(
        negation_i=ni,
        negation_j=nj,
        negation_mismatch=mismatch,
        negation_conflict_score=float(mismatch),
    )


# =============================================================================
# Aggregate features
# =============================================================================

def compute_entity_presence_coverage(entity: EntityPairFeatures) -> float:
    return (entity.entity_presence_i + entity.entity_presence_j) / 2.0


def compute_ea_alignment_score(
    entity: EntityPairFeatures,
    attribute: Optional[AttributePairFeatures],
    temporal: Optional[TemporalPairFeatures],
    number: Optional[NumberPairFeatures],
    negation: Optional[NegationPairFeatures],
) -> float:
    """
    Entity-Attribute alignment score.

    Cao nghĩa là hai claim có vẻ nói về cùng factual target.
    """

    scores: List[float] = []
    weights: List[float] = []

    scores.append(entity.entity_match)
    weights.append(0.40)

    if attribute is not None:
        scores.append(attribute.attribute_match)
        weights.append(0.25)

    if temporal is not None:
        if temporal.temporal_relation in {
            TemporalRelation.equal,
            TemporalRelation.overlap,
            TemporalRelation.contains,
            TemporalRelation.within,
        }:
            scores.append(1.0)
        elif temporal.temporal_relation in {
            TemporalRelation.before,
            TemporalRelation.after,
            TemporalRelation.disjoint,
        }:
            scores.append(0.0)
        else:
            scores.append(0.5)

        weights.append(0.15)

    if number is not None:
        if number.number_presence_i == 0 or number.number_presence_j == 0:
            scores.append(0.5)
        elif number.number_match == 1:
            scores.append(1.0)
        elif number.number_mismatch == 1:
            scores.append(0.0)
        else:
            scores.append(0.5)

        weights.append(0.10)

    if negation is not None:
        scores.append(1.0 - negation.negation_conflict_score)
        weights.append(0.10)

    total_weight = sum(weights)

    if total_weight <= 0:
        return 0.0

    return clamp01(sum(s * w for s, w in zip(scores, weights)) / total_weight)


def compute_conflict_intensity_score(
    relation_type: RelationType,
    relation_confidence: float,
    entity: EntityPairFeatures,
    attribute: Optional[AttributePairFeatures],
    number: Optional[NumberPairFeatures],
    temporal: Optional[TemporalPairFeatures],
    negation: Optional[NegationPairFeatures],
) -> float:
    """
    Conflict intensity cao khi:
    - relation_type là contradiction/contextual_conflict
    - có nhiều mismatch ở entity/attribute/number/temporal/negation
    """

    mismatch_scores: List[float] = []

    mismatch_scores.append(float(entity.entity_mismatch))

    if attribute is not None:
        attr_conflict = max(
            float(attribute.relation_mismatch),
            float(attribute.attribute_exclusivity_conflict or 0.0),
        )
        mismatch_scores.append(attr_conflict)

    if number is not None:
        mismatch_scores.append(float(number.number_mismatch))

    if temporal is not None:
        mismatch_scores.append(float(temporal.temporal_mismatch))

    if negation is not None:
        mismatch_scores.append(float(negation.negation_mismatch))

    if not mismatch_scores:
        base = 0.0
    else:
        base = sum(mismatch_scores) / len(mismatch_scores)

    if relation_type == RelationType.contradiction:
        multiplier = 1.0
    elif relation_type == RelationType.contextual_conflict:
        multiplier = 0.8
    elif relation_type in {RelationType.entailment, RelationType.support}:
        multiplier = 0.3
    else:
        multiplier = 0.0

    return clamp01(base * multiplier * relation_confidence)


def compute_diagnostic_vector(
    entity: EntityPairFeatures,
    attribute: Optional[AttributePairFeatures],
    number: Optional[NumberPairFeatures],
    temporal: Optional[TemporalPairFeatures],
    negation: Optional[NegationPairFeatures],
) -> List[int]:
    """
    diagnostic_vector = [
        entity_mismatch,
        attribute_or_relation_mismatch,
        number_mismatch,
        temporal_mismatch,
        negation_mismatch
    ]
    """

    entity_fail = entity.entity_mismatch

    attr_fail = 0
    if attribute is not None:
        attr_fail = binary(
            attribute.relation_mismatch == 1
            or (attribute.attribute_exclusivity_conflict or 0.0) > 0.0
        )

    number_fail = 0
    if number is not None:
        number_fail = number.number_mismatch

    temporal_fail = 0
    if temporal is not None:
        temporal_fail = temporal.temporal_mismatch

    negation_fail = 0
    if negation is not None:
        negation_fail = negation.negation_mismatch

    return [
        entity_fail,
        attr_fail,
        number_fail,
        temporal_fail,
        negation_fail,
    ]


def infer_dominant_fail_type(diagnostic_vector: List[int]) -> DominantFailType:
    fail_count = sum(diagnostic_vector)

    if fail_count == 0:
        return DominantFailType.none

    if fail_count > 1:
        return DominantFailType.multiple

    labels = [
        DominantFailType.entity,
        DominantFailType.attribute,
        DominantFailType.number,
        DominantFailType.temporal,
        DominantFailType.negation,
    ]

    for flag, label in zip(diagnostic_vector, labels):
        if flag == 1:
            return label

    return DominantFailType.none


def compute_aggregate_edge_features(
    relation_type: RelationType,
    relation_confidence: float,
    entity: EntityPairFeatures,
    attribute: Optional[AttributePairFeatures],
    number: Optional[NumberPairFeatures],
    temporal: Optional[TemporalPairFeatures],
    negation: Optional[NegationPairFeatures],
) -> AggregateEdgeFeatures:
    diagnostic_vector = compute_diagnostic_vector(
        entity=entity,
        attribute=attribute,
        number=number,
        temporal=temporal,
        negation=negation,
    )

    ea_alignment_score = compute_ea_alignment_score(
        entity=entity,
        attribute=attribute,
        temporal=temporal,
        number=number,
        negation=negation,
    )

    conflict_intensity_score = compute_conflict_intensity_score(
        relation_type=relation_type,
        relation_confidence=relation_confidence,
        entity=entity,
        attribute=attribute,
        number=number,
        temporal=temporal,
        negation=negation,
    )

    return AggregateEdgeFeatures(
        ea_alignment_score=ea_alignment_score,
        conflict_intensity_score=conflict_intensity_score,
        entity_presence_coverage=compute_entity_presence_coverage(entity),
        diagnostic_vector=diagnostic_vector,
        dominant_fail_type=infer_dominant_fail_type(diagnostic_vector),
    )


# =============================================================================
# Claim-pair feature builder
# =============================================================================

def build_claim_pair_features(
    claim_i: Claim,
    claim_j: Claim,
    relation_type: RelationType,
    relation_confidence: float,
) -> ClaimPairFeatures:
    entity = compute_entity_pair_features(
        claim_i=claim_i,
        claim_j=claim_j,
    )

    attribute = compute_attribute_pair_features(
        claim_i=claim_i,
        claim_j=claim_j,
        relation_type=relation_type,
    )

    number = compute_number_pair_features(
        claim_i=claim_i,
        claim_j=claim_j,
    )

    temporal = compute_temporal_pair_features(
        claim_i=claim_i,
        claim_j=claim_j,
    )

    negation = compute_negation_pair_features(
        claim_i=claim_i,
        claim_j=claim_j,
    )

    aggregate = compute_aggregate_edge_features(
        relation_type=relation_type,
        relation_confidence=relation_confidence,
        entity=entity,
        attribute=attribute,
        number=number,
        temporal=temporal,
        negation=negation,
    )

    return ClaimPairFeatures(
        entity=entity,
        attribute=attribute,
        number=number,
        temporal=temporal,
        negation=negation,
        aggregate=aggregate,
    )


def build_graph_edges(
    claims_by_id: Dict[str, Claim],
    nli_edges: List[RawNLIEdge],
    drop_neutral: bool = True,
    strict_missing_claim: bool = False,
) -> List[Edge]:
    graph_edges: List[Edge] = []

    skipped_neutral = 0
    skipped_missing_claim = 0

    for raw in nli_edges:
        relation_type = safe_relation_type(raw.relation_type)

        if drop_neutral and relation_type == RelationType.neutral:
            skipped_neutral += 1
            continue

        claim_i = claims_by_id.get(raw.source_claim_id)
        claim_j = claims_by_id.get(raw.target_claim_id)

        if claim_i is None or claim_j is None:
            skipped_missing_claim += 1

            if strict_missing_claim:
                raise KeyError(
                    f"Missing claim for edge {raw.edge_id}: "
                    f"{raw.source_claim_id} or {raw.target_claim_id}"
                )

            continue

        pair_features = build_claim_pair_features(
            claim_i=claim_i,
            claim_j=claim_j,
            relation_type=relation_type,
            relation_confidence=raw.relation_confidence,
        )

        graph_edges.append(
            Edge(
                edge_id=raw.edge_id,
                source_claim_id=raw.source_claim_id,
                target_claim_id=raw.target_claim_id,
                relation_type=relation_type,
                relation_confidence=raw.relation_confidence,
                nli_features=raw.get_nli_features(),
                claim_pair_features=pair_features,
            )
        )

    print(f"Built graph edges: {len(graph_edges)}")
    print(f"Skipped neutral edges: {skipped_neutral}")
    print(f"Skipped missing-claim edges: {skipped_missing_claim}")

    return graph_edges


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--claims",
        type=str,
        required=True,
        help="Path tới claims JSONL, ví dụ factoid_claims_revised.jsonl",
    )

    parser.add_argument(
        "--nli_edges",
        type=str,
        required=True,
        help="Path tới output từ nli_inferencer.py, ví dụ nli_edges.jsonl",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="claim_graph_edges.jsonl",
        help="Path output edge features JSONL",
    )

    parser.add_argument(
        "--keep_neutral",
        action="store_true",
        help="Nếu bật, vẫn giữ neutral edges. Mặc định là drop neutral.",
    )

    parser.add_argument(
        "--strict_missing_claim",
        action="store_true",
        help="Nếu bật, lỗi ngay khi edge trỏ tới claim_id không tồn tại.",
    )

    args = parser.parse_args()

    claims_by_id = read_claims(args.claims)
    print(f"Loaded claims: {len(claims_by_id)}")

    nli_edges = read_nli_edges(args.nli_edges)
    print(f"Loaded NLI edges: {len(nli_edges)}")

    graph_edges = build_graph_edges(
        claims_by_id=claims_by_id,
        nli_edges=nli_edges,
        drop_neutral=not args.keep_neutral,
        strict_missing_claim=args.strict_missing_claim,
    )

    write_jsonl(graph_edges, args.output)
    print(f"Saved graph edges to: {args.output}")


if __name__ == "__main__":
    main()