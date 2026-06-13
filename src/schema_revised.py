"""
src/schema_revised.py
Pydantic models theo conflict_state_guided_rag_schema.json.
Pipeline: Conflict-State-Guided Retrieval for Conflict-Aware RAG.
Lưu ý: Document, ActionLabel, LoopResult đã bị loại bỏ theo schema revise.
"""

from __future__ import annotations
from typing import Annotated, Any, Literal, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Primitive type aliases
# ---------------------------------------------------------------------------

BinaryFlag = Literal[0, 1]
UnitInterval = Annotated[float, Field(ge=0.0, le=1.0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]
EmbeddingVector = list[float]

RelationType = Literal["support", "entailment", "contradiction", "neutral"]
ConflictState = Literal["Resolvable", "Underdetermined", "Contextual"]
QueryType = Literal[
    "resolution_verification",
    "evidence_coverage_expansion",
    "context_disambiguation",
]
TemporalRelation = Literal["before", "after", "during", "disjoint", "equal"]
TemporalGranularity = Literal["day", "month", "year", "range", "none"]
ContextSlotType = Literal["time", "condition", "scope", "location", "entity", "effective_date"]
ContextMismatchType = Literal["time", "condition", "scope", "location", "entity", "modality"]
DominantFailType = Literal[
    "entity_mismatch",
    "relation_mismatch",
    "number_mismatch",
    "temporal_mismatch",
    "negation_mismatch",
    "attribute_exclusivity_conflict",
    "none",
]
FeatureLevel = Literal["claim", "edge", "cluster", "region"]


# ---------------------------------------------------------------------------
# Factoid sub-models (Module 3, mục 5.3)
# ---------------------------------------------------------------------------

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
    granularity: Optional[TemporalGranularity] = None


class FactoidNegation(BaseModel):
    polarity: BinaryFlag  # 0 = khẳng định, 1 = phủ định


class FactoidFeatures(BaseModel):
    number: list[FactoidNumberValue] = Field(default_factory=list)
    entity: list[FactoidEntityMention] = Field(default_factory=list)
    temporal: Optional[FactoidTemporal] = None
    negation: Optional[FactoidNegation] = None
    verb: Optional[FactoidVerb] = None


# ---------------------------------------------------------------------------
# Claim (Modules 2, 3, 4)
# ---------------------------------------------------------------------------

class ClaimFeatures(BaseModel):
    """Module 4, mục 6.2 — Node Features."""
    retrieval_relevance: UnitInterval
    claim_confidence: UnitInterval
    claim_evidence_coverage: UnitInterval
    context_completeness: UnitInterval


class Claim(BaseModel):
    """Node trong claim graph — hợp nhất Atomic Claim, Factoid Claim, Claim Features, Claim ID Alignment."""
    claim_id: str
    claim_text: str
    canonical_claim_id: Optional[str] = None
    duplicate_of: Optional[str] = None
    factoid_features: Optional[FactoidFeatures] = None
    claim_features: ClaimFeatures


# ---------------------------------------------------------------------------
# Edge Features (Module 5, mục 7.3 và 7.4)
# ---------------------------------------------------------------------------

class NLIEdgeFeatures(BaseModel):
    """Module 5, mục 7.3 — NLI inference probabilities."""
    entailment_prob: UnitInterval
    neutral_prob: UnitInterval
    support_prob: UnitInterval
    contradiction_prob: UnitInterval


class EntityPairFeatures(BaseModel):
    """Module 5, mục 7.4 — bảng Entity."""
    entity_match: UnitInterval
    entity_presence_i: BinaryFlag
    entity_presence_j: BinaryFlag
    entity_mismatch: BinaryFlag


class AttributePairFeatures(BaseModel):
    """Module 5, mục 7.4 — bảng Relation (attribute a)."""
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
    temporal_granularity_i: Optional[TemporalGranularity] = None
    temporal_granularity_j: Optional[TemporalGranularity] = None
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
    diagnostic_vector: Annotated[list[BinaryFlag], Field(min_length=5, max_length=5)]
    dominant_fail_type: DominantFailType


class ClaimPairFeatures(BaseModel):
    """Module 5, mục 7.4 — Edge Features by Claim-Pair."""
    entity: EntityPairFeatures
    attribute: Optional[AttributePairFeatures] = None
    number: Optional[NumberPairFeatures] = None
    temporal: Optional[TemporalPairFeatures] = None
    negation: Optional[NegationPairFeatures] = None
    aggregate: AggregateEdgeFeatures


# ---------------------------------------------------------------------------
# Edge (Module 5)
# ---------------------------------------------------------------------------

class Edge(BaseModel):
    """Relation giữa hai claim — hợp nhất NLI features và claim-pair features."""
    edge_id: str
    source_claim_id: str
    target_claim_id: str
    relation_type: RelationType
    relation_confidence: UnitInterval
    nli_features: Optional[NLIEdgeFeatures] = None
    claim_pair_features: Optional[ClaimPairFeatures] = None


# ---------------------------------------------------------------------------
# Cluster (Module 7)
# ---------------------------------------------------------------------------

class ClusterFeatures(BaseModel):
    """Module 7, mục 9.6 — Cluster Features."""
    num_claims: int = Field(ge=0)
    avg_claim_confidence: UnitInterval
    cluster_evidence_coverage: UnitInterval
    support_density: UnitInterval
    internal_consistency: UnitInterval
    avg_context_completeness: UnitInterval


class Cluster(BaseModel):
    """Module 7, mục 9.1 — một stance/phía lập luận trong conflict region."""
    cluster_id: str
    conflict_region_id: str
    claim_ids: list[str]
    cluster_stance_summary: Optional[str] = None
    cluster_features: ClusterFeatures


# ---------------------------------------------------------------------------
# Conflict Region (Modules 6, 8, 10)
# ---------------------------------------------------------------------------

class RegionDetection(BaseModel):
    """Module 6, mục 8.2 — Graph Features cho region detection."""
    core_contradiction_edges: list[str] = Field(default_factory=list)
    region_seed_claims: list[str] = Field(default_factory=list)
    region_detection_method: Optional[str] = None


class RegionFeatures(BaseModel):
    """Module 8, mục 10.12/10.13 — input chính cho Conflict State Classification."""
    num_claims: int = Field(ge=0)
    num_edges: Optional[int] = Field(default=None, ge=0)
    num_claim_clusters: int = Field(ge=0)
    num_contradiction_edges: Optional[int] = Field(default=None, ge=0)
    contradiction_density: UnitInterval
    support_density: UnitInterval
    max_cluster_confidence: Optional[UnitInterval] = None
    min_cluster_confidence: Optional[UnitInterval] = None
    cluster_confidence_gap: UnitInterval
    max_cluster_coverage: Optional[UnitInterval] = None
    min_cluster_coverage: Optional[UnitInterval] = None
    cluster_coverage_gap: UnitInterval
    coverage_balance_score: UnitInterval
    avg_context_completeness: UnitInterval
    missing_time_ratio: Optional[UnitInterval] = None
    missing_condition_ratio: Optional[UnitInterval] = None
    missing_scope_ratio: Optional[UnitInterval] = None
    dominant_context_mismatch_type: list[ContextMismatchType] = Field(default_factory=list)


class AttentionClaimEntry(BaseModel):
    claim_id: str
    attention_weight: UnitInterval


class AttentionEdgeEntry(BaseModel):
    edge_id: str
    edge_type: RelationType
    attention_weight: UnitInterval


class AttentionDiagnostics(BaseModel):
    """Mục 10.11 — hỗ trợ interpretability cho Conflict Region Encoding."""
    top_claims: list[AttentionClaimEntry] = Field(default_factory=list)
    top_edges: list[AttentionEdgeEntry] = Field(default_factory=list)


class ConflictRegionEncoding(BaseModel):
    """Module 8, mục 10.11 — output của graph encoder."""
    conflict_region_id: str
    encoding_method: str
    region_embedding: EmbeddingVector
    node_embeddings: dict[str, EmbeddingVector] = Field(default_factory=dict)
    cluster_embeddings: dict[str, EmbeddingVector] = Field(default_factory=dict)
    attention_diagnostics: Optional[AttentionDiagnostics] = None
    aggregated_region_features: RegionFeatures


class ConflictRegion(BaseModel):
    """Region-level representation — hợp nhất Modules 6, 8, 10."""
    conflict_region_id: str
    claim_ids: list[str]
    detection: Optional[RegionDetection] = None
    region_features: RegionFeatures
    region_embedding: Optional[EmbeddingVector] = None
    encoding: Optional[ConflictRegionEncoding] = None
    predicted_state: Optional[ConflictState] = None


# ---------------------------------------------------------------------------
# Conflict State Classification (Module 10)
# ---------------------------------------------------------------------------

class StateProbabilities(BaseModel):
    """Mục 12.4 — state_probabilities."""
    Resolvable: UnitInterval
    Underdetermined: UnitInterval
    Contextual: UnitInterval


class StateRationale(BaseModel):
    """Mục 12.4 — giải thích phân loại conflict state."""
    main_signal: str
    missing_context_slots: list[ContextSlotType] = Field(default_factory=list)
    confidence_gap: Optional[UnitInterval] = None
    coverage_gap: Optional[UnitInterval] = None


class ClassificationRegionFeatures(BaseModel):
    """Mục 12.3 — tập con region_features dùng trực tiếp cho phân loại state."""
    contradiction_density: Optional[UnitInterval] = None
    support_density: Optional[UnitInterval] = None
    cluster_confidence_gap: Optional[UnitInterval] = None
    cluster_coverage_gap: Optional[UnitInterval] = None
    coverage_balance_score: Optional[UnitInterval] = None
    avg_context_completeness: Optional[UnitInterval] = None
    missing_time_ratio: Optional[UnitInterval] = None
    missing_condition_ratio: Optional[UnitInterval] = None
    dominant_context_mismatch_type: list[str] = Field(default_factory=list)


class ConflictStateClassificationInput(BaseModel):
    """Module 10, mục 12.3 — Input schema."""
    conflict_region_id: str
    region_embedding: Optional[EmbeddingVector] = None
    region_features: ClassificationRegionFeatures


class ConflictStateClassificationOutput(BaseModel):
    """Module 10, mục 12.4 — Output schema."""
    conflict_region_id: str
    predicted_state: ConflictState
    state_probabilities: StateProbabilities
    state_rationale: Optional[StateRationale] = None


# ---------------------------------------------------------------------------
# Targeted Query Generation (Module 11)
# ---------------------------------------------------------------------------

class ConflictingClaimRef(BaseModel):
    """Mục 13.4 — tham chiếu claim đang xung đột."""
    claim_id: str
    claim_text: str
    claim_confidence: UnitInterval


class TargetedQueryGenerationInput(BaseModel):
    """Module 11, mục 13.4 — Input schema."""
    query_id: str
    user_query: str
    conflict_region_id: str
    predicted_state: ConflictState
    conflicting_claims: list[ConflictingClaimRef]
    missing_context_slots: list[ContextSlotType] = Field(default_factory=list)
    dominant_context_mismatch_type: list[ContextMismatchType] = Field(default_factory=list)


class TargetedQuery(BaseModel):
    """Mục 16.6 / 13.5 — targeted query điều khiển retrieval vòng tiếp theo."""
    targeted_query_id: str
    conflict_region_id: str
    predicted_state: ConflictState
    query_text: str
    query_type: QueryType
    target_claim_ids: list[str]
    missing_slots_targeted: list[ContextSlotType] = Field(default_factory=list)
    priority: int = Field(ge=1)
    expected_evidence_type: Optional[str] = None


class TargetedQueryGenerationOutput(BaseModel):
    """Module 11, mục 13.5 — Output schema."""
    conflict_region_id: str
    predicted_state: ConflictState
    targeted_queries: list[TargetedQuery]


# ---------------------------------------------------------------------------
# Iterative Reasoning-Retrieval Loop (Module 12)
# ---------------------------------------------------------------------------

class TargetedQueryRef(BaseModel):
    """Tham chiếu rút gọn tới một targeted query, dùng trong IterationInput."""
    targeted_query_id: str
    query_text: str


class IterationInput(BaseModel):
    """Module 12, mục 14.3 — Input schema."""
    iteration_id: int = Field(ge=1)
    current_graph_id: str
    unresolved_conflict_regions: list[str]
    targeted_queries: list[TargetedQueryRef]


class UpdatedConflictRegion(BaseModel):
    """Mục 14.4 — kết quả re-classification sau iteration."""
    conflict_region_id: str
    old_state: ConflictState
    new_state: ConflictState
    state_change_reason: str


class GraphUpdate(BaseModel):
    """Mục 14.4 — graph_update."""
    new_documents: list[str]
    new_claims: list[str]
    new_edges: list[str]
    updated_conflict_regions: list[UpdatedConflictRegion]


class IterationOutput(BaseModel):
    """Module 12, mục 14.4 — Output schema."""
    iteration_id: int = Field(ge=1)
    graph_update: GraphUpdate


class StoppingCriteriaFields(BaseModel):
    major_conflicts_resolved: bool
    state_stable_for_n_iterations: int = Field(ge=0)
    confidence_delta_below_threshold: float = Field(ge=0)
    coverage_gain_below_threshold: float = Field(ge=0)
    no_new_relevant_claims: bool
    max_iterations_reached: bool


class StoppingCriteria(BaseModel):
    """Module 12, mục 14.5 — Stopping criteria schema."""
    stopping_criteria: StoppingCriteriaFields


# ---------------------------------------------------------------------------
# Final Evidence Selection and Answer Generation (Module 13)
# ---------------------------------------------------------------------------

class ResolvedGraph(BaseModel):
    """Mục 15.3 — trạng thái cuối cùng của claim graph sau khi refine."""
    graph_id: str
    claims: list[str]
    resolved_conflict_regions: list[str]
    unresolved_conflict_regions: list[str]


class FinalAnswerInput(BaseModel):
    """Module 13, mục 15.3 — Input schema."""
    query_id: str
    user_query: str
    resolved_graph: ResolvedGraph


class FinalAnswerOutput(BaseModel):
    """Module 13, mục 15.4/15.5 — Output schema."""
    query_id: str
    answer: str
    supporting_claims: list[str]
    resolved_conflict_regions: list[str]
    answer_confidence: UnitInterval
    remaining_uncertainties: list[str]


# ---------------------------------------------------------------------------
# Alignment utilities (Mục 17)
# ---------------------------------------------------------------------------

class EntityAlignment(BaseModel):
    """Mục 17.2 — chuẩn hóa entity mentions khác cách viết."""
    raw_mentions: list[str]
    canonical_entity: str


class TemporalAlignment(BaseModel):
    """Mục 17.3 — normalize time về start, end, granularity."""
    raw_time: str
    start: str
    end: str
    granularity: str  # TemporalGranularity + "fuzzy_range"


# ---------------------------------------------------------------------------
# Feature Update Log (Module 9)
# ---------------------------------------------------------------------------

class FeatureUpdateLogEntry(BaseModel):
    """Module 9, mục 11.3 / 17.5 — per-entry trong update log."""
    level: FeatureLevel
    target_id: str
    iteration: Optional[int] = Field(default=None, ge=0)
    feature_name: str
    old_value: Any
    new_value: Any
    update_reason: str


class FeatureUpdateLog(BaseModel):
    """Module 9, mục 11.3 — Update log schema."""
    iteration_id: int = Field(ge=0)
    feature_update_log: list[FeatureUpdateLogEntry]


# ---------------------------------------------------------------------------
# ConflictGraph — top-level container
# ---------------------------------------------------------------------------

class ConflictGraph(BaseModel):
    """Biểu diễn tổng thể claim graph tại một iteration nhất định."""
    graph_id: str
    claims: list[Claim]
    edges: list[Edge]
    clusters: list[Cluster] = Field(default_factory=list)
    conflict_regions: list[ConflictRegion]