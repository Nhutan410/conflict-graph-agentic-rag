# conflict_region_builder.py

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict, deque
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Set, Tuple, Iterable, Any

from pydantic import BaseModel, Field, ConfigDict
from typing_extensions import Annotated


# =============================================================================
# Common types
# =============================================================================

UnitInterval = Annotated[float, Field(ge=0.0, le=1.0)]
BinaryFlag = Annotated[int, Field(ge=0, le=1)]
EmbeddingVector = List[float]


# =============================================================================
# Enums
# =============================================================================

class RelationType(str, Enum):
    neutral = "neutral"
    entailment = "entailment"
    support = "support"
    contradiction = "contradiction"
    contextual_conflict = "contextual_conflict"


class ContextMismatchType(str, Enum):
    entity = "entity"
    attribute = "attribute"
    number = "number"
    temporal = "temporal"
    negation = "negation"
    condition = "condition"
    scope = "scope"
    multiple = "multiple"
    none = "none"


class ConflictState(str, Enum):
    resolvable = "resolvable"
    underdetermined = "underdetermined"
    contextual = "contextual"
    no_conflict = "no_conflict"


class DominantFailType(str, Enum):
    none = "none"
    entity = "entity"
    attribute = "attribute"
    number = "number"
    temporal = "temporal"
    negation = "negation"
    multiple = "multiple"


# =============================================================================
# Claim schema
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
    granularity: Optional[str] = None


class FactoidNegation(BaseModel):
    polarity: BinaryFlag


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
# Edge schemas from edge_feature_builder.py
# =============================================================================

class NLIEdgeFeatures(BaseModel):
    entailment_prob: UnitInterval
    neutral_prob: UnitInterval
    support_prob: UnitInterval
    contradiction_prob: UnitInterval


class EntityPairFeatures(BaseModel):
    entity_match: UnitInterval
    entity_presence_i: BinaryFlag
    entity_presence_j: BinaryFlag
    entity_mismatch: BinaryFlag


class AttributePairFeatures(BaseModel):
    attribute_match: UnitInterval
    attribute_presence_i: BinaryFlag
    attribute_presence_j: BinaryFlag
    relation_mismatch: BinaryFlag
    implied_attribute_missing: Optional[BinaryFlag] = None
    attribute_exclusivity_conflict: Optional[UnitInterval] = None


class NumberPairFeatures(BaseModel):
    number_presence_i: BinaryFlag
    number_presence_j: BinaryFlag
    number_match: Optional[BinaryFlag] = None
    number_diff_ratio: Optional[float] = None
    unit_mismatch: Optional[BinaryFlag] = None
    number_mismatch: BinaryFlag


class TemporalPairFeatures(BaseModel):
    temporal_presence_i: BinaryFlag
    temporal_presence_j: BinaryFlag
    temporal_relation: Optional[str] = None
    temporal_granularity_i: Optional[str] = None
    temporal_granularity_j: Optional[str] = None
    temporal_granularity_match: Optional[BinaryFlag] = None
    temporal_order_mismatch: Optional[BinaryFlag] = None
    temporal_mismatch: BinaryFlag


class NegationPairFeatures(BaseModel):
    negation_i: BinaryFlag
    negation_j: BinaryFlag
    negation_mismatch: BinaryFlag
    negation_conflict_score: UnitInterval


class AggregateEdgeFeatures(BaseModel):
    ea_alignment_score: UnitInterval
    conflict_intensity_score: UnitInterval
    entity_presence_coverage: Optional[UnitInterval] = None
    diagnostic_vector: Annotated[List[BinaryFlag], Field(min_length=5, max_length=5)]
    dominant_fail_type: DominantFailType


class ClaimPairFeatures(BaseModel):
    entity: EntityPairFeatures
    attribute: Optional[AttributePairFeatures] = None
    number: Optional[NumberPairFeatures] = None
    temporal: Optional[TemporalPairFeatures] = None
    negation: Optional[NegationPairFeatures] = None
    aggregate: AggregateEdgeFeatures


class Edge(BaseModel):
    model_config = ConfigDict(extra="allow")

    edge_id: str
    source_claim_id: str
    target_claim_id: str
    relation_type: RelationType
    relation_confidence: UnitInterval
    nli_features: Optional[NLIEdgeFeatures] = None
    claim_pair_features: Optional[ClaimPairFeatures] = None


# =============================================================================
# Cluster schema
# =============================================================================

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
    claim_ids: List[str]
    cluster_stance_summary: Optional[str] = None
    cluster_features: ClusterFeatures


# =============================================================================
# Conflict region schema
# =============================================================================

class RegionDetection(BaseModel):
    """Module 6, mục 8.2 — Graph Features cho region detection."""
    core_contradiction_edges: List[str] = Field(default_factory=list)
    region_seed_claims: List[str] = Field(default_factory=list)
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
    dominant_context_mismatch_type: List[ContextMismatchType] = Field(default_factory=list)


class AttentionClaimEntry(BaseModel):
    claim_id: str
    attention_weight: UnitInterval


class AttentionEdgeEntry(BaseModel):
    edge_id: str
    edge_type: RelationType
    attention_weight: UnitInterval


class AttentionDiagnostics(BaseModel):
    """Mục 10.11 — hỗ trợ interpretability cho Conflict Region Encoding."""
    top_claims: List[AttentionClaimEntry] = Field(default_factory=list)
    top_edges: List[AttentionEdgeEntry] = Field(default_factory=list)


class ConflictRegionEncoding(BaseModel):
    """Module 8, mục 10.11 — output của graph encoder."""
    conflict_region_id: str
    encoding_method: str
    region_embedding: EmbeddingVector
    node_embeddings: Dict[str, EmbeddingVector] = Field(default_factory=dict)
    cluster_embeddings: Dict[str, EmbeddingVector] = Field(default_factory=dict)
    attention_diagnostics: Optional[AttentionDiagnostics] = None
    aggregated_region_features: RegionFeatures


class ConflictRegion(BaseModel):
    """Region-level representation — hợp nhất Modules 6, 8, 10."""
    conflict_region_id: str
    claim_ids: List[str]
    detection: Optional[RegionDetection] = None
    region_features: RegionFeatures
    region_embedding: Optional[EmbeddingVector] = None
    encoding: Optional[ConflictRegionEncoding] = None
    predicted_state: Optional[ConflictState] = None


class ConflictRegionRecord(BaseModel):
    conflict_region: ConflictRegion
    clusters: List[Cluster] = Field(default_factory=list)
    edge_ids: List[str] = Field(default_factory=list)


# =============================================================================
# IO helpers
# =============================================================================

def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
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


def write_jsonl(records: Iterable[BaseModel | Dict[str, Any]], path: str | Path) -> None:
    path = Path(path)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            if isinstance(record, BaseModel):
                obj = record.model_dump(mode="json")
            else:
                obj = record
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def read_claims(path: str | Path) -> Dict[str, Claim]:
    claims: Dict[str, Claim] = {}

    for i, obj in enumerate(read_jsonl(path), start=1):
        try:
            claim = Claim.model_validate(obj)
        except Exception as e:
            raise ValueError(
                f"Invalid Claim at line {i}: {e}\n"
                f"{json.dumps(obj, ensure_ascii=False)[:500]}"
            )
        claims[claim.claim_id] = claim

    return claims


def read_edges(path: str | Path) -> Dict[str, Edge]:
    edges: Dict[str, Edge] = {}

    for i, obj in enumerate(read_jsonl(path), start=1):
        try:
            edge = Edge.model_validate(obj)
        except Exception as e:
            raise ValueError(
                f"Invalid Edge at line {i}: {e}\n"
                f"{json.dumps(obj, ensure_ascii=False)[:500]}"
            )
        edges[edge.edge_id] = edge

    return edges


# =============================================================================
# Basic utilities
# =============================================================================

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def safe_mean(values: List[float], default: float = 0.0) -> float:
    if not values:
        return default
    return float(sum(values) / len(values))


def safe_max(values: List[float], default: Optional[float] = None) -> Optional[float]:
    if not values:
        return default
    return float(max(values))


def safe_min(values: List[float], default: Optional[float] = None) -> Optional[float]:
    if not values:
        return default
    return float(min(values))


def normalize_vector(xs: List[float]) -> List[float]:
    """
    L2 normalize feature vector để dùng như region_embedding prototype.
    """
    norm = math.sqrt(sum(x * x for x in xs))
    if norm <= 1e-12:
        return xs
    return [float(x / norm) for x in xs]


def edge_is_conflict(edge: Edge) -> bool:
    return edge.relation_type in {
        RelationType.contradiction,
        RelationType.contextual_conflict,
    }


def edge_is_positive(edge: Edge) -> bool:
    return edge.relation_type in {
        RelationType.support,
        RelationType.entailment,
    }


def get_claim_confidence(claim: Claim) -> float:
    return float(claim.claim_features.claim_confidence)


def get_claim_coverage(claim: Claim) -> float:
    return float(claim.claim_features.claim_evidence_coverage)


def get_context_completeness(claim: Claim) -> float:
    return float(claim.claim_features.context_completeness)


def get_claim_doc_key(claim: Claim) -> Optional[str]:
    """
    Document key dùng cho option require_distinct_docs_per_region.

    Ưu tiên doc_id; nếu thiếu thì fallback source_id.
    Nếu cả hai đều thiếu thì trả về None và region sẽ bị loại
    khi bật require_distinct_docs_per_region.
    """
    if claim.doc_id:
        return str(claim.doc_id)
    if claim.source_id:
        return str(claim.source_id)
    return None


def claims_have_distinct_docs(
    claim_ids: Iterable[str],
    claims: Dict[str, Claim],
) -> bool:
    """
    True nếu mỗi claim trong tập có document key khác nhau.
    Claim không tồn tại hoặc thiếu cả doc_id/source_id được xem là không đạt.
    """
    seen_docs: Set[str] = set()

    for cid in claim_ids:
        claim = claims.get(cid)
        if claim is None:
            return False

        doc_key = get_claim_doc_key(claim)
        if doc_key is None:
            return False

        if doc_key in seen_docs:
            return False

        seen_docs.add(doc_key)

    return True


def has_time(claim: Claim) -> bool:
    f = claim.factoid_features
    if f is None or f.temporal is None:
        return False
    t = f.temporal
    return bool(t.raw_time or t.start or t.end)


def has_condition_or_scope(claim: Claim) -> Tuple[bool, bool]:
    """
    Schema factoid hiện tại chưa có condition/scope field riêng.
    Dùng entity.attribute heuristic:
    - attribute chứa condition -> condition
    - attribute chứa scope / category / area / domain -> scope
    """
    f = claim.factoid_features
    if f is None:
        return False, False

    has_condition = False
    has_scope_value = False

    for ent in f.entity:
        attr = (ent.attribute or "").lower()
        if "condition" in attr or "criteria" in attr:
            has_condition = True
        if "scope" in attr or "category" in attr or "domain" in attr or "area" in attr:
            has_scope_value = True

    return has_condition, has_scope_value


def get_context_mismatch_types_from_edge(edge: Edge) -> List[ContextMismatchType]:
    if edge.claim_pair_features is None:
        return []

    cpf = edge.claim_pair_features
    output: List[ContextMismatchType] = []

    if cpf.entity and cpf.entity.entity_mismatch == 1:
        output.append(ContextMismatchType.entity)

    if cpf.attribute:
        attr_fail = (
            cpf.attribute.relation_mismatch == 1
            or (cpf.attribute.attribute_exclusivity_conflict or 0.0) > 0.0
        )
        if attr_fail:
            output.append(ContextMismatchType.attribute)

    if cpf.number and cpf.number.number_mismatch == 1:
        output.append(ContextMismatchType.number)

    if cpf.temporal and cpf.temporal.temporal_mismatch == 1:
        output.append(ContextMismatchType.temporal)

    if cpf.negation and cpf.negation.negation_mismatch == 1:
        output.append(ContextMismatchType.negation)

    return output


# =============================================================================
# Graph construction
# =============================================================================

class ClaimGraph(BaseModel):
    claim_ids: Set[str]
    edge_ids: Set[str]
    adjacency: Dict[str, Set[str]]
    positive_adjacency: Dict[str, Set[str]]
    conflict_edges: Set[str]
    edges_by_pair: Dict[Tuple[str, str], List[str]]


def build_claim_graph(
    claims: Dict[str, Claim],
    edges: Dict[str, Edge],
) -> ClaimGraph:
    claim_ids: Set[str] = set(claims.keys())
    edge_ids: Set[str] = set(edges.keys())

    adjacency: Dict[str, Set[str]] = defaultdict(set)
    positive_adjacency: Dict[str, Set[str]] = defaultdict(set)
    conflict_edges: Set[str] = set()
    edges_by_pair: Dict[Tuple[str, str], List[str]] = defaultdict(list)

    for edge_id, e in edges.items():
        a = e.source_claim_id
        b = e.target_claim_id

        claim_ids.add(a)
        claim_ids.add(b)

        adjacency[a].add(b)
        adjacency[b].add(a)

        pair_key = tuple(sorted([a, b]))
        edges_by_pair[pair_key].append(edge_id)

        if edge_is_positive(e):
            positive_adjacency[a].add(b)
            positive_adjacency[b].add(a)

        if edge_is_conflict(e):
            conflict_edges.add(edge_id)

    return ClaimGraph(
        claim_ids=claim_ids,
        edge_ids=edge_ids,
        adjacency=dict(adjacency),
        positive_adjacency=dict(positive_adjacency),
        conflict_edges=conflict_edges,
        edges_by_pair=dict(edges_by_pair),
    )


# =============================================================================
# Module 6: Conflict Region Detection
# =============================================================================

def detect_conflict_regions(
    claims: Dict[str, Claim],
    edges: Dict[str, Edge],
    graph: ClaimGraph,
    expansion_hops: int = 1,
    min_conflict_confidence: float = 0.50,
    require_distinct_docs_per_region: bool = False,
) -> List[RegionDetection]:
    """
    Module 6:
    - seed = contradiction/contextual_conflict edge
    - expand sang neighbors qua support/entailment edges
    - merge nếu regions share claims
    - nếu require_distinct_docs_per_region=True thì mọi claim trong region
      phải đến từ các document khác nhau (doc_id, fallback source_id).
    """

    seed_regions: List[RegionDetection] = []

    for edge_id in graph.conflict_edges:
        edge = edges[edge_id]

        if edge.relation_confidence < min_conflict_confidence:
            continue

        seed_claims = [edge.source_claim_id, edge.target_claim_id]

        if require_distinct_docs_per_region and not claims_have_distinct_docs(seed_claims, claims):
            continue

        region_claims: Set[str] = set(seed_claims)

        # Expand quanh seed qua positive edges.
        # Khi bật require_distinct_docs_per_region, chỉ thêm neighbor nếu doc của neighbor
        # chưa xuất hiện trong region hiện tại.
        q = deque([(c, 0) for c in seed_claims])
        visited = set(seed_claims)

        while q:
            cur, depth = q.popleft()
            if depth >= expansion_hops:
                continue

            for nb in graph.positive_adjacency.get(cur, set()):
                if nb in visited:
                    continue

                candidate_claims = set(region_claims)
                candidate_claims.add(nb)

                if require_distinct_docs_per_region and not claims_have_distinct_docs(candidate_claims, claims):
                    continue

                visited.add(nb)
                region_claims.add(nb)
                q.append((nb, depth + 1))

        seed_regions.append(
            RegionDetection(
                core_contradiction_edges=[edge_id],
                region_seed_claims=seed_claims,
                region_detection_method=f"contradiction_seed_expansion_hops_{expansion_hops}",
            )
        )

        # Gắn tạm claim_ids vào object bằng extra không được phép trong schema,
        # nên merge sẽ dùng mapping ngoài.
        seed_regions[-1].__dict__["_claim_ids"] = list(region_claims)

    # Merge overlapping regions.
    merged: List[Tuple[Set[str], List[str], List[str]]] = []

    for det in seed_regions:
        claim_set = set(det.__dict__.get("_claim_ids", []))
        core_edges = list(det.core_contradiction_edges)
        seed_claims = list(det.region_seed_claims)

        if require_distinct_docs_per_region and not claims_have_distinct_docs(claim_set, claims):
            continue

        merged_into_existing = False

        for idx, (m_claims, m_edges, m_seeds) in enumerate(merged):
            if not (claim_set & m_claims):
                continue

            merged_claims = set(m_claims)
            merged_claims.update(claim_set)

            if require_distinct_docs_per_region and not claims_have_distinct_docs(merged_claims, claims):
                continue

            m_claims.update(claim_set)
            m_edges.extend(core_edges)
            m_seeds.extend(seed_claims)
            merged[idx] = (m_claims, sorted(set(m_edges)), sorted(set(m_seeds)))
            merged_into_existing = True
            break

        if not merged_into_existing:
            merged.append((claim_set, sorted(set(core_edges)), sorted(set(seed_claims))))

    detections: List[RegionDetection] = []

    for region_idx, (claim_set, core_edges, seed_claims) in enumerate(merged, start=1):
        if require_distinct_docs_per_region and not claims_have_distinct_docs(claim_set, claims):
            continue

        det = RegionDetection(
            core_contradiction_edges=core_edges,
            region_seed_claims=seed_claims,
            region_detection_method=f"contradiction_seed_expansion_hops_{expansion_hops}_merged",
        )
        det.__dict__["_claim_ids"] = sorted(claim_set)
        detections.append(det)

    return detections

def collect_region_edges(
    region_claim_ids: Set[str],
    edges: Dict[str, Edge],
) -> List[str]:
    edge_ids = []

    for edge_id, e in edges.items():
        if e.source_claim_id in region_claim_ids and e.target_claim_id in region_claim_ids:
            edge_ids.append(edge_id)

    return sorted(edge_ids)


# =============================================================================
# Module 7: Claim Cluster Construction
# =============================================================================

def connected_components_from_adjacency(
    nodes: Set[str],
    adjacency: Dict[str, Set[str]],
) -> List[Set[str]]:
    seen: Set[str] = set()
    components: List[Set[str]] = []

    for n in sorted(nodes):
        if n in seen:
            continue

        comp: Set[str] = set()
        q = deque([n])
        seen.add(n)

        while q:
            cur = q.popleft()
            comp.add(cur)

            for nb in adjacency.get(cur, set()):
                if nb not in nodes:
                    continue
                if nb in seen:
                    continue
                seen.add(nb)
                q.append(nb)

        components.append(comp)

    return components


def build_clusters_for_region(
    conflict_region_id: str,
    region_claim_ids: Set[str],
    region_edge_ids: List[str],
    claims: Dict[str, Claim],
    edges: Dict[str, Edge],
) -> List[Cluster]:
    """
    Cluster construction theo rule stance-based có overlap:

    - Mỗi contradiction/contextual_conflict edge tạo hai phía stance seed.
    - Mỗi cluster bắt đầu từ một claim ở phía contradiction seed và được mở rộng
      bằng support/entailment edges.
    - Không đưa hai claim có contradiction/contextual_conflict edge trực tiếp
      vào cùng một cluster.
    - Nếu claim X support/entail cả hai contradiction claims A và B,
      thì X được đưa vào cả hai clusters.
    - Các claim không nằm trong vùng mở rộng của conflict seed sẽ được gom bằng
      connected components positive-only, vẫn tránh contradiction nội bộ.
    """

    positive_adj: Dict[str, Set[str]] = defaultdict(set)
    conflict_adj: Dict[str, Set[str]] = defaultdict(set)
    conflict_seed_claims: Set[str] = set()

    # 1) Tách positive edges và conflict edges trong phạm vi region.
    for edge_id in region_edge_ids:
        if edge_id not in edges:
            continue

        e = edges[edge_id]
        a = e.source_claim_id
        b = e.target_claim_id

        if a not in region_claim_ids or b not in region_claim_ids:
            continue

        if edge_is_positive(e):
            positive_adj[a].add(b)
            positive_adj[b].add(a)

        elif edge_is_conflict(e):
            conflict_adj[a].add(b)
            conflict_adj[b].add(a)
            conflict_seed_claims.add(a)
            conflict_seed_claims.add(b)

    def has_conflict_with_cluster(candidate: str, cluster_claims: Set[str]) -> bool:
        """
        True nếu candidate có contradiction/contextual_conflict trực tiếp
        với bất kỳ claim nào đã nằm trong cluster.
        """
        candidate_conflicts = conflict_adj.get(candidate, set())
        return bool(candidate_conflicts & cluster_claims)

    def expand_positive_cluster(seed: str) -> Set[str]:
        """
        BFS qua support/entailment từ một seed.
        Không thêm candidate nếu candidate contradiction trực tiếp với claim đã có trong cluster.
        Không dùng assigned global để cho phép một claim xuất hiện ở nhiều clusters
        khi nó support/entail nhiều phía contradiction khác nhau.
        """
        comp: Set[str] = {seed}
        visited: Set[str] = {seed}
        q = deque([seed])

        while q:
            cur = q.popleft()

            for nb in sorted(positive_adj.get(cur, set())):
                if nb in visited:
                    continue

                visited.add(nb)

                if has_conflict_with_cluster(nb, comp):
                    continue

                comp.add(nb)
                q.append(nb)

        return comp

    def component_has_internal_conflict(comp: Set[str]) -> bool:
        for cid in comp:
            if conflict_adj.get(cid, set()) & comp:
                return True
        return False

    def add_component_unique(components: List[Set[str]], comp: Set[str]) -> None:
        if not comp:
            return
        if comp not in components:
            components.append(comp)

    components: List[Set[str]] = []

    # 2) Ưu tiên tạo cluster từ từng claim nằm trong contradiction/contextual_conflict seed.
    #    Với A contradiction B, ta có cluster phía A và cluster phía B.
    #    Nếu X positive với cả A và B, X sẽ được đưa vào cả hai clusters.
    for seed in sorted(conflict_seed_claims):
        add_component_unique(components, expand_positive_cluster(seed))

    # 3) Safety net cho claim chưa thuộc cluster nào:
    #    gom bằng positive connected components nhưng vẫn tránh contradiction nội bộ.
    already_covered: Set[str] = set()
    for comp in components:
        already_covered.update(comp)

    remaining_claims = set(region_claim_ids) - already_covered

    for seed in sorted(remaining_claims):
        if seed not in remaining_claims:
            continue

        comp = expand_positive_cluster(seed)
        comp = comp & remaining_claims

        # Nếu positive component còn chứa conflict nội bộ do cấu trúc phức tạp,
        # tách lại bằng BFS có check conflict từng seed.
        if component_has_internal_conflict(comp):
            for inner_seed in sorted(comp):
                inner_comp = expand_positive_cluster(inner_seed) & remaining_claims
                if not component_has_internal_conflict(inner_comp):
                    add_component_unique(components, inner_comp)
        else:
            add_component_unique(components, comp)

        remaining_claims -= comp

    # 4) Nếu region không có conflict seed vì dữ liệu đã bị filter lạ,
    #    fallback về singleton/positive-safe components.
    if not components:
        for claim_id in sorted(region_claim_ids):
            add_component_unique(components, {claim_id})

    clusters: List[Cluster] = []

    for idx, comp in enumerate(components, start=1):
        cluster_id = f"{conflict_region_id}_cl_{idx:03d}"
        claim_ids = sorted(comp)

        features = compute_cluster_features(
            claim_ids=claim_ids,
            region_edge_ids=region_edge_ids,
            claims=claims,
            edges=edges,
        )

        summary = build_simple_cluster_summary(claim_ids, claims)

        clusters.append(
            Cluster(
                cluster_id=cluster_id,
                conflict_region_id=conflict_region_id,
                claim_ids=claim_ids,
                cluster_stance_summary=summary,
                cluster_features=features,
            )
        )

    return clusters

def build_simple_cluster_summary(
    claim_ids: List[str],
    claims: Dict[str, Claim],
    max_claims: int = 2,
) -> Optional[str]:
    texts = []

    for cid in claim_ids[:max_claims]:
        if cid in claims:
            texts.append(claims[cid].claim_text)

    if not texts:
        return None

    if len(texts) == 1:
        return texts[0]

    return " / ".join(texts)


def compute_cluster_features(
    claim_ids: List[str],
    region_edge_ids: List[str],
    claims: Dict[str, Claim],
    edges: Dict[str, Edge],
) -> ClusterFeatures:
    claim_set = set(claim_ids)
    n = len(claim_ids)

    confidences = [
        get_claim_confidence(claims[cid])
        for cid in claim_ids
        if cid in claims
    ]

    coverages = [
        get_claim_coverage(claims[cid])
        for cid in claim_ids
        if cid in claims
    ]

    contexts = [
        get_context_completeness(claims[cid])
        for cid in claim_ids
        if cid in claims
    ]

    internal_edges = [
        edges[eid]
        for eid in region_edge_ids
        if eid in edges
        and edges[eid].source_claim_id in claim_set
        and edges[eid].target_claim_id in claim_set
    ]

    possible_edges = n * (n - 1) / 2 if n >= 2 else 1

    support_edges = [
        e for e in internal_edges
        if e.relation_type in {RelationType.support, RelationType.entailment}
    ]

    contradiction_edges = [
        e for e in internal_edges
        if edge_is_conflict(e)
    ]

    support_density = len(support_edges) / possible_edges
    contradiction_density = len(contradiction_edges) / possible_edges

    internal_consistency = clamp01(1.0 - contradiction_density)

    return ClusterFeatures(
        num_claims=n,
        avg_claim_confidence=clamp01(safe_mean(confidences, 0.0)),
        cluster_evidence_coverage=clamp01(safe_mean(coverages, 0.0)),
        support_density=clamp01(support_density),
        internal_consistency=clamp01(internal_consistency),
        avg_context_completeness=clamp01(safe_mean(contexts, 0.0)),
    )


# =============================================================================
# Module 8: Feature Aggregation
# =============================================================================

REGION_FEATURE_NAMES = [
    "num_claims_norm",
    "num_edges_norm",
    "num_claim_clusters_norm",
    "num_contradiction_edges_norm",
    "contradiction_density",
    "support_density",
    "max_cluster_confidence",
    "min_cluster_confidence",
    "cluster_confidence_gap",
    "max_cluster_coverage",
    "min_cluster_coverage",
    "cluster_coverage_gap",
    "coverage_balance_score",
    "avg_context_completeness",
    "missing_time_ratio",
    "missing_condition_ratio",
    "missing_scope_ratio",
    "dominant_entity_mismatch",
    "dominant_attribute_mismatch",
    "dominant_number_mismatch",
    "dominant_temporal_mismatch",
    "dominant_negation_mismatch",
]


def compute_region_features(
    region_claim_ids: List[str],
    region_edge_ids: List[str],
    clusters: List[Cluster],
    claims: Dict[str, Claim],
    edges: Dict[str, Edge],
) -> RegionFeatures:
    claim_set = set(region_claim_ids)

    region_edges = [
        edges[eid]
        for eid in region_edge_ids
        if eid in edges
    ]

    num_claims = len(claim_set)
    num_edges = len(region_edges)
    num_clusters = len(clusters)

    possible_edges = num_claims * (num_claims - 1) / 2 if num_claims >= 2 else 1

    contradiction_edges = [
        e for e in region_edges
        if edge_is_conflict(e)
    ]

    support_edges = [
        e for e in region_edges
        if edge_is_positive(e)
    ]

    contradiction_density = clamp01(len(contradiction_edges) / possible_edges)
    support_density = clamp01(len(support_edges) / possible_edges)

    cluster_confidences = [
        c.cluster_features.avg_claim_confidence
        for c in clusters
    ]

    cluster_coverages = [
        c.cluster_features.cluster_evidence_coverage
        for c in clusters
    ]

    max_conf = safe_max(cluster_confidences, None)
    min_conf = safe_min(cluster_confidences, None)

    max_cov = safe_max(cluster_coverages, None)
    min_cov = safe_min(cluster_coverages, None)

    conf_gap = 0.0
    if max_conf is not None and min_conf is not None:
        conf_gap = clamp01(max_conf - min_conf)

    cov_gap = 0.0
    if max_cov is not None and min_cov is not None:
        cov_gap = clamp01(max_cov - min_cov)

    coverage_balance_score = clamp01(1.0 - cov_gap)

    context_values = [
        get_context_completeness(claims[cid])
        for cid in claim_set
        if cid in claims
    ]

    avg_context = clamp01(safe_mean(context_values, 0.0))

    missing_time_ratio = compute_missing_time_ratio(claim_set, claims)
    missing_condition_ratio = compute_missing_condition_ratio(claim_set, claims)
    missing_scope_ratio = compute_missing_scope_ratio(claim_set, claims)

    dominant_context_mismatch = infer_dominant_context_mismatch(region_edges)

    return RegionFeatures(
        num_claims=num_claims,
        num_edges=num_edges,
        num_claim_clusters=num_clusters,
        num_contradiction_edges=len(contradiction_edges),
        contradiction_density=contradiction_density,
        support_density=support_density,
        max_cluster_confidence=max_conf,
        min_cluster_confidence=min_conf,
        cluster_confidence_gap=conf_gap,
        max_cluster_coverage=max_cov,
        min_cluster_coverage=min_cov,
        cluster_coverage_gap=cov_gap,
        coverage_balance_score=coverage_balance_score,
        avg_context_completeness=avg_context,
        missing_time_ratio=missing_time_ratio,
        missing_condition_ratio=missing_condition_ratio,
        missing_scope_ratio=missing_scope_ratio,
        dominant_context_mismatch_type=dominant_context_mismatch,
    )


def compute_missing_time_ratio(claim_ids: Set[str], claims: Dict[str, Claim]) -> float:
    if not claim_ids:
        return 0.0

    missing = 0
    total = 0

    for cid in claim_ids:
        if cid not in claims:
            continue
        total += 1
        if not has_time(claims[cid]):
            missing += 1

    if total == 0:
        return 0.0

    return clamp01(missing / total)


def compute_missing_condition_ratio(claim_ids: Set[str], claims: Dict[str, Claim]) -> float:
    if not claim_ids:
        return 0.0

    missing = 0
    total = 0

    for cid in claim_ids:
        if cid not in claims:
            continue
        total += 1
        has_condition, _ = has_condition_or_scope(claims[cid])
        if not has_condition:
            missing += 1

    if total == 0:
        return 0.0

    return clamp01(missing / total)


def compute_missing_scope_ratio(claim_ids: Set[str], claims: Dict[str, Claim]) -> float:
    if not claim_ids:
        return 0.0

    missing = 0
    total = 0

    for cid in claim_ids:
        if cid not in claims:
            continue
        total += 1
        _, has_scope_value = has_condition_or_scope(claims[cid])
        if not has_scope_value:
            missing += 1

    if total == 0:
        return 0.0

    return clamp01(missing / total)


def infer_dominant_context_mismatch(region_edges: List[Edge]) -> List[ContextMismatchType]:
    counts: Dict[ContextMismatchType, int] = defaultdict(int)

    for edge in region_edges:
        for t in get_context_mismatch_types_from_edge(edge):
            counts[t] += 1

    if not counts:
        return []

    max_count = max(counts.values())

    return [
        t for t, c in sorted(counts.items(), key=lambda x: x[0].value)
        if c == max_count and c > 0
    ]


def region_features_to_vector(f: RegionFeatures) -> List[float]:
    """
    Feature Aggregation vector for Option 1.
    Đây là region_embedding dạng feature vector đã normalize.
    """

    num_claims_norm = min(f.num_claims / 20.0, 1.0)
    num_edges_norm = min((f.num_edges or 0) / 50.0, 1.0)
    num_clusters_norm = min(f.num_claim_clusters / 10.0, 1.0)
    num_contra_norm = min((f.num_contradiction_edges or 0) / 20.0, 1.0)

    dominant_set = set(f.dominant_context_mismatch_type)

    vec = [
        num_claims_norm,
        num_edges_norm,
        num_clusters_norm,
        num_contra_norm,
        f.contradiction_density,
        f.support_density,
        f.max_cluster_confidence or 0.0,
        f.min_cluster_confidence or 0.0,
        f.cluster_confidence_gap,
        f.max_cluster_coverage or 0.0,
        f.min_cluster_coverage or 0.0,
        f.cluster_coverage_gap,
        f.coverage_balance_score,
        f.avg_context_completeness,
        f.missing_time_ratio or 0.0,
        f.missing_condition_ratio or 0.0,
        f.missing_scope_ratio or 0.0,
        1.0 if ContextMismatchType.entity in dominant_set else 0.0,
        1.0 if ContextMismatchType.attribute in dominant_set else 0.0,
        1.0 if ContextMismatchType.number in dominant_set else 0.0,
        1.0 if ContextMismatchType.temporal in dominant_set else 0.0,
        1.0 if ContextMismatchType.negation in dominant_set else 0.0,
    ]

    return [clamp01(x) for x in vec]


# =============================================================================
# Option 1: Feature Aggregation + MLP
# =============================================================================

class MLPConflictStatePredictor:
    """
    Wrapper cho model MLP đã train.

    Kỳ vọng model được lưu bằng joblib và có method:
        predict_proba(X)
        classes_

    Nếu không truyền model_path, pipeline sẽ dùng heuristic fallback.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model = None

        if model_path:
            try:
                import joblib
            except ImportError as e:
                raise ImportError(
                    "Bạn cần cài joblib để load MLP model: pip install joblib"
                ) from e

            self.model = joblib.load(model_path)

    def predict(self, vector: List[float]) -> Optional[ConflictState]:
        if self.model is None:
            return None

        y = self.model.predict([vector])[0]
        return ConflictState(str(y))

    def predict_attention_like_weights(
        self,
        claim_ids: List[str],
        region_edge_ids: List[str],
        claims: Dict[str, Claim],
        edges: Dict[str, Edge],
        top_k: int = 5,
    ) -> AttentionDiagnostics:
        """
        Option 1 không có attention thật.
        Ta tạo diagnostics dạng heuristic:
        - claim weight dựa trên degree + confidence
        - edge weight dựa trên relation_confidence + conflict_intensity
        """

        claim_scores: Dict[str, float] = defaultdict(float)

        for eid in region_edge_ids:
            e = edges[eid]
            score = e.relation_confidence
            if edge_is_conflict(e):
                score += 0.5
            claim_scores[e.source_claim_id] += score
            claim_scores[e.target_claim_id] += score

        for cid in claim_ids:
            if cid in claims:
                claim_scores[cid] += get_claim_confidence(claims[cid])

        claim_entries = normalize_ranked_claim_scores(claim_scores, top_k=top_k)

        edge_scores: Dict[str, float] = {}

        for eid in region_edge_ids:
            e = edges[eid]
            conflict_intensity = 0.0

            if e.claim_pair_features is not None:
                conflict_intensity = e.claim_pair_features.aggregate.conflict_intensity_score

            edge_scores[eid] = e.relation_confidence + conflict_intensity

        edge_entries = normalize_ranked_edge_scores(edge_scores, edges, top_k=top_k)

        return AttentionDiagnostics(
            top_claims=claim_entries,
            top_edges=edge_entries,
        )


def normalize_ranked_claim_scores(
    scores: Dict[str, float],
    top_k: int = 5,
) -> List[AttentionClaimEntry]:
    if not scores:
        return []

    items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    total = sum(v for _, v in items)

    if total <= 0:
        weight = 1.0 / len(items)
        return [
            AttentionClaimEntry(claim_id=cid, attention_weight=clamp01(weight))
            for cid, _ in items
        ]

    return [
        AttentionClaimEntry(claim_id=cid, attention_weight=clamp01(v / total))
        for cid, v in items
    ]


def normalize_ranked_edge_scores(
    scores: Dict[str, float],
    edges: Dict[str, Edge],
    top_k: int = 5,
) -> List[AttentionEdgeEntry]:
    if not scores:
        return []

    items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    total = sum(v for _, v in items)

    output: List[AttentionEdgeEntry] = []

    for eid, v in items:
        if eid not in edges:
            continue

        weight = v / total if total > 0 else 1.0 / len(items)

        output.append(
            AttentionEdgeEntry(
                edge_id=eid,
                edge_type=edges[eid].relation_type,
                attention_weight=clamp01(weight),
            )
        )

    return output


def heuristic_conflict_state(region_features: RegionFeatures) -> ConflictState:
    """
    Fallback khi chưa có trained MLP.

    Logic:
    - Contextual nếu context thiếu nhiều hoặc mismatch temporal/entity nổi bật.
    - Resolvable nếu cluster confidence/coverage gap lớn.
    - Underdetermined nếu hai phía cân bằng coverage và confidence gap nhỏ.
    """

    mismatch_set = set(region_features.dominant_context_mismatch_type)

    context_heavy = (
        region_features.avg_context_completeness < 0.55
        or (region_features.missing_time_ratio or 0.0) > 0.50
        or (region_features.missing_condition_ratio or 0.0) > 0.50
        or ContextMismatchType.temporal in mismatch_set
        or ContextMismatchType.entity in mismatch_set
    )

    if context_heavy:
        return ConflictState.contextual

    if (
        region_features.cluster_confidence_gap >= 0.25
        or region_features.cluster_coverage_gap >= 0.25
    ):
        return ConflictState.resolvable

    if (
        region_features.coverage_balance_score >= 0.75
        and region_features.cluster_confidence_gap < 0.20
        and region_features.contradiction_density > 0.0
    ):
        return ConflictState.underdetermined

    if region_features.contradiction_density > 0.0:
        return ConflictState.underdetermined

    return ConflictState.no_conflict


def build_cluster_embedding(cluster: Cluster) -> EmbeddingVector:
    f = cluster.cluster_features

    vec = [
        min(f.num_claims / 20.0, 1.0),
        f.avg_claim_confidence,
        f.cluster_evidence_coverage,
        f.support_density,
        f.internal_consistency,
        f.avg_context_completeness,
    ]

    return normalize_vector(vec)


def build_node_embeddings(
    region_claim_ids: List[str],
    claims: Dict[str, Claim],
    region_edge_ids: List[str],
    edges: Dict[str, Edge],
) -> Dict[str, EmbeddingVector]:
    """
    Option 1 không học node embedding.
    Ta tạo node vector từ claim_features + degree stats.
    """

    degree = defaultdict(int)
    contradiction_degree = defaultdict(int)
    support_degree = defaultdict(int)

    for eid in region_edge_ids:
        e = edges[eid]
        a = e.source_claim_id
        b = e.target_claim_id

        degree[a] += 1
        degree[b] += 1

        if edge_is_conflict(e):
            contradiction_degree[a] += 1
            contradiction_degree[b] += 1

        if edge_is_positive(e):
            support_degree[a] += 1
            support_degree[b] += 1

    output: Dict[str, EmbeddingVector] = {}

    for cid in region_claim_ids:
        if cid not in claims:
            continue

        c = claims[cid]

        vec = [
            c.claim_features.retrieval_relevance,
            c.claim_features.claim_confidence,
            c.claim_features.claim_evidence_coverage,
            c.claim_features.context_completeness,
            min(degree[cid] / 10.0, 1.0),
            min(contradiction_degree[cid] / 5.0, 1.0),
            min(support_degree[cid] / 5.0, 1.0),
        ]

        output[cid] = normalize_vector([float(x) for x in vec])

    return output


def encode_conflict_region(
    conflict_region_id: str,
    region_claim_ids: List[str],
    region_edge_ids: List[str],
    clusters: List[Cluster],
    region_features: RegionFeatures,
    claims: Dict[str, Claim],
    edges: Dict[str, Edge],
    predictor: MLPConflictStatePredictor,
) -> ConflictRegionEncoding:
    raw_region_vector = region_features_to_vector(region_features)
    region_embedding = normalize_vector(raw_region_vector)

    node_embeddings = build_node_embeddings(
        region_claim_ids=region_claim_ids,
        claims=claims,
        region_edge_ids=region_edge_ids,
        edges=edges,
    )

    cluster_embeddings = {
        cluster.cluster_id: build_cluster_embedding(cluster)
        for cluster in clusters
    }

    diagnostics = predictor.predict_attention_like_weights(
        claim_ids=region_claim_ids,
        region_edge_ids=region_edge_ids,
        claims=claims,
        edges=edges,
    )

    return ConflictRegionEncoding(
        conflict_region_id=conflict_region_id,
        encoding_method="feature_aggregation_mlp_option_1",
        region_embedding=region_embedding,
        node_embeddings=node_embeddings,
        cluster_embeddings=cluster_embeddings,
        attention_diagnostics=diagnostics,
        aggregated_region_features=region_features,
    )


# =============================================================================
# Full pipeline
# =============================================================================

def run_pipeline(
    claims: Dict[str, Claim],
    edges: Dict[str, Edge],
    expansion_hops: int = 1,
    min_conflict_confidence: float = 0.50,
    mlp_model_path: Optional[str] = None,
    use_heuristic_if_no_mlp: bool = True,
    require_distinct_docs_per_region: bool = False,
) -> List[ConflictRegionRecord]:
    graph = build_claim_graph(claims=claims, edges=edges)

    detections = detect_conflict_regions(
        claims=claims,
        edges=edges,
        graph=graph,
        expansion_hops=expansion_hops,
        min_conflict_confidence=min_conflict_confidence,
        require_distinct_docs_per_region=require_distinct_docs_per_region,
    )

    predictor = MLPConflictStatePredictor(model_path=mlp_model_path)

    records: List[ConflictRegionRecord] = []

    for idx, det in enumerate(detections, start=1):
        conflict_region_id = f"r_{idx:04d}"
        region_claim_ids = sorted(set(det.__dict__.get("_claim_ids", [])))
        region_claim_set = set(region_claim_ids)

        region_edge_ids = collect_region_edges(
            region_claim_ids=region_claim_set,
            edges=edges,
        )

        clusters = build_clusters_for_region(
            conflict_region_id=conflict_region_id,
            region_claim_ids=region_claim_set,
            region_edge_ids=region_edge_ids,
            claims=claims,
            edges=edges,
        )

        region_features = compute_region_features(
            region_claim_ids=region_claim_ids,
            region_edge_ids=region_edge_ids,
            clusters=clusters,
            claims=claims,
            edges=edges,
        )

        encoding = encode_conflict_region(
            conflict_region_id=conflict_region_id,
            region_claim_ids=region_claim_ids,
            region_edge_ids=region_edge_ids,
            clusters=clusters,
            region_features=region_features,
            claims=claims,
            edges=edges,
            predictor=predictor,
        )

        predicted_state = predictor.predict(encoding.region_embedding)

        if predicted_state is None and use_heuristic_if_no_mlp:
            predicted_state = heuristic_conflict_state(region_features)

        region = ConflictRegion(
            conflict_region_id=conflict_region_id,
            claim_ids=region_claim_ids,
            detection=RegionDetection(
                core_contradiction_edges=det.core_contradiction_edges,
                region_seed_claims=det.region_seed_claims,
                region_detection_method=det.region_detection_method,
            ),
            region_features=region_features,
            region_embedding=encoding.region_embedding,
            encoding=encoding,
            predicted_state=predicted_state,
        )

        records.append(
            ConflictRegionRecord(
                conflict_region=region,
                clusters=clusters,
                edge_ids=region_edge_ids,
            )
        )

    return records


# =============================================================================
# Optional: export region feature matrix for MLP training
# =============================================================================

def export_region_feature_matrix(
    records: List[ConflictRegionRecord],
    output_path: str | Path,
) -> None:
    rows: List[Dict[str, Any]] = []

    for rec in records:
        region = rec.conflict_region
        vec = region.region_embedding or []

        row = {
            "conflict_region_id": region.conflict_region_id,
            "predicted_state": region.predicted_state.value if region.predicted_state else None,
        }

        for name, value in zip(REGION_FEATURE_NAMES, vec):
            row[name] = value

        rows.append(row)

    write_jsonl(rows, output_path)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--claims",
        type=str,
        required=True,
        help="Path tới factoid claims JSONL, ví dụ factoid_claims_revised.jsonl",
    )

    parser.add_argument(
        "--edges",
        type=str,
        required=True,
        help="Path tới claim graph edges JSONL từ edge_feature_builder.py",
    )

    parser.add_argument(
        "--output_regions",
        type=str,
        default="conflict_regions.jsonl",
    )

    parser.add_argument(
        "--output_region_features",
        type=str,
        default="region_feature_matrix.jsonl",
    )

    parser.add_argument(
        "--expansion_hops",
        type=int,
        default=1,
        help="Số hop mở rộng từ contradiction seed qua support/entailment edges",
    )

    parser.add_argument(
        "--min_conflict_confidence",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--require_distinct_docs_per_region",
        action="store_true",
        help=(
            "Nếu bật, chỉ xây/giữ conflict region mà tất cả claims trong region "
            "đến từ các document khác nhau. Ưu tiên doc_id, fallback source_id."
        ),
    )

    parser.add_argument(
        "--mlp_model_path",
        type=str,
        default=None,
        help="Optional path tới sklearn MLP model lưu bằng joblib",
    )

    parser.add_argument(
        "--disable_heuristic_state",
        action="store_true",
        help="Nếu bật, không predict state bằng heuristic khi chưa có MLP.",
    )

    args = parser.parse_args()

    claims = read_claims(args.claims)
    print(f"Loaded claims: {len(claims)}")

    edges = read_edges(args.edges)
    print(f"Loaded graph edges: {len(edges)}")

    records = run_pipeline(
        claims=claims,
        edges=edges,
        expansion_hops=args.expansion_hops,
        min_conflict_confidence=args.min_conflict_confidence,
        mlp_model_path=args.mlp_model_path,
        use_heuristic_if_no_mlp=not args.disable_heuristic_state,
        require_distinct_docs_per_region=args.require_distinct_docs_per_region,
    )

    write_jsonl(records, args.output_regions)
    export_region_feature_matrix(records, args.output_region_features)

    print(f"Conflict regions: {len(records)}")
    print(f"Saved regions to: {args.output_regions}")
    print(f"Saved region feature matrix to: {args.output_region_features}")


if __name__ == "__main__":
    main()