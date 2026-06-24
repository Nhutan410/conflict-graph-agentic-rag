# nli_inferencer.py

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional, List, Dict, Literal, Tuple, Iterable, Any
from tqdm.auto import tqdm
from collections import defaultdict
import time

from sentence_transformers import SentenceTransformer
import numpy as np

import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field, ConfigDict
from typing_extensions import Annotated
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# =============================================================================
# Common types
# =============================================================================

UnitInterval = Annotated[float, Field(ge=0.0, le=1.0)]
BinaryFlag = Annotated[int, Field(ge=0, le=1)]


# =============================================================================
# Factoid schema
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
    polarity: BinaryFlag  # 0 = khẳng định, 1 = phủ định


class FactoidFeatures(BaseModel):
    number: List[FactoidNumberValue] = Field(default_factory=list)
    entity: List[FactoidEntityMention] = Field(default_factory=list)
    temporal: Optional[FactoidTemporal] = None
    negation: Optional[FactoidNegation] = None
    verb: Optional[FactoidVerb] = None


# =============================================================================
# Claim schema
# =============================================================================

class ClaimFeatures(BaseModel):
    retrieval_relevance: UnitInterval
    claim_confidence: UnitInterval
    claim_evidence_coverage: UnitInterval
    context_completeness: UnitInterval


class Claim(BaseModel):
    """
    Claim schema tương thích với JSONL:

    {
      "claim_id": "...",
      "claim_text": "...",
      "canonical_claim_id": "...",
      "duplicate_of": null,
      "factoid_features": {...},
      "claim_features": {...},
      "evidence": "...",
      "doc_id": "...",
      "source_id": "..."
    }
    """

    model_config = ConfigDict(extra="allow")

    claim_id: str
    claim_text: str
    canonical_claim_id: Optional[str] = None
    duplicate_of: Optional[str] = None
    factoid_features: Optional[FactoidFeatures] = None
    claim_features: ClaimFeatures

    # optional metadata từ claims.jsonl
    evidence: Optional[str] = None
    doc_id: Optional[str] = None
    source_id: Optional[str] = None


class QueryRecord(BaseModel):
    """
    Query schema tương thích với queries_revised.jsonl:

    {
      "query_id": "q_000",
      "user_query": "What is the population of Broken Bow?"
    }
    """

    query_id: str
    user_query: str


# =============================================================================
# Output schema
# =============================================================================

class NLIEdgeFeatures(BaseModel):
    entailment_prob: UnitInterval
    neutral_prob: UnitInterval
    support_prob: UnitInterval
    contradiction_prob: UnitInterval


class AlignmentFeatures(BaseModel):
    semantic_similarity: UnitInterval

    entity_overlap: UnitInterval
    subject_entity_overlap: UnitInterval
    verb_match: UnitInterval
    event_alignment: UnitInterval
    temporal_overlap: UnitInterval
    number_compatibility: UnitInterval
    negation_mismatch: bool

    same_factual_target: bool
    force_neutral: bool
    alignment_reason: List[str] = Field(default_factory=list)


class ClaimPair(BaseModel):
    claim_pair_id: str
    claim_a: Claim
    claim_b: Claim


class NLIEdge(BaseModel):
    edge_id: str
    source_claim_id: str
    target_claim_id: str

    source_doc_id: Optional[str] = None
    target_doc_id: Optional[str] = None
    source_id: Optional[str] = None
    target_source_id: Optional[str] = None

    relation_type: Literal[
        "entailment",
        "support",
        "neutral",
        "contradiction",
        # "contextual_conflict",
    ]

    relation_confidence: UnitInterval

    nli_edge_features: NLIEdgeFeatures
    alignment_features: AlignmentFeatures

    nli_skipped: bool = False
    skip_reason: Optional[str] = None

    entailment_a_to_b: UnitInterval
    entailment_b_to_a: UnitInterval
    neutral_a_to_b: UnitInterval
    neutral_b_to_a: UnitInterval
    contradiction_a_to_b: UnitInterval
    contradiction_b_to_a: UnitInterval


# =============================================================================
# IO helpers
# =============================================================================

def read_claims_jsonl(path: str | Path) -> List[Claim]:
    path = Path(path)
    claims: List[Claim] = []

    # Đếm số dòng để tqdm có total
    with path.open("r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in tqdm(
            enumerate(f, start=1),
            total=total_lines,
            desc="Reading claims",
            unit="line",
        ):
            line = line.strip()

            if not line:
                continue

            try:
                obj = json.loads(line)
                claim = Claim.model_validate(obj)
                claims.append(claim)

            except Exception as e:
                raise ValueError(
                    f"Invalid JSONL at line {line_no}: {e}\n"
                    f"Line content: {line[:500]}"
                )

    return claims


def read_queries_jsonl(path: str | Path) -> Dict[str, QueryRecord]:
    """
    Đọc queries_revised.jsonl và trả về dict:
        normalized_query_id -> QueryRecord

    Ví dụ:
        q_000 -> q000
        q000  -> q000
    """

    path = Path(path)
    queries: Dict[str, QueryRecord] = {}

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in tqdm(
            enumerate(f, start=1),
            desc="Reading queries",
            unit="line",
        ):
            line = line.strip()

            if not line:
                continue

            try:
                obj = json.loads(line)
                query = QueryRecord.model_validate(obj)

                normalized_qid = normalize_query_id(query.query_id)
                if normalized_qid is None:
                    continue

                queries[normalized_qid] = query

            except Exception as e:
                raise ValueError(
                    f"Invalid queries JSONL at line {line_no}: {e}\n"
                    f"Line content: {line[:500]}"
                )

    return queries


def write_jsonl(records: Iterable[BaseModel | Dict[str, Any]], path: str | Path) -> None:
    path = Path(path)

    records = list(records)

    with path.open("w", encoding="utf-8") as f:
        for record in tqdm(records, desc="Writing output", unit="edge"):
            if isinstance(record, BaseModel):
                obj = record.model_dump()
            else:
                obj = record

            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def filter_claims_by_query_relevance(
    claims: List[Claim],
    queries_by_id: Dict[str, QueryRecord],
    semantic_scorer: SemanticSimilarityScorer,
    min_query_relevance: float = 0.60,
    top_k_claims_per_query: Optional[int] = None,
    keep_claims_without_query: bool = False,
) -> List[Claim]:
    """
    Lọc claims theo mức độ relevance với user_query.

    Flow:
        claim_id -> query_id
        query_id -> user_query
        similarity(claim_text, user_query)
        giữ claim nếu similarity >= min_query_relevance

    Nếu top_k_claims_per_query != None:
        Sau khi lọc threshold, chỉ giữ top-k claim liên quan nhất trong từng query.
    """

    grouped: Dict[str, List[Tuple[Claim, float]]] = defaultdict(list)
    kept_without_query: List[Claim] = []
    missing_query_count = 0

    for claim in tqdm(
        claims,
        desc="Filtering claims by query relevance",
        unit="claim",
    ):
        qid = extract_query_id_from_claim_id(claim.claim_id)

        if qid is None or qid not in queries_by_id:
            missing_query_count += 1
            if keep_claims_without_query:
                kept_without_query.append(claim)
            continue

        query = queries_by_id[qid]

        relevance = semantic_scorer.score_claim_to_query(
            claim=claim,
            query=query,
        )

        # Claim có model_config extra='allow' nên có thể gắn field này để debug.
        claim.query_relevance_score = relevance

        if relevance >= min_query_relevance:
            grouped[qid].append((claim, relevance))

    filtered_claims: List[Claim] = []

    for qid, rows in grouped.items():
        rows = sorted(rows, key=lambda x: x[1], reverse=True)

        if top_k_claims_per_query is not None:
            rows = rows[:top_k_claims_per_query]

        filtered_claims.extend([claim for claim, _ in rows])

    filtered_claims.extend(kept_without_query)

    print(f"Claims before query relevance filtering: {len(claims)}")
    print(f"Claims after query relevance filtering: {len(filtered_claims)}")
    print(f"Claims without matched query: {missing_query_count}")
    print(f"Min query relevance: {min_query_relevance}")

    if top_k_claims_per_query is not None:
        print(f"Top-k claims per query: {top_k_claims_per_query}")

    if keep_claims_without_query:
        print(f"Kept claims without query: {len(kept_without_query)}")

    return filtered_claims


# =============================================================================
# Pair construction
# =============================================================================

def normalize_query_id(query_id: Optional[str]) -> Optional[str]:
    """
    Chuẩn hóa query_id về dạng q000.

    Ví dụ:
        q_000 -> q000
        q000  -> q000
        q_13  -> q013
        q13   -> q013
    """

    if query_id is None:
        return None

    text = str(query_id).lower().strip()

    m = re.search(r"q_?(\d+)", text)
    if not m:
        return text

    return f"q{int(m.group(1)):03d}"


def extract_query_id_from_claim_id(claim_id: str) -> Optional[str]:
    """
    Ví dụ:
        c_q000_d00_s00 -> q000
        c_q002_d05_s10 -> q002
    """
    m = re.search(r"c_(q\d+)_", claim_id)
    if m:
        return normalize_query_id(m.group(1))
    return None



def build_claim_pairs(
    claims: List[Claim],
    same_query_only: bool = True,
    skip_duplicates: bool = True,
    skip_same_doc: bool = False,
    max_pairs: Optional[int] = None,
) -> List[ClaimPair]:

    if skip_duplicates:
        claims = [c for c in claims if c.duplicate_of is None]

    pairs: List[ClaimPair] = []

    if same_query_only:
        groups = defaultdict(list)

        for claim in claims:
            qid = extract_query_id_from_claim_id(claim.claim_id)
            groups[qid].append(claim)

        group_items = list(groups.items())

        for qid, group_claims in tqdm(
            group_items,
            desc="Building claim pairs by query",
            unit="query",
        ):
            for i in range(len(group_claims)):
                for j in range(i + 1, len(group_claims)):
                    a = group_claims[i]
                    b = group_claims[j]

                    if skip_same_doc and a.doc_id is not None and a.doc_id == b.doc_id:
                        continue

                    pair_id = f"{a.claim_id}__{b.claim_id}"

                    pairs.append(
                        ClaimPair(
                            claim_pair_id=pair_id,
                            claim_a=a,
                            claim_b=b,
                        )
                    )

                    if max_pairs is not None and len(pairs) >= max_pairs:
                        return pairs

        return pairs

    # fallback nếu không same_query_only
    for i in tqdm(range(len(claims)), desc="Building claim pairs", unit="claim"):
        for j in range(i + 1, len(claims)):
            a = claims[i]
            b = claims[j]

            if skip_same_doc and a.doc_id is not None and a.doc_id == b.doc_id:
                continue

            pair_id = f"{a.claim_id}__{b.claim_id}"

            pairs.append(
                ClaimPair(
                    claim_pair_id=pair_id,
                    claim_a=a,
                    claim_b=b,
                )
            )

            if max_pairs is not None and len(pairs) >= max_pairs:
                return pairs

    return pairs


def build_claim_pairs_topk_semantic(
    claims: List[Claim],
    semantic_scorer: SemanticSimilarityScorer,
    same_query_only: bool = True,
    skip_duplicates: bool = True,
    skip_same_doc: bool = False,
    top_k: int = 10,
    min_similarity: float = 0.50,
    max_pairs: Optional[int] = None,
) -> List[ClaimPair]:

    if skip_duplicates:
        claims = [c for c in claims if c.duplicate_of is None]

    groups = defaultdict(list)

    if same_query_only:
        for claim in claims:
            qid = extract_query_id_from_claim_id(claim.claim_id) or "__unknown__"
            groups[qid].append(claim)
    else:
        groups["__all__"] = claims

    pair_map: Dict[str, ClaimPair] = {}

    for qid, group_claims in tqdm(
        groups.items(),
        desc="Top-k semantic candidate selection",
        unit="query",
    ):
        if len(group_claims) < 2:
            continue

        embeddings = []

        for claim in group_claims:
            emb = semantic_scorer.get_embedding(claim)
            embeddings.append(emb)

        embeddings = np.vstack(embeddings)

        # Vì embedding đã normalize, dot product = cosine similarity gốc [-1, 1].
        # Đưa về [0, 1] để dùng cùng scale với SemanticSimilarityScorer.score().
        raw_sim_matrix = embeddings @ embeddings.T
        sim_matrix = (raw_sim_matrix + 1.0) / 2.0

        n = len(group_claims)

        for i in range(n):
            # sort giảm dần similarity, bỏ chính nó
            neighbor_indices = np.argsort(-sim_matrix[i])

            selected = 0

            for j in neighbor_indices:
                if i == j:
                    continue

                sim = float(sim_matrix[i, j])

                if sim < min_similarity:
                    continue

                a = group_claims[i]
                b = group_claims[j]

                if skip_same_doc and a.doc_id is not None and a.doc_id == b.doc_id:
                    continue

                # Chuẩn hóa thứ tự để tránh duplicate A-B và B-A
                if a.claim_id < b.claim_id:
                    left, right = a, b
                else:
                    left, right = b, a

                pair_id = f"{left.claim_id}__{right.claim_id}"

                if pair_id not in pair_map:
                    pair_map[pair_id] = ClaimPair(
                        claim_pair_id=pair_id,
                        claim_a=left,
                        claim_b=right,
                    )

                selected += 1

                if max_pairs is not None and len(pair_map) >= max_pairs:
                    return list(pair_map.values())

                if selected >= top_k:
                    break

    return list(pair_map.values())

# =============================================================================
# Alignment helpers
# =============================================================================

def _norm_text(x: Optional[str]) -> str:
    if not x:
        return ""
    return " ".join(str(x).lower().strip().split())


def _safe_set(values: List[str]) -> set[str]:
    return {v for v in values if v}


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def string_similarity(a: Optional[str], b: Optional[str]) -> float:
    a = _norm_text(a)
    b = _norm_text(b)

    if not a or not b:
        return 0.0

    if a == b:
        return 1.0

    return clamp01(SequenceMatcher(None, a, b).ratio())


def get_entities(claim: Claim) -> List[str]:
    if not claim.factoid_features:
        return []

    return [
        _norm_text(e.canonical_entity)
        for e in claim.factoid_features.entity
        if e.canonical_entity
    ]


def get_subject_like_entities(claim: Claim) -> List[str]:
    """
    Heuristic:
    - Nếu attribute có subject/main/main_subject/person/organization thì lấy làm subject-like.
    - Nếu không có attribute rõ ràng, lấy entity đầu tiên làm subject-like.
    """
    if not claim.factoid_features or not claim.factoid_features.entity:
        return []

    subject_entities: List[str] = []

    subject_attrs = {
        "subject",
        "main",
        "main_subject",
        "person",
        "organization",
        "org",
        "location",
        "country",
        "team",
        "player",
    }

    for e in claim.factoid_features.entity:
        attr = _norm_text(e.attribute)

        if attr in subject_attrs:
            subject_entities.append(_norm_text(e.canonical_entity))

    if subject_entities:
        return subject_entities

    return [_norm_text(claim.factoid_features.entity[0].canonical_entity)]


def get_verb_lemma(claim: Claim) -> Optional[str]:
    if not claim.factoid_features or not claim.factoid_features.verb:
        return None
    return _norm_text(claim.factoid_features.verb.lemma)


def get_negation_polarity(claim: Claim) -> int:
    if not claim.factoid_features or not claim.factoid_features.negation:
        return 0
    return int(claim.factoid_features.negation.polarity)


def get_numbers(claim: Claim) -> List[FactoidNumberValue]:
    if not claim.factoid_features:
        return []
    return claim.factoid_features.number


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


def temporal_overlap_score(
    t1: Optional[FactoidTemporal],
    t2: Optional[FactoidTemporal],
) -> float:
    """
    Return:
    - 1.0 nếu thiếu time ở một trong hai phía: không dùng time để chặn relation
    - 1.0 nếu khoảng thời gian overlap
    - 0.0 nếu chắc chắn disjoint
    """
    if t1 is None or t2 is None:
        return 1.0

    s1 = parse_date_safe(t1.start)
    e1 = parse_date_safe(t1.end) or s1

    s2 = parse_date_safe(t2.start)
    e2 = parse_date_safe(t2.end) or s2

    if s1 is None or s2 is None:
        return 1.0

    if e1 is None:
        e1 = s1
    if e2 is None:
        e2 = s2

    latest_start = max(s1, s2)
    earliest_end = min(e1, e2)

    if latest_start <= earliest_end:
        return 1.0

    return 0.0


def number_compatibility_score(
    nums_a: List[FactoidNumberValue],
    nums_b: List[FactoidNumberValue],
    tolerance: float = 1e-6,
) -> float:
    """
    Nếu không có number ở một trong hai claim thì return 1.0,
    nghĩa là không dùng number để chặn relation.

    Return:
    - 1.0: compatible hoặc không đủ thông tin
    - 0.5: khác unit nhưng chưa convert được
    - 0.0: cùng unit nhưng value khác
    """
    if not nums_a or not nums_b:
        return 1.0

    best = 0.0

    for x in nums_a:
        for y in nums_b:
            unit_x = _norm_text(x.unit)
            unit_y = _norm_text(y.unit)

            if unit_x and unit_y and unit_x != unit_y:
                best = max(best, 0.5)
                continue

            if abs(float(x.value) - float(y.value)) <= tolerance:
                best = max(best, 1.0)
            else:
                best = max(best, 0.0)

    return clamp01(best)

class SemanticSimilarityScorer:
    """
    Tính semantic similarity bằng sentence-transformers.

    Dùng cho:
    - claim vs claim
    - claim vs query
    - raw text vs raw text

    Embedding được cache theo cache_key để không encode lại nhiều lần.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        self.cache: Dict[str, np.ndarray] = {}

    def get_text_embedding(
        self,
        text: str,
        cache_key: Optional[str] = None,
    ) -> np.ndarray:
        if cache_key is not None and cache_key in self.cache:
            return self.cache[cache_key]

        emb = self.model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        if cache_key is not None:
            self.cache[cache_key] = emb

        return emb

    def get_embedding(self, claim: Claim) -> np.ndarray:
        return self.get_text_embedding(
            text=claim.claim_text,
            cache_key=f"claim::{claim.claim_id}",
        )

    def score_texts(
        self,
        text_a: str,
        text_b: str,
        cache_key_a: Optional[str] = None,
        cache_key_b: Optional[str] = None,
    ) -> float:
        emb_a = self.get_text_embedding(text_a, cache_key=cache_key_a)
        emb_b = self.get_text_embedding(text_b, cache_key=cache_key_b)

        sim = float(np.dot(emb_a, emb_b))

        # cosine normalized embeddings có thể nằm trong [-1, 1]
        # đưa về [0, 1] để đồng nhất với UnitInterval
        sim_01 = (sim + 1.0) / 2.0

        return clamp01(sim_01)

    def score(self, claim_a: Claim, claim_b: Claim) -> float:
        return self.score_texts(
            text_a=claim_a.claim_text,
            text_b=claim_b.claim_text,
            cache_key_a=f"claim::{claim_a.claim_id}",
            cache_key_b=f"claim::{claim_b.claim_id}",
        )

    def score_claim_to_query(
        self,
        claim: Claim,
        query: QueryRecord,
    ) -> float:
        normalized_qid = normalize_query_id(query.query_id) or query.query_id

        return self.score_texts(
            text_a=claim.claim_text,
            text_b=query.user_query,
            cache_key_a=f"claim::{claim.claim_id}",
            cache_key_b=f"query::{normalized_qid}",
        )


def compute_alignment_features(
    claim_a: Claim,
    claim_b: Claim,
    semantic_similarity: float = 1.0,
    min_semantic_similarity: float = 0.60,
    min_entity_overlap: float = 0.25,
    min_subject_overlap: float = 0.25,
    min_event_alignment: float = 0.25,
) -> AlignmentFeatures:
    """
    Tính entity/event alignment từ factoid_features.

    Mục tiêu:
    - Nếu khác entity/event rõ ràng thì force_neutral=True.
    - Nếu cùng target nhưng khác time/number/negation thì giữ tín hiệu conflict.
    """

    reasons: List[str] = []

    entities_a = _safe_set(get_entities(claim_a))
    entities_b = _safe_set(get_entities(claim_b))

    subjects_a = _safe_set(get_subject_like_entities(claim_a))
    subjects_b = _safe_set(get_subject_like_entities(claim_b))

    entity_overlap = jaccard(entities_a, entities_b)
    subject_overlap = jaccard(subjects_a, subjects_b)

    verb_a = get_verb_lemma(claim_a)
    verb_b = get_verb_lemma(claim_b)
    verb_match = string_similarity(verb_a, verb_b)

    temporal_a = claim_a.factoid_features.temporal if claim_a.factoid_features else None
    temporal_b = claim_b.factoid_features.temporal if claim_b.factoid_features else None
    temporal_overlap = temporal_overlap_score(temporal_a, temporal_b)

    nums_a = get_numbers(claim_a)
    nums_b = get_numbers(claim_b)
    number_compatibility = number_compatibility_score(nums_a, nums_b)

    negation_a = get_negation_polarity(claim_a)
    negation_b = get_negation_polarity(claim_b)
    negation_mismatch = negation_a != negation_b

    # Nếu không có verb ở một phía thì không phạt quá mạnh.
    # Khi thiếu verb, dùng 0.5 làm unknown similarity.
    verb_signal = verb_match
    if not verb_a or not verb_b:
        verb_signal = 0.5

    # Event alignment tổng hợp.
    # Có thể tuning weight sau bằng validation set.
    event_alignment = clamp01(
        0.45 * max(entity_overlap, subject_overlap)
        + 0.35 * verb_signal
        + 0.20 * temporal_overlap
    )

    same_factual_target = True

    # Case 0: semantic similarity quá thấp.
    # Chỉ force_neutral khi semantic thấp VÀ không có overlap factoid nào.
    # Nếu vẫn có entity/subject overlap, giữ lại để NLI + alignment xử lý tiếp.
    low_semantic = semantic_similarity < min_semantic_similarity
    if low_semantic and entity_overlap == 0.0 and subject_overlap == 0.0:
        same_factual_target = False
        reasons.append("low_semantic_similarity")
    elif low_semantic:
        reasons.append("low_semantic_similarity_but_factoid_overlap")

    # Case 1: cả hai có entity nhưng gần như không overlap
    if entities_a and entities_b and entity_overlap < min_entity_overlap:
        same_factual_target = False
        reasons.append("low_entity_overlap")

    # Case 2: cả hai có subject chính nhưng khác nhau
    if subjects_a and subjects_b and subject_overlap < min_subject_overlap:
        same_factual_target = False
        reasons.append("low_subject_entity_overlap")

    # Case 3: event/predicate quá khác nhau
    if verb_a and verb_b and event_alignment < min_event_alignment:
        same_factual_target = False
        reasons.append("low_event_alignment")

    # Case 4: time chắc chắn disjoint
    if temporal_overlap == 0.0:
        reasons.append("temporal_disjoint")

    # Case 5: number khác nhau
    if number_compatibility == 0.0:
        reasons.append("number_mismatch")

    if negation_mismatch:
        reasons.append("negation_mismatch")

    force_neutral = not same_factual_target

    if not reasons:
        reasons.append("aligned_or_insufficient_factoid")

    return AlignmentFeatures(
        semantic_similarity=clamp01(semantic_similarity),
        entity_overlap=clamp01(entity_overlap),
        subject_entity_overlap=clamp01(subject_overlap),
        verb_match=clamp01(verb_match),
        event_alignment=clamp01(event_alignment),
        temporal_overlap=clamp01(temporal_overlap),
        number_compatibility=clamp01(number_compatibility),
        negation_mismatch=negation_mismatch,
        same_factual_target=same_factual_target,
        force_neutral=force_neutral,
        alignment_reason=reasons,
    )

def should_run_nli(
    alignment: AlignmentFeatures,
    semantic_similarity_threshold: float = 0.50,
    moderate_semantic_threshold: float = 0.35,
    verb_match_threshold: float = 0.50,
) -> Tuple[bool, str]:
    """
    Quyết định có cần chạy NLI cho claim pair hay không.

    Ý tưởng:
    - Nếu semantic similarity cao -> chạy NLI.
    - Nếu semantic thấp nhưng cùng entity / cùng subject / có mismatch số hoặc thời gian
      thì vẫn chạy NLI để không bỏ sót contradiction.
    - Nếu khác semantic + khác entity/event -> gán neutral, skip NLI.
    """

    sem = alignment.semantic_similarity
    ent = alignment.entity_overlap
    subj = alignment.subject_entity_overlap
    verb = alignment.verb_match
    temporal = alignment.temporal_overlap
    number = alignment.number_compatibility

    if sem >= semantic_similarity_threshold:
        return True, "high_semantic_similarity"

    if ent > 0 and sem >= moderate_semantic_threshold:
        return True, "entity_overlap_with_moderate_semantic_similarity"

    if subj > 0 and verb >= verb_match_threshold:
        return True, "same_subject_and_similar_verb"

    if ent > 0 and number == 0.0:
        return True, "same_entity_with_number_mismatch"

    if ent > 0 and temporal == 0.0:
        return True, "same_entity_with_temporal_mismatch"

    return False, "low_candidate_relevance"

def build_skipped_neutral_edge(
    pair: ClaimPair,
    alignment: AlignmentFeatures,
    skip_reason: str,
) -> NLIEdge:
    """
    Tạo edge neutral mà không cần chạy NLI.
    Dùng cho pair có semantic/entity/event quá thấp.
    """

    return NLIEdge(
        edge_id=f"e_{pair.claim_pair_id}",
        source_claim_id=pair.claim_a.claim_id,
        target_claim_id=pair.claim_b.claim_id,

        source_doc_id=pair.claim_a.doc_id,
        target_doc_id=pair.claim_b.doc_id,
        source_id=pair.claim_a.source_id,
        target_source_id=pair.claim_b.source_id,

        relation_type="neutral",
        relation_confidence=1.0,

        nli_edge_features=NLIEdgeFeatures(
            entailment_prob=0.0,
            neutral_prob=1.0,
            support_prob=0.0,
            contradiction_prob=0.0,
        ),

        alignment_features=alignment,

        nli_skipped=True,
        skip_reason=skip_reason,

        entailment_a_to_b=0.0,
        entailment_b_to_a=0.0,
        neutral_a_to_b=1.0,
        neutral_b_to_a=1.0,
        contradiction_a_to_b=0.0,
        contradiction_b_to_a=0.0,

    )

# =============================================================================
# Relation decision
# =============================================================================

def decide_relation_with_alignment(
    entailment_a_to_b: float,
    entailment_b_to_a: float,
    support_prob: float,
    neutral_prob: float,
    contradiction_prob: float,
    alignment: AlignmentFeatures,
    entailment_threshold: float = 0.70,
    support_threshold: float = 0.6,
    contradiction_threshold: float = 0.60,
) -> Tuple[str, float]:

    # 1. Khác factual target thì neutral trước
    if alignment.force_neutral:
        return "neutral", clamp01(max(neutral_prob, contradiction_prob))

    # 2. Contradiction
    if contradiction_prob >= contradiction_threshold:
        has_hard_conflict_signal = (
            alignment.negation_mismatch
            or alignment.number_compatibility == 0.0
        )

        has_contextual_mismatch = (
            alignment.temporal_overlap == 0.0
        )

        if has_hard_conflict_signal:
            return "contradiction", clamp01(contradiction_prob)

        if has_contextual_mismatch:
            return "contradiction", clamp01(contradiction_prob)

        if alignment.event_alignment >= 0.50:
            return "contradiction", clamp01(contradiction_prob)

        return "neutral", clamp01(max(neutral_prob, contradiction_prob))

    # 3. Entailment có hướng
    if entailment_a_to_b >= entailment_threshold:
        return "entailment", clamp01(entailment_a_to_b)

    # 4. Support
    if support_prob >= support_threshold:
        return "support", clamp01(support_prob)

    return "neutral", clamp01(neutral_prob)


# =============================================================================
# NLI inference engine
# =============================================================================

class NLIInferencer:
    def __init__(
        self,
        model_name: str = "roberta-large-mnli",
        device: Optional[str] = None,
        max_length: int = 256,
        batch_size: int = 16,
        entailment_threshold: float = 0.70,
        support_threshold: float = 0.55,
        contradiction_threshold: float = 0.60,
        semantic_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        semantic_similarity_threshold: float = 0.60,
        disable_semantic_filter: bool = False,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name
        ).to(self.device)

        self.model.eval()

        self.entailment_threshold = entailment_threshold
        self.support_threshold = support_threshold
        self.contradiction_threshold = contradiction_threshold

        self.semantic_similarity_threshold = semantic_similarity_threshold
        self.disable_semantic_filter = disable_semantic_filter

        if disable_semantic_filter:
            self.semantic_scorer = None
        else:
            self.semantic_scorer = SemanticSimilarityScorer(
                model_name=semantic_model_name,
                device=self.device,
            )

        self.label_map = self._build_label_map()

    def _build_label_map(self) -> Dict[str, int]:
        """
        Tự nhận diện label mapping.

        Yêu cầu model 3-class:
        - contradiction
        - neutral
        - entailment
        """

        id2label = self.model.config.id2label
        normalized: Dict[str, int] = {}

        for idx, label in id2label.items():
            label_lower = str(label).lower()

            if "entail" in label_lower and "not" not in label_lower:
                normalized["entailment"] = int(idx)
            elif "contrad" in label_lower:
                normalized["contradiction"] = int(idx)
            elif "neutral" in label_lower:
                normalized["neutral"] = int(idx)

        if {"entailment", "neutral", "contradiction"} <= set(normalized):
            return normalized

        if "nli-deberta-v3-base" in self.model_name.lower():
            return {
                "contradiction": 0,
                "entailment": 1,
                "neutral": 2,
            }
        # Fallback cho MNLI label order phổ biến:
        # roberta-large-mnli: contradiction=0, neutral=1, entailment=2
        if len(id2label) == 3:
            return {
                "contradiction": 0,
                "neutral": 1,
                "entailment": 2,
            }

        if len(id2label) == 2:
            raise ValueError(
                f"Model {self.model_name} is a binary entailment model with labels {id2label}. "
                "This pipeline requires a 3-class MNLI model with labels: "
                "entailment, neutral, contradiction. "
                "Use --model_name roberta-large-mnli or another 3-class MNLI model."
            )

        raise ValueError(
            f"Cannot infer NLI label mapping from model labels: {id2label}"
        )

    @torch.no_grad()
    def predict_directional_batch(
        self,
        premises: List[str],
        hypotheses: List[str],
        desc: str = "NLI inference",
    ) -> List[Dict[str, float]]:
        """
        Chạy NLI batch cho một chiều:
            premise -> hypothesis
        """

        outputs: List[Dict[str, float]] = []

        total_batches = (len(premises) + self.batch_size - 1) // self.batch_size

        for start in tqdm(
            range(0, len(premises), self.batch_size),
            total=total_batches,
            desc=desc,
            unit="batch",
        ):
            end = start + self.batch_size

            batch_premises = premises[start:end]
            batch_hypotheses = hypotheses[start:end]

            encoded = self.tokenizer(
                batch_premises,
                batch_hypotheses,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            logits = self.model(**encoded).logits
            probs = F.softmax(logits, dim=-1).detach().cpu()

            for row in probs:
                outputs.append(
                    {
                        "entailment": float(row[self.label_map["entailment"]]),
                        "neutral": float(row[self.label_map["neutral"]]),
                        "contradiction": float(row[self.label_map["contradiction"]]),
                    }
                )

        return outputs

    def infer_batch(self, pairs: List[ClaimPair]) -> List[NLIEdge]:
        """
        Flow mới:
        1. Tính semantic similarity + alignment trước.
        2. Pair không đáng xét -> tạo neutral edge, skip NLI.
        3. Pair đáng xét -> chạy NLI hai chiều.
        4. Merge kết quả theo thứ tự ban đầu.
        """

        if not pairs:
            return []

        nli_pairs: List[ClaimPair] = []
        nli_alignments: List[AlignmentFeatures] = []

        skipped_edges_by_pair_id: Dict[str, NLIEdge] = {}

        for pair in tqdm(
            pairs,
            total=len(pairs),
            desc="Candidate filtering",
            unit="pair",
        ):
            if self.semantic_scorer is None:
                semantic_similarity = 1.0
            else:
                semantic_similarity = self.semantic_scorer.score(
                    claim_a=pair.claim_a,
                    claim_b=pair.claim_b,
                )

            alignment = compute_alignment_features(
                claim_a=pair.claim_a,
                claim_b=pair.claim_b,
                semantic_similarity=semantic_similarity,
                min_semantic_similarity=self.semantic_similarity_threshold,
            )

            run_nli, reason = should_run_nli(
                alignment=alignment,
                semantic_similarity_threshold=self.semantic_similarity_threshold,
            )

            if run_nli:
                nli_pairs.append(pair)
                nli_alignments.append(alignment)
            else:
                skipped_edges_by_pair_id[pair.claim_pair_id] = build_skipped_neutral_edge(
                    pair=pair,
                    alignment=alignment,
                    skip_reason=reason,
                )

        print(f"Pairs total: {len(pairs)}")
        print(f"Pairs skipped as neutral: {len(skipped_edges_by_pair_id)}")
        print(f"Pairs sent to NLI: {len(nli_pairs)}")

        nli_edges_by_pair_id: Dict[str, NLIEdge] = {}

        if nli_pairs:
            premises_ab = [p.claim_a.claim_text for p in nli_pairs]
            hypotheses_ab = [p.claim_b.claim_text for p in nli_pairs]

            probs_ab = self.predict_directional_batch(
                premises_ab,
                hypotheses_ab,
                desc="NLI A -> B",
            )

            premises_ba = [p.claim_b.claim_text for p in nli_pairs]
            hypotheses_ba = [p.claim_a.claim_text for p in nli_pairs]

            probs_ba = self.predict_directional_batch(
                premises_ba,
                hypotheses_ba,
                desc="NLI B -> A",
            )

            for pair, alignment, ab, ba in tqdm(
                list(zip(nli_pairs, nli_alignments, probs_ab, probs_ba)),
                total=len(nli_pairs),
                desc="Building NLI edges",
                unit="edge",
            ):
                edge = self._build_edge_from_probs(
                    pair=pair,
                    ab=ab,
                    ba=ba,
                    alignment=alignment,
                )
                nli_edges_by_pair_id[pair.claim_pair_id] = edge

        # Giữ thứ tự output giống thứ tự pairs ban đầu
        final_edges: List[NLIEdge] = []

        for pair in pairs:
            if pair.claim_pair_id in nli_edges_by_pair_id:
                final_edges.append(nli_edges_by_pair_id[pair.claim_pair_id])
            else:
                final_edges.append(skipped_edges_by_pair_id[pair.claim_pair_id])

        return final_edges

    def _build_edge_from_probs(
        self,
        pair: ClaimPair,
        ab: Dict[str, float],
        ba: Dict[str, float],
        alignment: AlignmentFeatures,
    ) -> NLIEdge:

        entailment_a_to_b = ab["entailment"]
        entailment_b_to_a = ba["entailment"]

        neutral_a_to_b = ab["neutral"]
        neutral_b_to_a = ba["neutral"]

        contradiction_a_to_b = ab["contradiction"]
        contradiction_b_to_a = ba["contradiction"]

        contradiction_prob = max(
            contradiction_a_to_b,
            contradiction_b_to_a,
        )

        neutral_prob = (neutral_a_to_b + neutral_b_to_a) / 2.0

        # support_prob là symmetric entailment signal
        support_prob = max(
            entailment_a_to_b,
            entailment_b_to_a,
        )

        # entailment_prob giữ nghĩa directional A -> B
        entailment_prob = entailment_a_to_b

        relation_type, relation_confidence = decide_relation_with_alignment(
            entailment_a_to_b=entailment_a_to_b,
            entailment_b_to_a=entailment_b_to_a,
            support_prob=support_prob,
            neutral_prob=neutral_prob,
            contradiction_prob=contradiction_prob,
            alignment=alignment,
            entailment_threshold=self.entailment_threshold,
            support_threshold=self.support_threshold,
            contradiction_threshold=self.contradiction_threshold,
        )

        return NLIEdge(
            edge_id=f"e_{pair.claim_pair_id}",
            source_claim_id=pair.claim_a.claim_id,
            target_claim_id=pair.claim_b.claim_id,

            source_doc_id=pair.claim_a.doc_id,
            target_doc_id=pair.claim_b.doc_id,
            source_id=pair.claim_a.source_id,
            target_source_id=pair.claim_b.source_id,

            relation_type=relation_type,
            relation_confidence=relation_confidence,

            nli_edge_features=NLIEdgeFeatures(
                entailment_prob=clamp01(entailment_prob),
                neutral_prob=clamp01(neutral_prob),
                support_prob=clamp01(support_prob),
                contradiction_prob=clamp01(contradiction_prob),
            ),

            alignment_features=alignment,

            entailment_a_to_b=clamp01(entailment_a_to_b),
            entailment_b_to_a=clamp01(entailment_b_to_a),
            neutral_a_to_b=clamp01(neutral_a_to_b),
            neutral_b_to_a=clamp01(neutral_b_to_a),
            contradiction_a_to_b=clamp01(contradiction_a_to_b),
            contradiction_b_to_a=clamp01(contradiction_b_to_a),
        )


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path tới claims.jsonl",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="nli_edges.jsonl",
        help="Path output JSONL",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="roberta-large-mnli",
        help="NLI model 3-class, ví dụ roberta-large-mnli",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda, cpu hoặc bỏ trống để auto detect",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--same_query_only",
        action="store_true",
        help="Chỉ tạo pair trong cùng query_id, ví dụ q000",
    )

    parser.add_argument(
        "--skip_same_doc",
        action="store_true",
        help="Bỏ qua pair thuộc cùng doc_id",
    )

    parser.add_argument(
        "--include_duplicates",
        action="store_true",
        help="Không bỏ duplicate claims",
    )

    parser.add_argument(
        "--max_pairs",
        type=int,
        default=None,
        help="Giới hạn số pair để test nhanh",
    )

    parser.add_argument(
        "--candidate_selection",
        type=str,
        default="topk_semantic",
        choices=["full_pairs", "topk_semantic"],
        help=(
            "Cách tạo candidate pairs. "
            "full_pairs = tạo tất cả pair trong cùng query. "
            "topk_semantic = mỗi claim chỉ lấy top-k semantic neighbors."
        ),
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Số semantic neighbors gần nhất lấy cho mỗi claim khi dùng topk_semantic",
    )

    parser.add_argument(
        "--min_candidate_similarity",
        type=float,
        default=0.60,
        help=(
            "Ngưỡng semantic similarity scale [0, 1] tối thiểu để giữ candidate pair "
            "khi dùng topk_semantic"
        ),
    )

    parser.add_argument(
        "--entailment_threshold",
        type=float,
        default=0.70,
    )

    parser.add_argument(
        "--support_threshold",
        type=float,
        default=0.55,
    )

    parser.add_argument(
        "--contradiction_threshold",
        type=float,
        default=0.60,
    )

    parser.add_argument(
        "--semantic_model_name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Model dùng để tính semantic similarity giữa claim pairs",
    )

    parser.add_argument(
        "--semantic_similarity_threshold",
        type=float,
        default=0.60,
        help="Ngưỡng semantic similarity scale [0, 1] dùng cho candidate filtering và alignment",
    )

    parser.add_argument(
        "--disable_semantic_filter",
        action="store_true",
        help="Tắt semantic similarity filter",
    )

    parser.add_argument(
        "--queries",
        type=str,
        default=None,
        help="Path tới queries_revised.jsonl để filter claim theo relevance với user_query",
    )

    parser.add_argument(
        "--query_relevance_threshold",
        type=float,
        default=None,
        help=(
            "Ngưỡng relevance giữa claim_text và user_query. "
            "Nếu không truyền thì không filter theo query relevance."
        ),
    )

    parser.add_argument(
        "--top_k_claims_per_query",
        type=int,
        default=None,
        help="Chỉ giữ top-k claims liên quan nhất cho mỗi query sau khi filter relevance",
    )

    parser.add_argument(
        "--keep_claims_without_query",
        action="store_true",
        help="Giữ lại claims không map được sang query_id trong queries file",
    )

    parser.add_argument(
        "--target_query_id",
        type=str,
        default=None,
        help="Chỉ chạy NLI cho một query_id, ví dụ q000 hoặc q_000",
    )

    args = parser.parse_args()

    job_start_dt = datetime.now()
    job_start_ts = time.perf_counter()

    print("=" * 80)
    print(f"Job started at: {job_start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


    claims = read_claims_jsonl(args.input)
    print(f"Loaded claims: {len(claims)}")

    if args.target_query_id is not None:
        target_qid = normalize_query_id(args.target_query_id)

        claims = [
            claim for claim in claims
            if extract_query_id_from_claim_id(claim.claim_id) == target_qid
        ]

        print(f"Filtered claims for target_query_id={target_qid}: {len(claims)}")
    # Khởi tạo inferencer trước để dùng chung semantic_scorer cho top-k candidate selection.
    # Như vậy SentenceTransformer chỉ load một lần và cache embedding được tái sử dụng trong infer_batch().
    inferencer = NLIInferencer(
        model_name=args.model_name,
        device=args.device,
        max_length=args.max_length,
        batch_size=args.batch_size,
        entailment_threshold=args.entailment_threshold,
        support_threshold=args.support_threshold,
        contradiction_threshold=args.contradiction_threshold,
        semantic_model_name=args.semantic_model_name,
        semantic_similarity_threshold=args.semantic_similarity_threshold,
        disable_semantic_filter=args.disable_semantic_filter,
    )

    # -------------------------------------------------------------------------
    # Query relevance filtering
    # -------------------------------------------------------------------------
    if args.queries is not None and args.query_relevance_threshold is not None:
        if inferencer.semantic_scorer is None:
            raise ValueError(
                "Query relevance filtering requires semantic_scorer, "
                "but --disable_semantic_filter was used."
            )

        queries_by_id = read_queries_jsonl(args.queries)
        print(f"Loaded queries: {len(queries_by_id)}")

        claims = filter_claims_by_query_relevance(
            claims=claims,
            queries_by_id=queries_by_id,
            semantic_scorer=inferencer.semantic_scorer,
            min_query_relevance=args.query_relevance_threshold,
            top_k_claims_per_query=args.top_k_claims_per_query,
            keep_claims_without_query=args.keep_claims_without_query,
        )
    elif args.queries is not None and args.query_relevance_threshold is None:
        print(
            "WARNING: --queries was provided but --query_relevance_threshold is None. "
            "Query relevance filtering is skipped."
        )

    # -------------------------------------------------------------------------
    # Candidate pair construction
    # -------------------------------------------------------------------------
    if args.candidate_selection == "topk_semantic":
        if inferencer.semantic_scorer is None:
            raise ValueError(
                "candidate_selection='topk_semantic' requires semantic_scorer, "
                "but --disable_semantic_filter was used. "
                "Remove --disable_semantic_filter or use --candidate_selection full_pairs."
            )

        pairs = build_claim_pairs_topk_semantic(
            claims=claims,
            semantic_scorer=inferencer.semantic_scorer,
            same_query_only=args.same_query_only,
            skip_duplicates=not args.include_duplicates,
            skip_same_doc=args.skip_same_doc,
            top_k=args.top_k,
            min_similarity=args.min_candidate_similarity,
            max_pairs=args.max_pairs,
        )
    else:
        pairs = build_claim_pairs(
            claims=claims,
            same_query_only=args.same_query_only,
            skip_duplicates=not args.include_duplicates,
            skip_same_doc=args.skip_same_doc,
            max_pairs=args.max_pairs,
        )

    print(f"Candidate pairs: {len(pairs)}")
    print(f"Candidate selection mode: {args.candidate_selection}")

    edges = inferencer.infer_batch(pairs)

    write_jsonl(edges, args.output)

    print(f"Saved NLI edges to: {args.output}")

    job_end_dt = datetime.now()
    job_end_ts = time.perf_counter()
    elapsed_seconds = job_end_ts - job_start_ts

    print("=" * 80)
    print(f"Job finished at: {job_end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total runtime: {elapsed_seconds:.2f} seconds")
    print(f"Total runtime: {elapsed_seconds / 60:.2f} minutes")
    print("=" * 80)

if __name__ == "__main__":
    main()