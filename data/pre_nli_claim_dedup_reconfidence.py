import json
import re
import argparse
from typing import Dict, Any, List, Tuple, Optional, Set
from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s\.\-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def token_set(text: str) -> Set[str]:
    stopwords = {
        "the", "a", "an", "of", "in", "on", "at", "to", "for", "with",
        "and", "or", "by", "from", "was", "were", "is", "are", "be",
        "been", "being", "that", "this", "these", "those", "as", "it"
    }
    toks = re.findall(r"\b[a-zA-Z0-9][a-zA-Z0-9\-]*\b", text.lower())
    return {t for t in toks if t not in stopwords and len(t) >= 2}


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)





def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def get_claim_doc_key(claim: Dict[str, Any]) -> Optional[str]:
    """
    Lấy document/source key của một claim để tính document diversity.
    Ưu tiên doc_id; nếu thiếu thì fallback source_id.
    """
    doc_id = claim.get("doc_id")
    if doc_id is not None and str(doc_id).strip():
        return str(doc_id).strip()

    source_id = claim.get("source_id")
    if source_id is not None and str(source_id).strip():
        return str(source_id).strip()

    return None


def get_claim_feature_value(
    claim: Dict[str, Any],
    feature_name: str,
    default: float = 0.0,
) -> float:
    features = claim.get("claim_features") or {}
    if not isinstance(features, dict):
        return default

    value = features.get(feature_name, default)
    try:
        return clamp01(float(value))
    except Exception:
        return default


def ensure_claim_features(claim: Dict[str, Any]) -> Dict[str, Any]:
    features = claim.get("claim_features")
    if not isinstance(features, dict):
        features = {}
        claim["claim_features"] = features
    return features


def normalized_doc_diversity(num_docs: int, doc_count_norm_cap: int = 5) -> float:
    """
    Chuẩn hóa số lượng distinct docs về [0, 1].
    Nếu cap=5, >=5 docs được xem là coverage/diversity tối đa.
    """
    if doc_count_norm_cap <= 0:
        return 0.0
    return clamp01(num_docs / float(doc_count_norm_cap))


def recompute_claim_confidence_from_canonical_docs(
    annotated_claims: List[Dict[str, Any]],
    context_weight: float = 0.35,
    retrieval_weight: float = 0.35,
    doc_diversity_weight: float = 0.30,
    doc_count_norm_cap: int = 5,
    round_digits: int = 4,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Tính lại claim_confidence sau bước dedup/canonicalization.

    Công thức:
        claim_confidence =
            context_weight * context_completeness
          + retrieval_weight * retrieval_relevance
          + doc_diversity_weight * normalized_distinct_doc_count

    Trong đó distinct_doc_count được tính trên toàn bộ canonical members:
        canonical_claim_id -> {doc_id/source_id của mọi member trong nhóm canonical}

    Hàm cập nhật cả representative claim và duplicate members để output annotated
    có cùng thông tin tracing; file kept vẫn chỉ chứa duplicate_of == None.
    """
    weight_sum = context_weight + retrieval_weight + doc_diversity_weight
    if weight_sum <= 0:
        raise ValueError("Tổng weights để tính claim_confidence phải > 0")

    # Normalize weights để tránh user truyền tổng khác 1.0.
    context_weight /= weight_sum
    retrieval_weight /= weight_sum
    doc_diversity_weight /= weight_sum

    canonical_to_docs: Dict[str, Set[str]] = defaultdict(set)

    for claim in annotated_claims:
        claim_id = claim.get("claim_id")
        canonical_id = claim.get("canonical_claim_id") or claim.get("duplicate_of") or claim_id
        if not canonical_id:
            continue

        doc_key = get_claim_doc_key(claim)
        if doc_key is not None:
            canonical_to_docs[str(canonical_id)].add(doc_key)

    updated_claims: List[Dict[str, Any]] = []
    changed = 0

    for claim in annotated_claims:
        c2 = dict(claim)
        claim_id = c2.get("claim_id")
        canonical_id = c2.get("canonical_claim_id") or c2.get("duplicate_of") or claim_id
        canonical_id = str(canonical_id) if canonical_id is not None else None

        features = ensure_claim_features(c2)

        context_completeness = get_claim_feature_value(c2, "context_completeness", 0.0)
        retrieval_relevance = get_claim_feature_value(c2, "retrieval_relevance", 0.0)

        num_docs = len(canonical_to_docs.get(canonical_id, set())) if canonical_id else 0
        doc_diversity_score = normalized_doc_diversity(
            num_docs=num_docs,
            doc_count_norm_cap=doc_count_norm_cap,
        )

        old_conf = features.get("claim_confidence")
        new_conf = clamp01(
            context_weight * context_completeness
            + retrieval_weight * retrieval_relevance
            + doc_diversity_weight * doc_diversity_score
        )

        features["claim_confidence"] = round(new_conf, round_digits)
        features["num_distinct_canonical_member_docs"] = num_docs
        features["canonical_doc_diversity_score"] = round(doc_diversity_score, round_digits)
        features["claim_confidence_components"] = {
            "context_completeness": round(context_completeness, round_digits),
            "retrieval_relevance": round(retrieval_relevance, round_digits),
            "num_distinct_canonical_member_docs": num_docs,
            "canonical_doc_diversity_score": round(doc_diversity_score, round_digits),
            "weights": {
                "context_completeness": round(context_weight, round_digits),
                "retrieval_relevance": round(retrieval_weight, round_digits),
                "canonical_doc_diversity_score": round(doc_diversity_weight, round_digits),
            },
            "old_claim_confidence": old_conf,
        }

        changed += 1
        updated_claims.append(c2)

    stats = {
        "updated_claims": changed,
        "num_canonical_groups": len(canonical_to_docs),
        "doc_count_norm_cap": doc_count_norm_cap,
        "weights": {
            "context_completeness": context_weight,
            "retrieval_relevance": retrieval_weight,
            "canonical_doc_diversity_score": doc_diversity_weight,
        },
    }

    return updated_claims, stats


def normalize_unit(unit: Any) -> str:
    if unit is None:
        return ""

    unit = str(unit).lower().strip()

    aliases = {
        "percent": "%",
        "percentage": "%",
        "dollar": "usd",
        "dollars": "usd",
        "$": "usd",
        "people": "person",
        "persons": "person",
        "inhabitants": "person",
        "households": "household",
        "families": "family",
    }

    return aliases.get(unit, unit)


def get_numbers(claim: Dict[str, Any]) -> List[Tuple[float, str]]:
    nums = claim.get("factoid_features", {}).get("number", [])
    out = []

    if not isinstance(nums, list):
        return out

    for n in nums:
        if not isinstance(n, dict):
            continue

        value = n.get("value")
        unit = n.get("unit")

        if value is None:
            continue

        try:
            value = float(value)
        except Exception:
            continue

        unit = normalize_unit(unit)
        out.append((value, unit))

    return sorted(out, key=lambda x: (x[0], x[1]))


def get_temporal_signature(claim: Dict[str, Any]) -> Optional[Tuple[str, str, str]]:
    temporal = claim.get("factoid_features", {}).get("temporal")

    if not isinstance(temporal, dict):
        return None

    start = temporal.get("start")
    end = temporal.get("end")
    granularity = temporal.get("granularity")

    if not start and not end:
        return None

    return (
        str(start) if start else "",
        str(end) if end else "",
        str(granularity) if granularity else "",
    )


def get_polarity(claim: Dict[str, Any]) -> Optional[int]:
    neg = claim.get("factoid_features", {}).get("negation", {})
    if not isinstance(neg, dict):
        return None

    polarity = neg.get("polarity")
    if polarity is None:
        return None

    try:
        return int(polarity)
    except Exception:
        return None


def get_verb_signature(claim: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    verb = claim.get("factoid_features", {}).get("verb", {})
    if not isinstance(verb, dict):
        return None

    lemma = verb.get("lemma")
    tense = verb.get("tense")

    if not lemma and not tense:
        return None

    return (
        str(lemma).lower().strip() if lemma else "",
        str(tense).lower().strip() if tense else "",
    )


def get_entity_signature(claim: Dict[str, Any]) -> Set[str]:
    entities = claim.get("factoid_features", {}).get("entity", [])
    out = set()

    if not isinstance(entities, list):
        return out

    for e in entities:
        if isinstance(e, dict):
            value = e.get("text") or e.get("value") or e.get("name")
            if value:
                out.add(normalize_text(str(value)))
        elif isinstance(e, str):
            out.add(normalize_text(e))

    return out


def has_hard_blocker(
    claim_a: Dict[str, Any],
    claim_b: Dict[str, Any],
    require_same_numbers: bool = True,
    require_same_temporal: bool = True,
    require_same_polarity: bool = True,
    require_same_verb: bool = False,
    require_entity_overlap: bool = False,
) -> Tuple[bool, List[str]]:
    """
    Return:
        blocked: True nếu không được merge
        reasons: lý do chặn merge
    """
    reasons = []

    nums_a = get_numbers(claim_a)
    nums_b = get_numbers(claim_b)

    if require_same_numbers and nums_a and nums_b and nums_a != nums_b:
        reasons.append("number_mismatch")

    temp_a = get_temporal_signature(claim_a)
    temp_b = get_temporal_signature(claim_b)

    if require_same_temporal and temp_a and temp_b and temp_a != temp_b:
        reasons.append("temporal_mismatch")

    pol_a = get_polarity(claim_a)
    pol_b = get_polarity(claim_b)

    if require_same_polarity and pol_a is not None and pol_b is not None and pol_a != pol_b:
        reasons.append("polarity_mismatch")

    verb_a = get_verb_signature(claim_a)
    verb_b = get_verb_signature(claim_b)

    if require_same_verb and verb_a and verb_b and verb_a != verb_b:
        reasons.append("verb_mismatch")

    ents_a = get_entity_signature(claim_a)
    ents_b = get_entity_signature(claim_b)

    if require_entity_overlap and ents_a and ents_b and not (ents_a & ents_b):
        reasons.append("entity_mismatch")

    return len(reasons) > 0, reasons


def choose_representative(claims: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Claim được giữ lại.
    Tiêu chí:
    1. Có claim_text dài hơn.
    2. Có nhiều number/entity/temporal feature hơn.
    3. Nếu vẫn bằng nhau, lấy claim xuất hiện trước.
    """

    def score(c: Dict[str, Any]) -> Tuple[int, int, int, int]:
        text = c.get("claim_text", "") or c.get("text", "")
        nums = get_numbers(c)
        ents = get_entity_signature(c)
        temporal = get_temporal_signature(c)

        return (
            len(str(text)),
            len(nums),
            len(ents),
            1 if temporal else 0,
        )

    return max(claims, key=score)


def build_exact_key(claim: Dict[str, Any]) -> Tuple:
    """
    Key rẻ để merge exact hoặc near-exact.
    Dùng text đã normalize + factoid signatures.
    """
    text = normalize_text(claim.get("claim_text", "") or claim.get("text", ""))

    return (
        text,
        tuple(get_numbers(claim)),
        get_temporal_signature(claim),
        get_polarity(claim),
    )


class UnionFind:
    def __init__(self, ids: List[str]):
        self.parent = {x: x for x in ids}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra

    def groups(self) -> Dict[str, List[str]]:
        out = defaultdict(list)
        for x in self.parent:
            out[self.find(x)].append(x)
        return dict(out)


def deduplicate_claims(
    claims: List[Dict[str, Any]],
    encoder_name: str = "BAAI/bge-large-en-v1.5",
    similarity_threshold: float = 0.94,
    token_overlap_threshold: float = 0.70,
    exact_first: bool = True,
    same_query_only: bool = True,
    recompute_claim_confidence: bool = True,
    confidence_context_weight: float = 0.35,
    confidence_retrieval_weight: float = 0.35,
    confidence_doc_weight: float = 0.30,
    confidence_doc_count_norm_cap: int = 5,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Returns:
        annotated_claims:
            toàn bộ claims, đã cập nhật canonical_claim_id và duplicate_of

        kept_claims:
            chỉ các claim được giữ lại, duplicate_of = None

        duplicate_log:
            log claim nào bị merge vào claim nào
    """

    valid_claims = []
    for c in claims:
        cid = c.get("claim_id")
        text = c.get("claim_text") or c.get("text")
        if cid and isinstance(text, str) and text.strip():
            valid_claims.append(c)

    id_to_claim = {c["claim_id"]: c for c in valid_claims}
    claim_ids = [c["claim_id"] for c in valid_claims]

    uf = UnionFind(claim_ids)
    duplicate_log = []

    # -------------------------
    # Stage 1: exact normalized merge
    # -------------------------
    if exact_first:
        buckets = defaultdict(list)

        for c in valid_claims:
            if same_query_only:
                query_id = c.get("query_id") or infer_query_id_from_claim_id(c["claim_id"])
            else:
                query_id = "__all__"

            key = (query_id, build_exact_key(c))
            buckets[key].append(c)

        for _, bucket in buckets.items():
            if len(bucket) <= 1:
                continue

            rep = choose_representative(bucket)
            rep_id = rep["claim_id"]

            for c in bucket:
                cid = c["claim_id"]
                if cid == rep_id:
                    continue

                uf.union(rep_id, cid)
                duplicate_log.append({
                    "removed_claim_id": cid,
                    "duplicate_of": rep_id,
                    "method": "exact_normalized_key",
                    "similarity": 1.0,
                    "reasons": ["same_normalized_text_and_factoid_signature"],
                })

    # -------------------------
    # Stage 2: embedding-based conservative merge
    # -------------------------
    if len(valid_claims) > 1:
        encoder = SentenceTransformer(encoder_name)

        texts = [
            c.get("claim_text") or c.get("text") or ""
            for c in valid_claims
        ]

        embeddings = encoder.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

        similarity_matrix = np.matmul(embeddings, embeddings.T)

        for i in range(len(valid_claims)):
            claim_i = valid_claims[i]
            id_i = claim_i["claim_id"]

            for j in range(i + 1, len(valid_claims)):
                claim_j = valid_claims[j]
                id_j = claim_j["claim_id"]

                if uf.find(id_i) == uf.find(id_j):
                    continue

                if same_query_only:
                    q_i = claim_i.get("query_id") or infer_query_id_from_claim_id(id_i)
                    q_j = claim_j.get("query_id") or infer_query_id_from_claim_id(id_j)

                    if q_i != q_j:
                        continue

                sim = float(similarity_matrix[i, j])

                if sim < similarity_threshold:
                    continue

                blocked, blocker_reasons = has_hard_blocker(
                    claim_i,
                    claim_j,
                    require_same_numbers=True,
                    require_same_temporal=True,
                    require_same_polarity=True,
                    require_same_verb=False,
                    require_entity_overlap=False,
                )

                if blocked:
                    duplicate_log.append({
                        "claim_a": id_i,
                        "claim_b": id_j,
                        "method": "embedding_candidate_rejected",
                        "similarity": sim,
                        "reasons": blocker_reasons,
                    })
                    continue

                toks_i = token_set(claim_i.get("claim_text") or claim_i.get("text") or "")
                toks_j = token_set(claim_j.get("claim_text") or claim_j.get("text") or "")
                overlap = jaccard(toks_i, toks_j)

                if overlap < token_overlap_threshold:
                    duplicate_log.append({
                        "claim_a": id_i,
                        "claim_b": id_j,
                        "method": "embedding_candidate_rejected",
                        "similarity": sim,
                        "token_overlap": overlap,
                        "reasons": ["low_token_overlap"],
                    })
                    continue

                # Chọn representative giữa 2 root groups hiện tại
                group_i = [id_to_claim[x] for x in uf.groups()[uf.find(id_i)]]
                group_j = [id_to_claim[x] for x in uf.groups()[uf.find(id_j)]]
                merged_group = group_i + group_j
                rep = choose_representative(merged_group)
                rep_id = rep["claim_id"]

                # Union tất cả vào representative
                for c in merged_group:
                    if c["claim_id"] != rep_id:
                        uf.union(rep_id, c["claim_id"])

                duplicate_log.append({
                    "removed_claim_id": id_j if rep_id == id_i else id_i,
                    "duplicate_of": rep_id,
                    "claim_a": id_i,
                    "claim_b": id_j,
                    "method": "embedding_similarity_with_hard_blockers",
                    "similarity": sim,
                    "token_overlap": overlap,
                    "reasons": ["high_similarity", "no_hard_blockers", "sufficient_token_overlap"],
                })

    # -------------------------
    # Stage 3: assign canonical_claim_id and duplicate_of
    # -------------------------
    groups = uf.groups()

    claim_to_canonical = {}

    for _, member_ids in groups.items():
        group_claims = [id_to_claim[x] for x in member_ids]
        rep = choose_representative(group_claims)
        rep_id = rep["claim_id"]

        for cid in member_ids:
            claim_to_canonical[cid] = rep_id

    annotated_claims = []

    for c in claims:
        c2 = dict(c)

        cid = c2.get("claim_id")

        if cid in claim_to_canonical:
            canonical_id = claim_to_canonical[cid]
            c2["canonical_claim_id"] = canonical_id

            if cid == canonical_id:
                c2["duplicate_of"] = None
            else:
                c2["duplicate_of"] = canonical_id
        else:
            # claim không hợp lệ hoặc thiếu text thì giữ nguyên
            c2.setdefault("canonical_claim_id", cid)
            c2.setdefault("duplicate_of", None)

        annotated_claims.append(c2)

    if recompute_claim_confidence:
        annotated_claims, confidence_stats = recompute_claim_confidence_from_canonical_docs(
            annotated_claims=annotated_claims,
            context_weight=confidence_context_weight,
            retrieval_weight=confidence_retrieval_weight,
            doc_diversity_weight=confidence_doc_weight,
            doc_count_norm_cap=confidence_doc_count_norm_cap,
        )
        duplicate_log.append({
            "method": "recompute_claim_confidence",
            "stats": confidence_stats,
        })

    kept_claims = [
        c for c in annotated_claims
        if c.get("duplicate_of") is None
    ]

    return annotated_claims, kept_claims, duplicate_log


def infer_query_id_from_claim_id(claim_id: str) -> Optional[str]:
    """
    Ví dụ claim_id: c_q000_d00_s00
    Trả về q000
    """
    m = re.search(r"(q\d+)", claim_id)
    if m:
        return m.group(1)
    return None


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Input factoid_claims.jsonl"
    )

    parser.add_argument(
        "--output_annotated",
        default="factoid_claims_dedup_annotated.jsonl",
        help="Output JSONL gồm tất cả claims, có canonical_claim_id và duplicate_of"
    )

    parser.add_argument(
        "--output_kept",
        default="factoid_claims_dedup_kept.jsonl",
        help="Output JSONL chỉ giữ claims canonical, tức duplicate_of = null"
    )

    parser.add_argument(
        "--output_log",
        default="duplicate_log.jsonl",
        help="Log duplicate decisions"
    )

    parser.add_argument(
        "--encoder_name",
        default="BAAI/bge-large-en-v1.5"
    )

    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.94,
        help="Threshold cao để pre-NLI merge an toàn hơn"
    )

    parser.add_argument(
        "--token_overlap_threshold",
        type=float,
        default=0.70
    )

    parser.add_argument(
        "--same_query_only",
        action="store_true",
        help="Chỉ merge claims trong cùng query_id hoặc cùng qXXX suy ra từ claim_id"
    )

    parser.add_argument(
        "--no_exact_first",
        action="store_true",
        help="Tắt exact normalized merge trước embedding merge"
    )


    parser.add_argument(
        "--no_recompute_claim_confidence",
        action="store_true",
        help=(
            "Không tính lại claim_confidence sau dedup. "
            "Mặc định script sẽ tính lại từ context_completeness, retrieval_relevance "
            "và số distinct docs của canonical members."
        ),
    )

    parser.add_argument(
        "--confidence_context_weight",
        type=float,
        default=0.35,
        help="Weight cho context_completeness khi tính lại claim_confidence",
    )

    parser.add_argument(
        "--confidence_retrieval_weight",
        type=float,
        default=0.35,
        help="Weight cho retrieval_relevance khi tính lại claim_confidence",
    )

    parser.add_argument(
        "--confidence_doc_weight",
        type=float,
        default=0.30,
        help="Weight cho canonical doc diversity khi tính lại claim_confidence",
    )

    parser.add_argument(
        "--confidence_doc_count_norm_cap",
        type=int,
        default=5,
        help="Số distinct docs để normalized doc diversity đạt 1.0",
    )

    args = parser.parse_args()

    claims = read_jsonl(args.input)

    annotated, kept, log = deduplicate_claims(
        claims=claims,
        encoder_name=args.encoder_name,
        similarity_threshold=args.similarity_threshold,
        token_overlap_threshold=args.token_overlap_threshold,
        exact_first=not args.no_exact_first,
        same_query_only=args.same_query_only,
        recompute_claim_confidence=not args.no_recompute_claim_confidence,
        confidence_context_weight=args.confidence_context_weight,
        confidence_retrieval_weight=args.confidence_retrieval_weight,
        confidence_doc_weight=args.confidence_doc_weight,
        confidence_doc_count_norm_cap=args.confidence_doc_count_norm_cap,
    )

    write_jsonl(args.output_annotated, annotated)
    write_jsonl(args.output_kept, kept)
    write_jsonl(args.output_log, log)

    print(f"Input claims: {len(claims)}")
    print(f"Annotated claims: {len(annotated)}")
    print(f"Kept canonical claims: {len(kept)}")
    print(f"Removed duplicate claims: {len(annotated) - len(kept)}")
    print(f"Saved annotated: {args.output_annotated}")
    print(f"Saved kept: {args.output_kept}")
    print(f"Saved log: {args.output_log}")


if __name__ == "__main__":
    main()