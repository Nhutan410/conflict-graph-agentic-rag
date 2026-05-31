"""WBS 14 — Retrieval and Generation Evaluation Metrics."""

import logging
import math
from collections import Counter
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of relevant docs found in the top-k retrieved results.

    Args:
        retrieved_ids: Ordered list of retrieved doc IDs.
        relevant_ids: Set of ground-truth relevant doc IDs.
        k: Cut-off rank.
    """
    if not relevant_ids:
        return 0.0
    hits = sum(1 for doc_id in retrieved_ids[:k] if doc_id in relevant_ids)
    return hits / len(relevant_ids)


def mean_reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Mean Reciprocal Rank: 1/rank of the first relevant document.

    Args:
        retrieved_ids: Ordered list of retrieved doc IDs.
        relevant_ids: Set of ground-truth relevant doc IDs.
    """
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at k.

    Assumes binary relevance (relevant = 1, not relevant = 0).

    Args:
        retrieved_ids: Ordered list of retrieved doc IDs.
        relevant_ids: Set of ground-truth relevant doc IDs.
        k: Cut-off rank.
    """
    def dcg(ids: list[str], relevance: set[str], cutoff: int) -> float:
        score = 0.0
        for rank, doc_id in enumerate(ids[:cutoff], start=1):
            if doc_id in relevance:
                score += 1.0 / math.log2(rank + 1)
        return score

    actual_dcg = dcg(retrieved_ids, relevant_ids, k)
    ideal_ids = list(relevant_ids) + [""] * k
    ideal_dcg = dcg(ideal_ids, relevant_ids, k)
    if ideal_dcg == 0.0:
        return 0.0
    return actual_dcg / ideal_dcg


# ---------------------------------------------------------------------------
# Generation metrics
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    return text.lower().strip().split()


def answer_f1(prediction: str, gold: str) -> float:
    """Token-level F1 between predicted answer and gold answer.

    Args:
        prediction: Model-generated answer string.
        gold: Reference answer string.
    """
    pred_tokens = _tokenize(prediction)
    gold_tokens = _tokenize(gold)
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def answer_exact_match(prediction: str, gold: str) -> float:
    """Exact match score after lowercase and strip.

    Returns 1.0 if strings match, 0.0 otherwise.
    """
    return 1.0 if prediction.lower().strip() == gold.lower().strip() else 0.0


# ---------------------------------------------------------------------------
# Aggregate evaluation
# ---------------------------------------------------------------------------

def evaluate_retrieval(
    results: list[dict],
    qrels: dict[str, set[str]],
    k_values: list[int] = None,
) -> dict:
    """Aggregate retrieval metrics across all queries.

    Args:
        results: List of pipeline output dicts with 'query_id' and 'retrieved_doc_ids'.
        qrels: Mapping from query_id -> set of relevant doc IDs.
        k_values: List of k values for Recall@k. Defaults to [5, 10].

    Returns:
        Dict of metric -> mean value across queries.
    """
    if k_values is None:
        k_values = [5, 10]

    recall_sums = {k: 0.0 for k in k_values}
    mrr_sum = 0.0
    ndcg_sum = 0.0
    count = 0

    for result in results:
        q_id = result["query_id"]
        retrieved = result.get("retrieved_doc_ids", [])
        relevant = qrels.get(q_id, set())
        if not relevant:
            continue
        for k in k_values:
            recall_sums[k] += recall_at_k(retrieved, relevant, k)
        mrr_sum += mean_reciprocal_rank(retrieved, relevant)
        ndcg_sum += ndcg_at_k(retrieved, relevant, 10)
        count += 1

    if count == 0:
        logger.warning("No valid qrels found for evaluation.")
        return {}

    metrics: dict = {}
    for k in k_values:
        metrics[f"Recall@{k}"] = round(recall_sums[k] / count, 4)
    metrics["MRR"] = round(mrr_sum / count, 4)
    metrics["nDCG@10"] = round(ndcg_sum / count, 4)
    return metrics


def evaluate_generation(
    results: list[dict],
    gold_answers: dict[str, str],
) -> dict:
    """Aggregate generation metrics across all queries.

    Args:
        results: List of pipeline output dicts with 'query_id' and 'answer'.
        gold_answers: Mapping from query_id -> gold answer string.

    Returns:
        Dict with mean Answer_F1 and Answer_EM.
    """
    f1_sum = 0.0
    em_sum = 0.0
    count = 0

    for result in results:
        q_id = result["query_id"]
        prediction = result.get("answer", "")
        gold = gold_answers.get(q_id, "")
        if not gold:
            continue
        f1_sum += answer_f1(prediction, gold)
        em_sum += answer_exact_match(prediction, gold)
        count += 1

    if count == 0:
        return {}

    return {
        "Answer_F1": round(f1_sum / count, 4),
        "Answer_EM": round(em_sum / count, 4),
    }
