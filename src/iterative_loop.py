"""
src/iterative_loop.py
Phase 6 — Iterative Retrieval Loop
WBS 31: Graph updater
WBS 32: Iterative loop implementation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import networkx as nx

from .embedder import ClaimEmbedder, ClaimCanonical
from .extractor import ClaimExtractor
from .conflict_zone import ConflictAnalysisResult, ConflictZoneAnalyzer
from .graph_builder import ClaimGraphBuilder, NLIInference
from .retriever import BalancedTopKSelector, HybridRetriever
from .schema import (
    Claim,
    ConflictLocalization,
    Document,
    LoopResult,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Conflict-Aware Query Formulator
# ---------------------------------------------------------------------------

class ConflictQueryFormulator:
    """Tạo conflict-aware query từ conflict localization info.

    Inject slot type, conflicting values, và original query intent
    để hướng dẫn retrieval có mục tiêu hơn.
    """

    def formulate(
        self,
        original_query: str,
        localization: ConflictLocalization,
        claim_texts: dict[str, str],   # claim_id → text
    ) -> str:
        """Tạo targeted query.

        Args:
            original_query: Query gốc từ user.
            localization: ConflictLocalization object.
            claim_texts: Mapping claim_id → claim text.

        Returns:
            Targeted query string.
        """
        text_i = claim_texts.get(localization.claim_i_id, localization.value_i)
        text_j = claim_texts.get(localization.claim_j_id, localization.value_j)

        slot = localization.slot
        val_i = localization.value_i
        val_j = localization.value_j

        # Template theo slot type
        if slot == "temporal":
            targeted = (
                f"Verify: {original_query} "
                f"Source A claims {val_i}, Source B claims {val_j}. "
                f"[temporal conflict] Which year is correct?"
            )
        elif slot == "numerical":
            targeted = (
                f"Verify numerical claim: {original_query} "
                f"One source says {val_i}, another says {val_j}. "
                f"Find authoritative evidence."
            )
        elif slot in ("entity_subject", "entity_object"):
            targeted = (
                f"Disambiguate entity conflict: {original_query} "
                f"Conflicting entities: '{val_i}' vs '{val_j}'. "
                f"Clarify which entity is correct in this context."
            )
        elif slot == "location":
            targeted = (
                f"Verify location: {original_query} "
                f"Source A: {val_i}, Source B: {val_j}. "
                f"Find evidence for correct location."
            )
        else:
            targeted = (
                f"Fact-check: {original_query} "
                f"Conflicting claims: '{text_i}' vs '{text_j}'. "
                f"Find supporting evidence."
            )

        logger.debug("Formulated targeted query: %s", targeted[:100])
        return targeted


# ---------------------------------------------------------------------------
# Graph Updater (WBS 31)
# ---------------------------------------------------------------------------

class GraphUpdater:
    """Insert new claims/edges vào existing graph sau mỗi retrieval iteration."""

    def __init__(
        self,
        extractor: ClaimExtractor,
        embedder: ClaimEmbedder,
        canonical: ClaimCanonical,
        graph_builder: ClaimGraphBuilder,
    ) -> None:
        self.extractor = extractor
        self.embedder = embedder
        self.canonical = canonical
        self.graph_builder = graph_builder

    def update(
        self,
        graph: nx.DiGraph,
        new_docs: list[Document],
        existing_claims: list[Claim],
    ) -> tuple[nx.DiGraph, list[Claim]]:
        """Extract claims từ new_docs và insert vào graph.

        Args:
            graph: Existing evidence graph.
            new_docs: Newly retrieved documents.
            existing_claims: All claims đã có trong graph.

        Returns:
            Tuple (updated_graph, all_claims_including_new).
        """
        # Extract + embed new claims
        new_claims_raw = self.extractor.extract_batch(new_docs)
        if not new_claims_raw:
            logger.info("No new claims extracted from %d docs", len(new_docs))
            return graph, existing_claims

        self.embedder.embed(new_claims_raw)
        new_claims = self.canonical.canonicalize(new_claims_raw)

        # Lọc claims đã có trong graph
        existing_ids = {c.claim_id for c in existing_claims}
        truly_new = [c for c in new_claims if c.claim_id not in existing_ids]

        if not truly_new:
            logger.info("All new claims are duplicates of existing ones")
            return graph, existing_claims

        # Build sub-graph với new + existing claims và merge
        all_claims = existing_claims + truly_new
        new_graph, _ = self.graph_builder.build(all_claims)

        logger.info(
            "Graph updated: +%d new claims, graph now has %d nodes, %d edges",
            len(truly_new),
            new_graph.number_of_nodes(),
            new_graph.number_of_edges(),
        )
        return new_graph, all_claims


# ---------------------------------------------------------------------------
# IterativeLoop (WBS 32)
# ---------------------------------------------------------------------------

@dataclass
class IterationLog:
    """Log cho mỗi iteration."""
    iteration: int
    n_new_docs: int
    n_new_claims: int
    n_conflict_pairs: int
    max_conflict_intensity: float
    avg_credibility_gap: float


class IterativeLoop:
    """Main iterative reasoning-retrieval loop.

    Stopping criteria (WBS 32):
    A. conflict_intensity_score < θ_intensity → resolved → break
    B. iterations ≥ MAX_ITERATIONS → unresolvable → break
    C. No new unique documents retrieved → break

    Args:
        graph_builder: ClaimGraphBuilder instance.
        retriever: HybridRetriever instance.
        selector: BalancedTopKSelector instance.
        conflict_analyzer: ConflictZoneAnalyzer instance.
        query_formulator: ConflictQueryFormulator instance.
        updater: Optional GraphUpdater for graph updates.
        max_iterations: Hardcoded ceiling.
        intensity_threshold: Break nếu max intensity < threshold.
        min_intensity_reduction: Break nếu reduction per iter < này.
    """

    def __init__(
        self,
        graph_builder: ClaimGraphBuilder,
        retriever: HybridRetriever,
        selector: BalancedTopKSelector,
        conflict_analyzer: ConflictZoneAnalyzer,
        query_formulator: ConflictQueryFormulator,
        updater: Optional[GraphUpdater] = None,
        max_iterations: int = 3,
        intensity_threshold: float = 0.1,
        min_intensity_reduction: float = 0.1,
    ) -> None:
        self.graph_builder = graph_builder
        self.retriever = retriever
        self.selector = selector
        self.conflict_analyzer = conflict_analyzer
        self.query_formulator = query_formulator
        self.updater = updater
        self.max_iterations = max_iterations
        self.intensity_threshold = intensity_threshold
        self.min_intensity_reduction = min_intensity_reduction

    def run(
        self,
        query: str,
        query_id: str,
        claims: list[Claim],
        graph: nx.DiGraph,
        top_k_claims: list[Claim],
        edge_index: dict,
    ) -> LoopResult:
        """Chạy iterative conflict resolution. Claims đã được processed trước.

        Args:
            query: Original user query.
            query_id: Query identifier.
            claims: Pre-processed list[Claim].
            graph: nx.DiGraph từ Phase 3.
            top_k_claims: Initial top-k từ Phase 4.
            edge_index: (claim_id_a, claim_id_b) → relation.

        Returns:
            LoopResult với validated claims và conflict localizations.
        """
        all_claims = list(claims)
        prev_max_intensity = float("inf")
        iteration_logs: list[IterationLog] = []

        for iteration in range(self.max_iterations):
            # Conflict zone analysis (Phase 5)
            analysis = self.conflict_analyzer.analyze(graph, top_k_claims)
            localizations = analysis.conflict_localizations
            max_intensity = max((loc.conflict_intensity for loc in localizations), default=0.0)
            cred_vals = list(analysis.credibility_scores.values())
            avg_cred_gap = (max(cred_vals) - min(cred_vals)) if cred_vals else 0.0

            iteration_logs.append(IterationLog(
                iteration=iteration + 1,
                n_new_docs=0,
                n_new_claims=len(all_claims),
                n_conflict_pairs=len(localizations),
                max_conflict_intensity=max_intensity,
                avg_credibility_gap=avg_cred_gap,
            ))
            logger.info(
                "[Loop] iter=%d conflicts=%d max_intensity=%.3f",
                iteration + 1, len(localizations), max_intensity,
            )

            # Stopping conditions

            # Condition A: intensity đủ thấp
            if max_intensity < self.intensity_threshold:
                logger.info(
                    "[Loop] Resolved: intensity %.3f < %.3f",
                    max_intensity, self.intensity_threshold,
                )
                break

            # Condition C: intensity không giảm đủ
            intensity_reduction = prev_max_intensity - max_intensity
            if iteration > 0 and intensity_reduction < self.min_intensity_reduction:
                logger.info(
                    "[Loop] Convergence: reduction %.3f < %.3f",
                    intensity_reduction, self.min_intensity_reduction,
                )
                break

            prev_max_intensity = max_intensity

            if not localizations:
                logger.info("[Loop] No conflicts found")
                break

            # Targeted retrieval (Phase 6) — formulate queries
            claim_texts = {c.claim_id: c.text for c in all_claims}
            new_docs: list[Document] = []
            for loc in localizations[:2]:
                targeted_query = self.query_formulator.formulate(query, loc, claim_texts)
                logger.info("[Loop] Targeted query: '%s...'", targeted_query[:80])
                # Note: new_docs retrieved from external source would go here
                # Current implementation: targeted queries formulated but no external retrieval

            if self.updater and new_docs:
                graph, all_claims = self.updater.update(graph, new_docs, all_claims)
                self.retriever.index(all_claims)
                retrieved = self.retriever.retrieve(query)
                edge_index = {
                    (u, v): d.get("relation", "neutral")
                    for u, v, d in graph.edges(data=True)
                }
                top_k_claims = self.selector.select(retrieved, edge_index)
            else:
                logger.info("[Loop] No new documents — stopping")
                break

        # Final analysis
        final_analysis = self.conflict_analyzer.analyze(graph, top_k_claims)
        validated_claims = [
            c for c in top_k_claims
            if c.claim_id in final_analysis.validated_claim_ids
        ]
        validated_claims.sort(
            key=lambda c: final_analysis.credibility_scores.get(c.claim_id, 0.0),
            reverse=True,
        )
        is_resolved = (
            not final_analysis.conflict_localizations
            or all(
                loc.conflict_intensity < self.intensity_threshold
                for loc in final_analysis.conflict_localizations
            )
        )
        logger.info(
            "[Loop] Done: %d iters, resolved=%s, %d validated",
            len(iteration_logs), is_resolved, len(validated_claims),
        )

        return LoopResult(
            query_id=query_id,
            iterations_run=len(iteration_logs),
            resolved=is_resolved,
            validated_claims=validated_claims,
            conflict_localizations=final_analysis.conflict_localizations,
        )
