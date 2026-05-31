"""
src/pipeline.py
Main Pipeline Orchestrator — kết nối tất cả phases theo proposal Method 1.

Luồng: Documents → Claim Extraction → Claim Canonical → Evidence Graph
       → Retrieve → Conflict Zone → Iterative Loop → Generation

Usage:
    pipeline = ConflictAwarePipeline(PipelineConfig())
    result = pipeline.run(query="...", query_id="q001", documents=[...])
    print(result.final_answer)
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .generator import AnswerGenerator, EvidenceSelector
from .embedder import ClaimCanonical, ClaimEmbedder
from .extractor import ClaimExtractor
from .conflict_zone import (
    ConflictZoneAnalyzer,
    CredibilityArbitrator,
    FactoidDecomposer,
)
from .graph_builder import ClaimGraphBuilder, NLIInference, PairGenerator
from .iterative_loop import (
    ConflictQueryFormulator,
    GraphUpdater,
    IterativeLoop,
)
from .retriever import BalancedTopKSelector, HybridRetriever
from .schema import Claim, Document, LoopResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline Config
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Tất cả hyperparameters của pipeline."""

    # Models
    embedding_model: str = "BAAI/bge-m3"
    nli_model: str = "cross-encoder/nli-deberta-v3-base"
    llm_model: str = "gpt-4o-mini"

    # Claim canonical
    canonical_threshold: float = 0.92

    # Graph construction
    pair_similarity_threshold: float = 0.3
    nli_edge_threshold: float = 0.5

    # Retrieval
    retrieval_top_k: int = 10
    retrieval_alpha: float = 0.5            # dense vs BM25 weight
    min_conflict_pairs: int = 1

    # Arbitration
    arbitration_max_iter: int = 10
    arbitration_convergence: float = 0.01
    arbitration_damping: float = 0.85
    credibility_threshold: float = 0.0

    # Iterative loop
    max_loop_iterations: int = 3
    intensity_threshold: float = 0.1
    min_intensity_reduction: float = 0.1

    # Generation
    max_evidence_claims: int = 10

    # Reproducibility
    seed: int = 42
    device: str = "cpu"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class ConflictAwarePipeline:
    """End-to-end conflict-aware RAG pipeline.

    Implements Method 1: Conflict-Aware Evidence Arbitration
    with Factoid-Level Resolution.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._set_seed(config.seed)
        self._build_components()

    @classmethod
    def from_config(cls, config_path: str) -> "ConflictAwarePipeline":
        """Load pipeline từ YAML config file.

        Args:
            config_path: Path to YAML config.

        Returns:
            Initialized ConflictAwarePipeline.
        """
        import yaml
        with open(config_path) as f:
            cfg_dict = yaml.safe_load(f)
        config = PipelineConfig(**cfg_dict)
        return cls(config)

    def _set_seed(self, seed: int) -> None:
        """Set seed cho reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass

    def _build_components(self) -> None:
        """Khởi tạo tất cả components."""
        cfg = self.config

        # Phase 1: Claim Extraction
        self.extractor = ClaimExtractor(model=cfg.llm_model)
        self.embedder = ClaimEmbedder(model_name=cfg.embedding_model)
        self.canonical = ClaimCanonical(similarity_threshold=cfg.canonical_threshold)

        # Phase 3: Graph
        self.nli = NLIInference(
            model_name=cfg.nli_model,
            threshold=cfg.nli_edge_threshold,
            device=cfg.device,
        )
        self.pair_gen = PairGenerator(
            similarity_threshold=cfg.pair_similarity_threshold
        )
        self.graph_builder = ClaimGraphBuilder(
            nli=self.nli,
            pair_generator=self.pair_gen,
            nli_threshold=cfg.nli_edge_threshold,
        )

        # Phase 4: Retrieval
        self.retriever = HybridRetriever(
            embedding_model=cfg.embedding_model,
            alpha=cfg.retrieval_alpha,
            top_k=cfg.retrieval_top_k,
        )
        self.selector = BalancedTopKSelector(
            target_k=cfg.retrieval_top_k,
            min_conflict_pairs=cfg.min_conflict_pairs,
        )

        # Phase 5: Conflict Zone
        arbitrator = CredibilityArbitrator(
            max_iterations=cfg.arbitration_max_iter,
            convergence_threshold=cfg.arbitration_convergence,
            damping=cfg.arbitration_damping,
        )
        decomposer = FactoidDecomposer(
            use_llm_for_entities=True,
            llm_model=cfg.llm_model,
        )
        self.conflict_analyzer = ConflictZoneAnalyzer(
            arbitrator=arbitrator,
            decomposer=decomposer,
            credibility_threshold=cfg.credibility_threshold,
        )

        # Phase 6: Loop
        self.query_formulator = ConflictQueryFormulator()
        self.loop = IterativeLoop(
            graph_builder=self.graph_builder,
            retriever=self.retriever,
            selector=self.selector,
            conflict_analyzer=self.conflict_analyzer,
            query_formulator=self.query_formulator,
            max_iterations=cfg.max_loop_iterations,
            intensity_threshold=cfg.intensity_threshold,
            min_intensity_reduction=cfg.min_intensity_reduction,
        )

        # Phase 7: Generation
        self.evidence_selector = EvidenceSelector(max_claims=cfg.max_evidence_claims)
        self.answer_gen = AnswerGenerator(
            model=cfg.llm_model,
            evidence_selector=self.evidence_selector,
        )

        logger.info("Pipeline components initialized with config: %s", cfg)

    def run(
        self,
        query: str,
        query_id: str,
        documents: Optional[list[Document]] = None,
        claims: Optional[list[Claim]] = None,
    ) -> LoopResult:
        """Run full pipeline.

        Args:
            query: User query string.
            query_id: Unique identifier cho query.
            documents: list[Document] → extract + embed từ đầu.
            claims: list[Claim] → skip Phase 1+2, bắt đầu từ Phase 3.

        Returns:
            LoopResult với validated_claims, conflict_localizations, final_answer.

        Raises:
            ValueError: Nếu cả documents và claims đều None.
        """
        if claims is None and documents is None:
            raise ValueError("Either documents or claims must be provided")

        logger.info("=" * 60)
        logger.info("Pipeline.run: query_id=%s", query_id)
        logger.info("Query: %s", query[:100])

        # Phase 1+2 (skip nếu claims đã có)
        if claims is None:
            logger.info("Phase 1: Extracting claims from %d documents", len(documents))
            claims = self.extractor.extract_batch(documents)
            logger.info("Phase 2a: Embedding %d claims", len(claims))
            self.embedder.embed(claims)
            logger.info("Phase 2b: Canonicalizing claims")
            claims = self.canonical.canonicalize(claims)
        else:
            # Embed nếu chưa có embedding
            if any(c.embedding is None for c in claims):
                logger.info("Phase 2a: Embedding %d pre-loaded claims", len(claims))
                self.embedder.embed(claims)

        logger.info("Using %d claims for pipeline", len(claims))

        # Phase 3: Build evidence graph
        logger.info("Phase 3: Building evidence graph")
        graph, edges = self.graph_builder.build(claims)

        # Phase 4: Index + retrieve
        logger.info("Phase 4: Hybrid retrieval")
        self.retriever.index(claims)
        edge_index = {(u, v): d.get("relation", "neutral") for u, v, d in graph.edges(data=True)}
        ranked = self.retriever.retrieve(query)
        top_k = self.selector.select(ranked, edge_index)

        # Phase 5+6: Iterative conflict resolution
        logger.info("Phase 5+6: Iterative conflict resolution")
        loop_result = self.loop.run(
            query=query,
            query_id=query_id,
            claims=claims,
            graph=graph,
            top_k_claims=top_k,
            edge_index=edge_index,
        )

        # Phase 7: Generate answer
        logger.info("Phase 7: Generating answer")
        final_graph_claims = loop_result.validated_claims or claims[:5]
        if final_graph_claims:
            final_graph, _ = self.graph_builder.build(final_graph_claims)
            final_analysis = self.conflict_analyzer.analyze(
                final_graph,
                final_graph_claims,
            )
        else:
            from .conflict_zone import ConflictAnalysisResult
            import networkx as nx
            final_analysis = ConflictAnalysisResult(
                credibility_scores={},
                conflict_localizations=[],
                validated_claim_ids=[],
                suppressed_claim_ids=[],
            )

        answer = self.answer_gen.generate(
            query=query,
            loop_result=loop_result,
            credibility_scores=final_analysis.credibility_scores,
        )
        loop_result.final_answer = answer

        logger.info(
            "Pipeline complete: resolved=%s, answer_len=%d",
            loop_result.resolved, len(answer),
        )
        logger.info("=" * 60)

        return loop_result
