"""
tests/test_smoke.py
Smoke tests — KHÔNG cần OpenAI hay GPU.
Tests schema, adapters, và logic components với mock data.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Helpers: mock Claim factory
# ---------------------------------------------------------------------------

def _unit_vector(dim: int, seed: int) -> list[float]:
    """Random unit vector with fixed seed."""
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    v /= np.linalg.norm(v)
    return v.tolist()


def _make_claim(claim_id: str, text: str, doc_id: str = "d001", seed: int = 0):
    from src.schema import Claim
    return Claim(
        claim_id=claim_id,
        doc_id=doc_id,
        text=text,
        embedding=_unit_vector(64, seed),
        retrieval_relevance=0.5,
        claim_confidence=0.8,
        source_credibility=0.7,
    )


# ---------------------------------------------------------------------------
# 1. Test schema validation
# ---------------------------------------------------------------------------

class TestSchemaValidation:
    def test_document_creation(self):
        from src.schema import Document
        doc = Document(doc_id="d001", source="wiki", text="The sky is blue.")
        assert doc.doc_id == "d001"
        assert doc.credibility_score == -1.0
        assert doc.url is None

    def test_claim_creation(self):
        from src.schema import Claim
        c = Claim(claim_id="c001", doc_id="d001", text="The sky is blue.")
        assert c.embedding is None
        assert c.retrieval_relevance == -1.0

    def test_edge_validation_valid(self):
        from src.schema import Edge
        e = Edge(edge_id="e001", claim_a="c001", claim_b="c002",
                 relation="contradiction", nli_score=0.9)
        assert e.relation == "contradiction"

    def test_edge_validation_invalid(self):
        from src.schema import Edge
        with pytest.raises(Exception):
            Edge(edge_id="e001", claim_a="c001", claim_b="c002",
                 relation="INVALID_RELATION", nli_score=0.9)

    def test_loop_result_creation(self):
        from src.schema import LoopResult
        result = LoopResult(
            query_id="q_001",
            iterations_run=2,
            resolved=True,
            validated_claims=[],
            conflict_localizations=[],
        )
        assert result.final_answer is None
        assert result.resolved is True

    def test_conflict_localization(self):
        from src.schema import ConflictLocalization
        loc = ConflictLocalization(
            claim_i_id="c001",
            claim_j_id="c002",
            slot="temporal",
            value_i="2019",
            value_j="2021",
            conflict_intensity=0.5,
            credibility_i=0.8,
            credibility_j=0.6,
        )
        assert loc.slot == "temporal"
        assert loc.conflict_intensity == 0.5

    def test_factoid_slots(self):
        from src.schema import FactoidSlots
        slots = FactoidSlots(temporal="2021", entity_subject="Biden")
        assert slots.temporal == "2021"
        assert slots.numerical is None


# ---------------------------------------------------------------------------
# 2. Test adapter (claim_from_record)
# ---------------------------------------------------------------------------

class TestAdapter:
    def test_claim_from_record_basic(self):
        from data.loaders import _claim_from_record
        rec = {
            "claim_id": "c_q000_d00_s00",
            "claim_text": "The sky is blue.",
            "doc_id": "d_abc123",
            "claim_embedding": None,
            "retrieval_relevance": 0.75,
            "claim_confidence": 0.85,
        }
        claim = _claim_from_record(rec)
        assert claim.claim_id == "c_q000_d00_s00"
        assert claim.text == "The sky is blue."
        assert claim.doc_id == "d_abc123"
        assert claim.retrieval_relevance == 0.75
        assert claim.claim_confidence == 0.85
        assert claim.source_credibility == -1.0
        assert claim.embedding is None

    def test_claim_from_record_with_embedding(self):
        from data.loaders import _claim_from_record
        emb = [0.1, 0.2, 0.3]
        rec = {
            "claim_id": "c_q001_d00_s01",
            "claim_text": "Water boils at 100C.",
            "doc_id": "d_xyz",
            "claim_embedding": emb,
            "retrieval_relevance": 0.6,
            "claim_confidence": 0.9,
        }
        claim = _claim_from_record(rec)
        assert claim.embedding == emb

    def test_doc_from_record_basic(self):
        from data.loaders import _doc_from_record
        rec = {
            "doc_id": "d_abc",
            "source_id": "s_q000_00",
            "text": "Some document text here.",
            "metadata": {"url": "http://example.com"},
        }
        doc = _doc_from_record(rec)
        assert doc.doc_id == "d_abc"
        assert doc.source == "s_q000_00"
        assert doc.url == "http://example.com"
        assert doc.credibility_score == -1.0

    def test_claim_from_record_defaults(self):
        """Test that missing optional fields use defaults."""
        from data.loaders import _claim_from_record
        rec = {
            "claim_id": "c_q000_d00_s00",
            "claim_text": "Default test.",
            "doc_id": "d001",
        }
        claim = _claim_from_record(rec)
        assert claim.retrieval_relevance == -1.0
        assert claim.claim_confidence == -1.0


# ---------------------------------------------------------------------------
# 3. Test ClaimCanonical with mock claims
# ---------------------------------------------------------------------------

class TestClaimCanonical:
    def test_canonicalize_identical_embeddings_merges(self):
        """Two claims with identical embeddings and no numbers → merged."""
        from src.embedder import ClaimCanonical
        from src.schema import Claim

        emb = _unit_vector(64, seed=0)
        c1 = Claim(claim_id="c001", doc_id="d1", text="The cat sat on the mat.",
                   embedding=emb)
        c2 = Claim(claim_id="c002", doc_id="d1", text="The cat sat on the mat.",
                   embedding=emb)  # exact same embedding

        canonical = ClaimCanonical(similarity_threshold=0.9)
        result = canonical.canonicalize([c1, c2])
        assert len(result) == 1

    def test_canonicalize_orthogonal_embeddings_keeps_both(self):
        """Two claims with orthogonal embeddings → both kept."""
        from src.embedder import ClaimCanonical

        # Build orthogonal unit vectors
        v1 = np.zeros(64, dtype=np.float32)
        v1[0] = 1.0
        v2 = np.zeros(64, dtype=np.float32)
        v2[1] = 1.0

        c1 = _make_claim("c001", "The cat sat on the mat.", seed=0)
        c2 = _make_claim("c002", "Quantum mechanics involves wave functions.", seed=1)
        c1.embedding = v1.tolist()
        c2.embedding = v2.tolist()

        canonical = ClaimCanonical(similarity_threshold=0.92)
        result = canonical.canonicalize([c1, c2])
        assert len(result) == 2

    def test_canonicalize_high_sim_different_numbers_keeps_both(self):
        """High similarity but different numbers → kept separate (conflict source)."""
        from src.embedder import ClaimCanonical

        # Create high-similarity embedding (nearly the same)
        base = _unit_vector(64, seed=42)
        perturbed = [x + 0.001 * i for i, x in enumerate(base)]
        norm = math.sqrt(sum(x**2 for x in perturbed))
        perturbed = [x / norm for x in perturbed]

        c1 = _make_claim("c001", "The event happened in 2019.", seed=0)
        c2 = _make_claim("c002", "The event happened in 2021.", seed=0)
        c1.embedding = base
        c2.embedding = perturbed  # very similar embedding but different year

        canonical = ClaimCanonical(similarity_threshold=0.5)  # low threshold so they'd merge
        result = canonical.canonicalize([c1, c2])
        # Different numbers (2019 vs 2021) → should NOT merge
        assert len(result) == 2

    def test_canonicalize_single_claim(self):
        """Single claim passes through unchanged."""
        from src.embedder import ClaimCanonical

        c = _make_claim("c001", "Only one claim.", seed=0)
        canonical = ClaimCanonical()
        result = canonical.canonicalize([c])
        assert len(result) == 1
        assert result[0].claim_id == "c001"

    def test_canonicalize_missing_embedding_raises(self):
        """Claim without embedding raises ValueError."""
        from src.embedder import ClaimCanonical
        from src.schema import Claim

        c1 = Claim(claim_id="c001", doc_id="d1", text="No embedding.", embedding=None)
        c2 = _make_claim("c002", "Has embedding.", seed=0)

        canonical = ClaimCanonical()
        with pytest.raises(ValueError, match="no embedding"):
            canonical.canonicalize([c1, c2])


# ---------------------------------------------------------------------------
# 4. Test PairGenerator
# ---------------------------------------------------------------------------

class TestPairGenerator:
    def test_pair_generator_high_similarity(self):
        """Two identical embeddings → one pair generated."""
        from src.graph_builder import PairGenerator

        emb = _unit_vector(64, seed=0)
        c1 = _make_claim("c001", "Claim A", seed=0)
        c2 = _make_claim("c002", "Claim B", seed=1)
        c1.embedding = emb
        c2.embedding = emb  # identical → sim=1.0

        gen = PairGenerator(similarity_threshold=0.5)
        pairs = gen.generate([c1, c2])
        assert len(pairs) == 1
        assert {pairs[0][0].claim_id, pairs[0][1].claim_id} == {"c001", "c002"}

    def test_pair_generator_low_similarity(self):
        """Orthogonal embeddings → no pairs."""
        from src.graph_builder import PairGenerator

        v1 = np.zeros(64, dtype=np.float32)
        v1[0] = 1.0
        v2 = np.zeros(64, dtype=np.float32)
        v2[1] = 1.0

        c1 = _make_claim("c001", "Claim A", seed=0)
        c2 = _make_claim("c002", "Claim B", seed=1)
        c1.embedding = v1.tolist()
        c2.embedding = v2.tolist()

        gen = PairGenerator(similarity_threshold=0.5)
        pairs = gen.generate([c1, c2])
        assert len(pairs) == 0

    def test_pair_generator_single_claim(self):
        """Single claim → no pairs."""
        from src.graph_builder import PairGenerator

        c = _make_claim("c001", "Only one claim.", seed=0)
        gen = PairGenerator()
        pairs = gen.generate([c])
        assert pairs == []

    def test_pair_generator_multiple_claims(self):
        """N claims with all similar embeddings → N*(N-1)/2 pairs."""
        from src.graph_builder import PairGenerator

        emb = _unit_vector(64, seed=0)
        claims = [_make_claim(f"c{i:03d}", f"Claim {i}") for i in range(4)]
        for c in claims:
            c.embedding = emb

        gen = PairGenerator(similarity_threshold=0.5)
        pairs = gen.generate(claims)
        assert len(pairs) == 6  # C(4,2) = 6


# ---------------------------------------------------------------------------
# 5. Test CredibilityArbitrator with mock graph
# ---------------------------------------------------------------------------

class TestCredibilityArbitrator:
    def _make_graph_with_contradiction(self):
        """Build a simple graph: c001 --contradiction--> c002."""
        import networkx as nx
        G = nx.DiGraph()
        G.add_node("c001")
        G.add_node("c002")
        G.add_edge("c001", "c002", relation="contradiction", nli_score=0.9)
        return G

    def _make_graph_with_support(self):
        """Build a simple graph: c001 --support--> c002."""
        import networkx as nx
        G = nx.DiGraph()
        G.add_node("c001")
        G.add_node("c002")
        G.add_edge("c001", "c002", relation="support", nli_score=0.9)
        return G

    def test_arbitration_returns_all_nodes(self):
        """Arbitration returns scores for all graph nodes."""
        from src.conflict_zone import CredibilityArbitrator
        arb = CredibilityArbitrator(max_iterations=5)
        G = self._make_graph_with_contradiction()
        scores = arb.compute(G)
        assert "c001" in scores
        assert "c002" in scores

    def test_arbitration_contradiction_reduces_score(self):
        """Node receiving contradiction edge should have reduced score."""
        from src.conflict_zone import CredibilityArbitrator
        arb = CredibilityArbitrator(max_iterations=10, damping=0.85)
        G = self._make_graph_with_contradiction()
        scores = arb.compute(G)
        # c002 receives contradiction from c001 → should have lower score than c001
        assert scores["c002"] < scores["c001"]

    def test_arbitration_support_increases_score(self):
        """Node receiving support edge should have positive score."""
        from src.conflict_zone import CredibilityArbitrator
        arb = CredibilityArbitrator(max_iterations=10, damping=0.85)
        G = self._make_graph_with_support()
        scores = arb.compute(G)
        # c002 receives support from c001 → should have higher score than c001
        assert scores["c002"] > scores["c001"]

    def test_arbitration_empty_graph(self):
        """Empty graph returns empty scores."""
        import networkx as nx
        from src.conflict_zone import CredibilityArbitrator
        arb = CredibilityArbitrator()
        G = nx.DiGraph()
        scores = arb.compute(G)
        assert scores == {}

    def test_arbitration_convergence(self):
        """Arbitrator should converge without error on a complex graph."""
        import networkx as nx
        from src.conflict_zone import CredibilityArbitrator
        arb = CredibilityArbitrator(max_iterations=20, convergence_threshold=0.001)
        G = nx.DiGraph()
        for i in range(5):
            G.add_node(f"c{i:03d}")
        G.add_edge("c000", "c001", relation="support", nli_score=0.8)
        G.add_edge("c001", "c002", relation="contradiction", nli_score=0.7)
        G.add_edge("c002", "c003", relation="support", nli_score=0.9)
        G.add_edge("c003", "c004", relation="neutral", nli_score=0.6)
        scores = arb.compute(G)
        assert len(scores) == 5
        for score in scores.values():
            assert math.isfinite(score)


# ---------------------------------------------------------------------------
# 6. Test ConflictQueryFormulator
# ---------------------------------------------------------------------------

class TestConflictQueryFormulator:
    def _make_localization(self, slot: str, val_i: str, val_j: str):
        from src.schema import ConflictLocalization
        return ConflictLocalization(
            claim_i_id="c001",
            claim_j_id="c002",
            slot=slot,
            value_i=val_i,
            value_j=val_j,
            conflict_intensity=0.5,
            credibility_i=0.8,
            credibility_j=0.6,
        )

    def test_temporal_slot_query(self):
        from src.iterative_loop import ConflictQueryFormulator
        formulator = ConflictQueryFormulator()
        loc = self._make_localization("temporal", "2019", "2021")
        claim_texts = {"c001": "Event happened in 2019.", "c002": "Event happened in 2021."}
        query = formulator.formulate("When did the event happen?", loc, claim_texts)
        assert "2019" in query
        assert "2021" in query
        assert "temporal" in query.lower() or "year" in query.lower()

    def test_numerical_slot_query(self):
        from src.iterative_loop import ConflictQueryFormulator
        formulator = ConflictQueryFormulator()
        loc = self._make_localization("numerical", "50km", "80km")
        claim_texts = {}
        query = formulator.formulate("How far is the city?", loc, claim_texts)
        assert "50km" in query
        assert "80km" in query

    def test_entity_slot_query(self):
        from src.iterative_loop import ConflictQueryFormulator
        formulator = ConflictQueryFormulator()
        loc = self._make_localization("entity_subject", "Einstein", "Newton")
        claim_texts = {}
        query = formulator.formulate("Who discovered gravity?", loc, claim_texts)
        assert "Einstein" in query
        assert "Newton" in query

    def test_location_slot_query(self):
        from src.iterative_loop import ConflictQueryFormulator
        formulator = ConflictQueryFormulator()
        loc = self._make_localization("location", "Paris", "London")
        claim_texts = {}
        query = formulator.formulate("Where was the meeting?", loc, claim_texts)
        assert "Paris" in query
        assert "London" in query

    def test_unknown_slot_query(self):
        from src.iterative_loop import ConflictQueryFormulator
        formulator = ConflictQueryFormulator()
        loc = self._make_localization("unknown", "value_a", "value_b")
        claim_texts = {
            "c001": "Claim text A.",
            "c002": "Claim text B.",
        }
        query = formulator.formulate("Some query", loc, claim_texts)
        assert len(query) > 0


# ---------------------------------------------------------------------------
# 7. Test EvidenceSelector
# ---------------------------------------------------------------------------

class TestEvidenceSelector:
    def test_select_returns_at_most_max_claims(self):
        from src.generator import EvidenceSelector

        claims = [_make_claim(f"c{i:03d}", f"Claim text {i}.", seed=i) for i in range(15)]
        selector = EvidenceSelector(max_claims=5)
        selected = selector.select(claims)
        assert len(selected) <= 5

    def test_select_empty_returns_empty(self):
        from src.generator import EvidenceSelector

        selector = EvidenceSelector(max_claims=10)
        selected = selector.select([])
        assert selected == []

    def test_select_orders_by_credibility(self):
        from src.generator import EvidenceSelector

        claims = [_make_claim(f"c{i:03d}", f"Claim {i}.", seed=i) for i in range(5)]
        # Set all source_credibility to same value so only credibility_scores matter
        for c in claims:
            c.source_credibility = -1.0  # will be treated as 0.5

        # Use scores spread within normalized range: (cred+2)/4 saturates at 1 for cred>=2
        # Use sub-1 values to keep ordering clear
        credibility_scores = {
            "c000": -1.0,   # cred_norm = 0.25
            "c001": -0.5,   # cred_norm = 0.375
            "c002": 0.0,    # cred_norm = 0.5
            "c003": 0.5,    # cred_norm = 0.625
            "c004": 1.0,    # cred_norm = 0.75
        }
        selector = EvidenceSelector(max_claims=5, credibility_weight=1.0, relevance_weight=0.0)
        selected = selector.select(claims, credibility_scores)

        # Should be ordered by credibility descending
        selected_ids = [c.claim_id for c in selected]
        assert selected_ids[0] == "c004"  # highest credibility

    def test_select_uses_credibility_scores_override(self):
        from src.generator import EvidenceSelector

        claims = [_make_claim(f"c{i:03d}", f"Claim {i}.", seed=i) for i in range(3)]
        credibility_scores = {
            "c000": 10.0,   # highest
            "c001": 5.0,
            "c002": 1.0,
        }
        selector = EvidenceSelector(max_claims=3, credibility_weight=1.0, relevance_weight=0.0)
        selected = selector.select(claims, credibility_scores)
        assert selected[0].claim_id == "c000"

    def test_select_with_negative_relevance(self):
        """Claims with default -1.0 retrieval_relevance handled gracefully."""
        from src.generator import EvidenceSelector

        claims = [_make_claim(f"c{i:03d}", f"Claim {i}.", seed=i) for i in range(3)]
        for c in claims:
            c.retrieval_relevance = -1.0

        selector = EvidenceSelector(max_claims=5)
        selected = selector.select(claims)
        assert len(selected) == 3
