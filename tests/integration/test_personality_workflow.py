"""Integration tests for personality mode flow through agent workflow nodes.

Verifies that personality modes set in state are correctly read by agent
nodes, passed to DSPy modules, and returned in the output dict.

Requirements: 3.3, 3.4, 5.1, 5.2
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from patent_system.agents.personality import AGENT_PERSONALITY_DEFAULTS, PersonalityMode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(**overrides) -> dict:
    """Create a minimal PatentWorkflowState dict for testing."""
    base: dict = {
        "topic_id": 1,
        "invention_disclosure": {
            "technical_problem": "Slow processing",
            "novel_features": ["Feature A"],
            "implementation_details": "Uses GPU",
            "potential_variations": ["CPU fallback"],
        },
        "interview_messages": [],
        "prior_art_results": [],
        "prior_art_summary": "Some prior art summary",
        "failed_sources": [],
        "novelty_analysis": "Novel: Feature A",
        "claims_text": "Claim 1: A method for fast processing.",
        "description_text": "Technical Field: Processing systems.",
        "review_feedback": "",
        "review_approved": False,
        "iteration_count": 0,
        "current_step": "",
        "personality_modes": {},
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Test: Personality modes persist through multiple agent nodes
# ---------------------------------------------------------------------------


class TestPersonalityModePersistsThroughSteps:
    """Verify that personality_modes in state remain constant across nodes."""

    @patch("patent_system.agents.claims_drafting.DraftClaimsModule")
    @patch("patent_system.agents.novelty_analysis.NoveltyAnalysisModule")
    @patch("patent_system.agents.consistency_review.ReviewConsistencyModule")
    def test_modes_persist_through_three_nodes(
        self,
        mock_review_cls,
        mock_novelty_cls,
        mock_claims_cls,
    ):
        """Run claims, novelty, and consistency nodes sequentially.

        The personality_modes dict in state should not be mutated by any node.
        """
        # Set up mocks
        mock_claims = MagicMock()
        mock_claims.return_value.claims_text = "Claim 1"
        mock_claims_cls.return_value = mock_claims

        mock_novelty = MagicMock()
        mock_novelty.return_value.novelty_assessment = "Novel"
        mock_novelty_cls.return_value = mock_novelty

        mock_review = MagicMock()
        mock_review.return_value.feedback = "OK"
        mock_review.return_value.approved = True
        mock_review_cls.return_value = mock_review

        modes = {
            "claims_drafting": "innovation_friendly",
            "novelty_analysis": "neutral",
            "consistency_review": "critical",
        }
        state = _make_state(personality_modes=modes)

        # Snapshot the modes dict before running nodes
        original_modes = dict(modes)

        from patent_system.agents.claims_drafting import claims_drafting_node
        from patent_system.agents.consistency_review import consistency_review_node
        from patent_system.agents.novelty_analysis import novelty_analysis_node

        result_claims = claims_drafting_node(state)
        result_novelty = novelty_analysis_node(state)
        result_review = consistency_review_node(state)

        # personality_modes in state must not have been mutated
        assert state["personality_modes"] == original_modes

        # Each node returns the correct mode it used
        assert result_claims["personality_mode_used"] == "innovation_friendly"
        assert result_novelty["personality_mode_used"] == "neutral"
        assert result_review["personality_mode_used"] == "critical"


# ---------------------------------------------------------------------------
# Test: Agent nodes pass personality_mode kwarg to DSPy modules
# ---------------------------------------------------------------------------


class TestAgentNodesPassModeToDSPy:
    """Verify each agent node passes personality_mode to its DSPy module."""

    @patch("patent_system.agents.claims_drafting.DraftClaimsModule")
    def test_claims_drafting_passes_mode(self, mock_cls):
        mock_instance = MagicMock()
        mock_instance.return_value.claims_text = "Claim 1"
        mock_cls.return_value = mock_instance

        from patent_system.agents.claims_drafting import claims_drafting_node

        state = _make_state(
            personality_modes={"claims_drafting": "innovation_friendly"}
        )
        result = claims_drafting_node(state)

        # Verify the DSPy module was called with personality_mode kwarg
        mock_instance.assert_called_once()
        call_kwargs = mock_instance.call_args
        assert call_kwargs.kwargs.get("personality_mode") == "innovation_friendly" or \
            "innovation_friendly" in str(call_kwargs)
        assert result["personality_mode_used"] == "innovation_friendly"

    @patch("patent_system.agents.novelty_analysis.NoveltyAnalysisModule")
    def test_novelty_analysis_passes_mode(self, mock_cls):
        mock_instance = MagicMock()
        mock_instance.return_value.novelty_assessment = "Assessment"
        mock_cls.return_value = mock_instance

        from patent_system.agents.novelty_analysis import novelty_analysis_node

        state = _make_state(
            personality_modes={"novelty_analysis": "neutral"}
        )
        result = novelty_analysis_node(state)

        mock_instance.assert_called_once()
        call_kwargs = mock_instance.call_args
        assert call_kwargs.kwargs.get("personality_mode") == "neutral" or \
            "neutral" in str(call_kwargs)
        assert result["personality_mode_used"] == "neutral"

    @patch("patent_system.agents.consistency_review.ReviewConsistencyModule")
    def test_consistency_review_passes_mode(self, mock_cls):
        mock_instance = MagicMock()
        mock_instance.return_value.feedback = "All good"
        mock_instance.return_value.approved = True
        mock_cls.return_value = mock_instance

        from patent_system.agents.consistency_review import consistency_review_node

        state = _make_state(
            personality_modes={"consistency_review": "innovation_friendly"}
        )
        result = consistency_review_node(state)

        mock_instance.assert_called_once()
        call_kwargs = mock_instance.call_args
        assert call_kwargs.kwargs.get("personality_mode") == "innovation_friendly" or \
            "innovation_friendly" in str(call_kwargs)
        assert result["personality_mode_used"] == "innovation_friendly"

    @patch("patent_system.agents.market_potential.MarketPotentialModule")
    def test_market_potential_passes_mode(self, mock_cls):
        mock_instance = MagicMock()
        mock_instance.return_value.market_assessment = "High potential"
        mock_cls.return_value = mock_instance

        from patent_system.agents.market_potential import market_potential_node

        state = _make_state(
            personality_modes={"market_potential": "critical"}
        )
        result = market_potential_node(state)

        mock_instance.assert_called_once()
        call_kwargs = mock_instance.call_args
        assert call_kwargs.kwargs.get("personality_mode") == "critical" or \
            "critical" in str(call_kwargs)
        assert result["personality_mode_used"] == "critical"

    @patch("patent_system.agents.legal_clarification.LegalClarificationModule")
    def test_legal_clarification_passes_mode(self, mock_cls):
        mock_instance = MagicMock()
        mock_instance.return_value.legal_assessment = "No issues"
        mock_cls.return_value = mock_instance

        from patent_system.agents.legal_clarification import legal_clarification_node

        state = _make_state(
            personality_modes={"legal_clarification": "neutral"}
        )
        result = legal_clarification_node(state)

        mock_instance.assert_called_once()
        call_kwargs = mock_instance.call_args
        assert call_kwargs.kwargs.get("personality_mode") == "neutral" or \
            "neutral" in str(call_kwargs)
        assert result["personality_mode_used"] == "neutral"

    @patch("patent_system.agents.disclosure_summary.DisclosureSummaryModule")
    def test_disclosure_summary_passes_mode(self, mock_cls):
        mock_instance = MagicMock()
        mock_instance.return_value.disclosure_summary = "Summary text"
        mock_cls.return_value = mock_instance

        from patent_system.agents.disclosure_summary import disclosure_summary_node

        state = _make_state(
            personality_modes={"disclosure_summary": "innovation_friendly"}
        )
        result = disclosure_summary_node(state)

        mock_instance.assert_called_once()
        call_kwargs = mock_instance.call_args
        assert call_kwargs.kwargs.get("personality_mode") == "innovation_friendly" or \
            "innovation_friendly" in str(call_kwargs)
        assert result["personality_mode_used"] == "innovation_friendly"


# ---------------------------------------------------------------------------
# Test: Mode remains constant across all nodes within a single run
# ---------------------------------------------------------------------------


class TestModeConstantAcrossRun:
    """Verify that the same personality_modes dict produces consistent results
    across all agent nodes in a single simulated workflow run."""

    @patch("patent_system.agents.description_drafting.DraftDescriptionModule")
    @patch("patent_system.agents.description_drafting.RefineClaimsModule")
    @patch("patent_system.agents.disclosure_summary.DisclosureSummaryModule")
    @patch("patent_system.agents.legal_clarification.LegalClarificationModule")
    @patch("patent_system.agents.market_potential.MarketPotentialModule")
    @patch("patent_system.agents.consistency_review.ReviewConsistencyModule")
    @patch("patent_system.agents.novelty_analysis.NoveltyAnalysisModule")
    @patch("patent_system.agents.claims_drafting.DraftClaimsModule")
    def test_all_nodes_use_assigned_modes(
        self,
        mock_claims_cls,
        mock_novelty_cls,
        mock_review_cls,
        mock_market_cls,
        mock_legal_cls,
        mock_summary_cls,
        mock_refine_cls,
        mock_desc_cls,
    ):
        # Configure all mocks
        mock_claims = MagicMock()
        mock_claims.return_value.claims_text = "Claim 1"
        mock_claims_cls.return_value = mock_claims

        mock_novelty = MagicMock()
        mock_novelty.return_value.novelty_assessment = "Novel"
        mock_novelty_cls.return_value = mock_novelty

        mock_review = MagicMock()
        mock_review.return_value.feedback = "OK"
        mock_review.return_value.approved = True
        mock_review_cls.return_value = mock_review

        mock_market = MagicMock()
        mock_market.return_value.market_assessment = "Good"
        mock_market_cls.return_value = mock_market

        mock_legal = MagicMock()
        mock_legal.return_value.legal_assessment = "Clear"
        mock_legal_cls.return_value = mock_legal

        mock_summary = MagicMock()
        mock_summary.return_value.disclosure_summary = "Summary"
        mock_summary_cls.return_value = mock_summary

        mock_desc = MagicMock()
        mock_desc.return_value.description_text = "Description"
        mock_desc_cls.return_value = mock_desc

        # All modes set to innovation_friendly for this run
        modes = {
            "claims_drafting": "innovation_friendly",
            "novelty_analysis": "innovation_friendly",
            "consistency_review": "innovation_friendly",
            "market_potential": "innovation_friendly",
            "legal_clarification": "innovation_friendly",
            "disclosure_summary": "innovation_friendly",
            "patent_draft": "innovation_friendly",
        }
        state = _make_state(personality_modes=modes)

        from patent_system.agents.claims_drafting import claims_drafting_node
        from patent_system.agents.consistency_review import consistency_review_node
        from patent_system.agents.description_drafting import description_drafting_node
        from patent_system.agents.disclosure_summary import disclosure_summary_node
        from patent_system.agents.legal_clarification import legal_clarification_node
        from patent_system.agents.market_potential import market_potential_node
        from patent_system.agents.novelty_analysis import novelty_analysis_node

        results = [
            claims_drafting_node(state),
            novelty_analysis_node(state),
            consistency_review_node(state),
            market_potential_node(state),
            legal_clarification_node(state),
            disclosure_summary_node(state),
            description_drafting_node(state),
        ]

        # Every node must report the same mode
        for result in results:
            assert result["personality_mode_used"] == "innovation_friendly"

        # State must not have been mutated
        assert state["personality_modes"] == modes

    @patch("patent_system.agents.claims_drafting.DraftClaimsModule")
    @patch("patent_system.agents.novelty_analysis.NoveltyAnalysisModule")
    def test_defaults_used_when_modes_empty(self, mock_novelty_cls, mock_claims_cls):
        """When personality_modes is empty, nodes fall back to AGENT_PERSONALITY_DEFAULTS."""
        mock_claims = MagicMock()
        mock_claims.return_value.claims_text = "Claim 1"
        mock_claims_cls.return_value = mock_claims

        mock_novelty = MagicMock()
        mock_novelty.return_value.novelty_assessment = "Novel"
        mock_novelty_cls.return_value = mock_novelty

        from patent_system.agents.claims_drafting import claims_drafting_node
        from patent_system.agents.novelty_analysis import novelty_analysis_node

        state = _make_state(personality_modes={})

        result_claims = claims_drafting_node(state)
        result_novelty = novelty_analysis_node(state)

        # Should use defaults from AGENT_PERSONALITY_DEFAULTS
        assert result_claims["personality_mode_used"] == AGENT_PERSONALITY_DEFAULTS["claims_drafting"].value
        assert result_novelty["personality_mode_used"] == AGENT_PERSONALITY_DEFAULTS["novelty_analysis"].value
