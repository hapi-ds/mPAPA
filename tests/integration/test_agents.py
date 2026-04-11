"""Integration tests for agent node functions with mocked DSPy modules.

Tests each agent node function to verify they return the expected
state updates when DSPy modules are mocked.

Requirements: 9.1, 9.2, 9.3
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(**overrides) -> dict:
    """Create a minimal PatentWorkflowState dict for testing."""
    base: dict = {
        "topic_id": 1,
        "invention_disclosure": None,
        "interview_messages": [],
        "prior_art_results": [],
        "failed_sources": [],
        "novelty_analysis": None,
        "claims_text": "",
        "description_text": "",
        "review_feedback": "",
        "review_approved": False,
        "iteration_count": 0,
        "current_step": "",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Disclosure Agent
# ---------------------------------------------------------------------------


class TestDisclosureNode:
    """Test disclosure_node with mocked DSPy modules."""

    @patch("patent_system.agents.disclosure.StructureDisclosureModule")
    @patch("patent_system.agents.disclosure.InterviewQuestionModule")
    def test_returns_disclosure_and_step(self, mock_interview_cls, mock_structure_cls):
        mock_interview = MagicMock()
        mock_interview.return_value.next_question = "What problem does it solve?"
        mock_interview_cls.return_value = mock_interview

        mock_structure = MagicMock()
        mock_structure.return_value.disclosure_json = (
            '{"technical_problem": "Slow processing", '
            '"novel_features": ["Feature A"], '
            '"implementation_details": "Uses GPU", '
            '"potential_variations": ["CPU fallback"]}'
        )
        mock_structure_cls.return_value = mock_structure

        from patent_system.agents.disclosure import disclosure_node

        state = _make_state(topic_id=10)
        result = disclosure_node(state)

        assert result["current_step"] == "disclosure"
        assert result["invention_disclosure"] is not None
        assert result["invention_disclosure"]["technical_problem"] == "Slow processing"

    @patch("patent_system.agents.disclosure.StructureDisclosureModule")
    @patch("patent_system.agents.disclosure.InterviewQuestionModule")
    def test_handles_invalid_json_disclosure(self, mock_interview_cls, mock_structure_cls):
        mock_interview = MagicMock()
        mock_interview.return_value.next_question = "Tell me more?"
        mock_interview_cls.return_value = mock_interview

        mock_structure = MagicMock()
        mock_structure.return_value.disclosure_json = "not valid json"
        mock_structure_cls.return_value = mock_structure

        from patent_system.agents.disclosure import disclosure_node

        state = _make_state()
        result = disclosure_node(state)

        assert result["current_step"] == "disclosure"
        # Falls back to wrapping the raw string
        assert "technical_problem" in result["invention_disclosure"]


# ---------------------------------------------------------------------------
# Prior Art Search Agent
# ---------------------------------------------------------------------------


class TestPriorArtSearchNode:
    """Test prior_art_search_node with mocked external queries."""

    @patch("patent_system.agents.prior_art_search._query_source")
    def test_returns_results_and_step(self, mock_query):
        mock_query.return_value = {"results": []}

        from patent_system.agents.prior_art_search import prior_art_search_node

        state = _make_state(
            invention_disclosure={
                "technical_problem": "Battery drain",
                "novel_features": ["Low-power mode"],
            }
        )
        result = prior_art_search_node(state)

        assert result["current_step"] == "prior_art_search"
        assert isinstance(result["prior_art_results"], list)
        assert isinstance(result["failed_sources"], list)

    @patch("patent_system.agents.prior_art_search._query_source")
    def test_handles_source_failure_gracefully(self, mock_query):
        mock_query.side_effect = ConnectionError("timeout")

        from patent_system.agents.prior_art_search import prior_art_search_node

        state = _make_state()
        result = prior_art_search_node(state)

        assert result["current_step"] == "prior_art_search"
        assert len(result["failed_sources"]) > 0
        # All 5 sources should fail
        assert len(result["failed_sources"]) == 5


# ---------------------------------------------------------------------------
# Novelty Analysis Agent
# ---------------------------------------------------------------------------


class TestNoveltyAnalysisNode:
    """Test novelty_analysis_node with a mock RAG engine."""

    def test_returns_analysis_and_step(self):
        from patent_system.agents.novelty_analysis import novelty_analysis_node

        mock_rag = MagicMock()
        mock_rag.query.return_value = []

        state = _make_state(
            invention_disclosure={
                "technical_problem": "Slow indexing",
                "novel_features": ["Parallel indexer"],
            }
        )
        result = novelty_analysis_node(state, rag_engine=mock_rag)

        assert result["current_step"] == "novelty_analysis"
        assert result["novelty_analysis"] is not None
        assert "novel_aspects" in result["novelty_analysis"]

    def test_uses_placeholder_rag_when_none(self):
        from patent_system.agents.novelty_analysis import novelty_analysis_node

        state = _make_state(
            invention_disclosure={
                "technical_problem": "Problem",
                "novel_features": ["Feature"],
            }
        )
        result = novelty_analysis_node(state, rag_engine=None)

        assert result["current_step"] == "novelty_analysis"
        assert result["novelty_analysis"] is not None


# ---------------------------------------------------------------------------
# Claims Drafting Agent
# ---------------------------------------------------------------------------


class TestClaimsDraftingNode:
    """Test claims_drafting_node with mocked DSPy module."""

    @patch("patent_system.agents.claims_drafting.DraftClaimsModule")
    def test_returns_claims_and_increments_iteration(self, mock_module_cls):
        mock_instance = MagicMock()
        mock_instance.return_value.claims_text = "Anspruch 1: Ein Verfahren..."
        mock_module_cls.return_value = mock_instance

        from patent_system.agents.claims_drafting import claims_drafting_node

        state = _make_state(iteration_count=1)
        result = claims_drafting_node(state)

        assert result["current_step"] == "claims_drafting"
        assert result["claims_text"] == "Anspruch 1: Ein Verfahren..."
        assert result["iteration_count"] == 2


# ---------------------------------------------------------------------------
# Consistency Review Agent
# ---------------------------------------------------------------------------


class TestConsistencyReviewNode:
    """Test consistency_review_node with mocked DSPy module."""

    @patch("patent_system.agents.consistency_review.ReviewConsistencyModule")
    def test_returns_approved_status(self, mock_module_cls):
        mock_instance = MagicMock()
        mock_instance.return_value.feedback = "All consistent"
        mock_instance.return_value.approved = True
        mock_module_cls.return_value = mock_instance

        from patent_system.agents.consistency_review import consistency_review_node

        state = _make_state(claims_text="Claim 1", description_text="Description")
        result = consistency_review_node(state)

        assert result["current_step"] == "consistency_review"
        assert result["review_approved"] is True
        assert result["review_feedback"] == "All consistent"

    @patch("patent_system.agents.consistency_review.ReviewConsistencyModule")
    def test_returns_not_approved_with_feedback(self, mock_module_cls):
        mock_instance = MagicMock()
        mock_instance.return_value.feedback = "Terminology mismatch in claim 2"
        mock_instance.return_value.approved = False
        mock_module_cls.return_value = mock_instance

        from patent_system.agents.consistency_review import consistency_review_node

        state = _make_state(claims_text="Claim 1", description_text="Desc")
        result = consistency_review_node(state)

        assert result["review_approved"] is False
        assert "Terminology mismatch" in result["review_feedback"]

    @patch("patent_system.agents.consistency_review.ReviewConsistencyModule")
    def test_normalizes_string_approved_to_bool(self, mock_module_cls):
        mock_instance = MagicMock()
        mock_instance.return_value.feedback = "OK"
        mock_instance.return_value.approved = "true"
        mock_module_cls.return_value = mock_instance

        from patent_system.agents.consistency_review import consistency_review_node

        state = _make_state()
        result = consistency_review_node(state)

        assert result["review_approved"] is True


# ---------------------------------------------------------------------------
# Description Drafting Agent
# ---------------------------------------------------------------------------


class TestDescriptionDraftingNode:
    """Test description_drafting_node with mocked DSPy module."""

    @patch("patent_system.agents.description_drafting.DraftDescriptionModule")
    def test_returns_description_and_step(self, mock_module_cls):
        mock_instance = MagicMock()
        mock_instance.return_value.description_text = (
            "Technical Field\nThis invention relates to..."
        )
        mock_module_cls.return_value = mock_instance

        from patent_system.agents.description_drafting import description_drafting_node

        state = _make_state(
            claims_text="Claim 1: A method...",
            prior_art_results=[{"title": "Prior Art 1"}],
            invention_disclosure={"technical_problem": "Problem"},
        )
        result = description_drafting_node(state)

        assert result["current_step"] == "description_drafting"
        assert "Technical Field" in result["description_text"]

    @patch("patent_system.agents.description_drafting.DraftDescriptionModule")
    def test_handles_empty_state(self, mock_module_cls):
        mock_instance = MagicMock()
        mock_instance.return_value.description_text = "Minimal description"
        mock_module_cls.return_value = mock_instance

        from patent_system.agents.description_drafting import description_drafting_node

        state = _make_state()
        result = description_drafting_node(state)

        assert result["current_step"] == "description_drafting"
        assert result["description_text"] == "Minimal description"
