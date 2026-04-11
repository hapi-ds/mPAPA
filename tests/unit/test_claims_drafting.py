"""Unit tests for the Claims Drafting Agent."""

import logging
from unittest.mock import MagicMock, patch

import pytest

from patent_system.agents.claims_drafting import (
    _prepare_disclosure_text,
    _prepare_novelty_text,
    claims_drafting_node,
)


def _make_state(**overrides):
    """Create a minimal PatentWorkflowState dict for testing."""
    base = {
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
# _prepare_disclosure_text
# ---------------------------------------------------------------------------


class TestPrepareDisclosureText:
    """Tests for _prepare_disclosure_text."""

    def test_none_returns_empty(self):
        assert _prepare_disclosure_text(None) == ""

    def test_empty_dict_returns_empty(self):
        assert _prepare_disclosure_text({}) == ""

    def test_serializes_disclosure(self):
        disclosure = {
            "technical_problem": "Slow processing",
            "novel_features": ["Feature A"],
        }
        result = _prepare_disclosure_text(disclosure)
        assert "Slow processing" in result
        assert "Feature A" in result


# ---------------------------------------------------------------------------
# _prepare_novelty_text
# ---------------------------------------------------------------------------


class TestPrepareNoveltyText:
    """Tests for _prepare_novelty_text."""

    def test_none_returns_empty(self):
        assert _prepare_novelty_text(None) == ""

    def test_empty_dict_returns_empty(self):
        assert _prepare_novelty_text({}) == ""

    def test_serializes_novelty(self):
        novelty = {
            "novel_aspects": ["Aspect A"],
            "potential_conflicts": [],
            "suggested_claim_scope": "Focus on Aspect A",
        }
        result = _prepare_novelty_text(novelty)
        assert "Aspect A" in result
        assert "suggested_claim_scope" in result


# ---------------------------------------------------------------------------
# claims_drafting_node
# ---------------------------------------------------------------------------


class TestClaimsDraftingNode:
    """Tests for claims_drafting_node."""

    @patch("patent_system.agents.claims_drafting.DraftClaimsModule")
    def test_returns_required_keys(self, mock_module_cls):
        mock_instance = MagicMock()
        mock_instance.return_value.claims_text = "Claim 1: A method..."
        mock_module_cls.return_value = mock_instance

        state = _make_state()
        result = claims_drafting_node(state)

        assert "claims_text" in result
        assert "iteration_count" in result
        assert result["current_step"] == "claims_drafting"

    @patch("patent_system.agents.claims_drafting.DraftClaimsModule")
    def test_increments_iteration_count(self, mock_module_cls):
        mock_instance = MagicMock()
        mock_instance.return_value.claims_text = "Claims text"
        mock_module_cls.return_value = mock_instance

        state = _make_state(iteration_count=0)
        result = claims_drafting_node(state)
        assert result["iteration_count"] == 1

    @patch("patent_system.agents.claims_drafting.DraftClaimsModule")
    def test_increments_from_existing_count(self, mock_module_cls):
        mock_instance = MagicMock()
        mock_instance.return_value.claims_text = "Claims text"
        mock_module_cls.return_value = mock_instance

        state = _make_state(iteration_count=2)
        result = claims_drafting_node(state)
        assert result["iteration_count"] == 3

    @patch("patent_system.agents.claims_drafting.DraftClaimsModule")
    def test_passes_disclosure_and_novelty_to_module(self, mock_module_cls):
        mock_instance = MagicMock()
        mock_instance.return_value.claims_text = "Generated claims"
        mock_module_cls.return_value = mock_instance

        disclosure = {"technical_problem": "Problem X", "novel_features": ["F1"]}
        novelty = {"novel_aspects": ["F1"], "potential_conflicts": [], "suggested_claim_scope": "Scope"}
        state = _make_state(invention_disclosure=disclosure, novelty_analysis=novelty)

        claims_drafting_node(state)

        mock_instance.assert_called_once()
        call_kwargs = mock_instance.call_args
        assert "Problem X" in call_kwargs.kwargs["invention_disclosure"]
        assert "F1" in call_kwargs.kwargs["novelty_analysis"]

    @patch("patent_system.agents.claims_drafting.DraftClaimsModule")
    def test_returns_claims_text_from_module(self, mock_module_cls):
        mock_instance = MagicMock()
        mock_instance.return_value.claims_text = "Anspruch 1: Ein Verfahren..."
        mock_module_cls.return_value = mock_instance

        state = _make_state()
        result = claims_drafting_node(state)
        assert result["claims_text"] == "Anspruch 1: Ein Verfahren..."

    @patch("patent_system.agents.claims_drafting.DraftClaimsModule")
    def test_handles_none_disclosure_and_novelty(self, mock_module_cls):
        mock_instance = MagicMock()
        mock_instance.return_value.claims_text = "Fallback claims"
        mock_module_cls.return_value = mock_instance

        state = _make_state(invention_disclosure=None, novelty_analysis=None)
        result = claims_drafting_node(state)

        assert result["claims_text"] == "Fallback claims"
        call_kwargs = mock_instance.call_args
        assert call_kwargs.kwargs["invention_disclosure"] == ""
        assert call_kwargs.kwargs["novelty_analysis"] == ""

    @patch("patent_system.agents.claims_drafting.DraftClaimsModule")
    def test_logs_agent_invocation(self, mock_module_cls, caplog):
        mock_instance = MagicMock()
        mock_instance.return_value.claims_text = "Claims"
        mock_module_cls.return_value = mock_instance

        state = _make_state()
        with caplog.at_level(
            logging.INFO, logger="patent_system.agents.claims_drafting"
        ):
            claims_drafting_node(state)

        assert any(
            "ClaimsDraftingAgent" in r.message for r in caplog.records
        )
