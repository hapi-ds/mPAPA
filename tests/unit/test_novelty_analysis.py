"""Unit tests for the Novelty Analysis Agent.

The node now uses DSPy NoveltyAnalysisModule (LLM-backed) instead of
the old deterministic placeholder. Tests mock the DSPy module.
"""

import logging
from unittest.mock import MagicMock, patch

import pytest

from patent_system.agents.novelty_analysis import (
    _PlaceholderRAG,
    novelty_analysis_node,
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
        "market_assessment": "",
        "legal_assessment": "",
        "disclosure_summary": "",
        "prior_art_summary": "",
        "workflow_step_statuses": {},
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# novelty_analysis_node (LLM-backed via DSPy)
# ---------------------------------------------------------------------------


class TestNoveltyAnalysisNode:
    """Tests for novelty_analysis_node with mocked DSPy module."""

    @patch("patent_system.agents.novelty_analysis.NoveltyAnalysisModule")
    def test_returns_required_keys(self, mock_module_cls):
        mock_instance = MagicMock()
        mock_instance.return_value.novelty_assessment = "Novel aspects found."
        mock_module_cls.return_value = mock_instance

        state = _make_state(
            invention_disclosure="Test invention",
            claims_text="Claim 1",
            prior_art_summary="Some prior art",
        )
        result = novelty_analysis_node(state)
        assert "novelty_analysis" in result
        assert result["current_step"] == "novelty_analysis"

    @patch("patent_system.agents.novelty_analysis.NoveltyAnalysisModule")
    def test_novelty_analysis_is_string(self, mock_module_cls):
        mock_instance = MagicMock()
        mock_instance.return_value.novelty_assessment = "Detailed analysis."
        mock_module_cls.return_value = mock_instance

        state = _make_state(invention_disclosure="Test")
        result = novelty_analysis_node(state)
        assert isinstance(result["novelty_analysis"], str)
        assert result["novelty_analysis"] == "Detailed analysis."

    @patch("patent_system.agents.novelty_analysis.NoveltyAnalysisModule")
    def test_passes_disclosure_claims_prior_art(self, mock_module_cls):
        mock_instance = MagicMock()
        mock_instance.return_value.novelty_assessment = "Assessment"
        mock_module_cls.return_value = mock_instance

        state = _make_state(
            invention_disclosure="My invention",
            claims_text="Claim 1: A method...",
            prior_art_summary="Found 5 references.",
        )
        novelty_analysis_node(state)

        mock_instance.assert_called_once()
        call_kwargs = mock_instance.call_args
        assert "My invention" in str(call_kwargs)
        assert "Claim 1: A method..." in str(call_kwargs)
        assert "Found 5 references." in str(call_kwargs)

    @patch("patent_system.agents.novelty_analysis.NoveltyAnalysisModule")
    def test_handles_dict_disclosure(self, mock_module_cls):
        """Dict disclosures are serialized to JSON string."""
        mock_instance = MagicMock()
        mock_instance.return_value.novelty_assessment = "Assessment"
        mock_module_cls.return_value = mock_instance

        disclosure = {"technical_problem": "Slow processing", "novel_features": ["GPU"]}
        state = _make_state(invention_disclosure=disclosure)
        novelty_analysis_node(state)

        call_args = str(mock_instance.call_args)
        assert "Slow processing" in call_args

    @patch("patent_system.agents.novelty_analysis.NoveltyAnalysisModule")
    def test_handles_none_disclosure(self, mock_module_cls):
        mock_instance = MagicMock()
        mock_instance.return_value.novelty_assessment = "No disclosure provided."
        mock_module_cls.return_value = mock_instance

        state = _make_state(invention_disclosure=None)
        result = novelty_analysis_node(state)
        assert result["current_step"] == "novelty_analysis"

    @patch("patent_system.agents.novelty_analysis.NoveltyAnalysisModule")
    def test_logs_agent_invocation(self, mock_module_cls, caplog):
        mock_instance = MagicMock()
        mock_instance.return_value.novelty_assessment = "Assessment"
        mock_module_cls.return_value = mock_instance

        state = _make_state(invention_disclosure="Test")
        with caplog.at_level(
            logging.INFO, logger="patent_system.agents.novelty_analysis"
        ):
            novelty_analysis_node(state)

        assert any(
            "NoveltyAnalysisAgent" in r.message for r in caplog.records
        )

    @patch("patent_system.agents.novelty_analysis.NoveltyAnalysisModule")
    def test_wraps_connection_error(self, mock_module_cls):
        mock_instance = MagicMock()
        mock_instance.side_effect = ConnectionError("refused")
        mock_module_cls.return_value = mock_instance

        from patent_system.exceptions import LLMConnectionError

        state = _make_state(invention_disclosure="Test")
        with pytest.raises(LLMConnectionError):
            novelty_analysis_node(state)


class TestPlaceholderRAG:
    """Tests for _PlaceholderRAG."""

    def test_returns_empty_list(self):
        rag = _PlaceholderRAG()
        assert rag.query(topic_id=1, query_text="test") == []
