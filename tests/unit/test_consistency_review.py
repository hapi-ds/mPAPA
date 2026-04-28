"""Unit tests for the Consistency Reviewer Agent."""

import logging
from unittest.mock import MagicMock, patch

import pytest
import requests.exceptions

from patent_system.agents.consistency_review import consistency_review_node
from patent_system.exceptions import LLMConnectionError


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


class TestConsistencyReviewNode:
    """Tests for consistency_review_node."""

    @patch("patent_system.agents.consistency_review.ReviewConsistencyModule")
    def test_returns_required_keys(self, mock_module_cls):
        mock_instance = MagicMock()
        mock_instance.return_value.feedback = "All consistent."
        mock_instance.return_value.approved = True
        mock_module_cls.return_value = mock_instance

        state = _make_state(
            claims_text="Claim 1: A method...",
            description_text="The invention relates to...",
        )
        result = consistency_review_node(state)

        assert "review_feedback" in result
        assert "review_approved" in result
        assert result["current_step"] == "consistency_review"

    @patch("patent_system.agents.consistency_review.ReviewConsistencyModule")
    def test_returns_approved_true(self, mock_module_cls):
        mock_instance = MagicMock()
        mock_instance.return_value.feedback = "No issues found."
        mock_instance.return_value.approved = True
        mock_module_cls.return_value = mock_instance

        state = _make_state(
            claims_text="Claim 1: A device...",
            description_text="The device comprises...",
        )
        result = consistency_review_node(state)

        assert result["review_approved"] is True
        assert result["review_feedback"] == "No issues found."

    @patch("patent_system.agents.consistency_review.ReviewConsistencyModule")
    def test_returns_approved_false_with_feedback(self, mock_module_cls):
        mock_instance = MagicMock()
        mock_instance.return_value.feedback = (
            "Terminology mismatch: 'widget' in claims vs 'component' in description."
        )
        mock_instance.return_value.approved = False
        mock_module_cls.return_value = mock_instance

        state = _make_state(
            claims_text="Claim 1: A widget...",
            description_text="The component is...",
        )
        result = consistency_review_node(state)

        assert result["review_approved"] is False
        assert "Terminology mismatch" in result["review_feedback"]

    @patch("patent_system.agents.consistency_review.ReviewConsistencyModule")
    def test_passes_claims_and_description_to_module(self, mock_module_cls):
        mock_instance = MagicMock()
        mock_instance.return_value.feedback = "OK"
        mock_instance.return_value.approved = True
        mock_module_cls.return_value = mock_instance

        state = _make_state(
            claims_text="Claim 1: A method for processing data.",
            description_text="The present invention relates to data processing.",
        )
        consistency_review_node(state)

        mock_instance.assert_called_once_with(
            claims="Claim 1: A method for processing data.",
            description="The present invention relates to data processing.",
            personality_mode="critical",
            review_notes_text=None,
            domain_profile_slug="general-patent-drafting",
        )

    @patch("patent_system.agents.consistency_review.ReviewConsistencyModule")
    def test_handles_empty_claims_and_description(self, mock_module_cls):
        mock_instance = MagicMock()
        mock_instance.return_value.feedback = "No content to review."
        mock_instance.return_value.approved = False
        mock_module_cls.return_value = mock_instance

        state = _make_state(claims_text="", description_text="")
        result = consistency_review_node(state)

        assert result["review_approved"] is False
        mock_instance.assert_called_once_with(claims="", description="", personality_mode="critical", review_notes_text=None, domain_profile_slug="general-patent-drafting")

    @patch("patent_system.agents.consistency_review.ReviewConsistencyModule")
    def test_normalises_string_approved_true(self, mock_module_cls):
        mock_instance = MagicMock()
        mock_instance.return_value.feedback = "Looks good."
        mock_instance.return_value.approved = "True"
        mock_module_cls.return_value = mock_instance

        state = _make_state(claims_text="C", description_text="D")
        result = consistency_review_node(state)

        assert result["review_approved"] is True

    @patch("patent_system.agents.consistency_review.ReviewConsistencyModule")
    def test_normalises_string_approved_false(self, mock_module_cls):
        mock_instance = MagicMock()
        mock_instance.return_value.feedback = "Issues found."
        mock_instance.return_value.approved = "no"
        mock_module_cls.return_value = mock_instance

        state = _make_state(claims_text="C", description_text="D")
        result = consistency_review_node(state)

        assert result["review_approved"] is False

    @patch("patent_system.agents.consistency_review.ReviewConsistencyModule")
    def test_logs_agent_invocation(self, mock_module_cls, caplog):
        mock_instance = MagicMock()
        mock_instance.return_value.feedback = "OK"
        mock_instance.return_value.approved = True
        mock_module_cls.return_value = mock_instance

        state = _make_state(
            claims_text="Claim 1",
            description_text="Description text",
        )
        with caplog.at_level(
            logging.INFO, logger="patent_system.agents.consistency_review"
        ):
            consistency_review_node(state)

        assert any(
            "ConsistencyReviewerAgent" in r.message for r in caplog.records
        )

    @patch("patent_system.agents.consistency_review.dspy")
    @patch("patent_system.agents.consistency_review.ReviewConsistencyModule")
    def test_raises_llm_connection_error_on_connection_failure(
        self, mock_module_cls, mock_dspy
    ):
        mock_instance = MagicMock()
        mock_instance.side_effect = requests.exceptions.ConnectionError(
            "Connection refused"
        )
        mock_module_cls.return_value = mock_instance

        mock_lm = MagicMock()
        mock_lm.kwargs = {"api_base": "http://localhost:1234"}
        mock_dspy.settings.lm = mock_lm

        state = _make_state(
            claims_text="Claim 1",
            description_text="Description text",
        )
        with pytest.raises(LLMConnectionError, match="localhost:1234"):
            consistency_review_node(state)
