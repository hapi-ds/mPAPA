"""Unit tests for the Invention Disclosure Agent."""

import json
import logging
from unittest.mock import MagicMock, patch

import pytest

from patent_system.agents.disclosure import (
    _INTERVIEW_TOPICS,
    disclosure_node,
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


class TestDisclosureNode:
    """Tests for the disclosure_node function."""

    @patch("patent_system.agents.disclosure.StructureDisclosureModule")
    @patch("patent_system.agents.disclosure.InterviewQuestionModule")
    def test_returns_disclosure_and_current_step(
        self, mock_interview_cls, mock_structure_cls
    ):
        """disclosure_node returns dict with invention_disclosure and current_step."""
        mock_interview = MagicMock()
        mock_interview.return_value.next_question = "What problem does it solve?"
        mock_interview_cls.return_value = mock_interview

        disclosure_data = {
            "technical_problem": "Slow data processing",
            "novel_features": ["parallel pipeline"],
            "implementation_details": "Uses GPU acceleration",
            "potential_variations": ["CPU fallback"],
        }
        mock_structure = MagicMock()
        mock_structure.return_value.disclosure_json = json.dumps(disclosure_data)
        mock_structure_cls.return_value = mock_structure

        state = _make_state()
        result = disclosure_node(state)

        assert "invention_disclosure" in result
        assert result["current_step"] == "disclosure"
        assert result["invention_disclosure"] == disclosure_data

    @patch("patent_system.agents.disclosure.StructureDisclosureModule")
    @patch("patent_system.agents.disclosure.InterviewQuestionModule")
    def test_generates_question_per_topic(
        self, mock_interview_cls, mock_structure_cls
    ):
        """Interview module is called once per topic area."""
        mock_interview = MagicMock()
        mock_interview.return_value.next_question = "A question"
        mock_interview_cls.return_value = mock_interview

        mock_structure = MagicMock()
        mock_structure.return_value.disclosure_json = json.dumps({"technical_problem": "x"})
        mock_structure_cls.return_value = mock_structure

        state = _make_state()
        disclosure_node(state)

        assert mock_interview.call_count == len(_INTERVIEW_TOPICS)

    @patch("patent_system.agents.disclosure.StructureDisclosureModule")
    @patch("patent_system.agents.disclosure.InterviewQuestionModule")
    def test_handles_invalid_json_from_llm(
        self, mock_interview_cls, mock_structure_cls
    ):
        """When LLM returns non-JSON, a fallback disclosure dict is produced."""
        mock_interview = MagicMock()
        mock_interview.return_value.next_question = "Q?"
        mock_interview_cls.return_value = mock_interview

        mock_structure = MagicMock()
        mock_structure.return_value.disclosure_json = "not valid json"
        mock_structure_cls.return_value = mock_structure

        state = _make_state()
        result = disclosure_node(state)

        disc = result["invention_disclosure"]
        assert disc["technical_problem"] == "not valid json"
        assert disc["novel_features"] == []
        assert disc["implementation_details"] == ""
        assert disc["potential_variations"] == []

    @patch("patent_system.agents.disclosure.StructureDisclosureModule")
    @patch("patent_system.agents.disclosure.InterviewQuestionModule")
    def test_logs_agent_invocation(
        self, mock_interview_cls, mock_structure_cls, caplog
    ):
        """Agent invocation is logged with the correct agent name."""
        mock_interview = MagicMock()
        mock_interview.return_value.next_question = "Q?"
        mock_interview_cls.return_value = mock_interview

        mock_structure = MagicMock()
        mock_structure.return_value.disclosure_json = json.dumps({"technical_problem": "x"})
        mock_structure_cls.return_value = mock_structure

        state = _make_state()
        with caplog.at_level(logging.INFO, logger="patent_system.agents.disclosure"):
            disclosure_node(state)

        assert any("InventionDisclosureAgent" in r.message for r in caplog.records)

    @patch("patent_system.agents.disclosure.StructureDisclosureModule")
    @patch("patent_system.agents.disclosure.InterviewQuestionModule")
    def test_uses_existing_messages_as_context(
        self, mock_interview_cls, mock_structure_cls
    ):
        """Existing interview_messages are included in conversation history."""
        mock_interview = MagicMock()
        mock_interview.return_value.next_question = "Follow-up?"
        mock_interview_cls.return_value = mock_interview

        mock_structure = MagicMock()
        mock_structure.return_value.disclosure_json = json.dumps({"technical_problem": "x"})
        mock_structure_cls.return_value = mock_structure

        state = _make_state(interview_messages=["User: My invention is about X"])
        disclosure_node(state)

        # First call should include the existing message in conversation_history
        first_call_args = mock_interview.call_args_list[0]
        conv_history = first_call_args.kwargs.get(
            "conversation_history",
            first_call_args[1].get("conversation_history", first_call_args[0][0] if first_call_args[0] else ""),
        )
        assert "My invention is about X" in conv_history

    @patch("patent_system.agents.disclosure.StructureDisclosureModule")
    @patch("patent_system.agents.disclosure.InterviewQuestionModule")
    def test_structure_module_receives_transcript(
        self, mock_interview_cls, mock_structure_cls
    ):
        """StructureDisclosureModule is called with the accumulated transcript."""
        mock_interview = MagicMock()
        mock_interview.return_value.next_question = "Q about topic"
        mock_interview_cls.return_value = mock_interview

        mock_structure = MagicMock()
        mock_structure.return_value.disclosure_json = json.dumps({"technical_problem": "x"})
        mock_structure_cls.return_value = mock_structure

        state = _make_state()
        disclosure_node(state)

        mock_structure.assert_called_once()
        call_kwargs = mock_structure.call_args
        transcript = call_kwargs.kwargs.get(
            "transcript",
            call_kwargs[1].get("transcript", call_kwargs[0][0] if call_kwargs[0] else ""),
        )
        # Transcript should contain the generated questions
        assert "Q about topic" in transcript
