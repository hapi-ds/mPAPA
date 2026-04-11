"""Unit tests for the Description Drafter Agent."""

import logging
from unittest.mock import MagicMock, patch

import pytest

from patent_system.agents.description_drafting import (
    _prepare_claims_text,
    _prepare_disclosure_text,
    _prepare_prior_art_summary,
    description_drafting_node,
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
# _prepare_claims_text
# ---------------------------------------------------------------------------


class TestPrepareClaimsText:
    """Tests for _prepare_claims_text."""

    def test_returns_claims_from_state(self):
        state = _make_state(claims_text="Claim 1: A method...")
        assert _prepare_claims_text(state) == "Claim 1: A method..."

    def test_empty_string_when_missing(self):
        state = _make_state(claims_text="")
        assert _prepare_claims_text(state) == ""

    def test_empty_string_when_none(self):
        state = _make_state()
        state["claims_text"] = None
        assert _prepare_claims_text(state) == ""


# ---------------------------------------------------------------------------
# _prepare_prior_art_summary
# ---------------------------------------------------------------------------


class TestPreparePriorArtSummary:
    """Tests for _prepare_prior_art_summary."""

    def test_empty_when_no_results(self):
        state = _make_state(prior_art_results=[])
        assert _prepare_prior_art_summary(state) == ""

    def test_empty_when_none(self):
        state = _make_state()
        state["prior_art_results"] = None
        assert _prepare_prior_art_summary(state) == ""

    def test_serializes_results(self):
        results = [{"title": "Patent A", "abstract": "About A"}]
        state = _make_state(prior_art_results=results)
        summary = _prepare_prior_art_summary(state)
        assert "Patent A" in summary
        assert "About A" in summary


# ---------------------------------------------------------------------------
# _prepare_disclosure_text
# ---------------------------------------------------------------------------


class TestPrepareDisclosureText:
    """Tests for _prepare_disclosure_text."""

    def test_none_returns_empty(self):
        state = _make_state(invention_disclosure=None)
        assert _prepare_disclosure_text(state) == ""

    def test_empty_dict_returns_empty(self):
        state = _make_state(invention_disclosure={})
        assert _prepare_disclosure_text(state) == ""

    def test_serializes_disclosure(self):
        disclosure = {"technical_problem": "Slow processing"}
        state = _make_state(invention_disclosure=disclosure)
        result = _prepare_disclosure_text(state)
        assert "Slow processing" in result


# ---------------------------------------------------------------------------
# description_drafting_node
# ---------------------------------------------------------------------------


class TestDescriptionDraftingNode:
    """Tests for description_drafting_node."""

    @patch("patent_system.agents.description_drafting.DraftDescriptionModule")
    def test_returns_required_keys(self, mock_module_cls):
        mock_instance = MagicMock()
        mock_instance.return_value.description_text = "Full specification..."
        mock_module_cls.return_value = mock_instance

        state = _make_state(claims_text="Claim 1: A method...")
        result = description_drafting_node(state)

        assert "description_text" in result
        assert "current_step" in result
        assert result["current_step"] == "description_drafting"

    @patch("patent_system.agents.description_drafting.DraftDescriptionModule")
    def test_returns_description_from_module(self, mock_module_cls):
        spec_text = (
            "Technical Field\nThe invention relates to...\n"
            "Background Art\n...\nSummary of Invention\n..."
        )
        mock_instance = MagicMock()
        mock_instance.return_value.description_text = spec_text
        mock_module_cls.return_value = mock_instance

        state = _make_state(claims_text="Claim 1")
        result = description_drafting_node(state)

        assert result["description_text"] == spec_text

    @patch("patent_system.agents.description_drafting.DraftDescriptionModule")
    def test_passes_claims_prior_art_disclosure_to_module(self, mock_module_cls):
        mock_instance = MagicMock()
        mock_instance.return_value.description_text = "Description"
        mock_module_cls.return_value = mock_instance

        disclosure = {"technical_problem": "Problem X"}
        prior_art = [{"title": "Prior Art 1"}]
        state = _make_state(
            claims_text="Claim 1: A device...",
            prior_art_results=prior_art,
            invention_disclosure=disclosure,
        )
        description_drafting_node(state)

        mock_instance.assert_called_once()
        call_kwargs = mock_instance.call_args
        assert "Claim 1: A device..." == call_kwargs.kwargs["claims"]
        assert "Prior Art 1" in call_kwargs.kwargs["prior_art_summary"]
        assert "Problem X" in call_kwargs.kwargs["invention_disclosure"]

    @patch("patent_system.agents.description_drafting.DraftDescriptionModule")
    def test_handles_empty_state(self, mock_module_cls):
        mock_instance = MagicMock()
        mock_instance.return_value.description_text = "Minimal description"
        mock_module_cls.return_value = mock_instance

        state = _make_state(
            claims_text="",
            prior_art_results=[],
            invention_disclosure=None,
        )
        result = description_drafting_node(state)

        assert result["description_text"] == "Minimal description"
        call_kwargs = mock_instance.call_args
        assert call_kwargs.kwargs["claims"] == ""
        assert call_kwargs.kwargs["prior_art_summary"] == ""
        assert call_kwargs.kwargs["invention_disclosure"] == ""

    @patch("patent_system.agents.description_drafting.DraftDescriptionModule")
    def test_logs_agent_invocation(self, mock_module_cls, caplog):
        mock_instance = MagicMock()
        mock_instance.return_value.description_text = "Description text"
        mock_module_cls.return_value = mock_instance

        state = _make_state(claims_text="Claim 1")
        with caplog.at_level(
            logging.INFO, logger="patent_system.agents.description_drafting"
        ):
            description_drafting_node(state)

        assert any(
            "DescriptionDrafterAgent" in r.message for r in caplog.records
        )

    @patch("patent_system.agents.description_drafting.DraftDescriptionModule")
    def test_does_not_modify_iteration_count(self, mock_module_cls):
        """Description drafting should not change iteration_count."""
        mock_instance = MagicMock()
        mock_instance.return_value.description_text = "Desc"
        mock_module_cls.return_value = mock_instance

        state = _make_state(iteration_count=2)
        result = description_drafting_node(state)

        assert "iteration_count" not in result
