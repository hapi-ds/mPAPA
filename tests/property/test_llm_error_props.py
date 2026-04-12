"""Property-based tests for LLMConnectionError propagation.

Feature: placeholder-to-real-implementation, Property 5: LLMConnectionError propagation

For each DSPy agent node, when the underlying DSPy module raises a
connection error, the node shall raise LLMConnectionError whose message
contains the configured LM Studio base URL string.

**Validates: Requirements 2.7**
"""

from __future__ import annotations

import re
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from patent_system.agents.claims_drafting import claims_drafting_node
from patent_system.agents.consistency_review import consistency_review_node
from patent_system.agents.description_drafting import description_drafting_node
from patent_system.agents.disclosure import disclosure_node
from patent_system.agents.state import PatentWorkflowState
from patent_system.exceptions import LLMConnectionError

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Generate random base URL strings (non-empty, printable, URL-like)
_base_url = st.from_regex(
    r"https?://[a-z0-9][a-z0-9.\-]{0,30}(:[0-9]{1,5})?(/[a-z0-9]*)?",
    fullmatch=True,
)

# Connection error types that the agent nodes catch
_connection_errors = st.sampled_from([
    ConnectionError("connection refused"),
    OSError("network unreachable"),
])


def _build_state(**overrides: Any) -> PatentWorkflowState:
    """Build a minimal PatentWorkflowState with sensible defaults."""
    defaults: dict[str, Any] = {
        "topic_id": 1,
        "invention_disclosure": {"technical_problem": "test", "novel_features": []},
        "interview_messages": [],
        "prior_art_results": [],
        "failed_sources": [],
        "novelty_analysis": {"novel_aspects": []},
        "claims_text": "Claim 1: A method.",
        "description_text": "Technical field: testing.",
        "review_feedback": "",
        "review_approved": False,
        "iteration_count": 0,
        "current_step": "",
    }
    defaults.update(overrides)
    return PatentWorkflowState(**defaults)


def _make_mock_lm(base_url: str) -> MagicMock:
    """Create a mock dspy.LM with the given base URL in kwargs."""
    lm = MagicMock()
    lm.kwargs = {"api_base": base_url}
    return lm


# ---------------------------------------------------------------------------
# Property 5: LLMConnectionError propagation
# Feature: placeholder-to-real-implementation, Property 5: LLMConnectionError propagation
# ---------------------------------------------------------------------------


class TestLLMConnectionErrorPropagation:
    """Property 5: LLMConnectionError propagation.

    For each DSPy agent node function, when the underlying DSPy module
    raises a connection error, the node shall raise LLMConnectionError
    whose message contains the configured LM Studio base URL string.

    **Validates: Requirements 2.7**
    """

    @given(base_url=_base_url, error=_connection_errors)
    @settings(max_examples=100)
    def test_disclosure_node_propagates_llm_connection_error(
        self,
        base_url: str,
        error: Exception,
    ) -> None:
        """disclosure_node raises LLMConnectionError containing base URL."""
        # Use empty technical_problem so the pass-through path is NOT taken
        # and the LLM interview code path is exercised.
        state = _build_state(invention_disclosure=None)
        mock_lm = _make_mock_lm(base_url)

        with (
            patch(
                "patent_system.agents.disclosure.InterviewQuestionModule"
            ) as mock_cls,
            patch("patent_system.agents.disclosure.dspy.settings") as mock_settings,
        ):
            mock_settings.lm = mock_lm
            # The module instance's __call__ (forward) raises the error
            mock_cls.return_value.side_effect = error

            with pytest.raises(LLMConnectionError) as exc_info:
                disclosure_node(state)

            assert base_url in str(exc_info.value)

    @given(base_url=_base_url, error=_connection_errors)
    @settings(max_examples=100)
    def test_claims_drafting_node_propagates_llm_connection_error(
        self,
        base_url: str,
        error: Exception,
    ) -> None:
        """claims_drafting_node raises LLMConnectionError containing base URL."""
        state = _build_state()
        mock_lm = _make_mock_lm(base_url)

        with (
            patch(
                "patent_system.agents.claims_drafting.DraftClaimsModule"
            ) as mock_cls,
            patch(
                "patent_system.agents.claims_drafting.dspy.settings"
            ) as mock_settings,
        ):
            mock_settings.lm = mock_lm
            mock_cls.return_value.side_effect = error

            with pytest.raises(LLMConnectionError) as exc_info:
                claims_drafting_node(state)

            assert base_url in str(exc_info.value)

    @given(base_url=_base_url, error=_connection_errors)
    @settings(max_examples=100)
    def test_consistency_review_node_propagates_llm_connection_error(
        self,
        base_url: str,
        error: Exception,
    ) -> None:
        """consistency_review_node raises LLMConnectionError containing base URL."""
        state = _build_state()
        mock_lm = _make_mock_lm(base_url)

        with (
            patch(
                "patent_system.agents.consistency_review.ReviewConsistencyModule"
            ) as mock_cls,
            patch(
                "patent_system.agents.consistency_review.dspy.settings"
            ) as mock_settings,
        ):
            mock_settings.lm = mock_lm
            mock_cls.return_value.side_effect = error

            with pytest.raises(LLMConnectionError) as exc_info:
                consistency_review_node(state)

            assert base_url in str(exc_info.value)

    @given(base_url=_base_url, error=_connection_errors)
    @settings(max_examples=100)
    def test_description_drafting_node_propagates_llm_connection_error(
        self,
        base_url: str,
        error: Exception,
    ) -> None:
        """description_drafting_node raises LLMConnectionError containing base URL."""
        state = _build_state()
        mock_lm = _make_mock_lm(base_url)

        with (
            patch(
                "patent_system.agents.description_drafting.DraftDescriptionModule"
            ) as mock_cls,
            patch(
                "patent_system.agents.description_drafting.dspy.settings"
            ) as mock_settings,
        ):
            mock_settings.lm = mock_lm
            mock_cls.return_value.side_effect = error

            with pytest.raises(LLMConnectionError) as exc_info:
                description_drafting_node(state)

            assert base_url in str(exc_info.value)
