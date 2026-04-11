"""Unit tests for the Novelty Analysis Agent."""

import logging
from unittest.mock import MagicMock

import pytest

from patent_system.agents.novelty_analysis import (
    _PlaceholderRAG,
    _analyze_prior_art,
    _extract_query_from_disclosure,
    novelty_analysis_node,
)
from patent_system.db.models import NoveltyAnalysisResult


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
# _extract_query_from_disclosure
# ---------------------------------------------------------------------------


class TestExtractQuery:
    """Tests for _extract_query_from_disclosure."""

    def test_none_disclosure_returns_empty(self):
        assert _extract_query_from_disclosure(None) == ""

    def test_empty_disclosure_returns_empty(self):
        assert _extract_query_from_disclosure({}) == ""

    def test_extracts_technical_problem(self):
        disclosure = {"technical_problem": "Slow data processing"}
        result = _extract_query_from_disclosure(disclosure)
        assert "Slow data processing" in result

    def test_extracts_novel_features(self):
        disclosure = {"novel_features": ["GPU acceleration", "parallel pipeline"]}
        result = _extract_query_from_disclosure(disclosure)
        assert "GPU acceleration" in result
        assert "parallel pipeline" in result

    def test_combines_problem_and_features(self):
        disclosure = {
            "technical_problem": "Slow processing",
            "novel_features": ["Feature A"],
        }
        result = _extract_query_from_disclosure(disclosure)
        assert "Slow processing" in result
        assert "Feature A" in result


# ---------------------------------------------------------------------------
# _analyze_prior_art
# ---------------------------------------------------------------------------


class TestAnalyzePriorArt:
    """Tests for _analyze_prior_art."""

    def test_returns_novelty_analysis_result(self):
        disclosure = {"novel_features": ["quantum error correction"]}
        result = _analyze_prior_art(disclosure, [])
        assert isinstance(result, NoveltyAnalysisResult)

    def test_no_disclosure_gives_inconclusive(self):
        result = _analyze_prior_art(None, [])
        assert "No novel features identified" in result.novel_aspects[0]

    def test_no_conflicts_all_features_novel(self):
        disclosure = {"novel_features": ["Feature A", "Feature B"]}
        result = _analyze_prior_art(disclosure, [])
        assert "Feature A" in result.novel_aspects
        assert "Feature B" in result.novel_aspects
        assert result.potential_conflicts == []

    def test_detects_conflict_with_prior_art(self):
        disclosure = {"novel_features": ["gpu acceleration"]}
        prior_art = [{"text": "GPU acceleration is well known", "score": 0.9}]
        result = _analyze_prior_art(disclosure, prior_art)
        assert len(result.potential_conflicts) == 1
        assert result.potential_conflicts[0]["feature"] == "gpu acceleration"

    def test_novel_aspects_exclude_conflicting(self):
        disclosure = {"novel_features": ["gpu acceleration", "novel cache"]}
        prior_art = [{"text": "GPU acceleration is well known", "score": 0.9}]
        result = _analyze_prior_art(disclosure, prior_art)
        assert "novel cache" in result.novel_aspects
        assert "gpu acceleration" not in result.novel_aspects

    def test_suggested_claim_scope_with_novel_aspects(self):
        disclosure = {"novel_features": ["Feature X"]}
        result = _analyze_prior_art(disclosure, [])
        assert "Feature X" in result.suggested_claim_scope

    def test_suggested_claim_scope_insufficient_data(self):
        result = _analyze_prior_art(None, [])
        assert "Insufficient" in result.suggested_claim_scope


# ---------------------------------------------------------------------------
# novelty_analysis_node
# ---------------------------------------------------------------------------


class TestNoveltyAnalysisNode:
    """Tests for novelty_analysis_node."""

    def test_returns_required_keys(self):
        state = _make_state()
        result = novelty_analysis_node(state)
        assert "novelty_analysis" in result
        assert result["current_step"] == "novelty_analysis"

    def test_novelty_analysis_is_dict(self):
        state = _make_state()
        result = novelty_analysis_node(state)
        assert isinstance(result["novelty_analysis"], dict)

    def test_analysis_has_required_fields(self):
        state = _make_state(
            invention_disclosure={"novel_features": ["Feature A"]}
        )
        result = novelty_analysis_node(state)
        analysis = result["novelty_analysis"]
        assert "novel_aspects" in analysis
        assert "potential_conflicts" in analysis
        assert "suggested_claim_scope" in analysis

    def test_uses_provided_rag_engine(self):
        mock_rag = MagicMock()
        mock_rag.query.return_value = [
            {"text": "Related prior art about Feature A", "score": 0.85}
        ]
        disclosure = {
            "technical_problem": "Problem X",
            "novel_features": ["feature a"],
        }
        state = _make_state(invention_disclosure=disclosure)
        result = novelty_analysis_node(state, rag_engine=mock_rag)

        mock_rag.query.assert_called_once()
        call_kwargs = mock_rag.query.call_args
        assert call_kwargs.kwargs["topic_id"] == 1

    def test_uses_placeholder_rag_when_none(self):
        """No error when rag_engine is None — uses placeholder."""
        state = _make_state(
            invention_disclosure={"technical_problem": "Test"}
        )
        result = novelty_analysis_node(state, rag_engine=None)
        assert result["current_step"] == "novelty_analysis"

    def test_logs_agent_invocation(self, caplog):
        state = _make_state()
        with caplog.at_level(
            logging.INFO, logger="patent_system.agents.novelty_analysis"
        ):
            novelty_analysis_node(state)

        assert any(
            "NoveltyAnalysisAgent" in r.message for r in caplog.records
        )

    def test_rag_query_uses_disclosure_content(self):
        mock_rag = MagicMock()
        mock_rag.query.return_value = []
        disclosure = {
            "technical_problem": "Quantum computing",
            "novel_features": ["error correction"],
        }
        state = _make_state(invention_disclosure=disclosure, topic_id=42)
        novelty_analysis_node(state, rag_engine=mock_rag)

        call_kwargs = mock_rag.query.call_args
        query_text = call_kwargs.kwargs["query_text"]
        assert "Quantum computing" in query_text
        assert "error correction" in query_text
        assert call_kwargs.kwargs["topic_id"] == 42

    def test_empty_disclosure_skips_rag_query(self):
        mock_rag = MagicMock()
        mock_rag.query.return_value = []
        state = _make_state(invention_disclosure=None)
        novelty_analysis_node(state, rag_engine=mock_rag)

        mock_rag.query.assert_not_called()

    def test_result_can_be_validated_as_model(self):
        """The returned dict can be validated back into NoveltyAnalysisResult."""
        state = _make_state(
            invention_disclosure={"novel_features": ["Feature A"]}
        )
        result = novelty_analysis_node(state)
        validated = NoveltyAnalysisResult.model_validate(
            result["novelty_analysis"]
        )
        assert isinstance(validated, NoveltyAnalysisResult)


class TestPlaceholderRAG:
    """Tests for _PlaceholderRAG."""

    def test_returns_empty_list(self):
        rag = _PlaceholderRAG()
        assert rag.query(topic_id=1, query_text="test") == []
