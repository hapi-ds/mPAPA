"""Unit tests for the Prior Art Search Agent."""

import logging
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from patent_system.agents.prior_art_search import (
    _derive_search_terms,
    _query_source,
    prior_art_search_node,
    sort_search_results,
    _SOURCE_REGISTRY,
)
from patent_system.exceptions import SourceUnavailableError


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
# _derive_search_terms
# ---------------------------------------------------------------------------

class TestDeriveSearchTerms:
    """Tests for _derive_search_terms."""

    def test_none_disclosure_returns_fallback(self):
        assert _derive_search_terms(None) == [""]

    def test_empty_disclosure_returns_fallback(self):
        assert _derive_search_terms({}) == [""]

    def test_extracts_technical_problem(self):
        disclosure = {"technical_problem": "Slow data processing"}
        terms = _derive_search_terms(disclosure)
        assert "Slow data processing" in terms

    def test_extracts_novel_features(self):
        disclosure = {"novel_features": ["GPU acceleration", "parallel pipeline"]}
        terms = _derive_search_terms(disclosure)
        assert "GPU acceleration" in terms
        assert "parallel pipeline" in terms

    def test_extracts_implementation_details(self):
        disclosure = {"implementation_details": "Uses CUDA kernels"}
        terms = _derive_search_terms(disclosure)
        assert "Uses CUDA kernels" in terms

    def test_full_disclosure(self):
        disclosure = {
            "technical_problem": "Slow processing",
            "novel_features": ["Feature A"],
            "implementation_details": "Detail B",
        }
        terms = _derive_search_terms(disclosure)
        assert len(terms) == 3


# ---------------------------------------------------------------------------
# prior_art_search_node
# ---------------------------------------------------------------------------

class TestPriorArtSearchNode:
    """Tests for prior_art_search_node."""

    @patch("patent_system.agents.prior_art_search._query_source")
    def test_returns_required_keys(self, mock_query):
        """Node returns dict with prior_art_results, failed_sources, current_step."""
        mock_query.return_value = {"results": []}
        state = _make_state()
        result = prior_art_search_node(state)

        assert "prior_art_results" in result
        assert "failed_sources" in result
        assert result["current_step"] == "prior_art_search"

    @patch("patent_system.agents.prior_art_search._query_source")
    def test_collects_results_from_patent_sources(self, mock_query):
        """Patent source results are parsed and serialized."""
        def side_effect(source_name, terms, **kwargs):
            if source_name == "DEPATISnet":
                return {
                    "results": [
                        {"patent_number": "DE123", "title": "Test Patent", "abstract": "Abstract"}
                    ]
                }
            return {"results": []}

        mock_query.side_effect = side_effect
        state = _make_state()
        result = prior_art_search_node(state)

        assert len(result["prior_art_results"]) >= 1
        assert any(r.get("patent_number") == "DE123" for r in result["prior_art_results"])

    @patch("patent_system.agents.prior_art_search._query_source")
    def test_collects_results_from_paper_sources(self, mock_query):
        """Paper source results are parsed and serialized."""
        def side_effect(source_name, terms, **kwargs):
            if source_name == "ArXiv":
                return {
                    "results": [
                        {"doi": "10.1234/test", "title": "Test Paper", "abstract": "Abstract"}
                    ]
                }
            return {"results": []}

        mock_query.side_effect = side_effect
        state = _make_state()
        result = prior_art_search_node(state)

        assert len(result["prior_art_results"]) >= 1
        assert any(r.get("doi") == "10.1234/test" for r in result["prior_art_results"])

    @patch("patent_system.agents.prior_art_search._query_source")
    def test_handles_source_failure_gracefully(self, mock_query):
        """Failed sources are logged and added to failed_sources list."""
        def side_effect(source_name, terms, **kwargs):
            if source_name == "DEPATISnet":
                raise ConnectionError("timeout")
            return {"results": []}

        mock_query.side_effect = side_effect
        state = _make_state()
        result = prior_art_search_node(state)

        assert "DEPATISnet" in result["failed_sources"]
        assert result["current_step"] == "prior_art_search"

    @patch("patent_system.agents.prior_art_search._query_source")
    def test_continues_after_source_failure(self, mock_query):
        """Results from remaining sources are still collected after a failure."""
        def side_effect(source_name, terms, **kwargs):
            if source_name == "DEPATISnet":
                raise ConnectionError("timeout")
            if source_name == "Google Patents":
                return {
                    "results": [
                        {"patent_number": "US456", "title": "Good Patent", "abstract": "Works"}
                    ]
                }
            return {"results": []}

        mock_query.side_effect = side_effect
        state = _make_state()
        result = prior_art_search_node(state)

        assert "DEPATISnet" in result["failed_sources"]
        assert any(r.get("patent_number") == "US456" for r in result["prior_art_results"])

    @patch("patent_system.agents.prior_art_search._query_source")
    def test_multiple_source_failures(self, mock_query):
        """Multiple failed sources are all tracked."""
        def side_effect(source_name, terms, **kwargs):
            if source_name in ("DEPATISnet", "ArXiv", "PubMed"):
                raise SourceUnavailableError(source_name, ConnectionError("down"))
            return {"results": []}

        mock_query.side_effect = side_effect
        state = _make_state()
        result = prior_art_search_node(state)

        assert set(result["failed_sources"]) == {"DEPATISnet", "ArXiv", "PubMed"}

    @patch("patent_system.agents.prior_art_search._query_source")
    def test_logs_agent_invocation(self, mock_query, caplog):
        """Agent invocation is logged with the correct agent name."""
        mock_query.return_value = {"results": []}
        state = _make_state()

        with caplog.at_level(logging.INFO, logger="patent_system.agents.prior_art_search"):
            prior_art_search_node(state)

        assert any("PriorArtSearchAgent" in r.message for r in caplog.records)

    @patch("patent_system.agents.prior_art_search._query_source")
    def test_logs_external_requests(self, mock_query, caplog):
        """Each source query is logged as an external request."""
        mock_query.return_value = {"results": []}
        state = _make_state()

        with caplog.at_level(logging.INFO, logger="patent_system.agents.prior_art_search"):
            prior_art_search_node(state)

        external_logs = [r for r in caplog.records if "External request" in r.message]
        assert len(external_logs) == len(_SOURCE_REGISTRY)

    @patch("patent_system.agents.prior_art_search._query_source")
    def test_uses_disclosure_for_search_terms(self, mock_query):
        """Search terms are derived from the invention disclosure."""
        mock_query.return_value = {"results": []}
        disclosure = {
            "technical_problem": "Quantum computing optimization",
            "novel_features": ["qubit error correction"],
            "implementation_details": "Surface code approach",
        }
        state = _make_state(invention_disclosure=disclosure)
        prior_art_search_node(state)

        # Verify _query_source was called with derived terms
        for call in mock_query.call_args_list:
            terms = call[0][1]
            assert "Quantum computing optimization" in terms


# ---------------------------------------------------------------------------
# sort_search_results
# ---------------------------------------------------------------------------

class TestSortSearchResults:
    """Tests for sort_search_results."""

    def test_sort_by_discovery_date_descending(self):
        results = [
            {"title": "Old", "discovered_date": "2020-01-01T00:00:00"},
            {"title": "New", "discovered_date": "2024-06-15T00:00:00"},
            {"title": "Mid", "discovered_date": "2022-03-10T00:00:00"},
        ]
        sorted_r = sort_search_results(results, "discovery_date")
        assert sorted_r[0]["title"] == "New"
        assert sorted_r[1]["title"] == "Mid"
        assert sorted_r[2]["title"] == "Old"

    def test_sort_by_citation_count_descending(self):
        results = [
            {"title": "Low", "citation_count": 5},
            {"title": "High", "citation_count": 100},
            {"title": "Mid", "citation_count": 42},
        ]
        sorted_r = sort_search_results(results, "citation_count")
        assert sorted_r[0]["title"] == "High"
        assert sorted_r[1]["title"] == "Mid"
        assert sorted_r[2]["title"] == "Low"

    def test_sort_by_relevance(self):
        results = [
            {"title": "Short", "abstract": "Brief"},
            {"title": "Long", "abstract": "A much longer abstract with more detail"},
            {"title": "Medium", "abstract": "Some abstract text"},
        ]
        sorted_r = sort_search_results(results, "relevance")
        assert sorted_r[0]["title"] == "Long"

    def test_unknown_criterion_returns_unchanged(self):
        results = [{"title": "A"}, {"title": "B"}]
        sorted_r = sort_search_results(results, "unknown")
        assert sorted_r == results

    def test_empty_list(self):
        assert sort_search_results([], "discovery_date") == []

    def test_missing_fields_handled(self):
        """Records missing the sort field don't cause errors."""
        results = [
            {"title": "Has date", "discovered_date": "2024-01-01T00:00:00"},
            {"title": "No date"},
        ]
        sorted_r = sort_search_results(results, "discovery_date")
        assert len(sorted_r) == 2
        assert sorted_r[0]["title"] == "Has date"

    def test_sort_does_not_mutate_original(self):
        results = [
            {"title": "B", "citation_count": 10},
            {"title": "A", "citation_count": 50},
        ]
        original = list(results)
        sort_search_results(results, "citation_count")
        assert results == original
