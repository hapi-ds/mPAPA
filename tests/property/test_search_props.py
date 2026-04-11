"""Property-based tests for partial results on source failure.

Feature: patent-analysis-drafting, Property 5: Partial results on source failure

For any non-empty subset of unreachable sources, the Prior Art Search Agent
returns results from the remaining reachable sources and the failed list
matches the unreachable subset exactly.

**Validates: Requirements 3.7**
"""

from unittest.mock import patch

from hypothesis import given, settings
from hypothesis import strategies as st

from patent_system.agents.prior_art_search import (
    _SOURCE_REGISTRY,
    prior_art_search_node,
)
from patent_system.exceptions import SourceUnavailableError

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_SOURCES = list(_SOURCE_REGISTRY.keys())

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Non-empty strict subsets of sources that will be unreachable
_unreachable_subset = (
    st.sets(st.sampled_from(ALL_SOURCES), min_size=1, max_size=len(ALL_SOURCES) - 1)
    .map(sorted)
    .map(list)
)


def _make_state() -> dict:
    """Create a minimal PatentWorkflowState dict for testing."""
    return {
        "topic_id": 1,
        "invention_disclosure": {"technical_problem": "test problem"},
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


def _build_query_side_effect(unreachable: list[str]):
    """Build a side-effect function for _query_source.

    Unreachable sources raise SourceUnavailableError.
    Reachable sources return a minimal valid result set so that
    parsers produce at least one record.
    """
    # Source type lookup from the registry
    source_types = {name: info["type"] for name, info in _SOURCE_REGISTRY.items()}

    def side_effect(source_name: str, search_terms: list[str]) -> dict:
        if source_name in unreachable:
            raise SourceUnavailableError(
                source_name, ConnectionError(f"{source_name} is down")
            )

        # Return a minimal result that the parser can handle
        if source_types.get(source_name) == "patent":
            return {
                "results": [
                    {
                        "patent_number": f"PAT-{source_name[:3]}",
                        "title": f"Result from {source_name}",
                        "abstract": "Test abstract",
                    }
                ]
            }
        else:
            return {
                "results": [
                    {
                        "doi": f"10.0000/{source_name[:3].lower()}",
                        "title": f"Result from {source_name}",
                        "abstract": "Test abstract",
                    }
                ]
            }

    return side_effect


# ---------------------------------------------------------------------------
# Property 5: Partial results on source failure
# Feature: patent-analysis-drafting, Property 5: Partial results on source failure
# ---------------------------------------------------------------------------


class TestPartialResultsOnSourceFailure:
    """Property 5: Partial results on source failure.

    For any non-empty subset of data sources that are unreachable, the
    Prior Art Search Agent still returns results from the remaining
    reachable sources, and the list of failed sources exactly matches
    the unreachable subset.

    **Validates: Requirements 3.7**
    """

    @given(unreachable=_unreachable_subset)
    @settings(max_examples=100)
    def test_failed_sources_match_unreachable_subset(
        self,
        unreachable: list[str],
    ) -> None:
        """The failed_sources list exactly equals the set of unreachable sources."""
        with patch(
            "patent_system.agents.prior_art_search._query_source",
            side_effect=_build_query_side_effect(unreachable),
        ):
            result = prior_art_search_node(_make_state())

        assert set(result["failed_sources"]) == set(unreachable)

    @given(unreachable=_unreachable_subset)
    @settings(max_examples=100)
    def test_reachable_sources_produce_results(
        self,
        unreachable: list[str],
    ) -> None:
        """Results are returned from every reachable source."""
        reachable = [s for s in ALL_SOURCES if s not in unreachable]

        with patch(
            "patent_system.agents.prior_art_search._query_source",
            side_effect=_build_query_side_effect(unreachable),
        ):
            result = prior_art_search_node(_make_state())

        # At least one result per reachable source
        assert len(result["prior_art_results"]) >= len(reachable)

    @given(unreachable=_unreachable_subset)
    @settings(max_examples=100)
    def test_no_results_from_unreachable_sources(
        self,
        unreachable: list[str],
    ) -> None:
        """No result record should originate from an unreachable source."""
        with patch(
            "patent_system.agents.prior_art_search._query_source",
            side_effect=_build_query_side_effect(unreachable),
        ):
            result = prior_art_search_node(_make_state())

        for record in result["prior_art_results"]:
            assert record.get("source") not in unreachable
