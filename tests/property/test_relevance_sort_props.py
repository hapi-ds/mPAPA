"""Property-based tests for relevance sort: scored results before unscored.

**Validates: Requirements 8.4**

Property 4 from the design document: Relevance sort stability — scored before
unscored.

For any list of search results with mixed scored and unscored entries, sorting
by relevance SHALL place all scored results before all unscored results.
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from patent_system.gui.research_panel import _sort_results

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# A result dict that HAS a relevance_score (scored)
_scored_result = st.fixed_dictionaries(
    {
        "title": st.text(min_size=1, max_size=80),
        "abstract": st.text(min_size=0, max_size=200),
        "source": st.sampled_from(
            ["ArXiv", "PubMed", "Google Patents", "EPO OPS", "Google Scholar"]
        ),
        "relevance_score": st.floats(
            min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False
        ),
    }
)

# A result dict that does NOT have a relevance_score key (unscored)
_unscored_result = st.fixed_dictionaries(
    {
        "title": st.text(min_size=1, max_size=80),
        "abstract": st.text(min_size=0, max_size=200),
        "source": st.sampled_from(
            ["ArXiv", "PubMed", "Google Patents", "EPO OPS", "Google Scholar"]
        ),
    }
)

# Mixed list: at least one scored and at least one unscored
_mixed_results = st.tuples(
    st.lists(_scored_result, min_size=1, max_size=15),
    st.lists(_unscored_result, min_size=1, max_size=15),
).map(lambda pair: pair[0] + pair[1])


# ---------------------------------------------------------------------------
# Property 4: Relevance sort stability — scored before unscored (Req 8.4)
# ---------------------------------------------------------------------------


class TestRelevanceSortScoredBeforeUnscored:
    """Property 4: Relevance sort stability — scored before unscored.

    For any list of search results with mixed relevance_score (some present,
    some absent), sorting by "relevance" places all scored results before all
    unscored results.

    **Validates: Requirements 8.4**
    """

    @given(results=_mixed_results)
    @settings(max_examples=200)
    def test_scored_results_appear_before_unscored(
        self,
        results: list[dict],
    ) -> None:
        """All scored results appear before all unscored results after relevance sort."""
        sorted_results = _sort_results(results, "relevance")

        scored = [r for r in sorted_results if r.get("relevance_score") is not None]
        unscored = [r for r in sorted_results if r.get("relevance_score") is None]

        assert scored and unscored, "Test requires both scored and unscored results"

        last_scored_idx = sorted_results.index(scored[-1])
        first_unscored_idx = sorted_results.index(unscored[0])
        assert last_scored_idx < first_unscored_idx, (
            f"Last scored result at index {last_scored_idx} should be before "
            f"first unscored result at index {first_unscored_idx}"
        )

    @given(results=_mixed_results)
    @settings(max_examples=200)
    def test_sort_preserves_all_elements(
        self,
        results: list[dict],
    ) -> None:
        """Sorting preserves all elements (no records lost or duplicated)."""
        sorted_results = _sort_results(results, "relevance")
        assert len(sorted_results) == len(results)

    @given(results=_mixed_results)
    @settings(max_examples=200)
    def test_scored_results_are_in_descending_order(
        self,
        results: list[dict],
    ) -> None:
        """Among scored results, they are sorted in descending relevance_score order."""
        sorted_results = _sort_results(results, "relevance")
        scored = [r for r in sorted_results if r.get("relevance_score") is not None]

        for i in range(len(scored) - 1):
            assert scored[i]["relevance_score"] >= scored[i + 1]["relevance_score"], (
                f"Scored results not in descending order: "
                f"{scored[i]['relevance_score']} < {scored[i + 1]['relevance_score']}"
            )
