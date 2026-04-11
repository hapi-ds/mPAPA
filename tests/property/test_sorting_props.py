"""Property-based tests for search result sorting.

Feature: patent-analysis-drafting, Property 4: Search result sorting

For any list of search results and any valid sort criterion, applying the
sort function produces a correctly ordered list.

**Validates: Requirements 3.6**
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from patent_system.agents.prior_art_search import sort_search_results

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# ISO-ish date strings that sort lexicographically the same as chronologically
_date_str = st.from_regex(
    r"20[0-2][0-9]-[01][0-9]-[0-3][0-9]T[0-2][0-9]:[0-5][0-9]:[0-5][0-9]",
    fullmatch=True,
)

_citation_count = st.integers(min_value=0, max_value=10_000)

_abstract_text = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "Z")),
    min_size=0,
    max_size=500,
)

_search_result = st.fixed_dictionaries(
    {
        "title": st.text(min_size=1, max_size=100),
        "discovered_date": _date_str,
        "citation_count": _citation_count,
        "abstract": _abstract_text,
        "source": st.sampled_from(
            ["DEPATISnet", "Google Patents", "Google Scholar", "ArXiv", "PubMed"]
        ),
    }
)

_results_list = st.lists(_search_result, min_size=0, max_size=30)

_sort_criterion = st.sampled_from(["discovery_date", "citation_count", "relevance"])


# ---------------------------------------------------------------------------
# Property 4: Search result sorting
# Feature: patent-analysis-drafting, Property 4: Search result sorting
# ---------------------------------------------------------------------------


class TestSearchResultSorting:
    """Property 4: Search result sorting.

    For any list of search results and any valid sort criterion
    (discovery_date, relevance, citation_count), applying
    ``sort_search_results`` produces a correctly ordered list.

    **Validates: Requirements 3.6**
    """

    @given(results=_results_list, criterion=_sort_criterion)
    @settings(max_examples=100)
    def test_sorted_output_is_correctly_ordered(
        self,
        results: list[dict],
        criterion: str,
    ) -> None:
        """Sorted results are in descending order for the chosen criterion."""
        sorted_results = sort_search_results(results, criterion)

        # Length must be preserved (no records lost or duplicated)
        assert len(sorted_results) == len(results)

        # Verify descending order for each criterion
        for i in range(len(sorted_results) - 1):
            a = sorted_results[i]
            b = sorted_results[i + 1]

            if criterion == "discovery_date":
                key_a = a.get("discovered_date", "") or ""
                key_b = b.get("discovered_date", "") or ""
                assert key_a >= key_b

            elif criterion == "citation_count":
                key_a = a.get("citation_count", 0) or 0
                key_b = b.get("citation_count", 0) or 0
                assert key_a >= key_b

            elif criterion == "relevance":
                key_a = len(a.get("abstract", "") or "")
                key_b = len(b.get("abstract", "") or "")
                assert key_a >= key_b

    @given(results=_results_list, criterion=_sort_criterion)
    @settings(max_examples=100)
    def test_sort_does_not_mutate_input(
        self,
        results: list[dict],
        criterion: str,
    ) -> None:
        """Sorting returns a new list; the original is unchanged."""
        original = list(results)
        sort_search_results(results, criterion)
        assert results == original

    @given(results=_results_list, criterion=_sort_criterion)
    @settings(max_examples=100)
    def test_sort_preserves_all_elements(
        self,
        results: list[dict],
        criterion: str,
    ) -> None:
        """Every element in the input appears in the output (multiset equality)."""
        sorted_results = sort_search_results(results, criterion)
        # Compare by converting dicts to frozensets of items for hashability
        input_ids = sorted(str(r) for r in results)
        output_ids = sorted(str(r) for r in sorted_results)
        assert input_ids == output_ids
