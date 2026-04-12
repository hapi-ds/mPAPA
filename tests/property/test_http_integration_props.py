"""Property-based tests for HTTP integration in prior art search.

Feature: placeholder-to-real-implementation, Property 1: Source URL construction

For any source name in ``_SOURCE_REGISTRY`` and any non-empty list of search
term strings, ``_query_source`` shall construct an HTTP request whose URL
contains the known base endpoint for that source and includes the URL-encoded
search terms.

**Validates: Requirements 1.1**
"""

from unittest.mock import MagicMock, patch
from typing import Any
from urllib.parse import quote_plus

from hypothesis import given, settings
from hypothesis import strategies as st

from patent_system.agents.prior_art_search import (
    _SOURCE_ENDPOINTS,
    _SOURCE_REGISTRY,
    _query_source,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_SOURCES = list(_SOURCE_REGISTRY.keys())

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Printable, non-empty search term strings (avoid control chars / surrogates)
_search_term = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z"), min_codepoint=32, max_codepoint=126),
    min_size=1,
    max_size=40,
).filter(lambda s: s.strip())

_search_terms_list = st.lists(_search_term, min_size=1, max_size=5)

_source_name = st.sampled_from(ALL_SOURCES)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_response(body: str = "<xml></xml>") -> MagicMock:
    """Create a mock urlopen response context manager."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = body.encode("utf-8")
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


# ---------------------------------------------------------------------------
# Property 1: Source URL construction
# Feature: placeholder-to-real-implementation, Property 1: Source URL construction
# ---------------------------------------------------------------------------


class TestSourceURLConstruction:
    """Property 1: Source URL construction.

    For any source name in ``_SOURCE_REGISTRY`` and any non-empty list of
    search term strings, ``_query_source`` shall construct an HTTP request
    whose URL contains the known base endpoint for that source and includes
    the URL-encoded search terms.

    **Validates: Requirements 1.1**
    """

    @given(source=_source_name, terms=_search_terms_list)
    @settings(max_examples=100)
    def test_url_contains_base_endpoint(
        self,
        source: str,
        terms: list[str],
    ) -> None:
        """The constructed URL contains the known base endpoint for the source."""
        expected_endpoint = _SOURCE_ENDPOINTS[source]

        with patch(
            "patent_system.agents.prior_art_search.urlopen",
            return_value=_make_mock_response(),
        ) as mock_urlopen:
            try:
                _query_source(source, terms)
            except Exception:
                pass  # We only care about the URL that was constructed

            # urlopen should have been called at least once
            assert mock_urlopen.call_count >= 1, (
                f"urlopen was not called for source {source}"
            )

            # Collect all URLs from all calls (PubMed makes 2 calls)
            all_urls = []
            for call in mock_urlopen.call_args_list:
                req = call[0][0]  # first positional arg
                url = req.full_url if hasattr(req, "full_url") else str(req)
                all_urls.append(url)

            # At least one URL must contain the base endpoint
            assert any(expected_endpoint in url for url in all_urls), (
                f"No URL contained endpoint '{expected_endpoint}'. "
                f"URLs: {all_urls}"
            )

    @given(source=_source_name, terms=_search_terms_list)
    @settings(max_examples=100)
    def test_url_contains_encoded_search_terms(
        self,
        source: str,
        terms: list[str],
    ) -> None:
        """The constructed URL contains URL-encoded search terms."""
        with patch(
            "patent_system.agents.prior_art_search.urlopen",
            return_value=_make_mock_response(),
        ) as mock_urlopen:
            try:
                _query_source(source, terms)
            except Exception:
                pass

            assert mock_urlopen.call_count >= 1

            # Collect all URLs across all calls
            all_urls = []
            for call in mock_urlopen.call_args_list:
                req = call[0][0]
                url = req.full_url if hasattr(req, "full_url") else str(req)
                all_urls.append(url)

            combined_url = " ".join(all_urls)

            # Each non-empty search term should appear URL-encoded in at
            # least one of the request URLs.
            for term in terms:
                encoded = quote_plus(term)
                assert encoded in combined_url, (
                    f"Encoded term '{encoded}' (from '{term}') not found "
                    f"in any URL for source {source}. URLs: {all_urls}"
                )


# ---------------------------------------------------------------------------
# Property 2: Parser data flow integrity
# Feature: placeholder-to-real-implementation, Property 2: Parser data flow integrity
# ---------------------------------------------------------------------------

from patent_system.agents.prior_art_search import (
    prior_art_search_node,
)
from patent_system.exceptions import SourceUnavailableError

# Strategies for generating valid raw response dicts per source type.
# Patent sources (DEPATISnet, Google Patents) use patent_number/title/abstract.
# Paper sources (ArXiv, PubMed, Google Scholar) use doi/title/abstract.

_nonempty_text = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N", "P", "Z"),
        min_codepoint=32,
        max_codepoint=126,
    ),
    min_size=1,
    max_size=60,
).filter(lambda s: s.strip())

_patent_entry = st.fixed_dictionaries({
    "patent_number": _nonempty_text,
    "title": _nonempty_text,
    "abstract": _nonempty_text,
})

_paper_entry = st.fixed_dictionaries({
    "doi": _nonempty_text,
    "title": _nonempty_text,
    "abstract": _nonempty_text,
})

_patent_response = st.fixed_dictionaries({
    "results": st.lists(_patent_entry, min_size=0, max_size=5),
})

_paper_response = st.fixed_dictionaries({
    "results": st.lists(_paper_entry, min_size=0, max_size=5),
})

# Map each source to the correct response strategy
_PATENT_SOURCES = {"DEPATISnet", "Google Patents"}
_PAPER_SOURCES = {"ArXiv", "PubMed", "Google Scholar"}


@st.composite
def _source_and_response(draw: st.DrawFn) -> tuple[str, dict]:
    """Draw a source name and a matching valid raw response dict."""
    source = draw(_source_name)
    if source in _PATENT_SOURCES:
        response = draw(_patent_response)
    else:
        response = draw(_paper_response)
    return source, response


class TestParserDataFlowIntegrity:
    """Property 2: Parser data flow integrity.

    For any source name and for any valid raw response dict (containing a
    ``results`` key with well-formed entries), the records returned by
    ``prior_art_search_node`` for that source shall be equal to the records
    produced by directly calling the corresponding parser's ``parse_patent``
    or ``parse_paper`` method on the same raw response.

    **Validates: Requirements 1.2**
    """

    @given(data=_source_and_response())
    @settings(max_examples=100)
    def test_node_records_match_direct_parser_output(
        self,
        data: tuple[str, dict],
    ) -> None:
        """Records from prior_art_search_node equal direct parser output."""
        source_name, raw_response = data

        # Get the parser and type from the registry
        source_info = _SOURCE_REGISTRY[source_name]
        parser = source_info["parser"]
        source_type = source_info["type"]

        # Compute expected records by calling the parser directly
        if source_type == "patent":
            expected_records = parser.parse_patent(raw_response)
        else:
            expected_records = parser.parse_paper(raw_response)
        expected_serialized = [parser.serialize(r) for r in expected_records]

        # Build a minimal state for the node
        state: PatentWorkflowState = {
            "topic_id": 1,
            "invention_disclosure": {"technical_problem": "test"},
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

        # Mock _query_source: return the generated response for the target
        # source, raise SourceUnavailableError for all others so they don't
        # contribute records.
        def mock_query_source(name: str, terms: list[str], **kwargs: Any) -> dict:
            if name == source_name:
                return raw_response
            raise SourceUnavailableError(name, RuntimeError("mocked failure"))

        with patch(
            "patent_system.agents.prior_art_search._query_source",
            side_effect=mock_query_source,
        ):
            result = prior_art_search_node(state)

        actual_results = result["prior_art_results"]

        # Remove non-deterministic fields (discovered_date, id, session_id,
        # embedding, full_text, pdf_path, claims) before comparison so we
        # compare only the fields the parser sets from the raw response.
        def _strip_volatile(record: dict) -> dict:
            return {
                k: v
                for k, v in record.items()
                if k not in (
                    "discovered_date", "id", "session_id",
                    "embedding", "full_text", "pdf_path", "claims",
                )
            }

        actual_stripped = [_strip_volatile(r) for r in actual_results]
        expected_stripped = [_strip_volatile(r) for r in expected_serialized]

        assert actual_stripped == expected_stripped, (
            f"Mismatch for source {source_name}.\n"
            f"Expected: {expected_stripped}\n"
            f"Actual:   {actual_stripped}"
        )

        # Also verify that the target source is NOT in failed_sources
        assert source_name not in result["failed_sources"], (
            f"Source {source_name} should not be in failed_sources"
        )


# ---------------------------------------------------------------------------
# Property 3: Search agent fault tolerance
# Feature: placeholder-to-real-implementation, Property 3: Search agent fault tolerance
# ---------------------------------------------------------------------------


@st.composite
def _failing_source_subset(draw: st.DrawFn) -> frozenset[str]:
    """Draw a subset (possibly empty) of source names that should fail."""
    return frozenset(draw(st.sets(st.sampled_from(ALL_SOURCES), min_size=0)))


class TestSearchAgentFaultTolerance:
    """Property 3: Search agent fault tolerance.

    For any subset of source names that raise ``SourceUnavailableError``,
    ``prior_art_search_node`` shall return ``failed_sources`` containing
    exactly those names, and ``prior_art_results`` containing all records
    from non-failing sources.

    **Validates: Requirements 1.3**
    """

    @given(failing_sources=_failing_source_subset())
    @settings(max_examples=100)
    def test_failed_sources_match_failing_subset(
        self,
        failing_sources: frozenset[str],
    ) -> None:
        """failed_sources contains exactly the names that raised errors."""
        # Build a minimal state for the node
        state: PatentWorkflowState = {
            "topic_id": 1,
            "invention_disclosure": {"technical_problem": "test"},
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

        # Build a fixed valid response per source for non-failing sources
        def _valid_response(source_name: str) -> dict:
            source_info = _SOURCE_REGISTRY[source_name]
            if source_info["type"] == "patent":
                return {"results": [{"patent_number": "P1", "title": "T", "abstract": "A"}]}
            return {"results": [{"doi": "D1", "title": "T", "abstract": "A"}]}

        def mock_query_source(name: str, terms: list[str], **kwargs: Any) -> dict:
            if name in failing_sources:
                raise SourceUnavailableError(name, RuntimeError("mocked failure"))
            return _valid_response(name)

        with patch(
            "patent_system.agents.prior_art_search._query_source",
            side_effect=mock_query_source,
        ):
            result = prior_art_search_node(state)

        assert set(result["failed_sources"]) == set(failing_sources), (
            f"Expected failed_sources={set(failing_sources)}, "
            f"got {set(result['failed_sources'])}"
        )

    @given(failing_sources=_failing_source_subset())
    @settings(max_examples=100)
    def test_results_contain_only_non_failing_source_records(
        self,
        failing_sources: frozenset[str],
    ) -> None:
        """prior_art_results contains records only from non-failing sources."""
        state: PatentWorkflowState = {
            "topic_id": 1,
            "invention_disclosure": {"technical_problem": "test"},
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

        # Each non-failing source returns exactly one record with a
        # distinguishable title so we can verify provenance.
        def _valid_response(source_name: str) -> dict:
            source_info = _SOURCE_REGISTRY[source_name]
            if source_info["type"] == "patent":
                return {"results": [
                    {"patent_number": f"PN-{source_name}", "title": f"Title-{source_name}", "abstract": "A"},
                ]}
            return {"results": [
                {"doi": f"DOI-{source_name}", "title": f"Title-{source_name}", "abstract": "A"},
            ]}

        def mock_query_source(name: str, terms: list[str], **kwargs: Any) -> dict:
            if name in failing_sources:
                raise SourceUnavailableError(name, RuntimeError("mocked failure"))
            return _valid_response(name)

        with patch(
            "patent_system.agents.prior_art_search._query_source",
            side_effect=mock_query_source,
        ):
            result = prior_art_search_node(state)

        # Verify every returned record comes from a non-failing source
        non_failing = set(ALL_SOURCES) - set(failing_sources)
        result_sources = {r["source"] for r in result["prior_art_results"]}

        assert result_sources <= non_failing, (
            f"Results contain records from failing sources. "
            f"Result sources: {result_sources}, failing: {failing_sources}"
        )

        # Verify we got records from ALL non-failing sources
        assert result_sources == non_failing, (
            f"Missing records from non-failing sources. "
            f"Expected: {non_failing}, got: {result_sources}"
        )


# ---------------------------------------------------------------------------
# Property 4: Privacy — only derived search terms transmitted
# Feature: placeholder-to-real-implementation, Property 4: Privacy — only derived search terms transmitted
# ---------------------------------------------------------------------------

import json

from patent_system.agents.prior_art_search import (
    _derive_search_terms,
    prior_art_search_node,
)

# Strategy for generating invention disclosure dicts with various field values.
_optional_text = st.one_of(st.none(), st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N", "P", "Z"),
        min_codepoint=32,
        max_codepoint=126,
    ),
    min_size=0,
    max_size=80,
))

_novel_features_list = st.lists(
    st.text(
        alphabet=st.characters(
            whitelist_categories=("L", "N", "P", "Z"),
            min_codepoint=32,
            max_codepoint=126,
        ),
        min_size=0,
        max_size=60,
    ),
    min_size=0,
    max_size=5,
)

_disclosure_strategy = st.fixed_dictionaries(
    {
        "technical_problem": _optional_text,
        "novel_features": _novel_features_list,
        "implementation_details": _optional_text,
    },
    optional={
        "potential_variations": _optional_text,
    },
)


class TestPrivacyDerivedSearchTerms:
    """Property 4: Privacy — only derived search terms transmitted.

    For any invention disclosure dict, the arguments passed to the HTTP
    layer by ``_query_source`` shall contain only strings that are
    substrings of ``_derive_search_terms(disclosure)`` output and shall
    never contain the full serialized disclosure JSON.

    **Validates: Requirements 1.5**
    """

    @given(disclosure=_disclosure_strategy)
    @settings(max_examples=100)
    def test_query_source_receives_only_derived_terms(
        self,
        disclosure: dict,
    ) -> None:
        """All search terms passed to _query_source are substrings of _derive_search_terms output."""
        derived_terms = _derive_search_terms(disclosure)

        # Collect all search_terms arguments passed to _query_source
        captured_terms: list[list[str]] = []

        def mock_query_source(source_name: str, search_terms: list[str], **kwargs: Any) -> dict:
            captured_terms.append(search_terms)
            # Return a minimal valid response so the node doesn't error
            return {"results": []}

        state = {
            "topic_id": 1,
            "invention_disclosure": disclosure,
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

        with patch(
            "patent_system.agents.prior_art_search._query_source",
            side_effect=mock_query_source,
        ):
            prior_art_search_node(state)

        # _query_source should have been called at least once (once per source)
        assert len(captured_terms) > 0, "_query_source was never called"

        # Every list of terms passed to _query_source must equal the
        # derived terms exactly — the node passes the same list each time.
        for call_terms in captured_terms:
            for term in call_terms:
                assert term in derived_terms, (
                    f"Term '{term}' passed to _query_source is not in "
                    f"derived terms {derived_terms}"
                )

    @given(disclosure=_disclosure_strategy)
    @settings(max_examples=100)
    def test_full_disclosure_json_never_transmitted(
        self,
        disclosure: dict,
    ) -> None:
        """The full serialized disclosure JSON is never passed as a search term."""
        full_json = json.dumps(disclosure)

        captured_terms: list[list[str]] = []

        def mock_query_source(source_name: str, search_terms: list[str], **kwargs: Any) -> dict:
            captured_terms.append(search_terms)
            return {"results": []}

        state = {
            "topic_id": 1,
            "invention_disclosure": disclosure,
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

        with patch(
            "patent_system.agents.prior_art_search._query_source",
            side_effect=mock_query_source,
        ):
            prior_art_search_node(state)

        # No search term should be the full serialized disclosure JSON
        for call_terms in captured_terms:
            for term in call_terms:
                assert term != full_json, (
                    f"Full serialized disclosure JSON was passed as a "
                    f"search term: '{term}'"
                )
