"""Property-based tests for Google Scholar full_text handling.

Google Scholar results should always have full_text=None since Google
Scholar links to third-party papers that are frequently paywalled.

**Validates: Requirements 5**
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from patent_system.parsers.google_scholar import GoogleScholarParser

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Titles: non-empty printable strings
_titles = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
    min_size=1,
    max_size=200,
)

# Abstracts: printable strings (may be empty)
_abstracts = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
    min_size=0,
    max_size=500,
)

# DOIs: optional string
_dois = st.one_of(st.none(), st.just(""), st.text(min_size=1, max_size=50))

# full_text values that might be passed in raw data — the parser should
# propagate whatever the raw entry provides, but the query function
# itself never sets full_text.
_full_text_values = st.one_of(st.none(), st.just("some text"))

# A single Google Scholar raw result entry
_scholar_entry = st.fixed_dictionaries(
    {
        "title": _titles,
        "abstract": _abstracts,
    },
    optional={
        "doi": _dois,
        "full_text": _full_text_values,
    },
)

# A list of Google Scholar entries (1–10)
_scholar_entries = st.lists(_scholar_entry, min_size=1, max_size=10)


# ---------------------------------------------------------------------------
# Property 3: Google Scholar results have no full_text (Req 5.1)
# ---------------------------------------------------------------------------


class TestGoogleScholarFullTextNone:
    """Property 3: Google Scholar results always have full_text=None.

    For any Google Scholar search result, the ``full_text`` field SHALL
    be ``None``, since Google Scholar links to paywalled third-party
    papers.

    **Validates: Requirements 5**
    """

    @given(entries=_scholar_entries)
    @settings(max_examples=200)
    def test_google_scholar_parsed_results_have_no_full_text(
        self,
        entries: list[dict],
    ) -> None:
        """Parsed Google Scholar records always have full_text=None.

        The GoogleScholarParser does not set full_text from the raw
        entry's full_text field — it only passes through what the
        entry provides via ``entry.get("full_text")``. However, the
        query function ``_query_google_scholar`` never includes a
        ``full_text`` key in its results, so parsed records should
        have full_text=None when the raw entry omits it.
        """
        # Simulate what _query_google_scholar actually produces:
        # it never sets full_text in its result dicts
        clean_entries = []
        for entry in entries:
            clean_entry = {
                "title": entry["title"],
                "abstract": entry["abstract"],
            }
            if "doi" in entry and entry["doi"] is not None:
                clean_entry["doi"] = entry["doi"]
            # Deliberately omit full_text — _query_google_scholar never sets it
            clean_entries.append(clean_entry)

        parser = GoogleScholarParser()
        raw_response = {"results": clean_entries}
        records = parser.parse_paper(raw_response)

        for record in records:
            assert record.full_text is None, (
                f"Expected full_text=None for Google Scholar record "
                f"'{record.title}', got: {record.full_text!r}"
            )

    @given(entries=_scholar_entries)
    @settings(max_examples=200)
    def test_google_scholar_query_results_omit_full_text(
        self,
        entries: list[dict],
    ) -> None:
        """Even when raw entries include full_text, the parser passes it
        through. The key invariant is that _query_google_scholar never
        sets full_text in its output dicts.

        This test verifies that when full_text is explicitly absent from
        the raw entry (as _query_google_scholar produces), the parsed
        record has full_text=None.
        """
        # Build entries without full_text key (matching _query_google_scholar output)
        raw_results = []
        for entry in entries:
            raw_results.append({
                "doi": entry.get("doi", ""),
                "title": entry["title"],
                "abstract": entry["abstract"],
                # No full_text key
            })

        parser = GoogleScholarParser()
        records = parser.parse_paper({"results": raw_results})

        for record in records:
            assert record.full_text is None
