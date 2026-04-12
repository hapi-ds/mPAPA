"""Unit tests for the research panel sort logic.

Requirements: 3.5, 3.6
"""

from patent_system.gui.research_panel import (
    SORT_OPTIONS,
    _sort_results,
    _build_source_url,
)


# --- Sort options and source URL sanity ---

def test_sort_options_keys() -> None:
    """SORT_OPTIONS contains the three required criteria."""
    assert set(SORT_OPTIONS) == {"discovery_date", "relevance", "citation_count"}


def test_build_source_url_arxiv() -> None:
    """ArXiv records get a proper arxiv.org link."""
    rec = {"source": "ArXiv", "doi": "2301.12345"}
    url = _build_source_url(rec)
    assert url is not None
    assert "arxiv.org" in url
    assert "2301.12345" in url


def test_build_source_url_unknown_id() -> None:
    """Records with UNKNOWN id return None."""
    rec = {"source": "ArXiv", "doi": "UNKNOWN"}
    assert _build_source_url(rec) is None


def test_build_source_url_missing_source() -> None:
    """Records with unknown source return None."""
    rec = {"source": "SomeNewSource", "doi": "123"}
    assert _build_source_url(rec) is None


# --- _sort_results ---

_SAMPLE_ROWS = [
    {"title": "A", "discovered_date": "2024-01-01", "source": "ArXiv", "relevance_score": 3, "citation_count": 10},
    {"title": "B", "discovered_date": "2024-06-15", "source": "PubMed", "relevance_score": 7, "citation_count": 2},
    {"title": "C", "discovered_date": "2024-03-10", "source": "Google Patents", "relevance_score": 5, "citation_count": 50},
]


def test_sort_by_discovery_date() -> None:
    result = _sort_results(_SAMPLE_ROWS, "discovery_date")
    assert [r["title"] for r in result] == ["B", "C", "A"]


def test_sort_by_relevance() -> None:
    result = _sort_results(_SAMPLE_ROWS, "relevance")
    assert [r["title"] for r in result] == ["B", "C", "A"]


def test_sort_by_citation_count() -> None:
    result = _sort_results(_SAMPLE_ROWS, "citation_count")
    assert [r["title"] for r in result] == ["C", "A", "B"]


def test_sort_unknown_criterion_returns_original_order() -> None:
    result = _sort_results(_SAMPLE_ROWS, "unknown")
    assert [r["title"] for r in result] == ["A", "B", "C"]


def test_sort_empty_list() -> None:
    assert _sort_results([], "discovery_date") == []


def test_sort_missing_optional_keys() -> None:
    """Rows without relevance_score/citation_count default to 0."""
    rows = [
        {"title": "X", "discovered_date": "2024-01-01", "source": "S"},
        {"title": "Y", "discovered_date": "2024-02-01", "source": "S", "citation_count": 5},
    ]
    result = _sort_results(rows, "citation_count")
    assert result[0]["title"] == "Y"
    assert result[1]["title"] == "X"
