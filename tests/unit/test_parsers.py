"""Unit tests for source-specific parsers.

Covers valid response parsing, malformed record handling, and empty results
for each of the five data source parsers.
"""

import logging

import pytest

from patent_system.db.models import PatentRecord, ScientificPaperRecord
from patent_system.parsers.epo_ops import EPOOPSParser
from patent_system.parsers.google_patents import GooglePatentsParser
from patent_system.parsers.google_scholar import GoogleScholarParser
from patent_system.parsers.arxiv_parser import ArXivParser
from patent_system.parsers.pubmed import PubMedParser


# ---------------------------------------------------------------------------
# EPO OPS
# ---------------------------------------------------------------------------

class TestEPOOPSParser:
    """Tests for the EPO OPS parser."""

    def setup_method(self) -> None:
        self.parser = EPOOPSParser()

    def test_parse_valid_patents(self) -> None:
        raw = {
            "results": [
                {"patent_number": "DE102020001A1", "title": "Widget", "abstract": "A widget."},
                {"patent_number": "EP1234567B1", "title": "Gadget", "abstract": "A gadget."},
            ]
        }
        records = self.parser.parse_patent(raw)
        assert len(records) == 2
        assert all(isinstance(r, PatentRecord) for r in records)
        assert records[0].patent_number == "DE102020001A1"
        assert records[0].source == "EPO OPS"
        assert records[1].title == "Gadget"

    def test_parse_paper_returns_empty(self) -> None:
        assert self.parser.parse_paper({"results": []}) == []

    def test_malformed_record_skipped(self, caplog: pytest.LogCaptureFixture) -> None:
        raw = {
            "results": [
                {"patent_number": "DE1", "title": "Good", "abstract": "OK"},
                {"title": "Missing patent_number"},  # malformed
                {"patent_number": "DE2", "title": "Also good", "abstract": "Fine"},
            ]
        }
        with caplog.at_level(logging.WARNING):
            records = self.parser.parse_patent(raw)
        assert len(records) == 2
        assert records[0].patent_number == "DE1"
        assert records[1].patent_number == "DE2"
        assert any("malformed" in msg.lower() for msg in caplog.messages)

    def test_empty_results(self) -> None:
        assert self.parser.parse_patent({"results": []}) == []

    def test_missing_results_key(self) -> None:
        assert self.parser.parse_patent({}) == []


# ---------------------------------------------------------------------------
# Google Patents
# ---------------------------------------------------------------------------

class TestGooglePatentsParser:
    """Tests for the Google Patents parser."""

    def setup_method(self) -> None:
        self.parser = GooglePatentsParser()

    def test_parse_valid_patents(self) -> None:
        raw = {
            "results": [
                {"patent_number": "US1234567A", "title": "Invention", "abstract": "An invention."},
            ]
        }
        records = self.parser.parse_patent(raw)
        assert len(records) == 1
        assert records[0].source == "Google Patents"
        assert records[0].patent_number == "US1234567A"

    def test_parse_paper_returns_empty(self) -> None:
        assert self.parser.parse_paper({"results": []}) == []

    def test_malformed_record_skipped(self, caplog: pytest.LogCaptureFixture) -> None:
        raw = {
            "results": [
                {"patent_number": "US1", "title": "OK", "abstract": "Fine"},
                {"abstract": "No number or title"},  # malformed
            ]
        }
        with caplog.at_level(logging.WARNING):
            records = self.parser.parse_patent(raw)
        assert len(records) == 1
        assert any("malformed" in msg.lower() for msg in caplog.messages)

    def test_empty_results(self) -> None:
        assert self.parser.parse_patent({"results": []}) == []


# ---------------------------------------------------------------------------
# Google Scholar
# ---------------------------------------------------------------------------

class TestGoogleScholarParser:
    """Tests for the Google Scholar parser."""

    def setup_method(self) -> None:
        self.parser = GoogleScholarParser()

    def test_parse_valid_papers(self) -> None:
        raw = {
            "results": [
                {
                    "title": "Deep Learning Survey",
                    "abstract": "A survey of deep learning.",
                    "citation_count": 500,
                    "source_url": "https://scholar.google.com/1",
                },
            ]
        }
        records = self.parser.parse_paper(raw)
        assert len(records) == 1
        assert isinstance(records[0], ScientificPaperRecord)
        assert records[0].source == "Google Scholar"
        assert records[0].title == "Deep Learning Survey"
        # doi defaults to empty string when not provided
        assert records[0].doi == ""

    def test_parse_patent_returns_empty(self) -> None:
        assert self.parser.parse_patent({"results": []}) == []

    def test_malformed_record_skipped(self, caplog: pytest.LogCaptureFixture) -> None:
        raw = {
            "results": [
                {"title": "Good Paper", "abstract": "OK", "citation_count": 10, "source_url": "http://x"},
                {"citation_count": 5},  # missing title and abstract
            ]
        }
        with caplog.at_level(logging.WARNING):
            records = self.parser.parse_paper(raw)
        assert len(records) == 1
        assert any("malformed" in msg.lower() for msg in caplog.messages)

    def test_empty_results(self) -> None:
        assert self.parser.parse_paper({"results": []}) == []


# ---------------------------------------------------------------------------
# ArXiv
# ---------------------------------------------------------------------------

class TestArXivParser:
    """Tests for the ArXiv parser."""

    def setup_method(self) -> None:
        self.parser = ArXivParser()

    def test_parse_valid_papers(self) -> None:
        raw = {
            "results": [
                {"doi": "10.1234/arxiv.5678", "title": "Quantum Computing", "abstract": "Qubits."},
            ]
        }
        records = self.parser.parse_paper(raw)
        assert len(records) == 1
        assert records[0].source == "ArXiv"
        assert records[0].doi == "10.1234/arxiv.5678"

    def test_parse_patent_returns_empty(self) -> None:
        assert self.parser.parse_patent({"results": []}) == []

    def test_malformed_record_skipped(self, caplog: pytest.LogCaptureFixture) -> None:
        raw = {
            "results": [
                {"doi": "10.1/a", "title": "Good", "abstract": "OK"},
                {"doi": "10.1/b"},  # missing title and abstract
            ]
        }
        with caplog.at_level(logging.WARNING):
            records = self.parser.parse_paper(raw)
        assert len(records) == 1
        assert any("malformed" in msg.lower() for msg in caplog.messages)

    def test_empty_results(self) -> None:
        assert self.parser.parse_paper({"results": []}) == []


# ---------------------------------------------------------------------------
# PubMed
# ---------------------------------------------------------------------------

class TestPubMedParser:
    """Tests for the PubMed parser."""

    def setup_method(self) -> None:
        self.parser = PubMedParser()

    def test_parse_valid_papers(self) -> None:
        raw = {
            "results": [
                {"doi": "10.1000/pubmed.1", "title": "Gene Therapy", "abstract": "CRISPR."},
                {"doi": "10.1000/pubmed.2", "title": "Immunology", "abstract": "T-cells."},
            ]
        }
        records = self.parser.parse_paper(raw)
        assert len(records) == 2
        assert records[0].source == "PubMed"
        assert records[1].title == "Immunology"

    def test_parse_patent_returns_empty(self) -> None:
        assert self.parser.parse_patent({"results": []}) == []

    def test_malformed_record_skipped(self, caplog: pytest.LogCaptureFixture) -> None:
        raw = {
            "results": [
                {"doi": "10.1/x", "title": "OK", "abstract": "Fine"},
                None,  # completely malformed entry
            ]
        }
        with caplog.at_level(logging.WARNING):
            records = self.parser.parse_paper(raw)
        assert len(records) == 1
        assert any("malformed" in msg.lower() for msg in caplog.messages)

    def test_empty_results(self) -> None:
        assert self.parser.parse_paper({"results": []}) == []
