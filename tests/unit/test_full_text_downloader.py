"""Unit tests for the FullTextDownloader service.

Tests cover ArXiv download failure handling, PubMed graceful degradation,
EPO OPS XML extraction, and the download_all skip behavior.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from patent_system.config import AppSettings
from patent_system.services.full_text_downloader import FullTextDownloader


def _make_settings(**overrides) -> AppSettings:
    """Create AppSettings with a temporary PDF download directory."""
    defaults = {
        "full_text_download_enabled": True,
        "search_request_delay_seconds": 0.0,  # no delay in tests
    }
    defaults.update(overrides)
    return AppSettings(**defaults)


# ---------------------------------------------------------------------------
# 4.9: ArXiv download failure returns None and logs warning
# ---------------------------------------------------------------------------


class TestArxivDownloadFailure:
    """ArXiv download failure returns None and logs a warning."""

    @patch("patent_system.services.full_text_downloader.urlopen")
    def test_network_error_returns_none_tuple(self, mock_urlopen, caplog):
        """When the HTTP request fails, both text and path are None."""
        mock_urlopen.side_effect = OSError("Connection refused")

        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _make_settings(pdf_download_dir=Path(tmpdir))
            downloader = FullTextDownloader(settings)

            with caplog.at_level(logging.WARNING, logger="patent_system.services.full_text_downloader"):
                text, pdf_path = downloader.download_arxiv_fulltext(
                    "https://arxiv.org/pdf/2301.12345",
                    "2301.12345",
                )

            assert text is None
            assert pdf_path is None
            assert any("Failed to download ArXiv full text" in r.message for r in caplog.records)

    @patch("patent_system.services.full_text_downloader.urlopen")
    def test_timeout_returns_none_and_logs_warning(self, mock_urlopen, caplog):
        """When the request times out, returns None and logs warning."""
        from urllib.error import URLError

        mock_urlopen.side_effect = URLError("timed out")

        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _make_settings(pdf_download_dir=Path(tmpdir))
            downloader = FullTextDownloader(settings)

            with caplog.at_level(logging.WARNING, logger="patent_system.services.full_text_downloader"):
                text, pdf_path = downloader.download_arxiv_fulltext(
                    "https://arxiv.org/pdf/9999.99999",
                    "9999.99999",
                )

            assert text is None
            assert pdf_path is None
            warning_records = [
                r for r in caplog.records
                if r.levelno >= logging.WARNING and "ArXiv" in r.message
            ]
            assert len(warning_records) >= 1


# ---------------------------------------------------------------------------
# 4.10: PubMed gracefully handles missing PMC full text (no error logged)
# ---------------------------------------------------------------------------


class TestPubmedMissingFullText:
    """PubMed gracefully handles missing PMC full text without error logging."""

    @patch("patent_system.services.full_text_downloader.urlopen")
    def test_no_pmc_id_returns_none_without_error(self, mock_urlopen, caplog):
        """When no PMC ID exists for a PMID, returns None without logging an error."""
        # Simulate elink response with no PMC link
        elink_response = b"""<?xml version="1.0"?>
        <eLinkResult>
            <LinkSet>
                <DbFrom>pubmed</DbFrom>
                <IdList><Id>12345678</Id></IdList>
            </LinkSet>
        </eLinkResult>"""

        mock_resp = MagicMock()
        mock_resp.read.return_value = elink_response
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _make_settings(pdf_download_dir=Path(tmpdir))
            downloader = FullTextDownloader(settings)

            with caplog.at_level(logging.DEBUG, logger="patent_system.services.full_text_downloader"):
                result = downloader.download_pubmed_fulltext("12345678")

            assert result is None
            # No ERROR-level records should be present
            error_records = [
                r for r in caplog.records
                if r.levelno >= logging.ERROR
            ]
            assert len(error_records) == 0

    @patch("patent_system.services.full_text_downloader.urlopen")
    def test_empty_link_set_returns_none_silently(self, mock_urlopen, caplog):
        """When elink returns an empty LinkSetDb, returns None without error."""
        elink_response = b"""<?xml version="1.0"?>
        <eLinkResult>
            <LinkSet>
                <DbFrom>pubmed</DbFrom>
                <IdList><Id>99999999</Id></IdList>
                <LinkSetDb>
                    <DbTo>pmc</DbTo>
                </LinkSetDb>
            </LinkSet>
        </eLinkResult>"""

        mock_resp = MagicMock()
        mock_resp.read.return_value = elink_response
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _make_settings(pdf_download_dir=Path(tmpdir))
            downloader = FullTextDownloader(settings)

            with caplog.at_level(logging.DEBUG, logger="patent_system.services.full_text_downloader"):
                result = downloader.download_pubmed_fulltext("99999999")

            assert result is None
            error_records = [
                r for r in caplog.records
                if r.levelno >= logging.ERROR
            ]
            assert len(error_records) == 0


# ---------------------------------------------------------------------------
# 4.11: EPO OPS fulltext extraction from sample XML
# ---------------------------------------------------------------------------


class TestEpoOpsFulltextExtraction:
    """EPO OPS fulltext extraction from sample XML."""

    def test_extracts_description_and_claims_from_xml(self):
        """Parses description and claims from a sample EPO OPS fulltext XML."""
        sample_xml = """<?xml version="1.0"?>
        <ops:world-patent-data xmlns:ops="http://ops.epo.org"
                               xmlns:ftxt="http://www.epo.org/fulltext">
            <ftxt:fulltext-documents>
                <ftxt:fulltext-document>
                    <ftxt:description lang="en">
                        <ftxt:p>This invention relates to a method for processing data.</ftxt:p>
                        <ftxt:p>The method comprises the steps of receiving input and generating output.</ftxt:p>
                    </ftxt:description>
                    <ftxt:claims lang="en">
                        <ftxt:claim-text>1. A method for processing data comprising receiving input.</ftxt:claim-text>
                        <ftxt:claim-text>2. The method of claim 1, further comprising generating output.</ftxt:claim-text>
                    </ftxt:claims>
                </ftxt:fulltext-document>
            </ftxt:fulltext-documents>
        </ops:world-patent-data>"""

        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _make_settings(pdf_download_dir=Path(tmpdir))
            downloader = FullTextDownloader(settings)
            description, claims = downloader._parse_epo_fulltext_xml(sample_xml)

        assert description is not None
        assert "method for processing data" in description
        assert "receiving input and generating output" in description

        assert claims is not None
        assert "A method for processing data" in claims
        assert "method of claim 1" in claims

    def test_returns_none_for_empty_fulltext(self):
        """Returns (None, None) when the XML has no description or claims."""
        empty_xml = """<?xml version="1.0"?>
        <ops:world-patent-data xmlns:ops="http://ops.epo.org"
                               xmlns:ftxt="http://www.epo.org/fulltext">
            <ftxt:fulltext-documents>
                <ftxt:fulltext-document>
                </ftxt:fulltext-document>
            </ftxt:fulltext-documents>
        </ops:world-patent-data>"""

        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _make_settings(pdf_download_dir=Path(tmpdir))
            downloader = FullTextDownloader(settings)
            description, claims = downloader._parse_epo_fulltext_xml(empty_xml)

        assert description is None
        assert claims is None

    def test_extracts_description_only_when_no_claims(self):
        """Extracts description when claims section is absent."""
        xml_no_claims = """<?xml version="1.0"?>
        <ops:world-patent-data xmlns:ops="http://ops.epo.org"
                               xmlns:ftxt="http://www.epo.org/fulltext">
            <ftxt:fulltext-documents>
                <ftxt:fulltext-document>
                    <ftxt:description lang="en">
                        <ftxt:p>A novel approach to widget manufacturing.</ftxt:p>
                    </ftxt:description>
                </ftxt:fulltext-document>
            </ftxt:fulltext-documents>
        </ops:world-patent-data>"""

        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _make_settings(pdf_download_dir=Path(tmpdir))
            downloader = FullTextDownloader(settings)
            description, claims = downloader._parse_epo_fulltext_xml(xml_no_claims)

        assert description is not None
        assert "widget manufacturing" in description
        assert claims is None


# ---------------------------------------------------------------------------
# 4.12: download_all skips downloads when full_text_download_enabled is False
# ---------------------------------------------------------------------------


class TestDownloadAllSkipsWhenDisabled:
    """download_all skips downloads when full_text_download_enabled is False."""

    def test_returns_results_unchanged_when_disabled(self):
        """When disabled, results are returned without modification."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _make_settings(
                pdf_download_dir=Path(tmpdir),
                full_text_download_enabled=False,
            )
            downloader = FullTextDownloader(settings)

            results = [
                {"source": "ArXiv", "doi": "2301.12345", "title": "Test", "pdf_path": "https://arxiv.org/pdf/2301.12345"},
                {"source": "PubMed", "doi": "99999999", "title": "Test 2"},
                {"source": "EPO OPS", "patent_number": "EP1234567A1", "title": "Test 3"},
            ]
            original_results = [dict(r) for r in results]  # deep copy

            returned = downloader.download_all(results)

            assert returned == original_results

    @patch.object(FullTextDownloader, "download_arxiv_fulltext")
    @patch.object(FullTextDownloader, "download_pubmed_fulltext")
    @patch.object(FullTextDownloader, "download_epo_ops_fulltext")
    @patch.object(FullTextDownloader, "download_google_patents_fulltext")
    def test_no_download_methods_called_when_disabled(
        self,
        mock_google,
        mock_epo,
        mock_pubmed,
        mock_arxiv,
    ):
        """When disabled, no source-specific download methods are called."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _make_settings(
                pdf_download_dir=Path(tmpdir),
                full_text_download_enabled=False,
            )
            downloader = FullTextDownloader(settings)

            results = [
                {"source": "ArXiv", "doi": "2301.12345", "pdf_path": "https://arxiv.org/pdf/2301.12345"},
                {"source": "PubMed", "doi": "99999999"},
                {"source": "EPO OPS", "patent_number": "EP1234567A1"},
                {"source": "Google Patents", "patent_number": "US10123456B2"},
            ]

            downloader.download_all(results)

            mock_arxiv.assert_not_called()
            mock_pubmed.assert_not_called()
            mock_epo.assert_not_called()
            mock_google.assert_not_called()

    def test_progress_callback_not_called_when_disabled(self):
        """When disabled, the progress callback is never invoked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _make_settings(
                pdf_download_dir=Path(tmpdir),
                full_text_download_enabled=False,
            )
            downloader = FullTextDownloader(settings)
            callback = MagicMock()

            results = [{"source": "ArXiv", "doi": "123", "pdf_path": "http://example.com"}]
            downloader.download_all(results, progress_callback=callback)

            callback.assert_not_called()
