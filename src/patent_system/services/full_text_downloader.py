"""Full-text downloader service for external prior art sources.

Downloads complete document content from ArXiv (PDF), PubMed (PMC XML),
EPO OPS (fulltext endpoint), and Google Patents (HTML scraping). Saves
PDFs to disk and extracts text using the shared text extraction module.

Requirements: 1, 2, 3, 4, 9, 10, 11
"""

from __future__ import annotations

import logging
import re
import time
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from patent_system.config import AppSettings
from patent_system.services.text_extraction import extract_text_from_pdf

logger = logging.getLogger(__name__)

_HTTP_TIMEOUT = 30  # seconds for download requests


def _sanitize_filename(name: str) -> str:
    """Sanitize a string for safe use as a filename component.

    Replaces characters that are unsafe on common filesystems with
    underscores and strips leading/trailing whitespace and dots.

    Args:
        name: Raw identifier or source name.

    Returns:
        A filesystem-safe string. Returns ``_unnamed`` if the result
        would be empty.
    """
    # Replace path separators, null bytes, and common unsafe chars
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)
    sanitized = sanitized.strip(". ")
    return sanitized if sanitized else "_unnamed"


class _PatentHTMLExtractor(HTMLParser):
    """Extract description and claims text from a Google Patents HTML page.

    Looks for ``<section itemprop="description">`` and
    ``<section itemprop="claims">`` and collects their text content.
    """

    def __init__(self) -> None:
        super().__init__()
        self._in_description = False
        self._in_claims = False
        self._description_parts: list[str] = []
        self._claims_parts: list[str] = []
        self._depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_dict = dict(attrs)
        if tag == "section":
            itemprop = attr_dict.get("itemprop", "")
            if itemprop == "description":
                self._in_description = True
                self._depth = 1
                return
            elif itemprop == "claims":
                self._in_claims = True
                self._depth = 1
                return
        if self._in_description or self._in_claims:
            self._depth += 1

    def handle_endtag(self, tag: str) -> None:
        if self._in_description or self._in_claims:
            self._depth -= 1
            if self._depth <= 0:
                self._in_description = False
                self._in_claims = False
                self._depth = 0

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if not text:
            return
        if self._in_description:
            self._description_parts.append(text)
        elif self._in_claims:
            self._claims_parts.append(text)

    @property
    def description(self) -> str:
        return "\n".join(self._description_parts)

    @property
    def claims(self) -> str:
        return "\n".join(self._claims_parts)


class FullTextDownloader:
    """Downloads full-text content from external prior art sources.

    Supports ArXiv (PDF download + text extraction), PubMed (PMC XML),
    EPO OPS (fulltext endpoint), and Google Patents (HTML scraping).

    Args:
        settings: Application settings providing rate limit delay,
            PDF download directory, and feature flags.
    """

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._rate_limit_delay = settings.search_request_delay_seconds
        self._pdf_download_dir = settings.pdf_download_dir
        self._full_text_download_enabled = settings.full_text_download_enabled

    # ------------------------------------------------------------------
    # PDF persistence
    # ------------------------------------------------------------------

    def save_pdf(
        self,
        content_bytes: bytes,
        source: str,
        identifier: str,
    ) -> Path:
        """Save PDF bytes to disk under ``{pdf_download_dir}/{source}/{id}.pdf``.

        Creates intermediate directories as needed.

        Args:
            content_bytes: Raw PDF file content.
            source: Source name (e.g. ``"arxiv"``).
            identifier: Document identifier (e.g. ``"2301.12345"``).

        Returns:
            Absolute path to the saved PDF file.
        """
        safe_source = _sanitize_filename(source)
        safe_id = _sanitize_filename(identifier)
        target_dir = self._pdf_download_dir / safe_source
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / f"{safe_id}.pdf"
        target_path.write_bytes(content_bytes)
        logger.info("Saved PDF to %s", target_path)
        return target_path.resolve()

    # ------------------------------------------------------------------
    # Source-specific downloaders
    # ------------------------------------------------------------------

    def download_arxiv_fulltext(
        self,
        pdf_url: str,
        identifier: str,
    ) -> tuple[str | None, Path | None]:
        """Download an ArXiv PDF, save to disk, and extract text.

        Args:
            pdf_url: URL to the ArXiv PDF.
            identifier: ArXiv paper identifier (e.g. ``"2301.12345"``).

        Returns:
            Tuple of (extracted_text, saved_pdf_path). Both are ``None``
            when the download or extraction fails.
        """
        try:
            req = Request(
                pdf_url,
                headers={"User-Agent": "mPAPA/1.0 (Patent Research Tool)"},
            )
            with urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
                pdf_bytes = resp.read()

            pdf_path = self.save_pdf(pdf_bytes, "arxiv", identifier)
            text = extract_text_from_pdf(pdf_bytes, f"{identifier}.pdf")
            if text and text.strip():
                return text, pdf_path
            logger.warning(
                "ArXiv PDF text extraction returned empty for %s",
                identifier,
            )
            return None, pdf_path
        except Exception:
            logger.warning(
                "Failed to download ArXiv full text for %s from %s",
                identifier,
                pdf_url,
                exc_info=True,
            )
            return None, None

    def download_pubmed_fulltext(self, pmid: str) -> str | None:
        """Attempt to retrieve full text for a PubMed paper via PMC.

        Queries the NCBI efetch API for the PMC full-text XML. Returns
        ``None`` silently (no error logged) when the paper is not
        available in PMC — this is expected for paywalled papers.

        Args:
            pmid: PubMed identifier.

        Returns:
            Extracted body text, or ``None`` if unavailable.
        """
        try:
            # First check if there's a PMC ID for this PMID
            elink_params = urlencode({
                "dbfrom": "pubmed",
                "db": "pmc",
                "id": pmid,
                "retmode": "xml",
            })
            elink_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi?{elink_params}"
            req = Request(
                elink_url,
                headers={"User-Agent": "mPAPA/1.0 (Patent Research Tool)"},
            )
            with urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
                elink_xml = resp.read().decode("utf-8", errors="replace")

            elink_root = ET.fromstring(elink_xml)
            # Look for PMC ID in the link set
            pmc_id = None
            for link_set in elink_root.findall(".//LinkSetDb"):
                db_to = link_set.findtext("DbTo", default="")
                if db_to == "pmc":
                    link_el = link_set.find(".//Link/Id")
                    if link_el is not None and link_el.text:
                        pmc_id = link_el.text
                        break

            if not pmc_id:
                # No PMC full text available — expected for paywalled papers
                logger.debug(
                    "No PMC full text available for PMID %s (expected for paywalled papers)",
                    pmid,
                )
                return None

            # Fetch the full-text XML from PMC
            efetch_params = urlencode({
                "db": "pmc",
                "id": pmc_id,
                "retmode": "xml",
            })
            efetch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?{efetch_params}"
            req = Request(
                efetch_url,
                headers={"User-Agent": "mPAPA/1.0 (Patent Research Tool)"},
            )
            with urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
                pmc_xml = resp.read().decode("utf-8", errors="replace")

            pmc_root = ET.fromstring(pmc_xml)
            # Extract body text from all <body> <p> elements
            body_parts: list[str] = []
            for body in pmc_root.findall(".//body"):
                for p in body.iter("p"):
                    # Collect all text content including tail text
                    text = "".join(p.itertext()).strip()
                    if text:
                        body_parts.append(text)

            if body_parts:
                return "\n".join(body_parts)

            logger.debug("PMC article %s has no extractable body text", pmc_id)
            return None

        except Exception:
            logger.warning(
                "Failed to fetch PMC full text for PMID %s",
                pmid,
                exc_info=True,
            )
            return None

    def download_epo_ops_fulltext(
        self,
        patent_number: str,
    ) -> tuple[str | None, str | None]:
        """Fetch full text from the EPO OPS published-data/fulltext endpoint.

        Uses the existing ``epo_ops`` client pattern from
        ``prior_art_search.py``.

        Args:
            patent_number: Patent number (e.g. ``"EP1234567A1"``).

        Returns:
            Tuple of (description, claims). Either or both may be
            ``None`` if the endpoint returns no content.
        """
        try:
            import epo_ops

            from patent_system.config import AppSettings

            settings = self._settings
            if not settings.epo_ops_key or not settings.epo_ops_secret:
                logger.debug("EPO OPS credentials not configured, skipping fulltext fetch")
                return None, None

            client = epo_ops.Client(
                key=settings.epo_ops_key,
                secret=settings.epo_ops_secret,
                accept_type="xml",
            )

            # Parse patent number into components for the API
            # Common formats: EP1234567A1, US20210012345A1, DE102020001234A1
            match = re.match(r"^([A-Z]{2})(\d+)([A-Z]\d?)?$", patent_number)
            if not match:
                logger.debug(
                    "Cannot parse patent number '%s' for EPO OPS fulltext",
                    patent_number,
                )
                return None, None

            country = match.group(1)
            doc_number = match.group(2)
            kind = match.group(3) or ""

            # Build the reference for the fulltext endpoint
            input_ref = epo_ops.models.Docdb(doc_number, country, kind)
            response = client.published_data(
                reference_type="publication",
                input=input_ref,
                endpoint="fulltext",
            )

            xml_text = response.text
            return self._parse_epo_fulltext_xml(xml_text)

        except Exception:
            logger.debug(
                "EPO OPS fulltext fetch failed for %s",
                patent_number,
                exc_info=True,
            )
            return None, None

    def _parse_epo_fulltext_xml(self, xml_text: str) -> tuple[str | None, str | None]:
        """Parse EPO OPS fulltext XML response for description and claims.

        Args:
            xml_text: Raw XML response from the EPO OPS fulltext endpoint.

        Returns:
            Tuple of (description, claims).
        """
        ns = {
            "ops": "http://ops.epo.org",
            "ftxt": "http://www.epo.org/fulltext",
        }

        root = ET.fromstring(xml_text)

        description_parts: list[str] = []
        for desc in root.findall(".//ftxt:description", ns):
            for p in desc.findall(".//ftxt:p", ns):
                text = "".join(p.itertext()).strip()
                if text:
                    description_parts.append(text)

        claims_parts: list[str] = []
        for claim_set in root.findall(".//ftxt:claims", ns):
            for claim in claim_set.findall(".//ftxt:claim-text", ns):
                text = "".join(claim.itertext()).strip()
                if text:
                    claims_parts.append(text)

        description = "\n".join(description_parts) if description_parts else None
        claims = "\n".join(claims_parts) if claims_parts else None
        return description, claims

    def download_google_patents_fulltext(
        self,
        patent_number: str,
    ) -> tuple[str | None, str | None]:
        """Fetch full text from a Google Patents page.

        Downloads the patent page HTML and extracts description and
        claims sections.

        Args:
            patent_number: Patent number (e.g. ``"US10123456B2"``).

        Returns:
            Tuple of (description, claims). Either or both may be
            ``None`` if extraction fails.
        """
        try:
            url = f"https://patents.google.com/patent/{patent_number}/en"
            req = Request(
                url,
                headers={"User-Agent": "mPAPA/1.0 (Patent Research Tool)"},
            )
            with urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
                html_bytes = resp.read()
                charset = resp.headers.get_content_charset() or "utf-8"
                try:
                    html_text = html_bytes.decode(charset)
                except (UnicodeDecodeError, LookupError):
                    html_text = html_bytes.decode("utf-8", errors="replace")

            extractor = _PatentHTMLExtractor()
            extractor.feed(html_text)

            description = extractor.description if extractor.description.strip() else None
            claims = extractor.claims if extractor.claims.strip() else None
            return description, claims

        except Exception:
            logger.debug(
                "Google Patents fulltext fetch failed for %s",
                patent_number,
                exc_info=True,
            )
            return None, None

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------

    def download_all(
        self,
        results: list[dict],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[dict]:
        """Download full text for all results, dispatching by source.

        Iterates through results, calls the appropriate source-specific
        download method, respects rate limits between requests, and
        reports progress via the callback.

        When ``full_text_download_enabled`` is ``False``, returns the
        results unchanged without attempting any downloads.

        Args:
            results: List of search result dicts. Each must have a
                ``"source"`` key.
            progress_callback: Optional callable receiving
                ``(current_index, total)`` after each download.

        Returns:
            The same list of result dicts, enriched with ``full_text``,
            ``claims``, and ``pdf_path`` fields where available.
        """
        if not self._full_text_download_enabled:
            return results

        total = len(results)
        for i, result in enumerate(results):
            source = result.get("source", "")
            self._download_single(result, source)

            if progress_callback is not None:
                progress_callback(i + 1, total)

            # Rate limit between consecutive downloads
            if i < total - 1:
                time.sleep(self._rate_limit_delay)

        return results

    def _download_single(self, result: dict, source: str) -> None:
        """Dispatch a single result to the appropriate downloader.

        Modifies the result dict in-place with ``full_text``,
        ``claims``, and/or ``pdf_path`` fields.
        """
        source_lower = source.lower()

        if source_lower == "arxiv":
            pdf_url = result.get("pdf_path", "")
            identifier = result.get("doi", "") or result.get("id", "")
            if pdf_url and identifier:
                text, pdf_path = self.download_arxiv_fulltext(pdf_url, identifier)
                if text:
                    result["full_text"] = text
                if pdf_path:
                    result["pdf_path"] = str(pdf_path)

        elif source_lower == "pubmed":
            pmid = result.get("doi", "") or result.get("pmid", "")
            if pmid:
                text = self.download_pubmed_fulltext(pmid)
                if text:
                    result["full_text"] = text

        elif source_lower == "epo ops":
            patent_number = result.get("patent_number", "")
            if patent_number:
                description, claims = self.download_epo_ops_fulltext(patent_number)
                if description:
                    result["full_text"] = description
                if claims:
                    result["claims"] = claims

        elif source_lower == "google patents":
            patent_number = result.get("patent_number", "")
            if patent_number:
                description, claims = self.download_google_patents_fulltext(patent_number)
                if description:
                    result["full_text"] = description
                if claims:
                    result["claims"] = claims

        # Google Scholar: no full-text download (paywalled third-party papers)
