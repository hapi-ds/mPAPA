"""Prior Art Search Agent for the patent drafting pipeline.

Queries DEPATISnet, Google Patents, Google Scholar, ArXiv, and PubMed
for prior art related to the invention disclosure. Uses source-specific
parsers to structure results and handles source failures gracefully.

Requirements: 1.1, 1.5, 1.6, 1.7, 1.8, 1.9, 1.10, 3.1, 3.2, 3.3, 3.4, 3.7
"""

import logging
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from html.parser import HTMLParser
from typing import Any
from urllib.error import URLError
from urllib.parse import quote_plus, urlencode
from urllib.request import Request, urlopen

from patent_system.exceptions import SourceUnavailableError
from patent_system.logging_config import log_agent_invocation, log_external_request
from patent_system.parsers.arxiv_parser import ArXivParser
from patent_system.parsers.epo_ops import EPOOPSParser
from patent_system.parsers.google_patents import GooglePatentsParser
from patent_system.parsers.google_scholar import GoogleScholarParser
from patent_system.parsers.pubmed import PubMedParser

from patent_system.agents.state import PatentWorkflowState

logger = logging.getLogger(__name__)

# Source registry: maps source name to (query function, parser instance)
_SOURCE_REGISTRY: dict[str, dict[str, Any]] = {
    "EPO OPS": {"parser": EPOOPSParser(), "type": "patent"},
    "Google Patents": {"parser": GooglePatentsParser(), "type": "patent"},
    "Google Scholar": {"parser": GoogleScholarParser(), "type": "paper"},
    "ArXiv": {"parser": ArXivParser(), "type": "paper"},
    "PubMed": {"parser": PubMedParser(), "type": "paper"},
}


def _derive_search_terms(disclosure: dict | None) -> list[str]:
    """Derive search terms from an invention disclosure.

    Uses the novel_features list as focused search terms. Falls back to
    a truncated version of the technical_problem if no features are
    provided. The implementation_details field is excluded — it contains
    prior art text, not search keywords.

    Args:
        disclosure: The invention disclosure dict, or None.

    Returns:
        List of search term strings. Returns a single empty-string
        term if disclosure is None or empty.
    """
    if not disclosure:
        return [""]

    # Handle string disclosures (e.g. from the interactive workflow
    # where _initial_idea_node serializes the dict to a string).
    if isinstance(disclosure, str):
        try:
            import json
            disclosure = json.loads(disclosure)
        except (json.JSONDecodeError, TypeError):
            # Plain text disclosure — use it directly as a search term
            truncated = disclosure[:200].strip()
            return [truncated] if truncated else [""]

    terms: list[str] = []

    # Use novel_features as the primary search terms (short, focused)
    novel_features = disclosure.get("novel_features", [])
    if isinstance(novel_features, list):
        for feature in novel_features:
            if feature and isinstance(feature, str) and feature.strip():
                terms.append(str(feature).strip())

    # If no features, fall back to a truncated technical_problem
    if not terms:
        technical_problem = disclosure.get("technical_problem", "")
        if technical_problem:
            # Truncate to first 200 chars to keep URLs reasonable
            truncated = technical_problem[:200].strip()
            if truncated:
                terms.append(truncated)

    return terms if terms else [""]


# Source endpoints mapping source names to their base URLs.
_SOURCE_ENDPOINTS: dict[str, str] = {
    "ArXiv": "export.arxiv.org/api/query",
    "PubMed": "eutils.ncbi.nlm.nih.gov/entrez/eutils",
    "Google Scholar": "scholar.google.com/scholar",
    "Google Patents": "patents.google.com",
}

_HTTP_TIMEOUT = 15  # seconds

# Default maximum results per source (overridden by AppSettings.search_max_results_per_source)
_DEFAULT_MAX_RESULTS = 10

# Minimum seconds between requests to the same source (rate limiting)
# Loaded from AppSettings at first use; default 3.0s
_REQUEST_DELAY: float | None = None


def _get_request_delay() -> float:
    """Lazily load the request delay from settings."""
    global _REQUEST_DELAY
    if _REQUEST_DELAY is None:
        try:
            from patent_system.config import AppSettings
            _REQUEST_DELAY = AppSettings().search_request_delay_seconds
        except Exception:
            _REQUEST_DELAY = 3.0
    return _REQUEST_DELAY

# EPO OPS client (lazily initialized)
_epo_ops_client = None


def _http_get(url: str) -> str:
    """Perform an HTTP GET request and return the response body as a string.

    Args:
        url: The full URL to request.

    Returns:
        Response body decoded using the charset declared by the server,
        falling back to UTF-8, then Latin-1 as a last resort.

    Raises:
        URLError: On network errors.
        OSError: On connection failures.
    """
    req = Request(url, headers={"User-Agent": "mPAPA/1.0 (Patent Research Tool)"})
    with urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
        raw = resp.read()
        # Respect the Content-Type charset when available
        charset = resp.headers.get_content_charset()
        if charset:
            try:
                return raw.decode(charset)
            except (UnicodeDecodeError, LookupError):
                pass
        # Try UTF-8 first, fall back to Latin-1 (never fails)
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return raw.decode("latin-1")


def _query_arxiv(search_terms: list[str], max_results: int = _DEFAULT_MAX_RESULTS) -> dict:
    """Query ArXiv API, one term at a time, and merge results.

    Args:
        search_terms: List of search term strings.
        max_results: Maximum number of results to return.

    Returns:
        Dict with ``results`` key containing parsed paper entries.
    """
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    all_results: list[dict] = []
    seen_ids: set[str] = set()

    for i, term in enumerate(search_terms):
        if not term or not term.strip():
            continue
        if len(all_results) >= max_results:
            break
        if i > 0:
            time.sleep(_get_request_delay())

        query = quote_plus(term.strip())
        remaining = max_results - len(all_results)
        url = f"http://{_SOURCE_ENDPOINTS['ArXiv']}?search_query=all:{query}&max_results={remaining}"

        log_external_request(
            logger=logger, source="ArXiv", query=term,
            status="requesting", response_time_ms=0,
        )

        try:
            xml_text = _http_get(url)
        except Exception:
            logger.debug("ArXiv query failed for term: %s", term, exc_info=True)
            continue

        root = ET.fromstring(xml_text)
        for entry in root.findall("atom:entry", ns):
            if len(all_results) >= max_results:
                break
            entry_id = entry.findtext("atom:id", default="", namespaces=ns)
            if entry_id in seen_ids:
                continue
            seen_ids.add(entry_id)
            title = entry.findtext("atom:title", default="", namespaces=ns).strip()
            abstract = entry.findtext("atom:summary", default="", namespaces=ns).strip()
            pdf_link = ""
            for link_el in entry.findall("atom:link", ns):
                if link_el.get("title") == "pdf":
                    pdf_link = link_el.get("href", "")
                    break
            if title:
                all_results.append({
                    "doi": entry_id,
                    "title": title,
                    "abstract": abstract,
                    "full_text": abstract,
                    "pdf_path": pdf_link,
                })

    return {"results": all_results}


def _query_pubmed(search_terms: list[str], max_results: int = _DEFAULT_MAX_RESULTS) -> dict:
    """Query PubMed, one term at a time, and merge results.

    Args:
        search_terms: List of search term strings.
        max_results: Maximum number of results to return.

    Returns:
        Dict with ``results`` key containing parsed paper entries.
    """
    base = _SOURCE_ENDPOINTS["PubMed"]
    all_results: list[dict] = []
    seen_pmids: set[str] = set()

    for i, term in enumerate(search_terms):
        if not term or not term.strip():
            continue
        if len(all_results) >= max_results:
            break
        if i > 0:
            time.sleep(_get_request_delay())

        remaining = max_results - len(all_results)
        esearch_params = urlencode({
            "db": "pubmed", "term": term.strip(),
            "retmax": str(remaining), "retmode": "xml",
        })
        esearch_url = f"https://{base}/esearch.fcgi?{esearch_params}"

        log_external_request(
            logger=logger, source="PubMed", query=term,
            status="requesting", response_time_ms=0,
        )

        try:
            esearch_xml = _http_get(esearch_url)
        except Exception:
            logger.debug("PubMed esearch failed for term: %s", term, exc_info=True)
            continue

        esearch_root = ET.fromstring(esearch_xml)
        id_list = esearch_root.findall(".//Id")
        pmids = [
            id_el.text for id_el in id_list
            if id_el.text and id_el.text not in seen_pmids
        ]

        if not pmids:
            continue

        # efetch to get details
        efetch_params = urlencode({
            "db": "pubmed", "id": ",".join(pmids), "retmode": "xml",
        })
        efetch_url = f"https://{base}/efetch.fcgi?{efetch_params}"

        try:
            efetch_xml = _http_get(efetch_url)
        except Exception:
            logger.debug("PubMed efetch failed for term: %s", term, exc_info=True)
            continue

        efetch_root = ET.fromstring(efetch_xml)
        for article in efetch_root.findall(".//PubmedArticle"):
            if len(all_results) >= max_results:
                break
            pmid_el = article.find(".//PMID")
            pmid = pmid_el.text if pmid_el is not None and pmid_el.text else ""
            if pmid in seen_pmids:
                continue
            seen_pmids.add(pmid)

            title_el = article.find(".//ArticleTitle")
            title = title_el.text if title_el is not None and title_el.text else ""

            abstract_parts: list[str] = []
            for abs_el in article.findall(".//AbstractText"):
                label = abs_el.get("Label", "")
                text = abs_el.text or ""
                if label and text:
                    abstract_parts.append(f"{label}: {text}")
                elif text:
                    abstract_parts.append(text)
            abstract = "\n".join(abstract_parts) if abstract_parts else ""

            if title:
                all_results.append({
                    "doi": pmid,
                    "title": title,
                    "abstract": abstract,
                    "full_text": abstract if abstract else None,
                })

    return {"results": all_results}


class _SimpleHTMLTextExtractor(HTMLParser):
    """Minimal HTML parser that extracts text content from tags."""

    def __init__(self) -> None:
        super().__init__()
        self._text_parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self._text_parts.append(data)

    def get_text(self) -> str:
        return " ".join(self._text_parts)


def _query_google_scholar(search_terms: list[str], max_results: int = _DEFAULT_MAX_RESULTS) -> dict:
    """Query Google Scholar, one term at a time, and merge results.

    Args:
        search_terms: List of search term strings.
        max_results: Maximum number of results to return.

    Returns:
        Dict with ``results`` key containing parsed paper entries.
    """
    all_results: list[dict] = []
    seen_titles: set[str] = set()

    for i, term in enumerate(search_terms):
        if not term or not term.strip():
            continue
        if len(all_results) >= max_results:
            break
        if i > 0:
            time.sleep(_get_request_delay())

        query = quote_plus(term.strip())
        url = f"https://{_SOURCE_ENDPOINTS['Google Scholar']}?q={query}&hl=en"

        log_external_request(
            logger=logger, source="Google Scholar", query=term,
            status="requesting", response_time_ms=0,
        )

        try:
            html_text = _http_get(url)
        except Exception:
            logger.debug("Google Scholar query failed for term: %s", term, exc_info=True)
            continue

        parts = html_text.split('class="gs_ri"')
        for part in parts[1:]:
            if len(all_results) >= max_results:
                break
            title = ""
            abstract = ""
            h3_start = part.find("<h3")
            if h3_start != -1:
                h3_end = part.find("</h3>", h3_start)
                if h3_end != -1:
                    extractor = _SimpleHTMLTextExtractor()
                    extractor.feed(part[h3_start:h3_end])
                    title = extractor.get_text().strip()
            gs_rs_start = part.find('class="gs_rs"')
            if gs_rs_start != -1:
                div_end = part.find("</div>", gs_rs_start)
                if div_end != -1:
                    extractor = _SimpleHTMLTextExtractor()
                    extractor.feed(part[gs_rs_start:div_end])
                    abstract = extractor.get_text().strip()
            title_key = title.lower().strip()
            if title and title_key not in seen_titles:
                seen_titles.add(title_key)
                all_results.append({"doi": "", "title": title, "abstract": abstract})

    return {"results": all_results[:max_results]}


def _query_google_patents(search_terms: list[str], max_results: int = _DEFAULT_MAX_RESULTS) -> dict:
    """Query Google Patents via their JSON XHR endpoint.

    Queries each term separately and merges results to avoid URL length
    limits and OR syntax issues.

    Args:
        search_terms: List of search term strings.
        max_results: Maximum number of results to return.

    Returns:
        Dict with ``results`` key containing parsed patent entries.
    """
    import json

    all_results: list[dict] = []
    seen_ids: set[str] = set()

    for i, term in enumerate(search_terms):
        if not term or not term.strip():
            continue
        if len(all_results) >= max_results:
            break
        if i > 0:
            time.sleep(_get_request_delay())

        encoded_q = quote_plus(term.strip())
        url = f"https://{_SOURCE_ENDPOINTS['Google Patents']}/xhr/query?url=q%3D{encoded_q}&exp="

        log_external_request(
            logger=logger, source="Google Patents", query=term,
            status="requesting", response_time_ms=0,
        )

        try:
            json_text = _http_get(url)
            data = json.loads(json_text)
        except Exception:
            logger.debug("Google Patents query failed for term: %s", term, exc_info=True)
            continue

        cluster = data.get("results", {}).get("cluster", [])
        for group in cluster:
            for item in group.get("result", []):
                if len(all_results) >= max_results:
                    break
                patent = item.get("patent", {})
                title = patent.get("title", "").strip()
                snippet = patent.get("snippet", "").strip()
                patent_id = item.get("id", "")
                patent_number = patent_id.replace("patent/", "").replace("/en", "").strip()
                if title and patent_number not in seen_ids:
                    seen_ids.add(patent_number)
                    all_results.append({
                        "patent_number": patent_number or "UNKNOWN",
                        "title": title,
                        "abstract": snippet,
                    })

    return {"results": all_results[:max_results]}


def _query_epo_ops(search_terms: list[str], max_results: int = _DEFAULT_MAX_RESULTS) -> dict:
    """Query EPO Open Patent Services (OPS) published-data search.

    Uses the python-epo-ops-client library with OAuth2 authentication.
    Requires PATENT_EPO_OPS_KEY and PATENT_EPO_OPS_SECRET in settings.

    Args:
        search_terms: List of search term strings.
        max_results: Maximum number of results to return.

    Returns:
        Dict with ``results`` key containing parsed patent entries.
    """
    import xml.etree.ElementTree as _ET

    import epo_ops

    global _epo_ops_client

    if _epo_ops_client is None:
        from patent_system.config import AppSettings
        settings = AppSettings()
        if not settings.epo_ops_key or not settings.epo_ops_secret:
            raise SourceUnavailableError(
                "EPO OPS",
                ValueError("EPO OPS credentials not configured. Set PATENT_EPO_OPS_KEY and PATENT_EPO_OPS_SECRET in .env"),
            )
        _epo_ops_client = epo_ops.Client(
            key=settings.epo_ops_key,
            secret=settings.epo_ops_secret,
            accept_type="xml",
        )

    # Query each term separately and merge results (same pattern as
    # Google Patents / Google Scholar) to avoid CQL length limits.
    import re as _re

    all_results: list[dict] = []
    seen_numbers: set[str] = set()
    ns = {
        "ops": "http://ops.epo.org",
        "exch": "http://www.epo.org/exchange",
    }

    for i, term in enumerate(search_terms):
        if not term or not term.strip():
            continue
        if len(all_results) >= max_results:
            break
        if i > 0:
            time.sleep(_get_request_delay())

        # Sanitize: strip quotes, apostrophes, wildcards for CQL safety
        cleaned = _re.sub(r'["\'\*]', '', term.strip())
        cleaned = _re.sub(r'\s+', ' ', cleaned).strip()
        if not cleaned or len(cleaned) < 3:
            continue
        cleaned = cleaned[:100]
        cql = f'ti="{cleaned}" OR ab="{cleaned}"'

        log_external_request(
            logger=logger, source="EPO OPS", query=term,
            status="requesting", response_time_ms=0,
        )

        try:
            remaining = max_results - len(all_results)
            response = _epo_ops_client.published_data_search(
                cql, range_begin=1, range_end=min(remaining, 10),
                constituents=["biblio"],
            )
        except Exception:
            logger.debug("EPO OPS query failed for term: %s", term, exc_info=True)
            continue

        root = _ET.fromstring(response.text)

        for doc in root.findall(".//exch:exchange-document", ns):
            if len(all_results) >= max_results:
                break

            country = doc.get("country", "")
            doc_number = doc.get("doc-number", "")
            kind = doc.get("kind", "")
            patent_number = f"{country}{doc_number}{kind}".strip()

            if patent_number in seen_numbers:
                continue
            seen_numbers.add(patent_number)

            title = ""
            for title_el in doc.findall(".//exch:invention-title", ns):
                lang = title_el.get("lang", "")
                text = title_el.text or ""
                if lang == "en" and text:
                    title = text.strip()
                    break
                if not title and text:
                    title = text.strip()

            abstract = ""
            for abs_el in doc.findall(".//exch:abstract", ns):
                lang = abs_el.get("lang", "")
                paragraphs = []
                for p in abs_el.findall(".//exch:p", ns):
                    if p.text:
                        paragraphs.append(p.text.strip())
                text = " ".join(paragraphs)
                if lang == "en" and text:
                    abstract = text
                    break
                if not abstract and text:
                    abstract = text

            if title or patent_number:
                all_results.append({
                    "patent_number": patent_number or "UNKNOWN",
                    "title": title or "Untitled",
                    "abstract": abstract,
                })

    return {"results": all_results[:max_results]}


# Dispatch table mapping source names to their query functions.
_SOURCE_QUERY_FUNCTIONS: dict[str, Any] = {
    "ArXiv": _query_arxiv,
    "PubMed": _query_pubmed,
    "Google Scholar": _query_google_scholar,
    "Google Patents": _query_google_patents,
    "EPO OPS": _query_epo_ops,
}


def _query_source(source_name: str, search_terms: list[str], max_results: int = _DEFAULT_MAX_RESULTS) -> dict:
    """Query an external data source with the given search terms.

    Dispatches to source-specific query functions that make real HTTP
    requests to the respective APIs. Only search terms derived from
    ``_derive_search_terms()`` are transmitted — no invention disclosure
    content is sent.

    Args:
        source_name: Name of the data source to query.
        search_terms: List of search term strings.
        max_results: Maximum number of results to return per source.

    Returns:
        Raw response dict with a ``results`` key from the source API.

    Raises:
        SourceUnavailableError: If the source is unreachable or returns
            an error.
    """
    query_fn = _SOURCE_QUERY_FUNCTIONS.get(source_name)
    if query_fn is None:
        raise SourceUnavailableError(
            source_name, ValueError(f"Unknown source: {source_name}")
        )

    try:
        return query_fn(search_terms, max_results=max_results)
    except SourceUnavailableError:
        raise
    except (URLError, OSError, ET.ParseError, ValueError) as exc:
        raise SourceUnavailableError(source_name, exc) from exc


def prior_art_search_node(
    state: PatentWorkflowState,
    rag_engine: Any | None = None,
    selected_sources: list[str] | None = None,
    max_results_per_source: int = _DEFAULT_MAX_RESULTS,
) -> dict[str, Any]:
    """Run the Prior Art Search Agent.

    1. Derives search terms from the invention disclosure in state.
    2. Queries each source (DEPATISnet, Google Patents, Google Scholar,
       ArXiv, PubMed) using real HTTP calls.
    3. Uses source-specific parsers to structure results.
    4. Handles source failures gracefully: catches exceptions, logs via
       log_external_request, adds to failed_sources list, continues
       with remaining sources.
    5. Indexes results in the RAG engine if available.
    6. Returns dict with prior_art_results, failed_sources, current_step.

    When ``selected_sources`` is provided and non-empty, only the
    sources whose names appear in the list are queried. When ``None``
    or empty, all sources in ``_SOURCE_REGISTRY`` are queried
    (backward compatible).

    Args:
        state: The current workflow state.
        rag_engine: Optional RAG engine for indexing search results.
            When provided, parsed results are converted to document
            format and indexed under the topic ID from state.
        selected_sources: Optional list of source names to query.
            When provided and non-empty, only matching entries in
            ``_SOURCE_REGISTRY`` are iterated. When ``None`` or empty,
            all sources are queried.

    Returns:
        Dict with ``prior_art_results`` (list of serialized records),
        ``failed_sources`` (list of source names that failed), and
        ``current_step`` set to ``"prior_art_search"``.
    """
    start = time.monotonic()

    disclosure = state.get("invention_disclosure")
    search_terms = _derive_search_terms(disclosure)

    all_results: list[dict] = []
    failed_sources: list[str] = []

    # Filter sources when selected_sources is provided and non-empty
    if selected_sources:
        sources_to_query = {
            name: info
            for name, info in _SOURCE_REGISTRY.items()
            if name in selected_sources
        }
    else:
        sources_to_query = _SOURCE_REGISTRY

    for src_idx, (source_name, source_info) in enumerate(sources_to_query.items()):
        parser = source_info["parser"]
        source_type = source_info["type"]
        req_start = time.monotonic()

        try:
            raw_response = _query_source(source_name, search_terms, max_results=max_results_per_source)
            req_duration = (time.monotonic() - req_start) * 1000

            log_external_request(
                logger=logger,
                source=source_name,
                query=", ".join(search_terms),
                status="success",
                response_time_ms=req_duration,
            )

            # Parse results using the appropriate parser method
            if source_type == "patent":
                records = parser.parse_patent(raw_response)
            else:
                records = parser.parse_paper(raw_response)

            # Serialize records to dicts for state storage
            for record in records:
                all_results.append(parser.serialize(record))

        except Exception as exc:
            req_duration = (time.monotonic() - req_start) * 1000

            # Wrap in SourceUnavailableError if not already
            if not isinstance(exc, SourceUnavailableError):
                exc = SourceUnavailableError(source_name, exc)

            log_external_request(
                logger=logger,
                source=source_name,
                query=", ".join(search_terms),
                status=f"failed: {exc}",
                response_time_ms=req_duration,
            )

            failed_sources.append(source_name)
            logger.warning(
                "Source %s unavailable, continuing with remaining sources: %s",
                source_name,
                exc,
            )

    # Index results in RAG engine if available
    if rag_engine is not None and all_results:
        topic_id = state.get("topic_id")
        if topic_id is not None:
            rag_docs = []
            for record in all_results:
                title = record.get("title", "")
                abstract = record.get("abstract", "")
                text = f"{title} {abstract}".strip()
                if not text:
                    continue
                metadata: dict[str, str | int | float] = {}
                for k, v in record.items():
                    if k in ("title", "abstract", "embedding"):
                        continue
                    if v is None:
                        continue
                    if isinstance(v, (str, int, float)):
                        metadata[k] = v
                    elif isinstance(v, datetime):
                        metadata[k] = v.isoformat()
                    else:
                        metadata[k] = str(v)
                rag_docs.append({"text": text, "metadata": metadata})
            if rag_docs:
                try:
                    rag_engine.index_documents(topic_id, rag_docs)
                    logger.info(
                        "Indexed %d documents in RAG for topic %d",
                        len(rag_docs),
                        topic_id,
                    )
                except Exception:
                    logger.warning(
                        "Failed to index documents in RAG for topic %d",
                        topic_id,
                        exc_info=True,
                    )

    duration_ms = (time.monotonic() - start) * 1000

    log_agent_invocation(
        logger=logger,
        name="PriorArtSearchAgent",
        input_summary=f"search_terms={search_terms}, sources={list(sources_to_query.keys())}",
        output_summary=f"results={len(all_results)}, failed={failed_sources}",
        duration_ms=duration_ms,
    )

    # Build a human-readable prior art summary for the interactive workflow
    summary_parts: list[str] = []
    summary_parts.append(f"Found {len(all_results)} references from {len(sources_to_query) - len(failed_sources)} sources.")
    if failed_sources:
        summary_parts.append(f"Failed sources: {', '.join(failed_sources)}.")
    for i, record in enumerate(all_results[:20], 1):
        title = record.get("title", "Untitled")
        source = record.get("source", "")
        abstract = (record.get("abstract", "") or "")[:200]
        entry = f"\n[{i}] {title}"
        if source:
            entry += f" ({source})"
        if abstract:
            entry += f"\n    {abstract}"
        summary_parts.append(entry)
    prior_art_summary = "\n".join(summary_parts)

    return {
        "prior_art_results": all_results,
        "prior_art_summary": prior_art_summary,
        "failed_sources": failed_sources,
        "current_step": "prior_art_search",
    }


def sort_search_results(results: list[dict], criterion: str) -> list[dict]:
    """Sort search results by the given criterion.

    Args:
        results: List of serialized search result dicts.
        criterion: One of ``"discovery_date"``, ``"relevance"``, or
            ``"citation_count"``.

    Returns:
        A new list sorted by the specified criterion. Unknown criteria
        return the list unchanged.
    """
    if criterion == "discovery_date":
        return sorted(
            results,
            key=lambda r: r.get("discovered_date", "") or "",
            reverse=True,
        )
    elif criterion == "relevance":
        # Relevance sorting: prioritize records with longer abstracts
        # as a proxy for relevance (in production, this would use
        # a proper relevance score from the search engine).
        return sorted(
            results,
            key=lambda r: len(r.get("abstract", "") or ""),
            reverse=True,
        )
    elif criterion == "citation_count":
        return sorted(
            results,
            key=lambda r: r.get("citation_count", 0) or 0,
            reverse=True,
        )
    return list(results)
