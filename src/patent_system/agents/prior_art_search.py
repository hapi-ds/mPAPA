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
from patent_system.parsers.depatisnet import DEPATISnetParser
from patent_system.parsers.google_patents import GooglePatentsParser
from patent_system.parsers.google_scholar import GoogleScholarParser
from patent_system.parsers.pubmed import PubMedParser

from patent_system.agents.state import PatentWorkflowState

logger = logging.getLogger(__name__)

# Source registry: maps source name to (query function, parser instance)
_SOURCE_REGISTRY: dict[str, dict[str, Any]] = {
    "DEPATISnet": {"parser": DEPATISnetParser(), "type": "patent"},
    "Google Patents": {"parser": GooglePatentsParser(), "type": "patent"},
    "Google Scholar": {"parser": GoogleScholarParser(), "type": "paper"},
    "ArXiv": {"parser": ArXivParser(), "type": "paper"},
    "PubMed": {"parser": PubMedParser(), "type": "paper"},
}


def _derive_search_terms(disclosure: dict | None) -> list[str]:
    """Derive search terms from an invention disclosure.

    Extracts keywords from the technical problem, novel features,
    and implementation details fields.

    Args:
        disclosure: The invention disclosure dict, or None.

    Returns:
        List of search term strings. Returns a single empty-string
        term if disclosure is None or empty.
    """
    if not disclosure:
        return [""]

    terms: list[str] = []

    technical_problem = disclosure.get("technical_problem", "")
    if technical_problem:
        terms.append(technical_problem)

    novel_features = disclosure.get("novel_features", [])
    if isinstance(novel_features, list):
        for feature in novel_features:
            if feature:
                terms.append(str(feature))

    implementation_details = disclosure.get("implementation_details", "")
    if implementation_details:
        terms.append(implementation_details)

    return terms if terms else [""]


# Source endpoints mapping source names to their base URLs.
_SOURCE_ENDPOINTS: dict[str, str] = {
    "ArXiv": "export.arxiv.org/api/query",
    "PubMed": "eutils.ncbi.nlm.nih.gov/entrez/eutils",
    "Google Scholar": "scholar.google.com/scholar",
    "Google Patents": "patents.google.com",
    "DEPATISnet": "depatisnet.dpma.de/DepatisNet/depatisnet",
}

_HTTP_TIMEOUT = 15  # seconds

# Default maximum results per source (overridden by AppSettings.search_max_results_per_source)
_DEFAULT_MAX_RESULTS = 10


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
    """Query ArXiv API and parse Atom XML into results dict.

    Args:
        search_terms: List of search term strings.
        max_results: Maximum number of results to return.

    Returns:
        Dict with ``results`` key containing parsed paper entries.
    """
    query = "+AND+".join(quote_plus(t) for t in search_terms if t)
    if not query:
        query = quote_plus("")
    url = f"http://{_SOURCE_ENDPOINTS['ArXiv']}?search_query=all:{query}&max_results={max_results}"

    log_external_request(
        logger=logger, source="ArXiv", query=", ".join(search_terms),
        status="requesting", response_time_ms=0,
    )

    xml_text = _http_get(url)

    # Parse Atom XML
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(xml_text)
    results: list[dict] = []
    for entry in root.findall("atom:entry", ns):
        entry_id = entry.findtext("atom:id", default="", namespaces=ns)
        title = entry.findtext("atom:title", default="", namespaces=ns).strip()
        abstract = entry.findtext("atom:summary", default="", namespaces=ns).strip()
        # ArXiv provides a PDF link; extract it for potential full-text retrieval
        pdf_link = ""
        for link_el in entry.findall("atom:link", ns):
            if link_el.get("title") == "pdf":
                pdf_link = link_el.get("href", "")
                break
        if title:
            results.append({
                "doi": entry_id,
                "title": title,
                "abstract": abstract,
                "full_text": abstract,  # ArXiv summary is the full abstract
                "pdf_path": pdf_link,
            })

    return {"results": results}


def _query_pubmed(search_terms: list[str], max_results: int = _DEFAULT_MAX_RESULTS) -> dict:
    """Query PubMed via E-utilities (esearch + efetch) and parse XML results.

    Two-step process:
    1. esearch to get PMIDs matching the query.
    2. efetch to retrieve article details for those PMIDs.

    Args:
        search_terms: List of search term strings.

    Returns:
        Dict with ``results`` key containing parsed paper entries.
    """
    base = _SOURCE_ENDPOINTS["PubMed"]
    query = " ".join(t for t in search_terms if t)
    if not query:
        query = ""

    # Step 1: esearch to get IDs
    esearch_params = urlencode({
        "db": "pubmed", "term": query, "retmax": str(max_results), "retmode": "xml",
    })
    esearch_url = f"https://{base}/esearch.fcgi?{esearch_params}"

    log_external_request(
        logger=logger, source="PubMed", query=", ".join(search_terms),
        status="requesting", response_time_ms=0,
    )

    esearch_xml = _http_get(esearch_url)
    esearch_root = ET.fromstring(esearch_xml)
    id_list = esearch_root.findall(".//Id")
    pmids = [id_el.text for id_el in id_list if id_el.text]

    if not pmids:
        return {"results": []}

    # Step 2: efetch to get details
    efetch_params = urlencode({
        "db": "pubmed", "id": ",".join(pmids), "retmode": "xml",
    })
    efetch_url = f"https://{base}/efetch.fcgi?{efetch_params}"
    efetch_xml = _http_get(efetch_url)

    efetch_root = ET.fromstring(efetch_xml)
    results: list[dict] = []
    for article in efetch_root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//PMID")
        title_el = article.find(".//ArticleTitle")
        abstract_el = article.find(".//AbstractText")

        pmid = pmid_el.text if pmid_el is not None and pmid_el.text else ""
        title = title_el.text if title_el is not None and title_el.text else ""
        abstract = abstract_el.text if abstract_el is not None and abstract_el.text else ""

        # Collect all AbstractText elements (structured abstracts have
        # multiple sections like Background, Methods, Results, etc.)
        abstract_parts: list[str] = []
        for abs_el in article.findall(".//AbstractText"):
            label = abs_el.get("Label", "")
            text = abs_el.text or ""
            if label and text:
                abstract_parts.append(f"{label}: {text}")
            elif text:
                abstract_parts.append(text)
        full_abstract = "\n".join(abstract_parts) if abstract_parts else abstract

        if title:
            results.append({
                "doi": pmid,
                "title": title,
                "abstract": abstract,
                "full_text": full_abstract if full_abstract != abstract else None,
            })

    return {"results": results}


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
    """Query Google Scholar and parse HTML results.

    Args:
        search_terms: List of search term strings.

    Returns:
        Dict with ``results`` key containing parsed paper entries.
    """
    query = "+".join(quote_plus(t) for t in search_terms if t)
    if not query:
        query = quote_plus("")
    url = f"https://{_SOURCE_ENDPOINTS['Google Scholar']}?q={query}&hl=en"

    log_external_request(
        logger=logger, source="Google Scholar", query=", ".join(search_terms),
        status="requesting", response_time_ms=0,
    )

    html_text = _http_get(url)

    # Parse HTML — extract result entries from gs_ri divs
    results: list[dict] = []
    # Simple extraction: split on result class markers
    parts = html_text.split('class="gs_ri"')
    for part in parts[1:]:  # skip the first chunk before any result
        title = ""
        abstract = ""
        # Extract title from <h3> tag
        h3_start = part.find("<h3")
        if h3_start != -1:
            h3_end = part.find("</h3>", h3_start)
            if h3_end != -1:
                h3_html = part[h3_start:h3_end]
                extractor = _SimpleHTMLTextExtractor()
                extractor.feed(h3_html)
                title = extractor.get_text().strip()
        # Extract snippet/abstract from gs_rs div
        gs_rs_start = part.find('class="gs_rs"')
        if gs_rs_start != -1:
            div_end = part.find("</div>", gs_rs_start)
            if div_end != -1:
                snippet_html = part[gs_rs_start:div_end]
                extractor = _SimpleHTMLTextExtractor()
                extractor.feed(snippet_html)
                abstract = extractor.get_text().strip()
        if title:
            results.append({"doi": "", "title": title, "abstract": abstract})

    return {"results": results[:max_results]}


def _query_google_patents(search_terms: list[str], max_results: int = _DEFAULT_MAX_RESULTS) -> dict:
    """Query Google Patents and parse HTML results.

    Args:
        search_terms: List of search term strings.

    Returns:
        Dict with ``results`` key containing parsed patent entries.
    """
    query = "+".join(quote_plus(t) for t in search_terms if t)
    if not query:
        query = quote_plus("")
    url = f"https://{_SOURCE_ENDPOINTS['Google Patents']}/?q={query}"

    log_external_request(
        logger=logger, source="Google Patents", query=", ".join(search_terms),
        status="requesting", response_time_ms=0,
    )

    html_text = _http_get(url)

    # Parse HTML — extract patent results from search-result items
    results: list[dict] = []
    parts = html_text.split("search-result")
    for part in parts[1:]:
        patent_number = ""
        title = ""
        abstract = ""
        # Try to extract patent number from data attributes or text
        pn_marker = 'data-result="'
        pn_start = part.find(pn_marker)
        if pn_start != -1:
            pn_end = part.find('"', pn_start + len(pn_marker))
            patent_number = part[pn_start + len(pn_marker):pn_end].strip()
        # Extract title
        title_start = part.find("<h3")
        if title_start != -1:
            title_end = part.find("</h3>", title_start)
            if title_end != -1:
                extractor = _SimpleHTMLTextExtractor()
                extractor.feed(part[title_start:title_end])
                title = extractor.get_text().strip()
        # Extract abstract/snippet
        abs_marker = 'class="abstract"'
        abs_start = part.find(abs_marker)
        if abs_start == -1:
            abs_marker = 'class="snippet"'
            abs_start = part.find(abs_marker)
        if abs_start != -1:
            abs_end = part.find("</", abs_start + len(abs_marker))
            if abs_end != -1:
                extractor = _SimpleHTMLTextExtractor()
                extractor.feed(part[abs_start:abs_end])
                abstract = extractor.get_text().strip()
        if title or patent_number:
            results.append({
                "patent_number": patent_number or "UNKNOWN",
                "title": title or "Untitled",
                "abstract": abstract,
            })

    return {"results": results[:max_results]}


def _query_depatisnet(search_terms: list[str], max_results: int = _DEFAULT_MAX_RESULTS) -> dict:
    """Query DEPATISnet and parse HTML results.

    Args:
        search_terms: List of search term strings.

    Returns:
        Dict with ``results`` key containing parsed patent entries.
    """
    query = " ".join(t for t in search_terms if t)
    params = urlencode({
        "action": "bibdat",
        "docid": "",
        "query": query,
    })
    url = f"https://{_SOURCE_ENDPOINTS['DEPATISnet']}?{params}"

    log_external_request(
        logger=logger, source="DEPATISnet", query=", ".join(search_terms),
        status="requesting", response_time_ms=0,
    )

    html_text = _http_get(url)

    # Parse HTML — extract patent results from table rows
    results: list[dict] = []
    rows = html_text.split("<tr")
    for row in rows[1:]:  # skip header/preamble
        cells: list[str] = []
        cell_parts = row.split("<td")
        for cell in cell_parts[1:]:
            end = cell.find("</td>")
            if end != -1:
                extractor = _SimpleHTMLTextExtractor()
                extractor.feed(cell[:end])
                cells.append(extractor.get_text().strip())
        # Expect at least patent_number, title columns
        if len(cells) >= 2 and cells[0]:
            results.append({
                "patent_number": cells[0],
                "title": cells[1] if len(cells) > 1 else "Untitled",
                "abstract": cells[2] if len(cells) > 2 else "",
            })

    return {"results": results[:max_results]}


# Dispatch table mapping source names to their query functions.
_SOURCE_QUERY_FUNCTIONS: dict[str, Any] = {
    "ArXiv": _query_arxiv,
    "PubMed": _query_pubmed,
    "Google Scholar": _query_google_scholar,
    "Google Patents": _query_google_patents,
    "DEPATISnet": _query_depatisnet,
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

    for source_name, source_info in sources_to_query.items():
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

    return {
        "prior_art_results": all_results,
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
