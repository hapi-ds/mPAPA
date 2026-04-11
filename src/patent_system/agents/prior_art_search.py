"""Prior Art Search Agent for the patent drafting pipeline.

Queries DEPATISnet, Google Patents, Google Scholar, ArXiv, and PubMed
for prior art related to the invention disclosure. Uses source-specific
parsers to structure results and handles source failures gracefully.

Requirements: 3.1, 3.2, 3.3, 3.4, 3.7
"""

import logging
import time
from datetime import datetime
from typing import Any

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


def _query_source(source_name: str, search_terms: list[str]) -> dict:
    """Query an external data source with the given search terms.

    This is a placeholder HTTP call that can be mocked in tests.
    In production, this would make actual HTTP requests to the
    respective APIs.

    Args:
        source_name: Name of the data source to query.
        search_terms: List of search term strings.

    Returns:
        Raw response dict from the source API.

    Raises:
        SourceUnavailableError: If the source is unreachable.
    """
    # Placeholder: in production, this would perform real HTTP calls.
    # For now, return an empty result set. Tests will mock this function.
    return {"results": []}


def prior_art_search_node(state: PatentWorkflowState) -> dict[str, Any]:
    """Run the Prior Art Search Agent.

    1. Derives search terms from the invention disclosure in state.
    2. Queries each source (DEPATISnet, Google Patents, Google Scholar,
       ArXiv, PubMed) using placeholder HTTP calls.
    3. Uses source-specific parsers to structure results.
    4. Handles source failures gracefully: catches exceptions, logs via
       log_external_request, adds to failed_sources list, continues
       with remaining sources.
    5. Returns dict with prior_art_results, failed_sources, current_step.

    Args:
        state: The current workflow state.

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

    for source_name, source_info in _SOURCE_REGISTRY.items():
        parser = source_info["parser"]
        source_type = source_info["type"]
        req_start = time.monotonic()

        try:
            raw_response = _query_source(source_name, search_terms)
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

    duration_ms = (time.monotonic() - start) * 1000

    log_agent_invocation(
        logger=logger,
        name="PriorArtSearchAgent",
        input_summary=f"search_terms={search_terms}, sources={list(_SOURCE_REGISTRY.keys())}",
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
