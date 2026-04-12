"""Research panel UI for the Patent Analysis & Drafting System.

Provides the search query input, "Start Research" button, sort controls,
and a sortable results table displaying prior art search results.

Requirements: 16.4, 3.5, 3.6
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Any

from nicegui import ui

from patent_system.agents.prior_art_search import prior_art_search_node
from patent_system.db.repository import PatentRepository, ResearchSessionRepository
from patent_system.db.schema import get_connection

logger = logging.getLogger(__name__)

# Sort criteria options (Req 3.6)
SORT_OPTIONS: dict[str, str] = {
    "discovery_date": "Discovery Date",
    "relevance": "Relevance",
    "citation_count": "Citation Count",
}

# Source URL templates for linking to the original record
_SOURCE_URLS: dict[str, str] = {
    "ArXiv": "https://arxiv.org/abs/{id}",
    "PubMed": "https://pubmed.ncbi.nlm.nih.gov/{id}",
    "Google Scholar": "https://scholar.google.com/scholar?q={id}",
    "Google Patents": "https://patents.google.com/patent/{id}",
    "DEPATISnet": "https://depatisnet.dpma.de/DepatisNet/depatisnet?action=bibdat&docid={id}",
}


def _build_source_url(record: dict[str, Any]) -> str | None:
    """Build a URL to the original source for a search result record."""
    source = record.get("source", "")
    template = _SOURCE_URLS.get(source)
    if not template:
        return None
    # Use patent_number for patent sources, doi for paper sources
    record_id = record.get("patent_number") or record.get("doi") or ""
    if not record_id or record_id == "UNKNOWN":
        return None
    return template.format(id=record_id)


def _sort_results(
    results: list[dict[str, Any]],
    criterion: str,
) -> list[dict[str, Any]]:
    """Sort search results by the given criterion."""
    if criterion == "discovery_date":
        return sorted(results, key=lambda r: r.get("discovered_date", ""), reverse=True)
    if criterion == "relevance":
        return sorted(results, key=lambda r: r.get("relevance_score", 0), reverse=True)
    if criterion == "citation_count":
        return sorted(results, key=lambda r: r.get("citation_count", 0), reverse=True)
    return list(results)


def create_research_panel(
    container: Any,
    topic_id: int,
    conn: sqlite3.Connection | None = None,
    rag_engine: Any | None = None,
) -> None:
    """Populate *container* with the Research panel UI components."""
    container.clear()

    panel_state: dict[str, Any] = {
        "results": [],
        "sort_criterion": "discovery_date",
    }

    # Load previously saved results from DB
    if conn is not None:
        try:
            session_repo = ResearchSessionRepository(conn)
            patent_repo = PatentRepository(conn)
            sessions = session_repo.get_by_topic(topic_id)
            saved_results: list[dict[str, Any]] = []
            for session in sessions:
                records = patent_repo.get_by_session(session["id"])
                for rec in records:
                    saved_results.append({
                        "title": rec.title,
                        "abstract": rec.abstract,
                        "source": rec.source,
                        "patent_number": rec.patent_number,
                        "discovered_date": rec.discovered_date.isoformat() if rec.discovered_date else "",
                    })
            panel_state["results"] = saved_results
        except Exception:
            logger.exception("Failed to load saved results for topic %d", topic_id)

    with container:
        ui.label("Prior Art Research").classes("text-h6 q-mb-sm")

        with ui.row().classes("w-full items-end gap-2"):
            search_input = ui.input(
                label="Search query",
                placeholder="Enter search terms…",
            ).classes("flex-grow")

            status_label = ui.label("").classes("text-caption")

            def _on_start_research() -> None:
                query = search_input.value.strip() if search_input.value else ""
                if not query:
                    ui.notify("Please enter a search query.", type="warning")
                    return

                status_label.set_text("Searching…")
                logger.info("Start Research for topic %d, query=%r", topic_id, query)

                # Build a minimal workflow state with the query as disclosure
                state = {
                    "topic_id": topic_id,
                    "invention_disclosure": {"technical_problem": query},
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

                result = prior_art_search_node(state)

                results = result.get("prior_art_results", [])
                failed = result.get("failed_sources", [])

                # Persist the search session and results to DB
                if conn is not None:
                    try:
                        session_repo = ResearchSessionRepository(conn)
                        patent_repo = PatentRepository(conn)
                        session_id = session_repo.create(topic_id, query=query)
                        for rec in results:
                            from patent_system.db.models import PatentRecord as PR
                            patent_record = PR(
                                session_id=session_id,
                                patent_number=rec.get("patent_number", rec.get("doi", "UNKNOWN")),
                                title=rec.get("title", "Untitled"),
                                abstract=rec.get("abstract", ""),
                                source=rec.get("source", "unknown"),
                            )
                            patent_repo.create(session_id, patent_record)
                    except Exception:
                        logger.exception("Failed to persist search results for topic %d", topic_id)

                # Index results in RAG so AI Chat can use them
                if rag_engine is not None and results:
                    rag_docs = []
                    for rec in results:
                        title = rec.get("title", "")
                        abstract = rec.get("abstract", "")
                        text = f"{title} {abstract}".strip()
                        if text:
                            rag_docs.append({"text": text, "metadata": rec})
                    if rag_docs:
                        try:
                            rag_engine.index_documents(topic_id, rag_docs)
                            logger.info("Indexed %d docs in RAG for topic %d", len(rag_docs), topic_id)
                        except Exception:
                            logger.exception("Failed to index in RAG for topic %d", topic_id)

                # Append new results to existing ones
                panel_state["results"].extend(results)
                _refresh_table()

                # Status feedback
                parts = []
                parts.append(f"{len(results)} result(s) found")
                if failed:
                    parts.append(f"{len(failed)} source(s) unavailable: {', '.join(failed)}")
                status_label.set_text(" · ".join(parts))

                if not results:
                    ui.notify(
                        "No results found. External data source connectors are not yet implemented.",
                        type="info",
                        close_button=True,
                    )

            ui.button("Start Research", on_click=_on_start_research).props("color=primary")

        def _on_sort_change(e: Any) -> None:
            panel_state["sort_criterion"] = e.value
            _refresh_table()

        ui.select(
            options=SORT_OPTIONS,
            value="discovery_date",
            label="Sort by",
            on_change=_on_sort_change,
        ).classes("w-48 q-mt-sm")

        results_container = ui.column().classes("w-full q-mt-md gap-2")

        def _render_results(results: list[dict[str, Any]]) -> None:
            """Render search results as expandable cards with source links."""
            results_container.clear()
            with results_container:
                if not results:
                    ui.label("No results yet.").classes("text-grey")
                    return
                for rec in results:
                    title = rec.get("title", "Untitled")
                    abstract = rec.get("abstract", "")
                    source = rec.get("source", "unknown")
                    record_id = rec.get("patent_number") or rec.get("doi") or ""
                    source_url = _build_source_url(rec)

                    with ui.card().classes("w-full"):
                        # Title + source badge
                        with ui.row().classes("w-full items-center justify-between"):
                            ui.label(title).classes("text-subtitle1 font-bold")
                            ui.badge(source).props("color=primary outline")

                        # Record ID and source link
                        if record_id and record_id != "UNKNOWN":
                            with ui.row().classes("items-center gap-2"):
                                ui.label(record_id).classes("text-caption text-grey")
                                if source_url:
                                    ui.link("Open in " + source, source_url, new_tab=True).classes(
                                        "text-caption"
                                    )

                        # Abstract — truncated with expansion
                        if abstract:
                            # Show first 200 chars, expand for full text
                            short = abstract[:200]
                            if len(abstract) > 200:
                                with ui.expansion(
                                    short + "…", icon="description"
                                ).classes("w-full text-body2"):
                                    ui.label(abstract).classes(
                                        "text-body2"
                                    ).style("white-space: pre-wrap; word-break: break-word;")
                            else:
                                ui.label(abstract).classes(
                                    "text-body2 text-grey-8"
                                ).style("white-space: pre-wrap; word-break: break-word;")

        def _refresh_table() -> None:
            sorted_rows = _sort_results(panel_state["results"], panel_state["sort_criterion"])
            _render_results(sorted_rows)

        def set_results(results: list[dict[str, Any]]) -> None:
            panel_state["results"] = list(results)
            _refresh_table()

        container.set_results = set_results  # type: ignore[attr-defined]

        # Render any previously saved results
        if panel_state["results"]:
            _refresh_table()
