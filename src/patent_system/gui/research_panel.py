"""Research panel UI for the Patent Analysis & Drafting System.

Provides the search query input, "Start Research" button, sort controls,
and a sortable results table displaying prior art search results.

Requirements: 16.4, 3.5, 3.6
"""

from __future__ import annotations

import logging
from typing import Any

from nicegui import ui

from patent_system.agents.prior_art_search import prior_art_search_node

logger = logging.getLogger(__name__)

# Sort criteria options (Req 3.6)
SORT_OPTIONS: dict[str, str] = {
    "discovery_date": "Discovery Date",
    "relevance": "Relevance",
    "citation_count": "Citation Count",
}

# Table column definitions (Req 3.5)
RESULT_COLUMNS: list[dict[str, str]] = [
    {"name": "title", "label": "Title", "field": "title", "align": "left"},
    {"name": "discovered_date", "label": "Discovery Date", "field": "discovered_date", "align": "left"},
    {"name": "source", "label": "Source", "field": "source", "align": "left"},
]


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


def create_research_panel(container: Any, topic_id: int) -> None:
    """Populate *container* with the Research panel UI components."""
    container.clear()

    panel_state: dict[str, Any] = {
        "results": [],
        "sort_criterion": "discovery_date",
    }

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

                panel_state["results"] = results
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

        table = ui.table(
            columns=RESULT_COLUMNS,
            rows=[],
            row_key="title",
        ).classes("w-full q-mt-md")

        def _refresh_table() -> None:
            sorted_rows = _sort_results(panel_state["results"], panel_state["sort_criterion"])
            table.rows = sorted_rows
            table.update()

        def set_results(results: list[dict[str, Any]]) -> None:
            panel_state["results"] = list(results)
            _refresh_table()

        container.set_results = set_results  # type: ignore[attr-defined]
