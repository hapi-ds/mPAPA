"""Research panel UI for the Patent Analysis & Drafting System.

Provides the search query input, "Start Research" button, sort controls,
and a sortable results table displaying prior art search results.

Requirements: 16.4, 3.5, 3.6
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from nicegui import ui

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
    """Sort search results by the given criterion.

    Args:
        results: List of result dicts with keys title, discovered_date, source,
                 and optionally relevance_score and citation_count.
        criterion: One of 'discovery_date', 'relevance', or 'citation_count'.

    Returns:
        A new list sorted by the chosen criterion (descending).
    """
    if criterion == "discovery_date":
        return sorted(
            results,
            key=lambda r: r.get("discovered_date", ""),
            reverse=True,
        )
    if criterion == "relevance":
        return sorted(
            results,
            key=lambda r: r.get("relevance_score", 0),
            reverse=True,
        )
    if criterion == "citation_count":
        return sorted(
            results,
            key=lambda r: r.get("citation_count", 0),
            reverse=True,
        )
    return list(results)


def create_research_panel(container: Any, topic_id: int) -> None:
    """Populate *container* with the Research panel UI components.

    The panel contains:
    - A search query input field
    - A "Start Research" button (wired in task 12.1)
    - A sort dropdown (discovery date / relevance / citation count)
    - A sortable results table (title, discovery date, source)

    Args:
        container: A NiceGUI container element (e.g. ``ui.column``) to
            populate.  The container is cleared before adding content.
        topic_id: The active topic ID whose research results are shown.
    """
    container.clear()

    # Panel-local state
    panel_state: dict[str, Any] = {
        "results": [],          # raw result dicts
        "sort_criterion": "discovery_date",
    }

    with container:
        ui.label("Prior Art Research").classes("text-h6 q-mb-sm")

        # --- Search input + button (Req 16.4) ---
        with ui.row().classes("w-full items-end gap-2"):
            search_input = ui.input(
                label="Search query",
                placeholder="Enter search terms…",
            ).classes("flex-grow")

            def _on_start_research() -> None:
                """Placeholder handler — actual search wired in task 12.1."""
                query = search_input.value.strip() if search_input.value else ""
                logger.info(
                    "Start Research clicked for topic %d, query=%r",
                    topic_id,
                    query,
                )

            ui.button("Start Research", on_click=_on_start_research).props(
                "color=primary"
            )

        # --- Sort controls (Req 3.6) ---
        def _on_sort_change(e: Any) -> None:
            """Re-sort the results table when the criterion changes."""
            panel_state["sort_criterion"] = e.value
            _refresh_table()

        ui.select(
            options=SORT_OPTIONS,
            value="discovery_date",
            label="Sort by",
            on_change=_on_sort_change,
        ).classes("w-48 q-mt-sm")

        # --- Results table (Req 3.5) ---
        table = ui.table(
            columns=RESULT_COLUMNS,
            rows=[],
            row_key="title",
        ).classes("w-full q-mt-md")

        def _refresh_table() -> None:
            """Re-sort and update the table rows."""
            sorted_rows = _sort_results(
                panel_state["results"],
                panel_state["sort_criterion"],
            )
            table.rows = sorted_rows
            table.update()

        def set_results(results: list[dict[str, Any]]) -> None:
            """Public helper to inject search results into the panel.

            Each dict should contain at least ``title``, ``discovered_date``,
            and ``source``.  Optional keys: ``relevance_score``,
            ``citation_count``.
            """
            panel_state["results"] = list(results)
            _refresh_table()

        # Expose helper on the container so callers can push results later.
        container.set_results = set_results  # type: ignore[attr-defined]
