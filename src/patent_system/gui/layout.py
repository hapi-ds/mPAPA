"""Main layout for the Patent Analysis & Drafting System GUI.

Provides the top-level page structure: header, left drawer with topic
management, and a tabbed content area for Research, AI Chat, and Patent
Draft panels.

Requirements: 16.1, 16.2, 16.3, 1.1, 1.3, 1.4, 1.5
"""

from __future__ import annotations

import logging
import sqlite3

from nicegui import ui

from patent_system.db.repository import ChatHistoryRepository, TopicRepository
from patent_system.gui.chat_panel import create_chat_panel
from patent_system.gui.draft_panel import create_draft_panel
from patent_system.gui.research_panel import create_research_panel

logger = logging.getLogger(__name__)


def create_layout(topic_repo: TopicRepository, conn: sqlite3.Connection) -> None:
    """Set up the full page layout with header, drawer, and tabs.

    Args:
        topic_repo: Repository for topic CRUD operations.
        conn: SQLite connection for creating per-request repositories.
    """
    # Shared state across the layout
    state: dict = {
        "selected_topic_id": None,
    }

    # --- Header (Req 16.1) ---
    with ui.header().classes("items-center justify-between"):
        ui.label("Patent Analysis & Drafting System").classes(
            "text-h5 font-bold"
        )

    # --- Left Drawer (Req 16.2) ---
    with ui.left_drawer(value=True).classes("p-4") as drawer:
        ui.label("Topics").classes("text-h6 q-mb-sm")

        topic_list_container = ui.column().classes("w-full gap-1")

        def _refresh_topic_list() -> None:
            """Reload the topic list from the database."""
            topic_list_container.clear()
            topics = topic_repo.get_all()
            with topic_list_container:
                if not topics:
                    ui.label("No topics yet.").classes("text-grey")
                for topic in topics:
                    _topic_id = topic.id
                    _topic_name = topic.name
                    btn = ui.button(
                        _topic_name,
                        on_click=lambda _, tid=_topic_id: _select_topic(tid),
                    ).classes("w-full justify-start")
                    if _topic_id == state["selected_topic_id"]:
                        btn.props("color=primary")
                    else:
                        btn.props("flat color=dark")

        def _select_topic(topic_id: int) -> None:
            """Handle topic selection and load associated data."""
            state["selected_topic_id"] = topic_id
            logger.info("Selected topic %d", topic_id)
            _refresh_topic_list()
            _on_topic_selected(topic_id)

        # Error label for duplicate name (Req 1.5)
        error_label = ui.label("").classes("text-negative text-caption")
        error_label.set_visibility(False)

        async def _open_new_topic_dialog() -> None:
            """Show dialog for creating a new topic (Req 1.1)."""
            error_label.set_visibility(False)
            error_label.set_text("")

            with ui.dialog() as dialog, ui.card().classes("min-w-[300px]"):
                ui.label("New Topic").classes("text-h6")
                name_input = ui.input(
                    label="Topic name",
                    placeholder="Enter topic name",
                ).classes("w-full")

                dialog_error = ui.label("").classes("text-negative text-caption")
                dialog_error.set_visibility(False)

                def _create_topic() -> None:
                    name = name_input.value.strip()
                    if not name:
                        dialog_error.set_text("Topic name cannot be empty.")
                        dialog_error.set_visibility(True)
                        return

                    # Check for duplicate name (Req 1.5)
                    if topic_repo.name_exists(name):
                        dialog_error.set_text(
                            f'Topic "{name}" already exists.'
                        )
                        dialog_error.set_visibility(True)
                        return

                    try:
                        new_topic = topic_repo.create(name)
                        logger.info("Created topic: %s (id=%d)", name, new_topic.id)
                        dialog.close()
                        state["selected_topic_id"] = new_topic.id
                        _refresh_topic_list()
                        _on_topic_selected(new_topic.id)
                    except sqlite3.IntegrityError:
                        dialog_error.set_text(
                            f'Topic "{name}" already exists.'
                        )
                        dialog_error.set_visibility(True)

                with ui.row().classes("w-full justify-end gap-2 q-mt-sm"):
                    ui.button("Cancel", on_click=dialog.close).props("flat")
                    ui.button("Create", on_click=_create_topic).props(
                        "color=primary"
                    )

            dialog.open()

        ui.button(
            "New Topic",
            on_click=_open_new_topic_dialog,
            icon="add",
        ).classes("w-full q-mt-md")

    # --- Tabs (Req 16.3) ---
    with ui.tabs().classes("w-full") as tabs:
        research_tab = ui.tab("Research")
        chat_tab = ui.tab("AI Chat")
        draft_tab = ui.tab("Patent Draft")

    with ui.tab_panels(tabs, value=research_tab).classes(
        "w-full flex-grow"
    ) as panels:
        with ui.tab_panel(research_tab):
            research_container = ui.column().classes("w-full p-4")
            with research_container:
                ui.label("Select a topic to view research results.").classes(
                    "text-grey"
                )

        with ui.tab_panel(chat_tab):
            chat_container = ui.column().classes("w-full p-4")
            with chat_container:
                ui.label("Select a topic to start chatting.").classes(
                    "text-grey"
                )

        with ui.tab_panel(draft_tab):
            draft_container = ui.column().classes("w-full p-4")
            with draft_container:
                ui.label("Select a topic to view the patent draft.").classes(
                    "text-grey"
                )

    def _on_topic_selected(topic_id: int) -> None:
        """Load data for the selected topic into the tab panels (Req 1.4)."""
        topic = topic_repo.get_by_id(topic_id)
        if topic is None:
            return

        chat_repo = ChatHistoryRepository(conn)

        create_research_panel(research_container, topic_id)
        create_chat_panel(chat_container, topic_id, chat_repo)
        create_draft_panel(draft_container, topic_id)

    # Initial load of topic list (Req 1.3)
    _refresh_topic_list()
