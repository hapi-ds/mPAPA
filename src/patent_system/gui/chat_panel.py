"""AI Chat panel UI for the Patent Analysis & Drafting System.

Provides a scrollable chat history with distinct styling for user and
assistant messages, a text input field, and a "Send" button.  Messages
are persisted via ``ChatHistoryRepository``.

The actual RAG integration and LLM response generation will be wired in
task 12.1.  For now the send button saves the user message and displays
a placeholder assistant response.

Requirements: 16.5, 8.1, 8.2, 8.3, 8.4
"""

from __future__ import annotations

import logging
from typing import Any

from nicegui import ui

from patent_system.db.repository import ChatHistoryRepository

logger = logging.getLogger(__name__)


def _render_message(role: str, text: str) -> None:
    """Render a single chat bubble with role-appropriate styling.

    Args:
        role: ``"user"`` or ``"assistant"``.
        text: The message content.
    """
    if role == "user":
        with ui.row().classes("w-full justify-end"):
            ui.chat_message(
                text=text,
                name="You",
                sent=True,
            ).classes("bg-blue-100")
    else:
        with ui.row().classes("w-full justify-start"):
            ui.chat_message(
                text=text,
                name="Assistant",
                sent=False,
            ).classes("bg-grey-200")


def create_chat_panel(
    container: Any,
    topic_id: int,
    chat_repo: ChatHistoryRepository,
) -> None:
    """Populate *container* with the AI Chat panel UI components.

    The panel contains:
    - A scrollable chat history area (Req 8.4)
    - Distinct styling for user (right-aligned, blue) and assistant
      (left-aligned, grey) messages
    - A text input field for new messages
    - A "Send" button
    - Existing chat history loaded from ``ChatHistoryRepository`` on
      creation (Req 8.3)
    - New messages persisted via ``ChatHistoryRepository`` (Req 8.3)

    Args:
        container: A NiceGUI container element (e.g. ``ui.column``) to
            populate.  The container is cleared before adding content.
        topic_id: The active topic ID whose chat history is shown.
        chat_repo: Repository for chat message persistence.
    """
    container.clear()

    with container:
        ui.label("AI Chat").classes("text-h6 q-mb-sm")

        # --- Scrollable chat history area (Req 8.4) ---
        scroll_area = ui.scroll_area().classes(
            "w-full border rounded-lg p-2"
        ).style("height: 500px;")

        chat_messages_container = None
        with scroll_area:
            chat_messages_container = ui.column().classes("w-full gap-2")

        # Load existing history (Req 8.3)
        existing = chat_repo.get_by_topic(topic_id)
        with chat_messages_container:
            for msg in existing:
                _render_message(msg.role, msg.message)

        # --- Input row ---
        with ui.row().classes("w-full items-end gap-2 q-mt-sm"):
            message_input = ui.input(
                label="Type a message…",
                placeholder="Ask about your patents and papers…",
            ).classes("flex-grow")

            def _on_send() -> None:
                """Handle send: persist user message, show placeholder response."""
                text = message_input.value.strip() if message_input.value else ""
                if not text:
                    return

                # Persist and display user message (Req 8.3)
                try:
                    chat_repo.save_message(topic_id, "user", text)
                except Exception:
                    logger.exception(
                        "Failed to save user message for topic %d", topic_id
                    )

                with chat_messages_container:
                    _render_message("user", text)

                # Clear input
                message_input.value = ""

                # Placeholder assistant response (real RAG wired in task 12.1)
                placeholder = (
                    "Thank you for your question. "
                    "RAG-powered responses will be available once the AI "
                    "backend is connected."
                )
                try:
                    chat_repo.save_message(topic_id, "assistant", placeholder)
                except Exception:
                    logger.exception(
                        "Failed to save assistant message for topic %d",
                        topic_id,
                    )

                with chat_messages_container:
                    _render_message("assistant", placeholder)

                # Scroll to bottom
                scroll_area.scroll_to(percent=1.0)

            ui.button("Send", on_click=_on_send).props("color=primary")
