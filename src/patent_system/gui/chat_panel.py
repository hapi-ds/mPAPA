"""AI Chat panel UI for the Patent Analysis & Drafting System.

Provides a scrollable chat history with distinct styling for user and
assistant messages, a text input field, and a "Send" button.  Messages
are persisted via ``ChatHistoryRepository``.

RAG context is retrieved via ``RAGEngine.query()`` and combined with the
user question into a prompt sent to LM Studio via the globally configured
``dspy.LM`` instance.

Requirements: 16.5, 8.1, 8.2, 8.3, 8.4, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import dspy
import httpx
import litellm.exceptions
import requests.exceptions
from nicegui import ui

from patent_system.db.repository import ChatHistoryRepository

if TYPE_CHECKING:
    from patent_system.config import AppSettings
    from patent_system.rag.engine import RAGEngine

logger = logging.getLogger(__name__)


def build_chat_prompt(context_docs: list[dict], question: str) -> str:
    """Build a chat prompt from RAG context documents and a user question.

    The prompt includes all retrieved context texts followed by the user
    question.  When no context documents are available, a note is added
    indicating that no prior art context was found.

    This function is extracted as a module-level helper so it can be
    independently tested (Property 7).

    Args:
        context_docs: List of dicts, each containing at least a
            ``"text"`` key with the document content.
        question: The user's question text.

    Returns:
        The assembled prompt string ready to send to the LLM.
    """
    parts: list[str] = []

    if context_docs:
        parts.append(
            "Use the following context documents to answer the question.\n"
        )
        for i, doc in enumerate(context_docs, 1):
            parts.append(f"[Document {i}]\n{doc['text']}\n")
    else:
        parts.append(
            "No prior art context is available for this topic. "
            "Answer the question based on your general knowledge.\n"
        )

    parts.append(f"Question: {question}")
    return "\n".join(parts)


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
    *,
    rag_engine: RAGEngine | None = None,
    settings: AppSettings | None = None,
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
        rag_engine: Optional RAG engine for retrieving context documents
            relevant to the active topic (Req 5.1).
        settings: Optional application settings for LLM configuration
            (Req 5.2).
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
                """Handle send: query RAG, call LLM, display response."""
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

                # --- RAG retrieval (Req 5.1) ---
                context_docs: list[dict] = []
                if rag_engine is not None:
                    try:
                        context_docs = rag_engine.query(topic_id, text)
                    except Exception:
                        logger.exception(
                            "RAG query failed for topic %d", topic_id
                        )

                # --- Build prompt and call LLM (Req 5.2, 5.4, 5.6) ---
                prompt = build_chat_prompt(context_docs, text)

                try:
                    lm = dspy.settings.lm
                    if lm is None:
                        raise ConnectionError("DSPy LM is not configured")
                    response = lm(prompt)
                    # dspy.LM returns a list of strings; take the first
                    if isinstance(response, list):
                        assistant_text = response[0] if response else ""
                    else:
                        assistant_text = str(response)
                except (
                    requests.exceptions.ConnectionError,
                    httpx.ConnectError,
                    litellm.exceptions.APIConnectionError,
                    ConnectionError,
                    OSError,
                ) as exc:
                    # LM Studio unreachable — show error, do NOT persist (Req 5.5)
                    logger.error(
                        "LM Studio unreachable for chat topic %d: %s",
                        topic_id,
                        exc,
                    )
                    error_msg = (
                        "⚠️ The LLM backend is currently unavailable. "
                        "Please ensure LM Studio is running and try again."
                    )
                    with chat_messages_container:
                        _render_message("assistant", error_msg)
                    scroll_area.scroll_to(percent=1.0)
                    return

                # --- Display and persist assistant response (Req 5.3) ---
                try:
                    chat_repo.save_message(
                        topic_id, "assistant", assistant_text
                    )
                except Exception:
                    logger.exception(
                        "Failed to save assistant message for topic %d",
                        topic_id,
                    )

                with chat_messages_container:
                    _render_message("assistant", assistant_text)

                # Scroll to bottom
                scroll_area.scroll_to(percent=1.0)

            ui.button("Send", on_click=_on_send).props("color=primary")
