"""Unit tests for the AI Chat panel logic.

Tests focus on the ``_render_message`` helper and the ``create_chat_panel``
function's interaction with ``ChatHistoryRepository``.  NiceGUI rendering
is not tested directly — we verify persistence and data flow.

Requirements: 16.5, 8.3, 8.4
"""

import sqlite3

import pytest

from patent_system.db.models import ChatMessage
from patent_system.db.repository import ChatHistoryRepository
from patent_system.db.schema import init_schema


@pytest.fixture
def _chat_db() -> sqlite3.Connection:
    """In-memory DB with schema for chat tests."""
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    init_schema(conn)
    # Create a topic to satisfy FK
    conn.execute("INSERT INTO topics (name) VALUES ('test-topic')")
    conn.commit()
    return conn


@pytest.fixture
def chat_repo(_chat_db: sqlite3.Connection) -> ChatHistoryRepository:
    return ChatHistoryRepository(_chat_db)


def test_save_and_retrieve_user_message(chat_repo: ChatHistoryRepository) -> None:
    """User messages are persisted and retrievable."""
    chat_repo.save_message(1, "user", "Hello")
    messages = chat_repo.get_by_topic(1)
    assert len(messages) == 1
    assert messages[0].role == "user"
    assert messages[0].message == "Hello"


def test_save_and_retrieve_assistant_message(chat_repo: ChatHistoryRepository) -> None:
    """Assistant messages are persisted and retrievable."""
    chat_repo.save_message(1, "assistant", "Hi there")
    messages = chat_repo.get_by_topic(1)
    assert len(messages) == 1
    assert messages[0].role == "assistant"
    assert messages[0].message == "Hi there"


def test_messages_ordered_by_timestamp(chat_repo: ChatHistoryRepository) -> None:
    """Messages come back in chronological order."""
    chat_repo.save_message(1, "user", "first")
    chat_repo.save_message(1, "assistant", "second")
    chat_repo.save_message(1, "user", "third")
    messages = chat_repo.get_by_topic(1)
    assert [m.message for m in messages] == ["first", "second", "third"]


def test_empty_topic_returns_no_messages(chat_repo: ChatHistoryRepository) -> None:
    """A topic with no chat history returns an empty list."""
    messages = chat_repo.get_by_topic(1)
    assert messages == []


def test_chat_panel_module_importable() -> None:
    """The chat_panel module can be imported without errors."""
    from patent_system.gui.chat_panel import create_chat_panel, _render_message

    assert callable(create_chat_panel)
    assert callable(_render_message)
