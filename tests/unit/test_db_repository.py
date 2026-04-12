"""Unit tests for database repository classes.

Tests cover all CRUD methods in TopicRepository, PatentRepository,
ResearchSessionRepository, and ChatHistoryRepository using an in-memory
SQLite database with the full schema.
"""

import sqlite3

import pytest

from patent_system.db.models import PatentRecord, Topic
from patent_system.db.repository import (
    ChatHistoryRepository,
    PatentRepository,
    ResearchSessionRepository,
    TopicRepository,
)


# ---------------------------------------------------------------------------
# TopicRepository
# ---------------------------------------------------------------------------


class TestTopicRepository:
    """Tests for TopicRepository."""

    def test_create_returns_topic_with_id(self, in_memory_db: sqlite3.Connection) -> None:
        repo = TopicRepository(in_memory_db)
        topic = repo.create("My Invention")
        assert topic.id is not None
        assert topic.name == "My Invention"
        assert topic.created_at is not None

    def test_create_duplicate_name_raises(self, in_memory_db: sqlite3.Connection) -> None:
        repo = TopicRepository(in_memory_db)
        repo.create("Duplicate")
        with pytest.raises(sqlite3.IntegrityError):
            repo.create("Duplicate")

    def test_get_all_returns_newest_first(self, in_memory_db: sqlite3.Connection) -> None:
        repo = TopicRepository(in_memory_db)
        # Insert with explicit timestamps to guarantee ordering
        in_memory_db.execute(
            "INSERT INTO topics (name, created_at) VALUES (?, ?)",
            ("First", "2024-01-01 00:00:00"),
        )
        in_memory_db.execute(
            "INSERT INTO topics (name, created_at) VALUES (?, ?)",
            ("Second", "2024-06-01 00:00:00"),
        )
        in_memory_db.commit()
        topics = repo.get_all()
        assert len(topics) == 2
        # Newest first
        assert topics[0].name == "Second"
        assert topics[1].name == "First"

    def test_get_all_empty(self, in_memory_db: sqlite3.Connection) -> None:
        repo = TopicRepository(in_memory_db)
        assert repo.get_all() == []

    def test_get_by_id_found(self, in_memory_db: sqlite3.Connection) -> None:
        repo = TopicRepository(in_memory_db)
        created = repo.create("Find Me")
        found = repo.get_by_id(created.id)  # type: ignore[arg-type]
        assert found is not None
        assert found.name == "Find Me"
        assert found.id == created.id

    def test_get_by_id_not_found(self, in_memory_db: sqlite3.Connection) -> None:
        repo = TopicRepository(in_memory_db)
        assert repo.get_by_id(9999) is None

    def test_name_exists_true(self, in_memory_db: sqlite3.Connection) -> None:
        repo = TopicRepository(in_memory_db)
        repo.create("Exists")
        assert repo.name_exists("Exists") is True

    def test_name_exists_false(self, in_memory_db: sqlite3.Connection) -> None:
        repo = TopicRepository(in_memory_db)
        assert repo.name_exists("Nope") is False


# ---------------------------------------------------------------------------
# PatentRepository
# ---------------------------------------------------------------------------


class TestPatentRepository:
    """Tests for PatentRepository."""

    @pytest.fixture()
    def session_id(self, in_memory_db: sqlite3.Connection) -> int:
        """Create a topic and research session, return the session ID."""
        topic_repo = TopicRepository(in_memory_db)
        topic = topic_repo.create("Patent Topic")
        session_repo = ResearchSessionRepository(in_memory_db)
        return session_repo.create(topic.id, "test query")  # type: ignore[arg-type]

    def _make_record(self) -> PatentRecord:
        return PatentRecord(
            patent_number="US1234567",
            title="Test Patent",
            abstract="An abstract",
            source="DEPATISnet",
        )

    def test_create_returns_id(
        self, in_memory_db: sqlite3.Connection, session_id: int
    ) -> None:
        repo = PatentRepository(in_memory_db)
        patent_id = repo.create(session_id, self._make_record())
        assert isinstance(patent_id, int)
        assert patent_id > 0

    def test_get_by_session(
        self, in_memory_db: sqlite3.Connection, session_id: int
    ) -> None:
        repo = PatentRepository(in_memory_db)
        repo.create(session_id, self._make_record())
        results = repo.get_by_session(session_id)
        assert len(results) == 1
        assert results[0].patent_number == "US1234567"
        assert results[0].title == "Test Patent"

    def test_get_by_session_empty(self, in_memory_db: sqlite3.Connection) -> None:
        repo = PatentRepository(in_memory_db)
        assert repo.get_by_session(9999) == []

    def test_update_embedding(
        self, in_memory_db: sqlite3.Connection, session_id: int
    ) -> None:
        repo = PatentRepository(in_memory_db)
        patent_id = repo.create(session_id, self._make_record())
        embedding = b"\x00\x01\x02\x03"
        repo.update_embedding(patent_id, embedding)
        results = repo.get_by_session(session_id)
        assert results[0].embedding == embedding

    def test_create_invalid_session_raises(
        self, in_memory_db: sqlite3.Connection
    ) -> None:
        repo = PatentRepository(in_memory_db)
        with pytest.raises(sqlite3.IntegrityError):
            repo.create(99999, self._make_record())


# ---------------------------------------------------------------------------
# ResearchSessionRepository
# ---------------------------------------------------------------------------


class TestResearchSessionRepository:
    """Tests for ResearchSessionRepository."""

    def test_create_returns_id(self, in_memory_db: sqlite3.Connection) -> None:
        topic = TopicRepository(in_memory_db).create("Session Topic")
        repo = ResearchSessionRepository(in_memory_db)
        sid = repo.create(topic.id, "search query")  # type: ignore[arg-type]
        assert isinstance(sid, int)
        assert sid > 0

    def test_get_by_topic(self, in_memory_db: sqlite3.Connection) -> None:
        topic = TopicRepository(in_memory_db).create("Session Topic")
        repo = ResearchSessionRepository(in_memory_db)
        repo.create(topic.id, "query one")  # type: ignore[arg-type]
        repo.create(topic.id, "query two")  # type: ignore[arg-type]
        sessions = repo.get_by_topic(topic.id)  # type: ignore[arg-type]
        assert len(sessions) == 2
        assert sessions[0]["query"] == "query one"
        assert sessions[1]["query"] == "query two"

    def test_get_by_topic_empty(self, in_memory_db: sqlite3.Connection) -> None:
        repo = ResearchSessionRepository(in_memory_db)
        assert repo.get_by_topic(9999) == []

    def test_create_invalid_topic_raises(self, in_memory_db: sqlite3.Connection) -> None:
        repo = ResearchSessionRepository(in_memory_db)
        with pytest.raises(sqlite3.IntegrityError):
            repo.create(99999, "bad query")


# ---------------------------------------------------------------------------
# ChatHistoryRepository
# ---------------------------------------------------------------------------


class TestChatHistoryRepository:
    """Tests for ChatHistoryRepository."""

    def test_save_message_returns_id(self, in_memory_db: sqlite3.Connection) -> None:
        topic = TopicRepository(in_memory_db).create("Chat Topic")
        repo = ChatHistoryRepository(in_memory_db)
        msg_id = repo.save_message(topic.id, "user", "Hello")  # type: ignore[arg-type]
        assert isinstance(msg_id, int)
        assert msg_id > 0

    def test_get_by_topic_ordered_by_timestamp(
        self, in_memory_db: sqlite3.Connection
    ) -> None:
        topic = TopicRepository(in_memory_db).create("Chat Topic")
        repo = ChatHistoryRepository(in_memory_db)
        repo.save_message(topic.id, "user", "First")  # type: ignore[arg-type]
        repo.save_message(topic.id, "assistant", "Second")  # type: ignore[arg-type]
        messages = repo.get_by_topic(topic.id)  # type: ignore[arg-type]
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[0].message == "First"
        assert messages[1].role == "assistant"
        assert messages[1].message == "Second"
        assert messages[0].timestamp <= messages[1].timestamp

    def test_get_by_topic_empty(self, in_memory_db: sqlite3.Connection) -> None:
        repo = ChatHistoryRepository(in_memory_db)
        assert repo.get_by_topic(9999) == []

    def test_save_message_invalid_topic_raises(
        self, in_memory_db: sqlite3.Connection
    ) -> None:
        repo = ChatHistoryRepository(in_memory_db)
        with pytest.raises(sqlite3.IntegrityError):
            repo.save_message(99999, "user", "bad message")


# ---------------------------------------------------------------------------
# InventionDisclosureRepository
# ---------------------------------------------------------------------------


class TestInventionDisclosureRepository:
    """Tests for InventionDisclosureRepository."""

    def test_upsert_creates_disclosure_and_returns_id(
        self, in_memory_db: sqlite3.Connection
    ) -> None:
        from patent_system.db.repository import InventionDisclosureRepository

        topic = TopicRepository(in_memory_db).create("Disclosure Topic")
        repo = InventionDisclosureRepository(in_memory_db)
        disc_id = repo.upsert(topic.id, "A novel widget", ["widget", "gadget"])  # type: ignore[arg-type]
        assert isinstance(disc_id, int)
        assert disc_id > 0

    def test_get_by_topic_returns_correct_data(
        self, in_memory_db: sqlite3.Connection
    ) -> None:
        from patent_system.db.repository import InventionDisclosureRepository

        topic = TopicRepository(in_memory_db).create("Disclosure Topic")
        repo = InventionDisclosureRepository(in_memory_db)
        repo.upsert(topic.id, "A novel widget", ["widget", "gadget"])  # type: ignore[arg-type]
        result = repo.get_by_topic(topic.id)  # type: ignore[arg-type]
        assert result is not None
        assert result["primary_description"] == "A novel widget"
        assert result["search_terms"] == ["widget", "gadget"]

    def test_get_by_topic_returns_none_when_missing(
        self, in_memory_db: sqlite3.Connection
    ) -> None:
        from patent_system.db.repository import InventionDisclosureRepository

        repo = InventionDisclosureRepository(in_memory_db)
        assert repo.get_by_topic(9999) is None

    def test_upsert_replaces_existing_disclosure(
        self, in_memory_db: sqlite3.Connection
    ) -> None:
        from patent_system.db.repository import InventionDisclosureRepository

        topic = TopicRepository(in_memory_db).create("Disclosure Topic")
        repo = InventionDisclosureRepository(in_memory_db)
        repo.upsert(topic.id, "First description", ["term1"])  # type: ignore[arg-type]
        repo.upsert(topic.id, "Updated description", ["term2", "term3"])  # type: ignore[arg-type]
        result = repo.get_by_topic(topic.id)  # type: ignore[arg-type]
        assert result is not None
        assert result["primary_description"] == "Updated description"
        assert result["search_terms"] == ["term2", "term3"]

    def test_upsert_with_empty_search_terms(
        self, in_memory_db: sqlite3.Connection
    ) -> None:
        from patent_system.db.repository import InventionDisclosureRepository

        topic = TopicRepository(in_memory_db).create("Disclosure Topic")
        repo = InventionDisclosureRepository(in_memory_db)
        repo.upsert(topic.id, "Description only", [])  # type: ignore[arg-type]
        result = repo.get_by_topic(topic.id)  # type: ignore[arg-type]
        assert result is not None
        assert result["primary_description"] == "Description only"
        assert result["search_terms"] == []

    def test_upsert_preserves_term_order(
        self, in_memory_db: sqlite3.Connection
    ) -> None:
        from patent_system.db.repository import InventionDisclosureRepository

        topic = TopicRepository(in_memory_db).create("Disclosure Topic")
        repo = InventionDisclosureRepository(in_memory_db)
        terms = ["zebra", "apple", "mango", "banana"]
        repo.upsert(topic.id, "Ordered terms", terms)  # type: ignore[arg-type]
        result = repo.get_by_topic(topic.id)  # type: ignore[arg-type]
        assert result is not None
        assert result["search_terms"] == terms

    def test_upsert_invalid_topic_raises(
        self, in_memory_db: sqlite3.Connection
    ) -> None:
        from patent_system.db.repository import InventionDisclosureRepository

        repo = InventionDisclosureRepository(in_memory_db)
        with pytest.raises(sqlite3.IntegrityError):
            repo.upsert(99999, "Bad topic", ["term"])


# ---------------------------------------------------------------------------
# SourcePreferenceRepository
# ---------------------------------------------------------------------------


class TestSourcePreferenceRepository:
    """Tests for SourcePreferenceRepository."""

    def test_save_and_get_preferences(
        self, in_memory_db: sqlite3.Connection
    ) -> None:
        from patent_system.db.repository import SourcePreferenceRepository

        topic = TopicRepository(in_memory_db).create("Pref Topic")
        repo = SourcePreferenceRepository(in_memory_db)
        prefs = {"ArXiv": True, "PubMed": False, "Google Scholar": True}
        repo.save(topic.id, prefs)  # type: ignore[arg-type]
        result = repo.get_by_topic(topic.id)  # type: ignore[arg-type]
        assert result == prefs

    def test_get_by_topic_returns_none_when_missing(
        self, in_memory_db: sqlite3.Connection
    ) -> None:
        from patent_system.db.repository import SourcePreferenceRepository

        repo = SourcePreferenceRepository(in_memory_db)
        assert repo.get_by_topic(9999) is None

    def test_save_replaces_existing_preferences(
        self, in_memory_db: sqlite3.Connection
    ) -> None:
        from patent_system.db.repository import SourcePreferenceRepository

        topic = TopicRepository(in_memory_db).create("Pref Topic")
        repo = SourcePreferenceRepository(in_memory_db)
        repo.save(topic.id, {"ArXiv": True, "PubMed": True})  # type: ignore[arg-type]
        repo.save(topic.id, {"ArXiv": False, "Google Patents": True})  # type: ignore[arg-type]
        result = repo.get_by_topic(topic.id)  # type: ignore[arg-type]
        assert result == {"ArXiv": False, "Google Patents": True}

    def test_save_empty_preferences(
        self, in_memory_db: sqlite3.Connection
    ) -> None:
        from patent_system.db.repository import SourcePreferenceRepository

        topic = TopicRepository(in_memory_db).create("Pref Topic")
        repo = SourcePreferenceRepository(in_memory_db)
        repo.save(topic.id, {})  # type: ignore[arg-type]
        result = repo.get_by_topic(topic.id)  # type: ignore[arg-type]
        # Empty dict means no rows → returns None
        assert result is None

    def test_save_boolean_values_preserved(
        self, in_memory_db: sqlite3.Connection
    ) -> None:
        from patent_system.db.repository import SourcePreferenceRepository

        topic = TopicRepository(in_memory_db).create("Pref Topic")
        repo = SourcePreferenceRepository(in_memory_db)
        prefs = {
            "ArXiv": True,
            "PubMed": False,
            "Google Scholar": True,
            "Google Patents": False,
            "DEPATISnet": True,
        }
        repo.save(topic.id, prefs)  # type: ignore[arg-type]
        result = repo.get_by_topic(topic.id)  # type: ignore[arg-type]
        assert result == prefs
