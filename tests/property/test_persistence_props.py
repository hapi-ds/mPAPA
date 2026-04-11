"""Property-based tests for persistence and data integrity.

Validates: Requirements 1.2, 1.3, 1.5, 2.4, 3.2, 4.4, 5.4, 7.4, 8.3, 15.2
"""

import sqlite3

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from patent_system.db.models import PatentRecord
from patent_system.db.repository import (
    ChatHistoryRepository,
    PatentRepository,
    ResearchSessionRepository,
    TopicRepository,
)
from patent_system.db.schema import init_schema

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh_db() -> sqlite3.Connection:
    """Create a fresh in-memory SQLite connection with FK enforcement and full schema."""
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    init_schema(conn)
    return conn


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Safe printable text — no control chars, no NUL bytes
_safe_text = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N", "P", "S"),
        whitelist_characters=" /-_:.",
    ),
    min_size=1,
    max_size=80,
)

# Unique topic names via UUID
_topic_name = st.uuids().map(lambda u: f"topic-{u}")

# Roles for chat messages
_chat_role = st.sampled_from(["user", "assistant"])

# Patent sources
_patent_source = st.sampled_from(
    ["DEPATISnet", "Google Patents", "ArXiv", "PubMed", "Google Scholar"]
)

# Non-existent IDs (very large to avoid collisions with auto-increment)
_nonexistent_id = st.integers(min_value=900_000, max_value=999_999)


# ---------------------------------------------------------------------------
# Property 1: Domain record persistence round-trip
# Feature: patent-analysis-drafting, Property 1: Domain record persistence round-trip
# ---------------------------------------------------------------------------


class TestDomainRecordRoundTrip:
    """Property 1: Domain record persistence round-trip.

    For any valid domain record, persisting and retrieving by ID produces
    equivalent field values.

    **Validates: Requirements 1.2, 2.4, 3.2, 4.4, 5.4, 7.4, 8.3**
    """

    @given(name=_topic_name)
    @settings(max_examples=100)
    def test_topic_round_trip(self, name: str) -> None:
        """For any valid topic name, create then get_by_id returns matching fields."""
        conn = _fresh_db()
        try:
            repo = TopicRepository(conn)
            created = repo.create(name)

            retrieved = repo.get_by_id(created.id)

            assert retrieved is not None
            assert retrieved.id == created.id
            assert retrieved.name == name
            assert retrieved.created_at == created.created_at
        finally:
            conn.close()

    @given(
        patent_number=_safe_text,
        title=_safe_text,
        abstract_text=_safe_text,
        source=_patent_source,
    )
    @settings(max_examples=100)
    def test_patent_record_round_trip(
        self,
        patent_number: str,
        title: str,
        abstract_text: str,
        source: str,
    ) -> None:
        """For any valid PatentRecord, create then get_by_session returns matching fields."""
        conn = _fresh_db()
        try:
            topic = TopicRepository(conn).create("patent-rt-topic")
            session_id = ResearchSessionRepository(conn).create(topic.id, "query")

            record = PatentRecord(
                patent_number=patent_number,
                title=title,
                abstract=abstract_text,
                source=source,
            )
            PatentRepository(conn).create(session_id, record)

            results = PatentRepository(conn).get_by_session(session_id)
            assert len(results) == 1
            r = results[0]
            assert r.patent_number == patent_number
            assert r.title == title
            assert r.abstract == abstract_text
            assert r.source == source
        finally:
            conn.close()

    @given(
        role=_chat_role,
        message=_safe_text,
    )
    @settings(max_examples=100)
    def test_chat_message_round_trip(
        self,
        role: str,
        message: str,
    ) -> None:
        """For any valid ChatMessage, save then get_by_topic returns matching fields."""
        conn = _fresh_db()
        try:
            topic = TopicRepository(conn).create("chat-rt-topic")
            repo = ChatHistoryRepository(conn)

            msg_id = repo.save_message(topic.id, role, message)
            messages = repo.get_by_topic(topic.id)

            assert len(messages) == 1
            m = messages[0]
            assert m.id == msg_id
            assert m.topic_id == topic.id
            assert m.role == role
            assert m.message == message
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Property 2: Topic list ordering
# Feature: patent-analysis-drafting, Property 2: Topic list ordering
# ---------------------------------------------------------------------------


class TestTopicListOrdering:
    """Property 2: Topic list ordering.

    For any set of Topics with distinct timestamps, get_all returns them
    ordered by created_at descending.

    **Validates: Requirements 1.3**
    """

    @given(
        names=st.lists(
            _safe_text,
            min_size=2,
            max_size=10,
            unique=True,
        ),
    )
    @settings(max_examples=100)
    def test_get_all_returns_descending_order(
        self,
        names: list[str],
    ) -> None:
        """For any N topics with distinct timestamps, get_all is descending by created_at."""
        conn = _fresh_db()
        try:
            # Insert with explicit, distinct timestamps via raw SQL
            base_year = 2020
            for i, name in enumerate(names):
                ts = f"{base_year + i}-06-15 12:00:00"
                conn.execute(
                    "INSERT INTO topics (name, created_at) VALUES (?, ?)",
                    (name, ts),
                )
            conn.commit()

            repo = TopicRepository(conn)
            topics = repo.get_all()

            assert len(topics) == len(names)
            # Verify descending order
            for i in range(len(topics) - 1):
                assert topics[i].created_at >= topics[i + 1].created_at
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Property 3: Duplicate topic name rejection
# Feature: patent-analysis-drafting, Property 3: Duplicate topic name rejection
# ---------------------------------------------------------------------------


class TestDuplicateTopicNameRejection:
    """Property 3: Duplicate topic name rejection.

    For any existing topic name, creating a duplicate fails and total
    count is unchanged.

    **Validates: Requirements 1.5**
    """

    @given(name=_topic_name)
    @settings(max_examples=100)
    def test_duplicate_name_raises_and_count_unchanged(
        self,
        name: str,
    ) -> None:
        """For any topic name, creating it twice raises IntegrityError and count stays at 1."""
        conn = _fresh_db()
        try:
            repo = TopicRepository(conn)
            repo.create(name)

            count_before = len(repo.get_all())

            with pytest.raises(sqlite3.IntegrityError):
                repo.create(name)

            count_after = len(repo.get_all())
            assert count_after == count_before
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Property 9: Foreign key constraint enforcement
# Feature: patent-analysis-drafting, Property 9: Foreign key constraint enforcement
# ---------------------------------------------------------------------------


class TestForeignKeyConstraintEnforcement:
    """Property 9: Foreign key constraint enforcement.

    For any record referencing a non-existent FK, inserting raises an
    integrity error.

    **Validates: Requirements 15.2**
    """

    @given(bad_topic_id=_nonexistent_id, query=_safe_text)
    @settings(max_examples=100)
    def test_research_session_bad_topic_id(
        self,
        bad_topic_id: int,
        query: str,
    ) -> None:
        """Inserting a research session with non-existent topic_id raises IntegrityError."""
        conn = _fresh_db()
        try:
            repo = ResearchSessionRepository(conn)
            with pytest.raises(sqlite3.IntegrityError):
                repo.create(bad_topic_id, query)
        finally:
            conn.close()

    @given(bad_session_id=_nonexistent_id)
    @settings(max_examples=100)
    def test_patent_bad_session_id(
        self,
        bad_session_id: int,
    ) -> None:
        """Inserting a patent with non-existent session_id raises IntegrityError."""
        conn = _fresh_db()
        try:
            record = PatentRecord(
                patent_number="US0000000",
                title="Bad FK Patent",
                abstract="Abstract",
                source="DEPATISnet",
            )
            repo = PatentRepository(conn)
            with pytest.raises(sqlite3.IntegrityError):
                repo.create(bad_session_id, record)
        finally:
            conn.close()

    @given(bad_topic_id=_nonexistent_id, role=_chat_role, message=_safe_text)
    @settings(max_examples=100)
    def test_chat_message_bad_topic_id(
        self,
        bad_topic_id: int,
        role: str,
        message: str,
    ) -> None:
        """Inserting a chat message with non-existent topic_id raises IntegrityError."""
        conn = _fresh_db()
        try:
            repo = ChatHistoryRepository(conn)
            with pytest.raises(sqlite3.IntegrityError):
                repo.save_message(bad_topic_id, role, message)
        finally:
            conn.close()
