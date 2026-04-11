"""Database repository classes for CRUD operations.

Each repository takes a sqlite3.Connection and provides typed methods
for creating, reading, and updating records. Write failures are logged
via the structured logging helpers and re-raised so callers can handle
them (e.g. keep data in memory for retry).

Requirements: 1.2, 1.3, 1.5, 3.2, 8.3, 15.1, 15.2, 15.3
"""

import logging
import sqlite3
from datetime import datetime, timezone

from patent_system.db.models import ChatMessage, PatentRecord, Topic
from patent_system.logging_config import log_db_error

logger = logging.getLogger(__name__)


class TopicRepository:
    """CRUD operations for topics."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def create(self, name: str) -> Topic:
        """Insert a new topic and return the created Topic model.

        Raises:
            sqlite3.IntegrityError: If the name already exists.
        """
        try:
            cursor = self._conn.execute(
                "INSERT INTO topics (name) VALUES (?)",
                (name,),
            )
            self._conn.commit()
            row = self._conn.execute(
                "SELECT id, name, created_at FROM topics WHERE id = ?",
                (cursor.lastrowid,),
            ).fetchone()
            return Topic(
                id=row[0],
                name=row[1],
                created_at=_parse_timestamp(row[2]),
            )
        except sqlite3.Error as exc:
            log_db_error(logger, "INSERT", "topics", str(exc))
            raise

    def get_all(self) -> list[Topic]:
        """Return all topics ordered by created_at descending."""
        rows = self._conn.execute(
            "SELECT id, name, created_at FROM topics ORDER BY created_at DESC",
        ).fetchall()
        return [
            Topic(id=r[0], name=r[1], created_at=_parse_timestamp(r[2]))
            for r in rows
        ]

    def get_by_id(self, topic_id: int) -> Topic | None:
        """Return a topic by its ID, or None if not found."""
        row = self._conn.execute(
            "SELECT id, name, created_at FROM topics WHERE id = ?",
            (topic_id,),
        ).fetchone()
        if row is None:
            return None
        return Topic(id=row[0], name=row[1], created_at=_parse_timestamp(row[2]))

    def name_exists(self, name: str) -> bool:
        """Return True if a topic with the given name already exists."""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM topics WHERE name = ?",
            (name,),
        ).fetchone()
        return row[0] > 0


class PatentRepository:
    """CRUD operations for patent records."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def create(self, session_id: int, record: PatentRecord) -> int:
        """Insert a patent record and return the new row ID.

        Raises:
            sqlite3.IntegrityError: If session_id references a non-existent session.
        """
        try:
            cursor = self._conn.execute(
                """INSERT INTO patents
                   (session_id, patent_number, title, abstract, full_text,
                    claims, pdf_path, source, discovered_date, embedding)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    record.patent_number,
                    record.title,
                    record.abstract,
                    record.full_text,
                    record.claims,
                    record.pdf_path,
                    record.source,
                    record.discovered_date.isoformat(),
                    record.embedding,
                ),
            )
            self._conn.commit()
            return cursor.lastrowid  # type: ignore[return-value]
        except sqlite3.Error as exc:
            log_db_error(logger, "INSERT", "patents", str(exc))
            raise

    def get_by_session(self, session_id: int) -> list[PatentRecord]:
        """Return all patent records for a given research session."""
        rows = self._conn.execute(
            """SELECT id, session_id, patent_number, title, abstract,
                      full_text, claims, pdf_path, source, discovered_date, embedding
               FROM patents WHERE session_id = ?""",
            (session_id,),
        ).fetchall()
        return [
            PatentRecord(
                id=r[0],
                session_id=r[1],
                patent_number=r[2],
                title=r[3],
                abstract=r[4],
                full_text=r[5],
                claims=r[6],
                pdf_path=r[7],
                source=r[8],
                discovered_date=_parse_timestamp(r[9]),
                embedding=r[10],
            )
            for r in rows
        ]

    def update_embedding(self, patent_id: int, embedding: bytes) -> None:
        """Update the embedding BLOB for a patent record.

        Raises:
            sqlite3.Error: On database failure.
        """
        try:
            self._conn.execute(
                "UPDATE patents SET embedding = ? WHERE id = ?",
                (embedding, patent_id),
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            log_db_error(logger, "UPDATE", "patents", str(exc))
            raise


class ResearchSessionRepository:
    """CRUD operations for research sessions."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def create(self, topic_id: int, query: str) -> int:
        """Insert a new research session and return the new row ID.

        Raises:
            sqlite3.IntegrityError: If topic_id references a non-existent topic.
        """
        try:
            cursor = self._conn.execute(
                "INSERT INTO research_sessions (topic_id, query) VALUES (?, ?)",
                (topic_id, query),
            )
            self._conn.commit()
            return cursor.lastrowid  # type: ignore[return-value]
        except sqlite3.Error as exc:
            log_db_error(logger, "INSERT", "research_sessions", str(exc))
            raise

    def get_by_topic(self, topic_id: int) -> list[dict]:
        """Return all research sessions for a topic as dicts."""
        rows = self._conn.execute(
            """SELECT id, topic_id, query, search_date, status
               FROM research_sessions WHERE topic_id = ?""",
            (topic_id,),
        ).fetchall()
        return [
            {
                "id": r[0],
                "topic_id": r[1],
                "query": r[2],
                "search_date": r[3],
                "status": r[4],
            }
            for r in rows
        ]


class ChatHistoryRepository:
    """CRUD operations for chat messages."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def save_message(self, topic_id: int, role: str, message: str) -> int:
        """Insert a chat message and return the new row ID.

        Raises:
            sqlite3.IntegrityError: If topic_id references a non-existent topic.
        """
        try:
            cursor = self._conn.execute(
                "INSERT INTO chat_history (topic_id, role, message) VALUES (?, ?, ?)",
                (topic_id, role, message),
            )
            self._conn.commit()
            return cursor.lastrowid  # type: ignore[return-value]
        except sqlite3.Error as exc:
            log_db_error(logger, "INSERT", "chat_history", str(exc))
            raise

    def get_by_topic(self, topic_id: int) -> list[ChatMessage]:
        """Return all chat messages for a topic, ordered by timestamp ascending."""
        rows = self._conn.execute(
            """SELECT id, topic_id, role, message, timestamp
               FROM chat_history WHERE topic_id = ? ORDER BY timestamp ASC""",
            (topic_id,),
        ).fetchall()
        return [
            ChatMessage(
                id=r[0],
                topic_id=r[1],
                role=r[2],
                message=r[3],
                timestamp=_parse_timestamp(r[4]),
            )
            for r in rows
        ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_timestamp(value: str | datetime) -> datetime:
    """Parse a SQLite TIMESTAMP string into a timezone-aware datetime.

    SQLite stores CURRENT_TIMESTAMP as ``YYYY-MM-DD HH:MM:SS`` (UTC).
    If the value is already a datetime, return it directly.
    """
    if isinstance(value, datetime):
        return value
    # Try ISO format first (used when we store via .isoformat())
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        # Fallback for SQLite CURRENT_TIMESTAMP format
        dt = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt
