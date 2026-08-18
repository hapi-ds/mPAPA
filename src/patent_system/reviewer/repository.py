"""Database repository classes for the patent reviewer module.

Provides CRUD operations for review sessions and findings, following
the same patterns as the core repository classes in db/repository.py.
"""

import json
import logging
import sqlite3
from datetime import datetime, timezone

from patent_system.logging_config import log_db_error

logger = logging.getLogger(__name__)


def _parse_timestamp(value: str | datetime) -> datetime:
    """Parse a SQLite timestamp string into a timezone-aware datetime."""
    if isinstance(value, datetime):
        return value
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        dt = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


class ReviewSessionRepository:
    """CRUD operations for review sessions."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def create_session(
        self,
        patent_text: str,
        card_ids: list[str],
        jurisdiction: str,
        *,
        topic_id: int | None = None,
    ) -> int:
        """Insert a new review session and return the row ID.

        Args:
            patent_text: The patent text being reviewed.
            card_ids: List of rule card IDs used in this review.
            jurisdiction: Primary jurisdiction code (e.g. "EP", "US").
            topic_id: Optional topic FK (None for standalone reviews).

        Returns:
            The new session's row ID.

        Raises:
            sqlite3.Error: On database failure.
        """
        try:
            cursor = self._conn.execute(
                """INSERT INTO review_sessions
                   (topic_id, patent_text, card_ids, jurisdiction, status)
                   VALUES (?, ?, ?, ?, 'pending')""",
                (topic_id, patent_text, json.dumps(card_ids), jurisdiction),
            )
            self._conn.commit()
            return cursor.lastrowid  # type: ignore[return-value]
        except sqlite3.Error as exc:
            log_db_error(logger, "INSERT", "review_sessions", str(exc))
            raise

    def update_status(
        self,
        session_id: int,
        status: str,
        *,
        completed_at: str | None = None,
    ) -> None:
        """Update the status of a review session.

        Args:
            session_id: The session row ID.
            status: New status (pending | running | completed | failed).
            completed_at: Optional ISO timestamp for completion. If status
                is 'completed' or 'failed' and this is None, sets it to now.

        Raises:
            ValueError: If status is not a valid value.
            sqlite3.Error: On database failure.
        """
        valid_statuses = {"pending", "running", "completed", "failed"}
        if status not in valid_statuses:
            raise ValueError(
                f"Invalid status {status!r}. Must be one of {sorted(valid_statuses)}"
            )

        if status in ("completed", "failed") and completed_at is None:
            completed_at = datetime.now(timezone.utc).isoformat()

        try:
            self._conn.execute(
                """UPDATE review_sessions
                   SET status = ?, completed_at = ?
                   WHERE id = ?""",
                (status, completed_at, session_id),
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            log_db_error(logger, "UPDATE", "review_sessions", str(exc))
            raise

    def get_session(self, session_id: int) -> dict | None:
        """Return a review session by ID, or None if not found.

        Returns:
            Dict with keys: id, topic_id, patent_text, card_ids (as list),
            jurisdiction, status, created_at, completed_at.
        """
        try:
            row = self._conn.execute(
                """SELECT id, topic_id, patent_text, card_ids, jurisdiction,
                          status, created_at, completed_at
                   FROM review_sessions WHERE id = ?""",
                (session_id,),
            ).fetchone()
        except sqlite3.Error as exc:
            log_db_error(logger, "SELECT", "review_sessions", str(exc))
            raise

        if row is None:
            return None
        return {
            "id": row[0],
            "topic_id": row[1],
            "patent_text": row[2],
            "card_ids": json.loads(row[3]),
            "jurisdiction": row[4],
            "status": row[5],
            "created_at": row[6],
            "completed_at": row[7],
        }

    def list_sessions(
        self,
        *,
        topic_id: int | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """Return review sessions, newest first.

        Args:
            topic_id: If provided, filter to sessions for this topic.
            limit: Maximum number of sessions to return.

        Returns:
            List of session dicts (same shape as get_session, but
            patent_text is truncated to 200 chars for listing).
        """
        try:
            if topic_id is not None:
                rows = self._conn.execute(
                    """SELECT id, topic_id, patent_text, card_ids, jurisdiction,
                              status, created_at, completed_at
                       FROM review_sessions
                       WHERE topic_id = ?
                       ORDER BY created_at DESC
                       LIMIT ?""",
                    (topic_id, limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    """SELECT id, topic_id, patent_text, card_ids, jurisdiction,
                              status, created_at, completed_at
                       FROM review_sessions
                       ORDER BY created_at DESC
                       LIMIT ?""",
                    (limit,),
                ).fetchall()
        except sqlite3.Error as exc:
            log_db_error(logger, "SELECT", "review_sessions", str(exc))
            raise

        return [
            {
                "id": r[0],
                "topic_id": r[1],
                "patent_text": r[2][:200] if r[2] else "",
                "card_ids": json.loads(r[3]),
                "jurisdiction": r[4],
                "status": r[5],
                "created_at": r[6],
                "completed_at": r[7],
            }
            for r in rows
        ]


class ReviewFindingRepository:
    """CRUD operations for review findings."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def save_findings(
        self,
        session_id: int,
        findings: list[dict],
    ) -> list[int]:
        """Bulk-insert findings for a review session.

        Args:
            session_id: The review session FK.
            findings: List of finding dicts with keys: card_id, rule_id,
                finding, severity, location, suggestion, compliant, reference.

        Returns:
            List of inserted row IDs.

        Raises:
            sqlite3.Error: On database failure.
        """
        row_ids: list[int] = []
        try:
            cursor = self._conn.cursor()
            for f in findings:
                cursor.execute(
                    """INSERT INTO review_findings
                       (session_id, card_id, rule_id, finding, severity,
                        location, suggestion, compliant, reference)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        session_id,
                        f["card_id"],
                        f["rule_id"],
                        f["finding"],
                        f["severity"],
                        f.get("location", ""),
                        f.get("suggestion", ""),
                        1 if f.get("compliant", False) else 0,
                        f.get("reference", ""),
                    ),
                )
                row_ids.append(cursor.lastrowid)  # type: ignore[arg-type]
            self._conn.commit()
            return row_ids
        except sqlite3.Error as exc:
            self._conn.rollback()
            log_db_error(logger, "INSERT", "review_findings", str(exc))
            raise

    def get_findings_by_session(self, session_id: int) -> list[dict]:
        """Return all findings for a review session.

        Returns:
            List of finding dicts with keys: id, session_id, card_id,
            rule_id, finding, severity, location, suggestion, compliant,
            reference, created_at.
        """
        try:
            rows = self._conn.execute(
                """SELECT id, session_id, card_id, rule_id, finding,
                          severity, location, suggestion, compliant,
                          reference, created_at
                   FROM review_findings
                   WHERE session_id = ?
                   ORDER BY id ASC""",
                (session_id,),
            ).fetchall()
        except sqlite3.Error as exc:
            log_db_error(logger, "SELECT", "review_findings", str(exc))
            raise

        return [
            {
                "id": r[0],
                "session_id": r[1],
                "card_id": r[2],
                "rule_id": r[3],
                "finding": r[4],
                "severity": r[5],
                "location": r[6],
                "suggestion": r[7],
                "compliant": bool(r[8]),
                "reference": r[9],
                "created_at": r[10],
            }
            for r in rows
        ]
