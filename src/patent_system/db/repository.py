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

from patent_system.db.models import ChatMessage, PatentRecord, ScientificPaperRecord, Topic
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
                    claims, pdf_path, source, discovered_date, embedding,
                    relevance_score)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                    record.relevance_score,
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
                      full_text, claims, pdf_path, source, discovered_date,
                      embedding, relevance_score
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
                relevance_score=r[11],
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

    def update_relevance_score(self, patent_id: int, score: float) -> None:
        """Update the relevance score for a patent record."""
        try:
            self._conn.execute(
                "UPDATE patents SET relevance_score = ? WHERE id = ?",
                (score, patent_id),
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            log_db_error(logger, "UPDATE", "patents", str(exc))
            raise

    def delete(self, patent_id: int) -> None:
        """Delete a patent record by its ID.

        Args:
            patent_id: The row ID of the patent to delete.

        Raises:
            sqlite3.Error: On database failure.
        """
        try:
            self._conn.execute("DELETE FROM patents WHERE id = ?", (patent_id,))
            self._conn.commit()
        except sqlite3.Error as exc:
            log_db_error(logger, "DELETE", "patents", str(exc))
            raise


class ScientificPaperRepository:
    """CRUD operations for scientific paper records."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def create(self, session_id: int, record: ScientificPaperRecord) -> int:
        """Insert a scientific paper record and return the new row ID.

        Raises:
            sqlite3.IntegrityError: If session_id references a non-existent session.
        """
        try:
            cursor = self._conn.execute(
                """INSERT INTO scientific_papers
                   (session_id, doi, title, abstract, full_text,
                    pdf_path, source, discovered_date, embedding,
                    relevance_score)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    record.doi,
                    record.title,
                    record.abstract,
                    record.full_text,
                    record.pdf_path,
                    record.source,
                    record.discovered_date.isoformat(),
                    record.embedding,
                    record.relevance_score,
                ),
            )
            self._conn.commit()
            return cursor.lastrowid  # type: ignore[return-value]
        except sqlite3.Error as exc:
            log_db_error(logger, "INSERT", "scientific_papers", str(exc))
            raise

    def get_by_session(self, session_id: int) -> list[ScientificPaperRecord]:
        """Return all scientific paper records for a given research session."""
        rows = self._conn.execute(
            """SELECT id, session_id, doi, title, abstract,
                      full_text, pdf_path, source, discovered_date,
                      embedding, relevance_score
               FROM scientific_papers WHERE session_id = ?""",
            (session_id,),
        ).fetchall()
        return [
            ScientificPaperRecord(
                id=r[0],
                session_id=r[1],
                doi=r[2],
                title=r[3],
                abstract=r[4],
                full_text=r[5],
                pdf_path=r[6],
                source=r[7],
                discovered_date=_parse_timestamp(r[8]),
                embedding=r[9],
                relevance_score=r[10],
            )
            for r in rows
        ]

    def update_embedding(self, paper_id: int, embedding: bytes) -> None:
        """Update the embedding BLOB for a scientific paper record."""
        try:
            self._conn.execute(
                "UPDATE scientific_papers SET embedding = ? WHERE id = ?",
                (embedding, paper_id),
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            log_db_error(logger, "UPDATE", "scientific_papers", str(exc))
            raise

    def update_relevance_score(self, paper_id: int, score: float) -> None:
        """Update the relevance score for a scientific paper record."""
        try:
            self._conn.execute(
                "UPDATE scientific_papers SET relevance_score = ? WHERE id = ?",
                (score, paper_id),
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            log_db_error(logger, "UPDATE", "scientific_papers", str(exc))
            raise

    def delete(self, paper_id: int) -> None:
        """Delete a scientific paper record by its ID."""
        try:
            self._conn.execute(
                "DELETE FROM scientific_papers WHERE id = ?", (paper_id,),
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            log_db_error(logger, "DELETE", "scientific_papers", str(exc))
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

    def delete_by_topic(self, topic_id: int) -> int:
        """Delete all chat messages for a topic.

        Args:
            topic_id: The topic whose chat history to clear.

        Returns:
            Number of deleted rows.

        Raises:
            sqlite3.Error: On database failure.
        """
        try:
            cursor = self._conn.execute(
                "DELETE FROM chat_history WHERE topic_id = ?",
                (topic_id,),
            )
            self._conn.commit()
            return cursor.rowcount
        except sqlite3.Error as exc:
            log_db_error(logger, "DELETE", "chat_history", str(exc))
            raise


class PatentDraftRepository:
    """CRUD operations for per-topic patent draft text (claims + description)."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def upsert(self, topic_id: int, claims_text: str, description_text: str) -> None:
        """Save or update the draft for a topic."""
        try:
            self._conn.execute(
                """INSERT OR REPLACE INTO patent_drafts
                   (topic_id, claims_text, description_text)
                   VALUES (?, ?, ?)""",
                (topic_id, claims_text, description_text),
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            log_db_error(logger, "UPSERT", "patent_drafts", str(exc))
            raise

    def get_by_topic(self, topic_id: int) -> dict | None:
        """Load the draft for a topic. Returns dict with claims_text and description_text, or None."""
        row = self._conn.execute(
            "SELECT claims_text, description_text FROM patent_drafts WHERE topic_id = ?",
            (topic_id,),
        ).fetchone()
        if row is None:
            return None
        return {"claims_text": row[0], "description_text": row[1]}


class LocalDocumentRepository:
    """CRUD operations for user-uploaded local documents."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def create(self, topic_id: int, filename: str, content: str) -> int:
        """Insert a local document and return the new row ID."""
        try:
            cursor = self._conn.execute(
                "INSERT INTO local_documents (topic_id, filename, content) VALUES (?, ?, ?)",
                (topic_id, filename, content),
            )
            self._conn.commit()
            return cursor.lastrowid  # type: ignore[return-value]
        except sqlite3.Error as exc:
            log_db_error(logger, "INSERT", "local_documents", str(exc))
            raise

    def get_by_topic(self, topic_id: int) -> list[dict]:
        """Return all local documents for a topic."""
        rows = self._conn.execute(
            "SELECT id, filename, content, uploaded_at FROM local_documents WHERE topic_id = ?",
            (topic_id,),
        ).fetchall()
        return [
            {"id": r[0], "filename": r[1], "content": r[2], "uploaded_at": r[3]}
            for r in rows
        ]

    def delete(self, doc_id: int) -> None:
        """Delete a local document by ID."""
        try:
            self._conn.execute("DELETE FROM local_documents WHERE id = ?", (doc_id,))
            self._conn.commit()
        except sqlite3.Error as exc:
            log_db_error(logger, "DELETE", "local_documents", str(exc))
            raise

    def update_embedding(self, doc_id: int, embedding: bytes) -> None:
        """Update the embedding for a local document."""
        try:
            self._conn.execute(
                "UPDATE local_documents SET embedding = ? WHERE id = ?",
                (embedding, doc_id),
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            log_db_error(logger, "UPDATE", "local_documents", str(exc))
            raise


class InventionDisclosureRepository:
    """CRUD operations for per-topic invention disclosures.

    Requirements: 2.3, 2.4, 9.1, 9.3
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def upsert(
        self,
        topic_id: int,
        primary_description: str,
        search_terms: list[str],
    ) -> int:
        """Upsert disclosure and replace all search terms atomically.

        Uses INSERT OR REPLACE on the invention_disclosures table (topic_id
        has a UNIQUE constraint). Existing search terms are deleted first,
        then new ones are inserted with sort_order preserving list order.

        Args:
            topic_id: The topic ID to associate the disclosure with.
            primary_description: The main invention description text.
            search_terms: Ordered list of additional search term strings.

        Returns:
            The disclosure row ID.

        Raises:
            sqlite3.Error: On database failure.
        """
        try:
            cursor = self._conn.cursor()
            cursor.execute("BEGIN")

            # Upsert the disclosure row. INSERT OR REPLACE works because
            # topic_id has a UNIQUE constraint.
            cursor.execute(
                """INSERT OR REPLACE INTO invention_disclosures
                   (topic_id, primary_description)
                   VALUES (?, ?)""",
                (topic_id, primary_description),
            )
            disclosure_id: int = cursor.lastrowid  # type: ignore[assignment]

            # Atomically replace all search terms.
            cursor.execute(
                "DELETE FROM disclosure_search_terms WHERE disclosure_id = ?",
                (disclosure_id,),
            )
            for sort_order, term in enumerate(search_terms):
                cursor.execute(
                    """INSERT INTO disclosure_search_terms
                       (disclosure_id, term, sort_order)
                       VALUES (?, ?, ?)""",
                    (disclosure_id, term, sort_order),
                )

            self._conn.commit()
            return disclosure_id
        except sqlite3.Error as exc:
            self._conn.rollback()
            log_db_error(logger, "UPSERT", "invention_disclosures", str(exc))
            raise

    def get_by_topic(self, topic_id: int) -> dict | None:
        """Load disclosure for a topic.

        Args:
            topic_id: The topic ID to look up.

        Returns:
            Dict with keys ``id``, ``primary_description``, and
            ``search_terms`` (ordered by sort_order), or None if no
            disclosure exists for the topic.
        """
        try:
            row = self._conn.execute(
                """SELECT id, primary_description
                   FROM invention_disclosures WHERE topic_id = ?""",
                (topic_id,),
            ).fetchone()
            if row is None:
                return None

            disclosure_id = row[0]
            term_rows = self._conn.execute(
                """SELECT term FROM disclosure_search_terms
                   WHERE disclosure_id = ? ORDER BY sort_order ASC""",
                (disclosure_id,),
            ).fetchall()

            return {
                "id": disclosure_id,
                "primary_description": row[1],
                "search_terms": [r[0] for r in term_rows],
            }
        except sqlite3.Error as exc:
            log_db_error(logger, "SELECT", "invention_disclosures", str(exc))
            raise


WORKFLOW_STEP_ORDER: list[str] = [
    "initial_idea",
    "claims_drafting",
    "prior_art_search",
    "novelty_analysis",
    "consistency_review",
    "market_potential",
    "legal_clarification",
    "disclosure_summary",
    "patent_draft",
]

VALID_STEP_KEYS: set[str] = set(WORKFLOW_STEP_ORDER)


class WorkflowStepRepository:
    """CRUD for per-topic workflow step content and status.

    Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 12a.3
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def upsert(
        self,
        topic_id: int,
        step_key: str,
        content: str,
        status: str,
        personality_mode: str = "critical",
        review_notes: str = "",
        domain_profile_slug: str = "",
    ) -> None:
        """Insert or update a workflow step.

        Validates step_key against VALID_STEP_KEYS and uses INSERT OR REPLACE
        semantics (the UNIQUE(topic_id, step_key) constraint enables this).

        Args:
            topic_id: The topic this step belongs to.
            step_key: One of the nine canonical step keys.
            content: The step's text content.
            status: Either "pending" or "completed".
            personality_mode: The personality mode active when this step ran.
                Defaults to "critical".
            review_notes: User-authored review notes for this step.
                Defaults to empty string.
            domain_profile_slug: The domain profile slug active when this step ran.
                Defaults to empty string.

        Raises:
            ValueError: If step_key is not in VALID_STEP_KEYS.
            sqlite3.Error: On database failure.
        """
        if step_key not in VALID_STEP_KEYS:
            raise ValueError(
                f"Invalid step_key {step_key!r}. "
                f"Must be one of {sorted(VALID_STEP_KEYS)}"
            )
        try:
            self._conn.execute(
                """INSERT OR REPLACE INTO workflow_steps
                   (topic_id, step_key, content, status, personality_mode,
                    review_notes, domain_profile_slug, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)""",
                (topic_id, step_key, content, status, personality_mode,
                 review_notes, domain_profile_slug),
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            log_db_error(logger, "UPSERT", "workflow_steps", str(exc))
            raise

    def get_by_topic(self, topic_id: int) -> list[dict]:
        """Return all steps for a topic, ordered by the canonical step sequence.

        Steps are sorted according to WORKFLOW_STEP_ORDER regardless of
        insertion order. Only steps that exist in the database are returned.

        Args:
            topic_id: The topic to query.

        Returns:
            List of dicts with keys: id, topic_id, step_key, content,
            status, personality_mode, review_notes, domain_profile_slug,
            updated_at — ordered by WORKFLOW_STEP_ORDER.
        """
        try:
            rows = self._conn.execute(
                """SELECT id, topic_id, step_key, content, status, updated_at,
                          personality_mode, review_notes, domain_profile_slug
                   FROM workflow_steps WHERE topic_id = ?""",
                (topic_id,),
            ).fetchall()
        except sqlite3.Error as exc:
            log_db_error(logger, "SELECT", "workflow_steps", str(exc))
            raise

        step_order = {key: idx for idx, key in enumerate(WORKFLOW_STEP_ORDER)}
        results = [
            {
                "id": r[0],
                "topic_id": r[1],
                "step_key": r[2],
                "content": r[3],
                "status": r[4],
                "updated_at": r[5],
                "personality_mode": r[6],
                "review_notes": r[7],
                "domain_profile_slug": r[8],
            }
            for r in rows
        ]
        results.sort(key=lambda d: step_order.get(d["step_key"], len(WORKFLOW_STEP_ORDER)))
        return results

    def get_step(self, topic_id: int, step_key: str) -> dict | None:
        """Return a single step record, or None if not found.

        Args:
            topic_id: The topic to query.
            step_key: The step key to look up.

        Returns:
            Dict with keys: id, topic_id, step_key, content, status,
            personality_mode, review_notes, domain_profile_slug,
            updated_at — or None.
        """
        try:
            row = self._conn.execute(
                """SELECT id, topic_id, step_key, content, status, updated_at,
                          personality_mode, review_notes, domain_profile_slug
                   FROM workflow_steps
                   WHERE topic_id = ? AND step_key = ?""",
                (topic_id, step_key),
            ).fetchone()
        except sqlite3.Error as exc:
            log_db_error(logger, "SELECT", "workflow_steps", str(exc))
            raise

        if row is None:
            return None
        return {
            "id": row[0],
            "topic_id": row[1],
            "step_key": row[2],
            "content": row[3],
            "status": row[4],
            "updated_at": row[5],
            "personality_mode": row[6],
            "review_notes": row[7],
            "domain_profile_slug": row[8],
        }

    def reset_from_step(self, topic_id: int, step_key: str) -> None:
        """Set status to 'pending' for the given step and all subsequent steps.

        Uses WORKFLOW_STEP_ORDER to determine which steps come at or after
        the specified step_key.

        Args:
            topic_id: The topic to reset.
            step_key: The step from which to reset (inclusive).

        Raises:
            ValueError: If step_key is not in VALID_STEP_KEYS.
            sqlite3.Error: On database failure.
        """
        if step_key not in VALID_STEP_KEYS:
            raise ValueError(
                f"Invalid step_key {step_key!r}. "
                f"Must be one of {sorted(VALID_STEP_KEYS)}"
            )
        idx = WORKFLOW_STEP_ORDER.index(step_key)
        keys_to_reset = WORKFLOW_STEP_ORDER[idx:]
        try:
            placeholders = ",".join("?" for _ in keys_to_reset)
            self._conn.execute(
                f"""UPDATE workflow_steps
                    SET status = 'pending', updated_at = CURRENT_TIMESTAMP
                    WHERE topic_id = ? AND step_key IN ({placeholders})""",
                [topic_id, *keys_to_reset],
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            log_db_error(logger, "UPDATE", "workflow_steps", str(exc))
            raise


class SourcePreferenceRepository:
    """CRUD operations for per-topic source selection preferences.

    Requirements: 3.4, 3.5, 9.2
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def save(self, topic_id: int, preferences: dict[str, bool]) -> None:
        """Replace all source preferences for a topic.

        Deletes existing preferences and inserts the new mapping
        within a single transaction.

        Args:
            topic_id: The topic ID to save preferences for.
            preferences: Mapping of source_name → enabled boolean.

        Raises:
            sqlite3.Error: On database failure.
        """
        try:
            cursor = self._conn.cursor()
            cursor.execute("BEGIN")

            cursor.execute(
                "DELETE FROM source_preferences WHERE topic_id = ?",
                (topic_id,),
            )
            for source_name, enabled in preferences.items():
                cursor.execute(
                    """INSERT INTO source_preferences
                       (topic_id, source_name, enabled)
                       VALUES (?, ?, ?)""",
                    (topic_id, source_name, enabled),
                )

            self._conn.commit()
        except sqlite3.Error as exc:
            self._conn.rollback()
            log_db_error(logger, "REPLACE", "source_preferences", str(exc))
            raise

    def get_by_topic(self, topic_id: int) -> dict[str, bool] | None:
        """Load source preferences for a topic.

        Args:
            topic_id: The topic ID to look up.

        Returns:
            Dict mapping source_name → enabled, or None if no
            preferences are saved (caller should default to all-enabled).
        """
        try:
            rows = self._conn.execute(
                """SELECT source_name, enabled
                   FROM source_preferences WHERE topic_id = ?""",
                (topic_id,),
            ).fetchall()
            if not rows:
                return None
            return {r[0]: bool(r[1]) for r in rows}
        except sqlite3.Error as exc:
            log_db_error(logger, "SELECT", "source_preferences", str(exc))
            raise


class PersonalityPreferenceRepository:
    """CRUD operations for per-topic agent personality preferences.

    Requirements: 9.1, 9.2, 9.3, 9.4
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def save(self, topic_id: int, preferences: dict[str, str]) -> None:
        """Replace all personality preferences for a topic atomically.

        Deletes existing preferences and inserts the new mapping
        within a single transaction (DELETE + INSERT).

        Args:
            topic_id: The topic ID to save preferences for.
            preferences: Mapping of agent_name → personality mode string.

        Raises:
            sqlite3.Error: On database failure.
        """
        try:
            cursor = self._conn.cursor()

            cursor.execute(
                "DELETE FROM personality_preferences WHERE topic_id = ?",
                (topic_id,),
            )
            for agent_name, personality_mode in preferences.items():
                cursor.execute(
                    """INSERT INTO personality_preferences
                       (topic_id, agent_name, personality_mode)
                       VALUES (?, ?, ?)""",
                    (topic_id, agent_name, personality_mode),
                )

            self._conn.commit()
        except sqlite3.Error as exc:
            self._conn.rollback()
            log_db_error(logger, "REPLACE", "personality_preferences", str(exc))
            raise

    def get_by_topic(self, topic_id: int) -> dict[str, str] | None:
        """Load personality preferences for a topic.

        Args:
            topic_id: The topic ID to look up.

        Returns:
            Dict mapping agent_name → personality mode string, or None
            if no preferences are saved for this topic.
        """
        try:
            rows = self._conn.execute(
                """SELECT agent_name, personality_mode
                   FROM personality_preferences WHERE topic_id = ?""",
                (topic_id,),
            ).fetchall()
            if not rows:
                return None
            return {r[0]: r[1] for r in rows}
        except sqlite3.Error as exc:
            log_db_error(logger, "SELECT", "personality_preferences", str(exc))
            raise


class TopicDomainProfileRepository:
    """Per-topic domain profile selection persistence.

    Only stores which profile slug is selected for each topic.
    Profile definitions live in YAML files, not in the database.

    Requirements: 6.1, 6.2, 6.3, 6.4
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def save(self, topic_id: int, slug: str) -> None:
        """Persist the domain profile selection for a topic.

        Uses INSERT OR REPLACE semantics (the UNIQUE(topic_id) constraint
        enables this).

        Args:
            topic_id: The topic ID to save the selection for.
            slug: The domain profile slug to associate with the topic.

        Raises:
            sqlite3.Error: On database failure.
        """
        try:
            self._conn.execute(
                """INSERT OR REPLACE INTO topic_domain_profile
                   (topic_id, domain_profile_slug, updated_at)
                   VALUES (?, ?, CURRENT_TIMESTAMP)""",
                (topic_id, slug),
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            log_db_error(logger, "REPLACE", "topic_domain_profile", str(exc))
            raise

    def get_by_topic(self, topic_id: int) -> str | None:
        """Load the domain profile slug for a topic.

        Args:
            topic_id: The topic ID to look up.

        Returns:
            The domain profile slug string, or None if no selection
            is saved for this topic.
        """
        try:
            row = self._conn.execute(
                """SELECT domain_profile_slug
                   FROM topic_domain_profile WHERE topic_id = ?""",
                (topic_id,),
            ).fetchone()
            if row is None:
                return None
            return row[0]
        except sqlite3.Error as exc:
            log_db_error(logger, "SELECT", "topic_domain_profile", str(exc))
            raise


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
