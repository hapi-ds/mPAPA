"""Unit tests for review notes schema migration and repository changes.

Requirements: 1.1, 1.2, 1.3, 1.5
"""

import sqlite3

from patent_system.db.repository import WorkflowStepRepository
from patent_system.db.schema import _migrate_workflow_steps_review_notes, init_schema


def _create_topic(conn: sqlite3.Connection, name: str = "test-topic") -> int:
    """Insert a topic and return its ID."""
    cursor = conn.execute("INSERT INTO topics (name) VALUES (?)", (name,))
    conn.commit()
    return cursor.lastrowid  # type: ignore[return-value]


class TestReviewNotesMigration:
    """Tests for _migrate_workflow_steps_review_notes migration."""

    def test_migration_adds_review_notes_column(self, in_memory_db: sqlite3.Connection) -> None:
        """The migration should ensure the review_notes column exists."""
        cursor = in_memory_db.execute("PRAGMA table_info(workflow_steps)")
        col_names = {row[1] for row in cursor.fetchall()}
        assert "review_notes" in col_names

    def test_migration_is_idempotent(self, in_memory_db: sqlite3.Connection) -> None:
        """Running the migration twice should not raise an error."""
        # init_schema already ran the migration; run it again
        _migrate_workflow_steps_review_notes(in_memory_db)
        cursor = in_memory_db.execute("PRAGMA table_info(workflow_steps)")
        col_names = {row[1] for row in cursor.fetchall()}
        assert "review_notes" in col_names

    def test_migration_on_fresh_schema_without_column(self) -> None:
        """Migration adds the column when it is missing from an older schema."""
        conn = sqlite3.connect(":memory:")
        conn.execute("PRAGMA foreign_keys = ON")
        # Create the table WITHOUT review_notes (simulating old schema)
        conn.execute(
            """CREATE TABLE IF NOT EXISTS workflow_steps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic_id INTEGER NOT NULL,
                step_key TEXT NOT NULL,
                content TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'pending',
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                personality_mode TEXT NOT NULL DEFAULT '',
                UNIQUE(topic_id, step_key)
            )"""
        )
        # Verify column is missing
        cursor = conn.execute("PRAGMA table_info(workflow_steps)")
        col_names = {row[1] for row in cursor.fetchall()}
        assert "review_notes" not in col_names

        # Run migration
        _migrate_workflow_steps_review_notes(conn)

        # Verify column was added
        cursor = conn.execute("PRAGMA table_info(workflow_steps)")
        col_names = {row[1] for row in cursor.fetchall()}
        assert "review_notes" in col_names
        conn.close()


class TestUpsertWithReviewNotes:
    """Tests for WorkflowStepRepository.upsert with review_notes parameter."""

    def test_upsert_persists_review_notes(self, in_memory_db: sqlite3.Connection) -> None:
        """Upsert with review_notes should persist the value."""
        topic_id = _create_topic(in_memory_db)
        repo = WorkflowStepRepository(in_memory_db)

        repo.upsert(
            topic_id=topic_id,
            step_key="claims_drafting",
            content="Some claims content",
            status="completed",
            review_notes="Please revise claim 3",
        )

        result = repo.get_step(topic_id, "claims_drafting")
        assert result is not None
        assert result["review_notes"] == "Please revise claim 3"

    def test_upsert_without_review_notes_defaults_to_empty(
        self, in_memory_db: sqlite3.Connection
    ) -> None:
        """Upsert without explicit review_notes should default to empty string."""
        topic_id = _create_topic(in_memory_db)
        repo = WorkflowStepRepository(in_memory_db)

        repo.upsert(
            topic_id=topic_id,
            step_key="novelty_analysis",
            content="Analysis content",
            status="completed",
        )

        result = repo.get_step(topic_id, "novelty_analysis")
        assert result is not None
        assert result["review_notes"] == ""

    def test_upsert_updates_review_notes_on_replace(
        self, in_memory_db: sqlite3.Connection
    ) -> None:
        """Upserting the same step again should update review_notes."""
        topic_id = _create_topic(in_memory_db)
        repo = WorkflowStepRepository(in_memory_db)

        repo.upsert(
            topic_id=topic_id,
            step_key="claims_drafting",
            content="v1",
            status="completed",
            review_notes="first note",
        )
        repo.upsert(
            topic_id=topic_id,
            step_key="claims_drafting",
            content="v2",
            status="completed",
            review_notes="updated note",
        )

        result = repo.get_step(topic_id, "claims_drafting")
        assert result is not None
        assert result["review_notes"] == "updated note"


class TestGetByTopicReturnsReviewNotes:
    """Tests for WorkflowStepRepository.get_by_topic returning review_notes."""

    def test_get_by_topic_includes_review_notes(
        self, in_memory_db: sqlite3.Connection
    ) -> None:
        """get_by_topic should include review_notes in each result dict."""
        topic_id = _create_topic(in_memory_db)
        repo = WorkflowStepRepository(in_memory_db)

        repo.upsert(
            topic_id=topic_id,
            step_key="claims_drafting",
            content="claims",
            status="completed",
            review_notes="note on claims",
        )
        repo.upsert(
            topic_id=topic_id,
            step_key="novelty_analysis",
            content="novelty",
            status="completed",
            review_notes="note on novelty",
        )

        results = repo.get_by_topic(topic_id)
        assert len(results) == 2

        by_key = {r["step_key"]: r for r in results}
        assert by_key["claims_drafting"]["review_notes"] == "note on claims"
        assert by_key["novelty_analysis"]["review_notes"] == "note on novelty"

    def test_get_by_topic_empty_review_notes(
        self, in_memory_db: sqlite3.Connection
    ) -> None:
        """get_by_topic should return empty string for steps without review_notes."""
        topic_id = _create_topic(in_memory_db)
        repo = WorkflowStepRepository(in_memory_db)

        repo.upsert(
            topic_id=topic_id,
            step_key="prior_art_search",
            content="prior art",
            status="completed",
        )

        results = repo.get_by_topic(topic_id)
        assert len(results) == 1
        assert results[0]["review_notes"] == ""


class TestGetStepReturnsReviewNotes:
    """Tests for WorkflowStepRepository.get_step returning review_notes."""

    def test_get_step_includes_review_notes(
        self, in_memory_db: sqlite3.Connection
    ) -> None:
        """get_step should include review_notes in the result dict."""
        topic_id = _create_topic(in_memory_db)
        repo = WorkflowStepRepository(in_memory_db)

        repo.upsert(
            topic_id=topic_id,
            step_key="market_potential",
            content="market analysis",
            status="completed",
            review_notes="expand on market size",
        )

        result = repo.get_step(topic_id, "market_potential")
        assert result is not None
        assert result["review_notes"] == "expand on market size"

    def test_get_step_returns_none_for_missing_step(
        self, in_memory_db: sqlite3.Connection
    ) -> None:
        """get_step should return None when the step doesn't exist."""
        topic_id = _create_topic(in_memory_db)
        repo = WorkflowStepRepository(in_memory_db)

        result = repo.get_step(topic_id, "claims_drafting")
        assert result is None


class TestDefaultReviewNotesValue:
    """Tests for default empty string value on rows without review_notes."""

    def test_default_value_is_empty_string(
        self, in_memory_db: sqlite3.Connection
    ) -> None:
        """Rows inserted without review_notes should have empty string default."""
        topic_id = _create_topic(in_memory_db)

        # Insert directly via SQL without review_notes column
        in_memory_db.execute(
            """INSERT INTO workflow_steps
               (topic_id, step_key, content, status, personality_mode)
               VALUES (?, ?, ?, ?, ?)""",
            (topic_id, "disclosure_summary", "summary content", "completed", "critical"),
        )
        in_memory_db.commit()

        repo = WorkflowStepRepository(in_memory_db)
        result = repo.get_step(topic_id, "disclosure_summary")
        assert result is not None
        assert result["review_notes"] == ""
