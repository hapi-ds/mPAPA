"""Unit tests for personality-related repository classes.

Tests PersonalityPreferenceRepository and WorkflowStepRepository
personality_mode persistence.

Requirements: 8.4, 9.1–9.4
"""

import logging
import sqlite3

import pytest

from patent_system.db.repository import (
    PersonalityPreferenceRepository,
    WorkflowStepRepository,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_topic(conn: sqlite3.Connection, name: str = "test-topic") -> int:
    """Insert a topic row, commit, and return its ID."""
    conn.execute("INSERT INTO topics (name) VALUES (?)", (name,))
    conn.commit()
    return conn.execute("SELECT id FROM topics WHERE name = ?", (name,)).fetchone()[0]


# ---------------------------------------------------------------------------
# PersonalityPreferenceRepository tests
# ---------------------------------------------------------------------------


class TestPersonalityPreferenceRepositorySaveLoad:
    """Test save and load cycle for PersonalityPreferenceRepository.

    Requirements: 9.1, 9.2
    """

    def test_save_and_load_single_agent(self, in_memory_db: sqlite3.Connection) -> None:
        """Saving a single agent preference and loading returns the same mapping."""
        topic_id = _create_topic(in_memory_db)
        repo = PersonalityPreferenceRepository(in_memory_db)

        prefs = {"novelty_analysis": "critical"}
        repo.save(topic_id, prefs)

        result = repo.get_by_topic(topic_id)
        assert result == prefs

    def test_save_and_load_multiple_agents(self, in_memory_db: sqlite3.Connection) -> None:
        """Saving multiple agent preferences and loading returns the same mapping."""
        topic_id = _create_topic(in_memory_db)
        repo = PersonalityPreferenceRepository(in_memory_db)

        prefs = {
            "novelty_analysis": "critical",
            "claims_drafting": "neutral",
            "market_potential": "innovation_friendly",
        }
        repo.save(topic_id, prefs)

        result = repo.get_by_topic(topic_id)
        assert result == prefs

    def test_save_replaces_previous_preferences(self, in_memory_db: sqlite3.Connection) -> None:
        """Saving new preferences replaces the old ones atomically."""
        topic_id = _create_topic(in_memory_db)
        repo = PersonalityPreferenceRepository(in_memory_db)

        repo.save(topic_id, {"novelty_analysis": "critical"})
        repo.save(topic_id, {"claims_drafting": "neutral"})

        result = repo.get_by_topic(topic_id)
        assert result == {"claims_drafting": "neutral"}


class TestPersonalityPreferenceRepositoryNoSavedPrefs:
    """Test that get_by_topic returns None when no preferences exist.

    Requirements: 9.3
    """

    def test_returns_none_for_topic_with_no_preferences(
        self, in_memory_db: sqlite3.Connection
    ) -> None:
        """get_by_topic returns None when no preferences have been saved."""
        topic_id = _create_topic(in_memory_db)
        repo = PersonalityPreferenceRepository(in_memory_db)

        result = repo.get_by_topic(topic_id)
        assert result is None

    def test_returns_none_for_nonexistent_topic(
        self, in_memory_db: sqlite3.Connection
    ) -> None:
        """get_by_topic returns None for a topic ID with no rows at all."""
        repo = PersonalityPreferenceRepository(in_memory_db)

        result = repo.get_by_topic(999_999)
        assert result is None


# ---------------------------------------------------------------------------
# WorkflowStepRepository personality_mode tests
# ---------------------------------------------------------------------------


class TestWorkflowStepRepositoryPersonalityMode:
    """Test WorkflowStepRepository upsert with personality_mode and retrieval.

    Requirements: 8.4
    """

    def test_upsert_with_explicit_personality_mode(
        self, in_memory_db: sqlite3.Connection
    ) -> None:
        """Upserting a step with an explicit personality_mode persists it."""
        topic_id = _create_topic(in_memory_db)
        repo = WorkflowStepRepository(in_memory_db)

        repo.upsert(topic_id, "claims_drafting", "draft content", "completed", personality_mode="neutral")

        result = repo.get_step(topic_id, "claims_drafting")
        assert result is not None
        assert result["personality_mode"] == "neutral"

    def test_upsert_default_personality_mode_is_critical(
        self, in_memory_db: sqlite3.Connection
    ) -> None:
        """Upserting a step without personality_mode defaults to 'critical'."""
        topic_id = _create_topic(in_memory_db)
        repo = WorkflowStepRepository(in_memory_db)

        repo.upsert(topic_id, "novelty_analysis", "analysis content", "completed")

        result = repo.get_step(topic_id, "novelty_analysis")
        assert result is not None
        assert result["personality_mode"] == "critical"

    def test_upsert_updates_personality_mode_on_rerun(
        self, in_memory_db: sqlite3.Connection
    ) -> None:
        """Re-upserting a step with a different mode updates the stored mode."""
        topic_id = _create_topic(in_memory_db)
        repo = WorkflowStepRepository(in_memory_db)

        repo.upsert(topic_id, "claims_drafting", "v1", "completed", personality_mode="critical")
        repo.upsert(topic_id, "claims_drafting", "v2", "completed", personality_mode="innovation_friendly")

        result = repo.get_step(topic_id, "claims_drafting")
        assert result is not None
        assert result["personality_mode"] == "innovation_friendly"
        assert result["content"] == "v2"

    def test_get_by_topic_includes_personality_mode(
        self, in_memory_db: sqlite3.Connection
    ) -> None:
        """get_by_topic returns personality_mode in each step dict."""
        topic_id = _create_topic(in_memory_db)
        repo = WorkflowStepRepository(in_memory_db)

        repo.upsert(topic_id, "claims_drafting", "c1", "completed", personality_mode="neutral")
        repo.upsert(topic_id, "novelty_analysis", "n1", "completed", personality_mode="critical")

        steps = repo.get_by_topic(topic_id)
        assert len(steps) == 2
        modes = {s["step_key"]: s["personality_mode"] for s in steps}
        assert modes["claims_drafting"] == "neutral"
        assert modes["novelty_analysis"] == "critical"


# ---------------------------------------------------------------------------
# Database error handling tests (log + re-raise pattern)
# ---------------------------------------------------------------------------


class TestRepositoryErrorHandling:
    """Test that database errors are logged and re-raised.

    Requirements: 9.4
    """

    def test_personality_pref_save_logs_and_reraises_on_db_error(
        self, in_memory_db: sqlite3.Connection, caplog: pytest.LogCaptureFixture
    ) -> None:
        """PersonalityPreferenceRepository.save logs and re-raises on DB error."""
        repo = PersonalityPreferenceRepository(in_memory_db)

        # Close the connection to force a DB error
        in_memory_db.close()

        with pytest.raises(sqlite3.Error):
            repo.save(1, {"agent": "critical"})

    def test_personality_pref_get_logs_and_reraises_on_db_error(
        self, in_memory_db: sqlite3.Connection, caplog: pytest.LogCaptureFixture
    ) -> None:
        """PersonalityPreferenceRepository.get_by_topic logs and re-raises on DB error."""
        repo = PersonalityPreferenceRepository(in_memory_db)

        in_memory_db.close()

        with pytest.raises(sqlite3.Error):
            repo.get_by_topic(1)

    def test_workflow_step_upsert_logs_and_reraises_on_db_error(
        self, in_memory_db: sqlite3.Connection, caplog: pytest.LogCaptureFixture
    ) -> None:
        """WorkflowStepRepository.upsert logs and re-raises on DB error."""
        repo = WorkflowStepRepository(in_memory_db)

        in_memory_db.close()

        with pytest.raises(sqlite3.Error):
            repo.upsert(1, "claims_drafting", "content", "completed")

    def test_workflow_step_get_step_logs_and_reraises_on_db_error(
        self, in_memory_db: sqlite3.Connection, caplog: pytest.LogCaptureFixture
    ) -> None:
        """WorkflowStepRepository.get_step logs and re-raises on DB error."""
        repo = WorkflowStepRepository(in_memory_db)

        in_memory_db.close()

        with pytest.raises(sqlite3.Error):
            repo.get_step(1, "claims_drafting")
