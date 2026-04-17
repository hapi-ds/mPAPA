"""Integration tests for personality settings panel save/load cycle.

Tests the persistence layer used by the Settings panel: saving personality
preferences to the database and loading them back, verifying round-trip
correctness and default population.

Requirements: 7.3, 7.4, 7.6, 7.7
"""

from __future__ import annotations

import sqlite3

import pytest

from patent_system.agents.personality import AGENT_PERSONALITY_DEFAULTS, PersonalityMode
from patent_system.db.repository import PersonalityPreferenceRepository, TopicRepository


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_topic(conn: sqlite3.Connection, name: str = "Test Topic") -> int:
    """Create a topic and return its ID."""
    repo = TopicRepository(conn)
    topic = repo.create(name)
    return topic.id


# ---------------------------------------------------------------------------
# Test: Save persists to DB and load restores correctly
# ---------------------------------------------------------------------------


class TestSaveLoadCycle:
    """Test that saving preferences via PersonalityPreferenceRepository
    persists to the database and loading restores them correctly."""

    def test_save_and_load_round_trip(self, in_memory_db: sqlite3.Connection):
        """Save a full set of preferences and verify load returns them."""
        topic_id = _create_topic(in_memory_db)
        repo = PersonalityPreferenceRepository(in_memory_db)

        preferences = {
            "claims_drafting": "innovation_friendly",
            "novelty_analysis": "neutral",
            "consistency_review": "critical",
            "market_potential": "innovation_friendly",
            "legal_clarification": "neutral",
            "disclosure_summary": "critical",
            "patent_draft": "neutral",
        }

        repo.save(topic_id, preferences)
        loaded = repo.get_by_topic(topic_id)

        assert loaded is not None
        assert loaded == preferences

    def test_save_overwrites_previous_preferences(self, in_memory_db: sqlite3.Connection):
        """Saving new preferences replaces the old ones entirely."""
        topic_id = _create_topic(in_memory_db)
        repo = PersonalityPreferenceRepository(in_memory_db)

        # First save
        repo.save(topic_id, {
            "claims_drafting": "critical",
            "novelty_analysis": "critical",
        })

        # Second save with different values
        new_prefs = {
            "claims_drafting": "innovation_friendly",
            "novelty_analysis": "neutral",
            "consistency_review": "innovation_friendly",
        }
        repo.save(topic_id, new_prefs)

        loaded = repo.get_by_topic(topic_id)
        assert loaded == new_prefs

    def test_save_all_three_modes(self, in_memory_db: sqlite3.Connection):
        """Each of the three personality modes can be saved and loaded."""
        topic_id = _create_topic(in_memory_db)
        repo = PersonalityPreferenceRepository(in_memory_db)

        preferences = {
            "claims_drafting": PersonalityMode.CRITICAL.value,
            "novelty_analysis": PersonalityMode.NEUTRAL.value,
            "consistency_review": PersonalityMode.INNOVATION_FRIENDLY.value,
        }

        repo.save(topic_id, preferences)
        loaded = repo.get_by_topic(topic_id)

        assert loaded is not None
        assert loaded["claims_drafting"] == "critical"
        assert loaded["novelty_analysis"] == "neutral"
        assert loaded["consistency_review"] == "innovation_friendly"


# ---------------------------------------------------------------------------
# Test: Default population when no saved preferences exist
# ---------------------------------------------------------------------------


class TestDefaultPopulation:
    """Test that when no preferences are saved, the system falls back
    to AGENT_PERSONALITY_DEFAULTS."""

    def test_get_by_topic_returns_none_when_empty(self, in_memory_db: sqlite3.Connection):
        """get_by_topic returns None when no preferences exist for a topic."""
        topic_id = _create_topic(in_memory_db)
        repo = PersonalityPreferenceRepository(in_memory_db)

        result = repo.get_by_topic(topic_id)
        assert result is None

    def test_defaults_used_when_no_saved_preferences(self, in_memory_db: sqlite3.Connection):
        """When get_by_topic returns None, AGENT_PERSONALITY_DEFAULTS should be used."""
        topic_id = _create_topic(in_memory_db)
        repo = PersonalityPreferenceRepository(in_memory_db)

        loaded = repo.get_by_topic(topic_id)
        assert loaded is None

        # Simulate what the settings panel / workflow does: fall back to defaults
        effective = {
            agent: mode.value
            for agent, mode in AGENT_PERSONALITY_DEFAULTS.items()
        }

        # Verify defaults are well-formed
        assert effective["novelty_analysis"] == "critical"
        assert effective["claims_drafting"] == "neutral"
        assert effective["consistency_review"] == "critical"
        assert effective["disclosure_summary"] == "neutral"

    def test_nonexistent_topic_returns_none(self, in_memory_db: sqlite3.Connection):
        """get_by_topic returns None for a topic ID that has no preferences."""
        repo = PersonalityPreferenceRepository(in_memory_db)
        # Topic ID 9999 doesn't exist
        result = repo.get_by_topic(9999)
        assert result is None


# ---------------------------------------------------------------------------
# Test: Changing mode persists correctly
# ---------------------------------------------------------------------------


class TestChangingModePersists:
    """Test that changing a personality mode and saving persists the new value."""

    def test_change_single_agent_mode(self, in_memory_db: sqlite3.Connection):
        """Change one agent's mode and verify it persists."""
        topic_id = _create_topic(in_memory_db)
        repo = PersonalityPreferenceRepository(in_memory_db)

        # Initial save
        initial = {
            "claims_drafting": "neutral",
            "novelty_analysis": "critical",
        }
        repo.save(topic_id, initial)

        # Change claims_drafting to innovation_friendly
        updated = dict(initial)
        updated["claims_drafting"] = "innovation_friendly"
        repo.save(topic_id, updated)

        loaded = repo.get_by_topic(topic_id)
        assert loaded is not None
        assert loaded["claims_drafting"] == "innovation_friendly"
        assert loaded["novelty_analysis"] == "critical"

    def test_change_all_modes_to_same_value(self, in_memory_db: sqlite3.Connection):
        """Set all agents to the same mode and verify persistence."""
        topic_id = _create_topic(in_memory_db)
        repo = PersonalityPreferenceRepository(in_memory_db)

        all_innovation = {
            agent: "innovation_friendly"
            for agent in AGENT_PERSONALITY_DEFAULTS
        }
        repo.save(topic_id, all_innovation)

        loaded = repo.get_by_topic(topic_id)
        assert loaded is not None
        for agent in AGENT_PERSONALITY_DEFAULTS:
            assert loaded[agent] == "innovation_friendly"

    def test_preferences_isolated_per_topic(self, in_memory_db: sqlite3.Connection):
        """Preferences for different topics are independent."""
        topic1_id = _create_topic(in_memory_db, "Topic 1")
        topic2_id = _create_topic(in_memory_db, "Topic 2")
        repo = PersonalityPreferenceRepository(in_memory_db)

        repo.save(topic1_id, {"claims_drafting": "critical"})
        repo.save(topic2_id, {"claims_drafting": "innovation_friendly"})

        loaded1 = repo.get_by_topic(topic1_id)
        loaded2 = repo.get_by_topic(topic2_id)

        assert loaded1 is not None
        assert loaded2 is not None
        assert loaded1["claims_drafting"] == "critical"
        assert loaded2["claims_drafting"] == "innovation_friendly"
