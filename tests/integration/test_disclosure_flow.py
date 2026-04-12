"""Integration tests for end-to-end disclosure data flow.

Tests that disclosure data saved in the research panel layer flows
correctly into the draft panel and chat panel layers, and that schema
migration creates new tables alongside existing ones.

Requirements: 7.1, 7.2, 8.1, 8.2
"""

from __future__ import annotations

import sqlite3

import pytest

from patent_system.db.repository import InventionDisclosureRepository
from patent_system.db.schema import init_schema
from patent_system.gui.chat_panel import build_chat_prompt


class TestDisclosureToDraftFlow:
    """Test: save disclosure → load in draft panel → verify data flows correctly.

    Validates Requirements 7.1, 7.2
    """

    def test_disclosure_flows_as_draft_invention_dict(self, in_memory_db: sqlite3.Connection) -> None:
        """Saved disclosure data maps correctly to the invention_disclosure dict
        that create_draft_panel's _on_generate handler would build."""
        # Arrange: create a topic
        in_memory_db.execute("INSERT INTO topics (name) VALUES (?)", ("Test Topic",))
        in_memory_db.commit()
        topic_id = in_memory_db.execute("SELECT id FROM topics").fetchone()[0]

        repo = InventionDisclosureRepository(in_memory_db)
        description = "A novel method for quantum error correction"
        terms = ["surface codes", "logical qubits", "fault tolerance"]

        # Act: save disclosure (as research panel would)
        repo.upsert(topic_id, description, terms)

        # Load disclosure (as draft panel's _on_generate would)
        saved = repo.get_by_topic(topic_id)
        assert saved is not None

        # Build the invention_disclosure dict the same way draft_panel does
        disclosure_dict = {
            "technical_problem": saved["primary_description"],
            "novel_features": saved.get("search_terms", []),
            "implementation_details": "",
            "potential_variations": [],
        }

        # Assert: data matches what was entered in research panel
        assert disclosure_dict["technical_problem"] == description
        assert disclosure_dict["novel_features"] == terms

    def test_disclosure_with_empty_terms(self, in_memory_db: sqlite3.Connection) -> None:
        """Disclosure with no search terms still flows correctly to draft."""
        in_memory_db.execute("INSERT INTO topics (name) VALUES (?)", ("Empty Terms",))
        in_memory_db.commit()
        topic_id = in_memory_db.execute("SELECT id FROM topics").fetchone()[0]

        repo = InventionDisclosureRepository(in_memory_db)
        repo.upsert(topic_id, "Some invention", [])

        saved = repo.get_by_topic(topic_id)
        assert saved is not None
        assert saved["primary_description"] == "Some invention"
        assert saved["search_terms"] == []

    def test_no_disclosure_returns_none(self, in_memory_db: sqlite3.Connection) -> None:
        """When no disclosure exists, get_by_topic returns None (draft panel
        should show warning per Req 7.4)."""
        in_memory_db.execute("INSERT INTO topics (name) VALUES (?)", ("No Disclosure",))
        in_memory_db.commit()
        topic_id = in_memory_db.execute("SELECT id FROM topics").fetchone()[0]

        repo = InventionDisclosureRepository(in_memory_db)
        assert repo.get_by_topic(topic_id) is None


class TestDisclosureToChatFlow:
    """Test: save disclosure → open chat → verify prompt includes invention context.

    Validates Requirements 8.1, 8.2
    """

    def test_chat_prompt_includes_disclosure_context(self, in_memory_db: sqlite3.Connection) -> None:
        """Saved disclosure data appears in the chat prompt built by
        build_chat_prompt when passed as invention_context."""
        # Arrange: create topic and save disclosure
        in_memory_db.execute("INSERT INTO topics (name) VALUES (?)", ("Chat Topic",))
        in_memory_db.commit()
        topic_id = in_memory_db.execute("SELECT id FROM topics").fetchone()[0]

        repo = InventionDisclosureRepository(in_memory_db)
        description = "An improved battery cathode using solid-state electrolytes"
        terms = ["lithium-ion", "solid electrolyte", "dendrite prevention"]
        repo.upsert(topic_id, description, terms)

        # Load disclosure (as chat panel initialization would)
        saved = repo.get_by_topic(topic_id)
        assert saved is not None
        invention_context = {
            "primary_description": saved["primary_description"],
            "search_terms": saved.get("search_terms", []),
        }

        # Act: build chat prompt with invention context
        prompt = build_chat_prompt(
            context_docs=[{"text": "Some prior art document"}],
            question="How does my invention compare?",
            invention_context=invention_context,
        )

        # Assert: prompt contains the description and all search terms
        assert description in prompt
        for term in terms:
            assert term in prompt

    def test_chat_prompt_without_disclosure(self) -> None:
        """When no disclosure exists, build_chat_prompt works without
        invention context (Req 8.3)."""
        prompt = build_chat_prompt(
            context_docs=[{"text": "Some document"}],
            question="What is prior art?",
            invention_context=None,
        )

        assert "What is prior art?" in prompt
        assert "Invention Description" not in prompt


class TestSchemaMigration:
    """Test: schema migration creates new tables alongside existing ones.

    Validates that init_schema creates the invention_disclosures,
    disclosure_search_terms, and source_preferences tables alongside
    the pre-existing tables (topics, research_sessions, patents, etc.).
    """

    def test_new_tables_created_alongside_existing(self) -> None:
        """init_schema creates all expected tables in a fresh database."""
        conn = sqlite3.connect(":memory:")
        conn.execute("PRAGMA foreign_keys = ON")
        init_schema(conn)

        # Query all table names
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        table_names = {r[0] for r in rows}

        # Existing tables
        assert "topics" in table_names
        assert "research_sessions" in table_names
        assert "patents" in table_names
        assert "scientific_papers" in table_names
        assert "chat_history" in table_names

        # New tables from enhanced-research-workflow
        assert "invention_disclosures" in table_names
        assert "disclosure_search_terms" in table_names
        assert "source_preferences" in table_names

        conn.close()

    def test_schema_idempotent(self) -> None:
        """Running init_schema twice does not raise errors."""
        conn = sqlite3.connect(":memory:")
        conn.execute("PRAGMA foreign_keys = ON")
        init_schema(conn)
        init_schema(conn)  # second call should be safe

        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        assert len(rows) >= 8  # at least the known tables
        conn.close()

    def test_existing_data_preserved_after_reinit(self, in_memory_db: sqlite3.Connection) -> None:
        """Re-running init_schema does not destroy existing data."""
        # Insert a topic
        in_memory_db.execute("INSERT INTO topics (name) VALUES (?)", ("Preserved",))
        in_memory_db.commit()

        # Re-init schema
        init_schema(in_memory_db)

        # Data should still be there
        row = in_memory_db.execute("SELECT name FROM topics").fetchone()
        assert row is not None
        assert row[0] == "Preserved"
