"""Unit tests for SQLite schema initialization and connection factory."""

import sqlite3
from pathlib import Path

import pytest

from patent_system.db.schema import SCHEMA_SQL, get_connection, init_schema, _initialized_databases


EXPECTED_TABLES = {
    "topics",
    "research_sessions",
    "patents",
    "scientific_papers",
    "chat_history",
    "invention_disclosures",
    "disclosure_search_terms",
    "source_preferences",
}


class TestInitSchema:
    """Tests for init_schema function."""

    def test_creates_all_tables(self, in_memory_db: sqlite3.Connection) -> None:
        """All five tables should exist after schema init."""
        cursor = in_memory_db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        tables = {row[0] for row in cursor.fetchall()}
        assert tables == EXPECTED_TABLES

    def test_idempotent(self, in_memory_db: sqlite3.Connection) -> None:
        """Calling init_schema twice should not raise."""
        init_schema(in_memory_db)
        cursor = in_memory_db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        tables = {row[0] for row in cursor.fetchall()}
        assert tables == EXPECTED_TABLES

    def test_foreign_keys_enforced(self, in_memory_db: sqlite3.Connection) -> None:
        """Inserting a research_session with invalid topic_id should raise IntegrityError."""
        with pytest.raises(sqlite3.IntegrityError):
            in_memory_db.execute(
                "INSERT INTO research_sessions (topic_id, query) VALUES (999, 'test')"
            )

    def test_topics_unique_name(self, in_memory_db: sqlite3.Connection) -> None:
        """Duplicate topic names should raise IntegrityError."""
        in_memory_db.execute("INSERT INTO topics (name) VALUES ('my-topic')")
        with pytest.raises(sqlite3.IntegrityError):
            in_memory_db.execute("INSERT INTO topics (name) VALUES ('my-topic')")


class TestGetConnection:
    """Tests for get_connection factory function."""

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        """get_connection should create parent directories."""
        db_path = tmp_path / "sub" / "dir" / "test.db"
        _initialized_databases.discard(str(db_path.resolve()))
        conn = get_connection(db_path)
        try:
            assert db_path.exists()
            assert db_path.parent.is_dir()
        finally:
            conn.close()

    def test_schema_initialized(self, tmp_path: Path) -> None:
        """get_connection should initialize the schema."""
        db_path = tmp_path / "test.db"
        _initialized_databases.discard(str(db_path.resolve()))
        conn = get_connection(db_path)
        try:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
            tables = {row[0] for row in cursor.fetchall()}
            assert tables == EXPECTED_TABLES
        finally:
            conn.close()

    def test_foreign_keys_enabled(self, tmp_path: Path) -> None:
        """get_connection should enable foreign key enforcement."""
        db_path = tmp_path / "test.db"
        _initialized_databases.discard(str(db_path.resolve()))
        conn = get_connection(db_path)
        try:
            result = conn.execute("PRAGMA foreign_keys").fetchone()
            assert result is not None
            assert result[0] == 1
        finally:
            conn.close()
