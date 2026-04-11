"""Shared test fixtures for the Patent Analysis & Drafting System."""

import sqlite3
from collections.abc import Generator

import pytest

from patent_system.db.schema import init_schema


@pytest.fixture
def in_memory_db() -> Generator[sqlite3.Connection, None, None]:
    """Provide a fresh in-memory SQLite connection with FK enforcement and full schema."""
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    init_schema(conn)
    yield conn
    conn.close()
