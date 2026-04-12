"""SQLite schema initialization and database connection factory.

Defines the CREATE TABLE statements for the patent system database and
provides a connection factory that initializes the schema on first use.
"""

import sqlite3
from pathlib import Path

SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS topics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS research_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_id INTEGER NOT NULL,
    query TEXT NOT NULL,
    search_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status TEXT DEFAULT 'pending',
    FOREIGN KEY (topic_id) REFERENCES topics(id)
);

CREATE TABLE IF NOT EXISTS patents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    patent_number TEXT NOT NULL,
    title TEXT NOT NULL,
    abstract TEXT,
    full_text TEXT,
    claims TEXT,
    pdf_path TEXT,
    source TEXT NOT NULL,
    discovered_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    embedding BLOB,
    FOREIGN KEY (session_id) REFERENCES research_sessions(id)
);

CREATE TABLE IF NOT EXISTS scientific_papers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    doi TEXT NOT NULL,
    title TEXT NOT NULL,
    abstract TEXT,
    full_text TEXT,
    pdf_path TEXT,
    source TEXT NOT NULL,
    discovered_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    embedding BLOB,
    FOREIGN KEY (session_id) REFERENCES research_sessions(id)
);

CREATE TABLE IF NOT EXISTS chat_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_id INTEGER NOT NULL,
    role TEXT NOT NULL,
    message TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (topic_id) REFERENCES topics(id)
);

CREATE TABLE IF NOT EXISTS invention_disclosures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_id INTEGER NOT NULL UNIQUE,
    primary_description TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (topic_id) REFERENCES topics(id)
);

CREATE TABLE IF NOT EXISTS disclosure_search_terms (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    disclosure_id INTEGER NOT NULL,
    term TEXT NOT NULL,
    sort_order INTEGER NOT NULL,
    FOREIGN KEY (disclosure_id) REFERENCES invention_disclosures(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS source_preferences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_id INTEGER NOT NULL,
    source_name TEXT NOT NULL,
    enabled BOOLEAN NOT NULL DEFAULT 1,
    FOREIGN KEY (topic_id) REFERENCES topics(id),
    UNIQUE(topic_id, source_name)
);
"""

_initialized_databases: set[str] = set()


def init_schema(conn: sqlite3.Connection) -> None:
    """Execute the schema SQL to create all tables.

    Uses executescript to handle multiple statements. Safe to call
    multiple times due to IF NOT EXISTS clauses.
    """
    conn.executescript(SCHEMA_SQL)


def get_connection(database_path: Path) -> sqlite3.Connection:
    """Create a SQLite connection and initialize the schema on first use.

    Args:
        database_path: Path to the SQLite database file. Parent directories
            are created automatically if they don't exist.

    Returns:
        A sqlite3.Connection with foreign keys enabled and schema initialized.
    """
    database_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(database_path), check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")

    db_key = str(database_path.resolve())
    if db_key not in _initialized_databases:
        init_schema(conn)
        _initialized_databases.add(db_key)

    return conn
