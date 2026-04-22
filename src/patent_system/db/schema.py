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

CREATE TABLE IF NOT EXISTS patent_drafts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_id INTEGER NOT NULL UNIQUE,
    claims_text TEXT NOT NULL DEFAULT '',
    description_text TEXT NOT NULL DEFAULT '',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (topic_id) REFERENCES topics(id)
);

CREATE TABLE IF NOT EXISTS local_documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_id INTEGER NOT NULL,
    filename TEXT NOT NULL,
    content TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'Local Document',
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    embedding BLOB,
    FOREIGN KEY (topic_id) REFERENCES topics(id)
);

CREATE TABLE IF NOT EXISTS workflow_steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_id INTEGER NOT NULL,
    step_key TEXT NOT NULL,
    content TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'pending',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (topic_id) REFERENCES topics(id),
    UNIQUE(topic_id, step_key)
);

CREATE TABLE IF NOT EXISTS personality_preferences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_id INTEGER NOT NULL,
    agent_name TEXT NOT NULL,
    personality_mode TEXT NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (topic_id) REFERENCES topics(id),
    UNIQUE(topic_id, agent_name)
);
"""

_initialized_databases: set[str] = set()


def _migrate_relevance_score(conn: sqlite3.Connection) -> None:
    """Add relevance_score column to patents and scientific_papers if missing."""
    for table in ("patents", "scientific_papers"):
        cursor = conn.execute(f"PRAGMA table_info({table})")
        col_names = {row[1] for row in cursor.fetchall()}
        if "relevance_score" not in col_names:
            conn.execute(
                f"ALTER TABLE {table} ADD COLUMN relevance_score REAL"
            )


def _migrate_workflow_steps_personality(conn: sqlite3.Connection) -> None:
    """Add or fix the personality_mode column on workflow_steps.

    Phase 1 (new databases): adds the column with DEFAULT '' so pre-existing
    rows are clearly distinguishable from rows created with personality support.

    Phase 2 (databases from the first migration that used DEFAULT 'critical'):
    resets those legacy default values to '' so the UI does not show a
    misleading "Critical" badge on steps that were never run with personality
    mode support.
    """
    cursor = conn.execute("PRAGMA table_info(workflow_steps)")
    col_info = {row[1]: row for row in cursor.fetchall()}

    if "personality_mode" not in col_info:
        # Column doesn't exist yet — add it with empty-string default.
        conn.execute(
            "ALTER TABLE workflow_steps ADD COLUMN personality_mode TEXT NOT NULL DEFAULT ''"
        )
    else:
        # Column exists. If the default is 'critical' (from the earlier
        # migration), clear those legacy values so badges are not shown
        # for steps that pre-date personality mode support.
        # We detect the old migration by checking the column's default value.
        default_val = col_info["personality_mode"][4]  # dflt_value is index 4
        if default_val == "'critical'":
            conn.execute(
                "UPDATE workflow_steps SET personality_mode = '' WHERE personality_mode = 'critical'"
            )


def _migrate_workflow_steps_review_notes(conn: sqlite3.Connection) -> None:
    """Add review_notes column to workflow_steps if missing.

    Adds a TEXT column with an empty-string default so existing rows
    get a sensible value and new rows without an explicit review_notes
    value are clearly empty.
    """
    cursor = conn.execute("PRAGMA table_info(workflow_steps)")
    col_names = {row[1] for row in cursor.fetchall()}
    if "review_notes" not in col_names:
        conn.execute(
            "ALTER TABLE workflow_steps ADD COLUMN review_notes TEXT NOT NULL DEFAULT ''"
        )


def init_schema(conn: sqlite3.Connection) -> None:
    """Execute the schema SQL to create all tables and run migrations.

    Uses executescript to handle multiple statements. Safe to call
    multiple times due to IF NOT EXISTS clauses. After creating tables,
    runs migration logic to add columns that may be missing in existing
    databases.
    """
    conn.executescript(SCHEMA_SQL)
    _migrate_relevance_score(conn)
    _migrate_workflow_steps_personality(conn)
    _migrate_workflow_steps_review_notes(conn)


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
