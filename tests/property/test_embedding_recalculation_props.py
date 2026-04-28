"""Property-based tests for embedding recalculation bug condition.

Validates: Requirements 1.1, 1.4, 2.1

Bug Condition: isBugCondition(input) where
  input.current_model ≠ input.model_used_for_embeddings
  OR input.current_disclosure ≠ input.disclosure_used_for_scores

This test is written BEFORE the fix is implemented. It is EXPECTED TO FAIL
on unfixed code because no `_bg_recalculate` function exists yet. Failure
confirms the bug exists: there is no bulk recalculation mechanism.
"""

import os
import sqlite3
import struct
from unittest.mock import patch

from hypothesis import given, settings
from hypothesis import strategies as st

from patent_system.db.schema import init_schema

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Non-empty text for titles and abstracts (at least one non-whitespace char)
_nonempty_text = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
    min_size=3,
    max_size=100,
).filter(lambda s: s.strip())

# Optional full text
_optional_full_text = st.one_of(st.none(), _nonempty_text)

# A single patent record strategy
_patent_record = st.fixed_dictionaries({
    "title": _nonempty_text,
    "abstract": _nonempty_text,
    "full_text": _optional_full_text,
    "patent_number": st.from_regex(r"US[0-9]{7}", fullmatch=True),
})

# A single scientific paper record strategy
_paper_record = st.fixed_dictionaries({
    "title": _nonempty_text,
    "abstract": _nonempty_text,
    "full_text": _optional_full_text,
    "doi": st.from_regex(r"10\.[0-9]{4}/[a-z]{4}", fullmatch=True),
})

# A single local document record strategy
_local_doc_record = st.fixed_dictionaries({
    "title": _nonempty_text,
    "abstract": _nonempty_text,
    "full_text": _optional_full_text,
})

# Generate a set of records (at least 1 of each type)
_record_sets = st.fixed_dictionaries({
    "patents": st.lists(_patent_record, min_size=1, max_size=5),
    "papers": st.lists(_paper_record, min_size=1, max_size=5),
    "local_docs": st.lists(_local_doc_record, min_size=1, max_size=5),
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_fake_embedding(dim: int = 128) -> bytes:
    """Create a fake embedding BLOB of given dimensionality (old model)."""
    vector = [0.1] * dim
    return struct.pack(f"{dim}f", *vector)


def _create_new_fake_embedding(dim: int = 256) -> bytes:
    """Create a fake embedding BLOB representing the NEW model output.

    Uses different dimensionality (256) and different values (0.5) to
    clearly distinguish from old embeddings (128-dim, 0.1 values).
    """
    vector = [0.5] * dim
    return struct.pack(f"{dim}f", *vector)


def _setup_db_with_records(records: dict) -> tuple[sqlite3.Connection, int, int]:
    """Set up an in-memory SQLite DB with records that have old embedding BLOBs.

    Returns:
        (connection, topic_id, session_id)
    """
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    init_schema(conn)

    # Create a topic
    conn.execute("INSERT INTO topics (name) VALUES (?)", ("Test Topic",))
    topic_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    # Create a research session
    conn.execute(
        "INSERT INTO research_sessions (topic_id, query, status) VALUES (?, ?, ?)",
        (topic_id, "test query", "completed"),
    )
    session_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    # Insert patents with old embeddings
    old_embedding = _create_fake_embedding(128)
    for patent in records["patents"]:
        conn.execute(
            "INSERT INTO patents (session_id, patent_number, title, abstract, full_text, source, embedding) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                session_id,
                patent["patent_number"],
                patent["title"],
                patent["abstract"],
                patent.get("full_text"),
                "Google Patents",
                old_embedding,
            ),
        )

    # Insert scientific papers with old embeddings
    for paper in records["papers"]:
        conn.execute(
            "INSERT INTO scientific_papers (session_id, doi, title, abstract, full_text, source, embedding) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                session_id,
                paper["doi"],
                paper["title"],
                paper["abstract"],
                paper.get("full_text"),
                "ArXiv",
                old_embedding,
            ),
        )

    # Insert local documents with old embeddings
    for doc in records["local_docs"]:
        conn.execute(
            "INSERT INTO local_documents (topic_id, filename, content, source, embedding) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                topic_id,
                doc["title"],
                doc["abstract"],
                "Local Document",
                old_embedding,
            ),
        )

    conn.commit()
    return conn, topic_id, session_id


# ---------------------------------------------------------------------------
# Property 1: Bug Condition — No Bulk Recalculation Mechanism Exists
# ---------------------------------------------------------------------------


class TestEmbeddingRecalculationBugCondition:
    """Property 1: Bug Condition - Embedding Regeneration Completeness.

    For any set of records in the database where the bug condition holds
    (model changed or disclosure changed), a _bg_recalculate function
    SHALL exist and, when called, SHALL regenerate ALL embedding BLOBs
    for every record using EmbeddingService.generate_embedding() with
    the currently configured model.

    After recalculation:
    (a) every record has a non-null embedding
    (b) all embeddings have consistent byte length (same dimensionality)
    (c) embeddings differ from the old values (proving regeneration occurred)

    **Validates: Requirements 1.1, 1.4, 2.1**
    """

    @given(records=_record_sets)
    @settings(max_examples=200, deadline=None)
    def test_bg_recalculate_regenerates_all_embeddings(
        self,
        records: dict,
    ) -> None:
        """Bug condition: _bg_recalculate must exist and regenerate all embeddings."""
        # Import the recalculation function — this will fail on unfixed code
        # because no such function exists yet, proving the bug.
        from patent_system.gui.research_panel import _bg_recalculate  # type: ignore[attr-defined]

        # Set up DB with records that have old embeddings (simulating stale state)
        conn, topic_id, session_id = _setup_db_with_records(records)
        old_embedding = _create_fake_embedding(128)
        new_embedding = _create_new_fake_embedding(256)

        try:
            # Mock EmbeddingService.generate_embedding to return a new, different
            # embedding without requiring LM Studio to be running. This tests the
            # LOGIC of _bg_recalculate, not the actual LM Studio connection.
            with patch(
                "patent_system.rag.embeddings.EmbeddingService.generate_embedding",
                return_value=new_embedding,
            ):
                # Call the recalculation function
                _bg_recalculate(conn=conn, topic_id=topic_id)

            # Verify (a): every record has a non-null embedding after recalculation
            patents = conn.execute(
                "SELECT embedding FROM patents WHERE session_id = ?", (session_id,)
            ).fetchall()
            for row in patents:
                assert row[0] is not None, "Patent embedding is None after recalculation"

            papers = conn.execute(
                "SELECT embedding FROM scientific_papers WHERE session_id = ?",
                (session_id,),
            ).fetchall()
            for row in papers:
                assert row[0] is not None, "Paper embedding is None after recalculation"

            local_docs = conn.execute(
                "SELECT embedding FROM local_documents WHERE topic_id = ?",
                (topic_id,),
            ).fetchall()
            for row in local_docs:
                assert row[0] is not None, "Local doc embedding is None after recalculation"

            # Verify (b): all embeddings have consistent byte length (same model = same dim)
            all_embeddings = [row[0] for row in patents + papers + local_docs]
            embedding_lengths = {len(emb) for emb in all_embeddings}
            assert len(embedding_lengths) == 1, (
                f"Inconsistent embedding dimensions after recalculation: {embedding_lengths}"
            )

            # Verify (c): embeddings differ from old values (proving regeneration)
            for emb in all_embeddings:
                assert emb != old_embedding, (
                    "Embedding unchanged after recalculation — regeneration did not occur"
                )
        finally:
            conn.close()
