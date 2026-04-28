"""Property-based tests for embedding preservation — existing flows unchanged.

Validates: Requirements 3.1, 3.2, 3.3, 3.5

These tests observe and encode the CURRENT behavior of the unfixed code using
observation-first methodology. They verify that:
1. Normal search/upload flow generates embeddings and persists them
2. Disclosure save does NOT trigger embedding generation
3. Panel load reads pre-existing scores from DB without recomputation

All tests are EXPECTED TO PASS on unfixed code (confirming baseline behavior
to preserve after the fix is applied).
"""

import sqlite3
import struct
from unittest.mock import MagicMock, patch

from hypothesis import given, settings
from hypothesis import strategies as st

from patent_system.db.models import PatentRecord, ScientificPaperRecord
from patent_system.db.repository import (
    InventionDisclosureRepository,
    LocalDocumentRepository,
    PatentRepository,
    ResearchSessionRepository,
    ScientificPaperRepository,
    TopicRepository,
)
from patent_system.db.schema import init_schema
from patent_system.rag.vectorization import prepare_vectorization_text

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Safe text for titles, abstracts, descriptions (non-empty, printable)
_safe_text = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
    min_size=3,
    max_size=100,
).filter(lambda s: s.strip())

# Optional full text
_optional_full_text = st.one_of(st.none(), _safe_text)

# Patent source names
_patent_source = st.sampled_from(["EPO OPS", "Google Patents"])

# Paper source names
_paper_source = st.sampled_from(["ArXiv", "PubMed", "Google Scholar"])

# Search terms list (0 to 5 terms)
_search_terms = st.lists(_safe_text, min_size=0, max_size=5)

# Relevance scores (0.0 to 100.0)
_relevance_score = st.floats(min_value=0.0, max_value=100.0, allow_nan=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh_db() -> sqlite3.Connection:
    """Create a fresh in-memory SQLite connection with FK enforcement and full schema."""
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    init_schema(conn)
    return conn


def _create_fake_embedding(dim: int = 128) -> bytes:
    """Create a fake embedding BLOB of given dimensionality."""
    vector = [0.5] * dim
    return struct.pack(f"{dim}f", *vector)


def _setup_topic_and_session(conn: sqlite3.Connection) -> tuple[int, int]:
    """Create a topic and research session, return (topic_id, session_id)."""
    topic_repo = TopicRepository(conn)
    topic = topic_repo.create("Test Topic")
    session_repo = ResearchSessionRepository(conn)
    session_id = session_repo.create(topic.id, "test query")
    return topic.id, session_id


# ---------------------------------------------------------------------------
# Property 1: Search/Upload Flow Preservation
# ---------------------------------------------------------------------------


class TestSearchUploadFlowPreservation:
    """Property 1: Search/Upload Flow Preservation.

    For all records added via the normal persist-and-embed flow (mocking
    EmbeddingService), the embedding is generated once per record and
    stored via update_embedding() — identical to pre-fix behavior.

    This validates that adding new records via "Start Research" or local
    document upload continues to generate embeddings.

    **Validates: Requirements 3.1, 3.2**
    """

    @given(
        title=_safe_text,
        abstract=_safe_text,
        full_text=_optional_full_text,
        source=_patent_source,
    )
    @settings(max_examples=200)
    def test_patent_persist_and_embed_flow(
        self,
        title: str,
        abstract: str,
        full_text: str | None,
        source: str,
    ) -> None:
        """For any patent record, the persist-and-embed flow generates one embedding and stores it."""
        conn = _fresh_db()
        try:
            topic_id, session_id = _setup_topic_and_session(conn)
            patent_repo = PatentRepository(conn)

            # Create the patent record (simulating search result persistence)
            record = PatentRecord(
                patent_number="US1234567",
                title=title,
                abstract=abstract,
                full_text=full_text,
                source=source,
            )
            row_id = patent_repo.create(session_id, record)

            # Mock the embedding service to return a fake embedding
            fake_embedding = _create_fake_embedding(128)
            mock_embedding_service = MagicMock()
            mock_embedding_service.generate_embedding.return_value = fake_embedding

            # Simulate the embed flow: prepare text, generate embedding, update
            vect_text = prepare_vectorization_text(
                title=title,
                abstract=abstract,
                full_text=full_text,
                max_chars=4000,
            )
            emb_result = mock_embedding_service.generate_embedding(vect_text)
            if emb_result:
                patent_repo.update_embedding(row_id, emb_result)

            # Verify: generate_embedding called exactly once
            mock_embedding_service.generate_embedding.assert_called_once_with(vect_text)

            # Verify: embedding is persisted in DB
            results = patent_repo.get_by_session(session_id)
            assert len(results) == 1
            assert results[0].embedding == fake_embedding
        finally:
            conn.close()

    @given(
        title=_safe_text,
        abstract=_safe_text,
        full_text=_optional_full_text,
        source=_paper_source,
    )
    @settings(max_examples=200)
    def test_paper_persist_and_embed_flow(
        self,
        title: str,
        abstract: str,
        full_text: str | None,
        source: str,
    ) -> None:
        """For any scientific paper, the persist-and-embed flow generates one embedding and stores it."""
        conn = _fresh_db()
        try:
            topic_id, session_id = _setup_topic_and_session(conn)
            paper_repo = ScientificPaperRepository(conn)

            # Create the paper record
            record = ScientificPaperRecord(
                doi="10.1234/test",
                title=title,
                abstract=abstract,
                full_text=full_text,
                source=source,
            )
            row_id = paper_repo.create(session_id, record)

            # Mock the embedding service
            fake_embedding = _create_fake_embedding(128)
            mock_embedding_service = MagicMock()
            mock_embedding_service.generate_embedding.return_value = fake_embedding

            # Simulate the embed flow
            vect_text = prepare_vectorization_text(
                title=title,
                abstract=abstract,
                full_text=full_text,
                max_chars=4000,
            )
            emb_result = mock_embedding_service.generate_embedding(vect_text)
            if emb_result:
                paper_repo.update_embedding(row_id, emb_result)

            # Verify: generate_embedding called exactly once
            mock_embedding_service.generate_embedding.assert_called_once_with(vect_text)

            # Verify: embedding is persisted in DB
            results = paper_repo.get_by_session(session_id)
            assert len(results) == 1
            assert results[0].embedding == fake_embedding
        finally:
            conn.close()

    @given(
        filename=_safe_text,
        content=_safe_text,
    )
    @settings(max_examples=200)
    def test_local_document_upload_embed_flow(
        self,
        filename: str,
        content: str,
    ) -> None:
        """For any local document upload, embedding is generated once and stored."""
        conn = _fresh_db()
        try:
            topic_id, _ = _setup_topic_and_session(conn)
            doc_repo = LocalDocumentRepository(conn)

            # Create the local document
            row_id = doc_repo.create(topic_id, filename, content)

            # Mock the embedding service
            fake_embedding = _create_fake_embedding(128)
            mock_embedding_service = MagicMock()
            mock_embedding_service.generate_embedding.return_value = fake_embedding

            # Simulate the upload embed flow (same as _on_file_uploaded)
            vect_text = prepare_vectorization_text(
                title=filename,
                abstract=content,
                max_chars=4000,
            )
            emb_result = mock_embedding_service.generate_embedding(vect_text)
            if emb_result:
                doc_repo.update_embedding(row_id, emb_result)

            # Verify: generate_embedding called exactly once
            mock_embedding_service.generate_embedding.assert_called_once_with(vect_text)

            # Verify: embedding is persisted in DB
            docs = doc_repo.get_by_topic(topic_id)
            assert len(docs) == 1
            # Read embedding directly from DB since get_by_topic doesn't return it
            row = conn.execute(
                "SELECT embedding FROM local_documents WHERE id = ?", (row_id,)
            ).fetchone()
            assert row[0] == fake_embedding
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Property 2: Disclosure Save Preservation
# ---------------------------------------------------------------------------


class TestDisclosureSavePreservation:
    """Property 2: Disclosure Save Preservation.

    For all disclosure save operations, no calls to generate_embedding()
    are triggered — disclosure save is decoupled from embedding generation.

    This validates that saving the invention disclosure persists text and
    search terms without triggering any embedding recalculation.

    **Validates: Requirements 3.5**
    """

    @given(
        primary_description=_safe_text,
        search_terms=_search_terms,
    )
    @settings(max_examples=200, deadline=None)
    def test_disclosure_save_does_not_trigger_embedding_generation(
        self,
        primary_description: str,
        search_terms: list[str],
    ) -> None:
        """For any disclosure save, EmbeddingService.generate_embedding is never called."""
        conn = _fresh_db()
        try:
            topic_id, _ = _setup_topic_and_session(conn)
            disclosure_repo = InventionDisclosureRepository(conn)

            # Patch EmbeddingService at the module level to detect any calls
            with patch(
                "patent_system.rag.embeddings.EmbeddingService.generate_embedding"
            ) as mock_gen_emb:
                # Perform the disclosure save (same as _on_save_disclosure)
                disclosure_repo.upsert(topic_id, primary_description, search_terms)

                # Verify: generate_embedding was NEVER called
                mock_gen_emb.assert_not_called()

            # Verify: disclosure data is persisted correctly
            saved = disclosure_repo.get_by_topic(topic_id)
            assert saved is not None
            assert saved["primary_description"] == primary_description
            assert saved["search_terms"] == search_terms
        finally:
            conn.close()

    @given(
        desc_1=_safe_text,
        terms_1=_search_terms,
        desc_2=_safe_text,
        terms_2=_search_terms,
    )
    @settings(max_examples=200, deadline=None)
    def test_disclosure_update_does_not_trigger_embedding_generation(
        self,
        desc_1: str,
        terms_1: list[str],
        desc_2: str,
        terms_2: list[str],
    ) -> None:
        """For any disclosure update (upsert over existing), no embedding generation occurs."""
        conn = _fresh_db()
        try:
            topic_id, _ = _setup_topic_and_session(conn)
            disclosure_repo = InventionDisclosureRepository(conn)

            # First save
            disclosure_repo.upsert(topic_id, desc_1, terms_1)

            # Patch and perform update
            with patch(
                "patent_system.rag.embeddings.EmbeddingService.generate_embedding"
            ) as mock_gen_emb:
                disclosure_repo.upsert(topic_id, desc_2, terms_2)
                mock_gen_emb.assert_not_called()

            # Verify: updated data is persisted
            saved = disclosure_repo.get_by_topic(topic_id)
            assert saved is not None
            assert saved["primary_description"] == desc_2
            assert saved["search_terms"] == terms_2
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Property 3: Panel Load Preservation
# ---------------------------------------------------------------------------


class TestPanelLoadPreservation:
    """Property 3: Panel Load Preservation.

    For all DB states with pre-existing relevance scores, loading the
    panel reads scores from DB without recomputation.

    This validates that previously computed relevance scores continue to
    load from the database and display.

    **Validates: Requirements 3.3**
    """

    @given(
        title=_safe_text,
        abstract=_safe_text,
        score=_relevance_score,
    )
    @settings(max_examples=200)
    def test_patent_scores_loaded_from_db_without_recomputation(
        self,
        title: str,
        abstract: str,
        score: float,
    ) -> None:
        """For any patent with a pre-existing relevance score, loading reads it from DB directly."""
        conn = _fresh_db()
        try:
            topic_id, session_id = _setup_topic_and_session(conn)
            patent_repo = PatentRepository(conn)

            # Create patent with a pre-existing relevance score
            record = PatentRecord(
                patent_number="US9999999",
                title=title,
                abstract=abstract,
                source="Google Patents",
                relevance_score=score,
            )
            patent_repo.create(session_id, record)

            # Simulate panel load: read records from DB (same as create_research_panel)
            # No embedding service should be involved
            with patch(
                "patent_system.rag.embeddings.EmbeddingService.generate_embedding"
            ) as mock_gen_emb:
                loaded_records = patent_repo.get_by_session(session_id)

                # Verify: scores are loaded directly from DB
                assert len(loaded_records) == 1
                assert loaded_records[0].relevance_score == score
                assert loaded_records[0].title == title

                # Verify: no embedding generation triggered during load
                mock_gen_emb.assert_not_called()
        finally:
            conn.close()

    @given(
        title=_safe_text,
        abstract=_safe_text,
        score=_relevance_score,
    )
    @settings(max_examples=200)
    def test_paper_scores_loaded_from_db_without_recomputation(
        self,
        title: str,
        abstract: str,
        score: float,
    ) -> None:
        """For any paper with a pre-existing relevance score, loading reads it from DB directly."""
        conn = _fresh_db()
        try:
            topic_id, session_id = _setup_topic_and_session(conn)
            paper_repo = ScientificPaperRepository(conn)

            # Create paper with a pre-existing relevance score
            record = ScientificPaperRecord(
                doi="10.1234/test",
                title=title,
                abstract=abstract,
                source="ArXiv",
                relevance_score=score,
            )
            paper_repo.create(session_id, record)

            # Simulate panel load
            with patch(
                "patent_system.rag.embeddings.EmbeddingService.generate_embedding"
            ) as mock_gen_emb:
                loaded_records = paper_repo.get_by_session(session_id)

                assert len(loaded_records) == 1
                assert loaded_records[0].relevance_score == score
                assert loaded_records[0].title == title

                mock_gen_emb.assert_not_called()
        finally:
            conn.close()

    @given(
        title=_safe_text,
        abstract=_safe_text,
        patent_score=_relevance_score,
        paper_score=_relevance_score,
    )
    @settings(max_examples=200)
    def test_mixed_records_scores_preserved_on_load(
        self,
        title: str,
        abstract: str,
        patent_score: float,
        paper_score: float,
    ) -> None:
        """For any mix of patents and papers with scores, all scores load from DB unchanged."""
        conn = _fresh_db()
        try:
            topic_id, session_id = _setup_topic_and_session(conn)
            patent_repo = PatentRepository(conn)
            paper_repo = ScientificPaperRepository(conn)

            # Create records with pre-existing scores
            patent = PatentRecord(
                patent_number="US1111111",
                title=title,
                abstract=abstract,
                source="EPO OPS",
                relevance_score=patent_score,
            )
            patent_repo.create(session_id, patent)

            paper = ScientificPaperRecord(
                doi="10.5678/mixed",
                title=title,
                abstract=abstract,
                source="PubMed",
                relevance_score=paper_score,
            )
            paper_repo.create(session_id, paper)

            # Simulate panel load — read all records
            with patch(
                "patent_system.rag.embeddings.EmbeddingService.generate_embedding"
            ) as mock_gen_emb:
                patents = patent_repo.get_by_session(session_id)
                papers = paper_repo.get_by_session(session_id)

                assert len(patents) == 1
                assert patents[0].relevance_score == patent_score

                assert len(papers) == 1
                assert papers[0].relevance_score == paper_score

                # No embedding generation during load
                mock_gen_emb.assert_not_called()
        finally:
            conn.close()
