"""Unit tests for the RAGEngine class."""

from unittest.mock import MagicMock, patch

import pytest

from patent_system.config import AppSettings
from patent_system.rag.engine import RAGEngine


@pytest.fixture
def settings() -> AppSettings:
    """Provide default AppSettings for tests."""
    return AppSettings()


class TestRAGEngineInit:
    """Verify constructor initializes state correctly."""

    def test_stores_settings(self, settings: AppSettings) -> None:
        engine = RAGEngine(settings)
        assert engine._settings is settings

    def test_creates_embedding_service(self, settings: AppSettings) -> None:
        engine = RAGEngine(settings)
        assert engine._embedding_service is not None
        assert engine._embedding_service._model_name == settings.embedding_model_name

    def test_indexes_dict_starts_empty(self, settings: AppSettings) -> None:
        engine = RAGEngine(settings)
        assert engine._indexes == {}


class TestQueryEmptyIndex:
    """Verify query returns empty list when no documents are indexed."""

    def test_query_unknown_topic_returns_empty(self, settings: AppSettings) -> None:
        engine = RAGEngine(settings)
        result = engine.query(topic_id=999, query_text="anything")
        assert result == []

    def test_query_different_topic_returns_empty(self, settings: AppSettings) -> None:
        """Even if topic 1 has docs, topic 2 should return empty."""
        engine = RAGEngine(settings)
        # Don't index anything for topic 2
        result = engine.query(topic_id=2, query_text="test")
        assert result == []


class TestIndexDocuments:
    """Verify index_documents creates and populates indexes."""

    @patch("patent_system.rag.engine.RAGEngine._ensure_embed_model")
    @patch("patent_system.rag.engine.VectorStoreIndex")
    def test_creates_new_index_for_topic(
        self, mock_vsi_cls: MagicMock, mock_ensure: MagicMock, settings: AppSettings
    ) -> None:
        engine = RAGEngine(settings)
        mock_index = MagicMock()
        mock_vsi_cls.from_documents.return_value = mock_index

        docs = [{"text": "hello world", "metadata": {"source": "test"}}]
        engine.index_documents(topic_id=1, documents=docs)

        mock_vsi_cls.from_documents.assert_called_once()
        assert 1 in engine._indexes
        assert engine._indexes[1] is mock_index

    @patch("patent_system.rag.engine.RAGEngine._ensure_embed_model")
    @patch("patent_system.rag.engine.VectorStoreIndex")
    def test_inserts_into_existing_index(
        self, mock_vsi_cls: MagicMock, mock_ensure: MagicMock, settings: AppSettings
    ) -> None:
        engine = RAGEngine(settings)
        mock_index = MagicMock()
        engine._indexes[1] = mock_index

        docs = [{"text": "new doc", "metadata": {}}]
        engine.index_documents(topic_id=1, documents=docs)

        mock_index.insert_nodes.assert_called_once()
        # from_documents should NOT be called since index already exists
        mock_vsi_cls.from_documents.assert_not_called()

    @patch("patent_system.rag.engine.RAGEngine._ensure_embed_model")
    def test_empty_documents_is_noop(
        self, mock_ensure: MagicMock, settings: AppSettings
    ) -> None:
        engine = RAGEngine(settings)
        engine.index_documents(topic_id=1, documents=[])
        # No index should be created
        assert 1 not in engine._indexes
        mock_ensure.assert_not_called()

    @patch("patent_system.rag.engine.RAGEngine._ensure_embed_model")
    @patch("patent_system.rag.engine.VectorStoreIndex")
    def test_missing_metadata_defaults_to_empty_dict(
        self, mock_vsi_cls: MagicMock, mock_ensure: MagicMock, settings: AppSettings
    ) -> None:
        engine = RAGEngine(settings)
        mock_vsi_cls.from_documents.return_value = MagicMock()

        docs = [{"text": "no metadata key"}]
        engine.index_documents(topic_id=1, documents=docs)

        call_args = mock_vsi_cls.from_documents.call_args
        created_docs = call_args[0][0]
        assert created_docs[0].metadata == {}


class TestQuery:
    """Verify query retrieves and formats results correctly."""

    @patch("patent_system.rag.engine.RAGEngine._ensure_embed_model")
    def test_returns_formatted_results(
        self, mock_ensure: MagicMock, settings: AppSettings
    ) -> None:
        engine = RAGEngine(settings)

        # Set up a mock index with a mock retriever
        mock_node = MagicMock()
        mock_node.get_text.return_value = "relevant text"
        mock_node.get_score.return_value = 0.95
        mock_node.metadata = {"source": "patent_db"}

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [mock_node]

        mock_index = MagicMock()
        mock_index.as_retriever.return_value = mock_retriever
        engine._indexes[1] = mock_index

        results = engine.query(topic_id=1, query_text="novel feature")

        assert len(results) == 1
        assert results[0]["text"] == "relevant text"
        assert results[0]["score"] == 0.95
        assert results[0]["metadata"] == {"source": "patent_db"}

    @patch("patent_system.rag.engine.RAGEngine._ensure_embed_model")
    def test_passes_top_k_to_retriever(
        self, mock_ensure: MagicMock, settings: AppSettings
    ) -> None:
        engine = RAGEngine(settings)

        mock_index = MagicMock()
        mock_index.as_retriever.return_value = MagicMock(retrieve=MagicMock(return_value=[]))
        engine._indexes[1] = mock_index

        engine.query(topic_id=1, query_text="test", top_k=10)

        mock_index.as_retriever.assert_called_once_with(similarity_top_k=10)

    @patch("patent_system.rag.engine.RAGEngine._ensure_embed_model")
    def test_default_top_k_is_five(
        self, mock_ensure: MagicMock, settings: AppSettings
    ) -> None:
        engine = RAGEngine(settings)

        mock_index = MagicMock()
        mock_index.as_retriever.return_value = MagicMock(retrieve=MagicMock(return_value=[]))
        engine._indexes[1] = mock_index

        engine.query(topic_id=1, query_text="test")

        mock_index.as_retriever.assert_called_once_with(similarity_top_k=5)
