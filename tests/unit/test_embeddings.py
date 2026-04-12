"""Unit tests for the EmbeddingService wrapper."""

import struct
from unittest.mock import MagicMock, patch

import pytest

from patent_system.db.models import PatentRecord, ScientificPaperRecord
from patent_system.rag.embeddings import EmbeddingService


class TestEmbeddingServiceLazyInit:
    """Verify the model is not loaded at construction time."""

    def test_model_not_loaded_on_init(self) -> None:
        service = EmbeddingService()
        assert service._model is None

    def test_custom_model_name_stored(self) -> None:
        service = EmbeddingService(model_name="some/other-model")
        assert service._model_name == "some/other-model"


class TestGenerateEmbedding:
    """Tests for generate_embedding using a mocked embedding model."""

    @patch("patent_system.rag.embeddings.EmbeddingService._ensure_model")
    def test_returns_packed_bytes(self, mock_ensure: MagicMock) -> None:
        service = EmbeddingService()
        fake_vector = [0.1, 0.2, 0.3]
        service._model = MagicMock()
        service._model._get_text_embedding.return_value = fake_vector

        result = service.generate_embedding("hello world")

        assert result is not None
        unpacked = struct.unpack(f"{len(fake_vector)}f", result)
        assert len(unpacked) == 3
        assert pytest.approx(unpacked, abs=1e-6) == tuple(fake_vector)

    @patch("patent_system.rag.embeddings.EmbeddingService._ensure_model")
    def test_returns_none_on_failure(self, mock_ensure: MagicMock) -> None:
        service = EmbeddingService()
        service._model = MagicMock()
        service._model._get_text_embedding.side_effect = RuntimeError("boom")

        result = service.generate_embedding("hello")

        assert result is None

    @patch("patent_system.rag.embeddings.EmbeddingService._ensure_model")
    def test_returns_none_when_model_load_fails(self, mock_ensure: MagicMock) -> None:
        mock_ensure.side_effect = RuntimeError("model load failed")
        service = EmbeddingService()

        result = service.generate_embedding("hello")

        assert result is None


class TestGenerateEmbeddingForRecord:
    """Tests for generate_embedding_for_record."""

    @patch("patent_system.rag.embeddings.EmbeddingService._ensure_model")
    def test_patent_record(self, mock_ensure: MagicMock) -> None:
        service = EmbeddingService()
        fake_vector = [1.0, 2.0]
        service._model = MagicMock()
        service._model._get_text_embedding.return_value = fake_vector

        record = PatentRecord(
            patent_number="US123",
            title="My Patent",
            abstract="A great invention",
            source="test",
        )
        result = service.generate_embedding_for_record(record)

        assert result is not None
        service._model._get_text_embedding.assert_called_once_with(
            "My Patent A great invention"
        )

    @patch("patent_system.rag.embeddings.EmbeddingService._ensure_model")
    def test_scientific_paper_record(self, mock_ensure: MagicMock) -> None:
        service = EmbeddingService()
        fake_vector = [3.0, 4.0]
        service._model = MagicMock()
        service._model._get_text_embedding.return_value = fake_vector

        record = ScientificPaperRecord(
            doi="10.1234/test",
            title="My Paper",
            abstract="Interesting findings",
            source="arxiv",
        )
        result = service.generate_embedding_for_record(record)

        assert result is not None
        service._model._get_text_embedding.assert_called_once_with(
            "My Paper Interesting findings"
        )

    @patch("patent_system.rag.embeddings.EmbeddingService._ensure_model")
    def test_returns_none_on_failure(self, mock_ensure: MagicMock) -> None:
        service = EmbeddingService()
        service._model = MagicMock()
        service._model._get_text_embedding.side_effect = ValueError("bad input")

        record = PatentRecord(
            patent_number="US999",
            title="Broken",
            abstract="Fails",
            source="test",
        )
        result = service.generate_embedding_for_record(record)

        assert result is None
