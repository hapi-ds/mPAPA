"""Embedding service using LM Studio's OpenAI-compatible API.

Uses the ``/v1/embeddings`` endpoint exposed by LM Studio via the
``openai`` Python client. No local model download required.

Requirements: 12.1, 12.3
"""

import logging
import struct
from typing import Any

import openai
from llama_index.core.embeddings import BaseEmbedding

from patent_system.db.models import PatentRecord, ScientificPaperRecord

logger = logging.getLogger(__name__)


class LMStudioEmbedding(BaseEmbedding):
    """LlamaIndex-compatible embedding model backed by LM Studio.

    Calls the OpenAI-compatible ``/v1/embeddings`` endpoint. Accepts
    any model name that LM Studio has loaded — no hardcoded model
    validation.
    """

    _client: Any = None
    _model_id: str = ""

    def __init__(
        self,
        model_name: str,
        api_base: str = "http://localhost:1234/v1",
        api_key: str = "not-needed",
        embed_batch_size: int = 10,
        **kwargs: Any,
    ) -> None:
        super().__init__(embed_batch_size=embed_batch_size, **kwargs)
        self._model_id = model_name
        self._client = openai.OpenAI(base_url=api_base, api_key=api_key)

    @classmethod
    def class_name(cls) -> str:
        return "LMStudioEmbedding"

    def _get_text_embedding(self, text: str) -> list[float]:
        """Get embedding for a single text string."""
        resp = self._client.embeddings.create(
            input=[text], model=self._model_id,
        )
        return resp.data[0].embedding

    def _get_query_embedding(self, query: str) -> list[float]:
        """Get embedding for a query string."""
        return self._get_text_embedding(query)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        """Async query embedding — delegates to sync version."""
        return self._get_query_embedding(query)

    def _get_text_embedding_batch(
        self, texts: list[str], **kwargs: Any,
    ) -> list[list[float]]:
        """Get embeddings for a batch of texts in a single API call."""
        resp = self._client.embeddings.create(
            input=texts, model=self._model_id,
        )
        # Sort by index to preserve order
        sorted_data = sorted(resp.data, key=lambda d: d.index)
        return [d.embedding for d in sorted_data]


class EmbeddingService:
    """Generate vector embeddings via LM Studio's embedding endpoint.

    Args:
        model_name: The embedding model name loaded in LM Studio.
        api_base: Base URL for the OpenAI-compatible API.
        api_key: API key (typically ``"not-needed"`` for local LM Studio).
    """

    def __init__(
        self,
        model_name: str = "text-embedding-nomic-embed-text-v1.5",
        api_base: str = "http://localhost:1234/v1",
        api_key: str = "not-needed",
    ) -> None:
        self._model_name = model_name
        self._api_base = api_base
        self._api_key = api_key
        self._model: LMStudioEmbedding | None = None

    def _ensure_model(self) -> None:
        """Lazily create the embedding client."""
        if self._model is not None:
            return
        self._model = LMStudioEmbedding(
            model_name=self._model_name,
            api_base=self._api_base,
            api_key=self._api_key,
        )

    def get_llama_index_model(self) -> LMStudioEmbedding:
        """Return the LlamaIndex embedding model instance.

        Used by RAGEngine to set ``Settings.embed_model``.
        """
        self._ensure_model()
        assert self._model is not None
        return self._model

    def generate_embedding(self, text: str) -> bytes | None:
        """Generate an embedding vector for *text* and pack it to bytes.

        Returns:
            A ``bytes`` object containing the float32 vector, or ``None``
            if embedding generation fails.
        """
        try:
            self._ensure_model()
            assert self._model is not None
            vector: list[float] = self._model._get_text_embedding(text)
            return struct.pack(f"{len(vector)}f", *vector)
        except Exception:
            logger.exception("Failed to generate embedding for text")
            return None

    def generate_embedding_for_record(
        self, record: PatentRecord | ScientificPaperRecord,
    ) -> bytes | None:
        """Generate an embedding for a patent or scientific paper record.

        Combines the record's ``title`` and ``abstract`` into a single
        string and delegates to :meth:`generate_embedding`.

        Returns:
            Packed float32 bytes, or ``None`` on failure.
        """
        text = f"{record.title} {record.abstract}"
        return self.generate_embedding(text)
