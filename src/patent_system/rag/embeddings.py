"""Embedding service wrapper for generating vector embeddings.

Wraps LlamaIndex's HuggingFaceEmbedding with the BAAI/bge-large-en-v1.5 model.
Uses lazy initialization to avoid loading the model at import time.

Requirements: 12.1, 12.3
"""

import logging
import struct

from patent_system.db.models import PatentRecord, ScientificPaperRecord

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Generate vector embeddings using a HuggingFace model.

    The underlying model is loaded lazily on first use so that importing
    this module does not trigger heavy model downloads or GPU allocation.

    Args:
        model_name: HuggingFace model identifier. Defaults to
            ``BAAI/bge-large-en-v1.5``.
    """

    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5") -> None:
        self._model_name = model_name
        self._model = None

    def _ensure_model(self) -> None:
        """Load the embedding model if it has not been loaded yet."""
        if self._model is not None:
            return
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        self._model = HuggingFaceEmbedding(model_name=self._model_name)

    def generate_embedding(self, text: str) -> bytes | None:
        """Generate an embedding vector for *text* and pack it to bytes.

        Returns:
            A ``bytes`` object containing the float32 vector, or ``None``
            if embedding generation fails.
        """
        try:
            self._ensure_model()
            assert self._model is not None
            vector: list[float] = self._model.get_text_embedding(text)
            return struct.pack(f"{len(vector)}f", *vector)
        except Exception:
            logger.exception("Failed to generate embedding for text")
            return None

    def generate_embedding_for_record(
        self, record: PatentRecord | ScientificPaperRecord
    ) -> bytes | None:
        """Generate an embedding for a patent or scientific paper record.

        Combines the record's ``title`` and ``abstract`` into a single
        string and delegates to :meth:`generate_embedding`.

        Returns:
            Packed float32 bytes, or ``None`` on failure.
        """
        text = f"{record.title} {record.abstract}"
        return self.generate_embedding(text)
