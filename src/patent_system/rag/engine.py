"""RAG engine for document indexing and semantic retrieval.

Manages per-topic VectorStoreIndex instances backed by LlamaIndex.
Uses the EmbeddingService for vector generation and stores embeddings
as BLOBs in SQLite via the embedding service layer.

Requirements: 4.1, 8.1, 12.2
"""

import logging

from llama_index.core import Document, Settings, VectorStoreIndex

from patent_system.config import AppSettings
from patent_system.rag.embeddings import EmbeddingService

logger = logging.getLogger(__name__)


class RAGEngine:
    """Manages document indexing and retrieval scoped by topic.

    Each topic gets its own ``VectorStoreIndex`` so that queries only
    return results relevant to the selected workspace.

    Args:
        settings: Application settings providing the embedding model name.
    """

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._embedding_service = EmbeddingService(
            model_name=settings.embedding_model_name,
        )
        self._indexes: dict[int, VectorStoreIndex] = {}

    def _ensure_embed_model(self) -> None:
        """Lazily load the embedding model into LlamaIndex global settings."""
        if Settings.embed_model is not None and not isinstance(
            Settings.embed_model, str
        ):
            return
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        Settings.embed_model = HuggingFaceEmbedding(
            model_name=self._settings.embedding_model_name,
        )

    def index_documents(self, topic_id: int, documents: list[dict]) -> None:
        """Index a batch of documents under the given topic.

        Each dict in *documents* must contain ``"text"`` and ``"metadata"``
        keys.  The method creates LlamaIndex ``Document`` objects, generates
        embeddings via the embedding service, and upserts them into the
        topic-scoped ``VectorStoreIndex``.

        Args:
            topic_id: The topic workspace to index into.
            documents: List of dicts with ``"text"`` and ``"metadata"`` keys.
        """
        if not documents:
            return

        self._ensure_embed_model()

        nodes = [
            Document(
                text=doc["text"],
                metadata=doc.get("metadata", {}),
            )
            for doc in documents
        ]

        if topic_id in self._indexes:
            self._indexes[topic_id].insert_nodes(nodes)
            logger.info(
                "Inserted %d documents into existing index for topic %d",
                len(nodes),
                topic_id,
            )
        else:
            self._indexes[topic_id] = VectorStoreIndex.from_documents(nodes)
            logger.info(
                "Created new index with %d documents for topic %d",
                len(nodes),
                topic_id,
            )

    def query(
        self, topic_id: int, query_text: str, top_k: int = 5
    ) -> list[dict]:
        """Retrieve relevant documents for a query within a topic scope.

        Args:
            topic_id: The topic workspace to search.
            query_text: The natural-language query string.
            top_k: Maximum number of results to return.

        Returns:
            A list of dicts, each containing ``"text"``, ``"score"``, and
            ``"metadata"`` keys.  Returns an empty list when no documents
            have been indexed for the topic.
        """
        if topic_id not in self._indexes:
            logger.debug(
                "No index found for topic %d; returning empty results",
                topic_id,
            )
            return []

        self._ensure_embed_model()

        retriever = self._indexes[topic_id].as_retriever(
            similarity_top_k=top_k,
        )
        results = retriever.retrieve(query_text)

        return [
            {
                "text": node.get_text(),
                "score": node.get_score(),
                "metadata": node.metadata,
            }
            for node in results
        ]
