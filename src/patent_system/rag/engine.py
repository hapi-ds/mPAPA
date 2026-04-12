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
            api_base=settings.lm_studio_base_url,
            api_key=settings.lm_studio_api_key,
        )
        self._indexes: dict[int, VectorStoreIndex] = {}

    def _ensure_embed_model(self) -> None:
        """Lazily set the LlamaIndex global embed model from our service."""
        # Set our custom model directly — avoid accessing Settings.embed_model
        # as a getter, because LlamaIndex's lazy resolution tries to load
        # OpenAI and fails without OPENAI_API_KEY.
        if getattr(self, "_embed_model_set", False):
            return
        Settings.embed_model = self._embedding_service.get_llama_index_model()
        self._embed_model_set = True

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

        from llama_index.core.schema import TextNode

        _MAX_EMB_TEXT = 4000
        nodes = []
        for doc in documents:
            text = doc["text"][:_MAX_EMB_TEXT] if len(doc.get("text", "")) > _MAX_EMB_TEXT else doc.get("text", "")
            meta = doc.get("metadata", {})
            node = TextNode(
                text=text,
                metadata=meta,
                excluded_embed_metadata_keys=list(meta.keys()),
                excluded_llm_metadata_keys=list(meta.keys()),
            )
            nodes.append(node)

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

    def index_with_embeddings(
        self, topic_id: int, documents: list[dict],
    ) -> None:
        """Build the index from documents that already have embeddings.

        Each dict must contain ``"text"``, ``"metadata"``, and
        ``"embedding"`` (list[float]) keys. Skips LM Studio calls
        entirely — uses the pre-computed vectors directly.

        Documents without an embedding are indexed normally (will
        trigger embedding generation).

        Args:
            topic_id: The topic workspace to index into.
            documents: List of dicts with text, metadata, and embedding.
        """
        if not documents:
            return

        self._ensure_embed_model()

        from llama_index.core.schema import TextNode

        # Max chars to send for embedding — prevents CUDA OOM on long docs
        _MAX_EMB_TEXT = 4000

        nodes_with_emb: list[TextNode] = []
        nodes_without_emb: list[Document] = []

        for doc in documents:
            text = doc["text"][:_MAX_EMB_TEXT] if len(doc.get("text", "")) > _MAX_EMB_TEXT else doc.get("text", "")
            meta = doc.get("metadata", {})
            emb = doc.get("embedding")
            if emb and isinstance(emb, list):
                node = TextNode(
                    text=text,
                    metadata=meta,
                    embedding=emb,
                    excluded_embed_metadata_keys=list(meta.keys()),
                    excluded_llm_metadata_keys=list(meta.keys()),
                )
                nodes_with_emb.append(node)
            else:
                node = TextNode(
                    text=text,
                    metadata=meta,
                    excluded_embed_metadata_keys=list(meta.keys()),
                    excluded_llm_metadata_keys=list(meta.keys()),
                )
                nodes_without_emb.append(node)

        all_nodes = nodes_with_emb + nodes_without_emb  # type: ignore[operator]

        if topic_id in self._indexes:
            self._indexes[topic_id].insert_nodes(all_nodes)
        else:
            # Build index from nodes with pre-set embeddings
            self._indexes[topic_id] = VectorStoreIndex(nodes=all_nodes)

        logger.info(
            "Indexed %d docs for topic %d (%d with stored embeddings, %d new)",
            len(documents), topic_id, len(nodes_with_emb), len(nodes_without_emb),
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

        output = []
        for node in results:
            try:
                text = node.get_text()
            except ValueError:
                # Node is not a TextNode — fall back to content or skip
                inner = getattr(node, "node", node)
                text = getattr(inner, "text", "") or str(node)
            score = node.get_score() if hasattr(node, "get_score") else 0.0
            metadata = getattr(node, "metadata", None)
            if metadata is None or callable(metadata):
                inner = getattr(node, "node", node)
                metadata = getattr(inner, "metadata", {})
            output.append({
                "text": text,
                "score": score,
                "metadata": metadata,
            })
        return output
