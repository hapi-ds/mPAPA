"""Property graph index for citation relationship tracking.

Provides a lightweight graph structure for indexing and querying citation
relationships between patent and scientific paper documents. Uses an
adjacency list internally; can be enhanced later with LlamaIndex's
PropertyGraphIndex for richer graph queries.

Requirements: 13.1, 13.2
"""

from __future__ import annotations

import logging
from collections import deque

logger = logging.getLogger(__name__)


class CitationGraphIndex:
    """Track citation relationships between documents.

    Maintains a directed graph where an edge from *source* to *target*
    means "source cites target".  Supports forward lookups (what does a
    document cite?), reverse lookups (who cites a document?), and
    shortest-path queries between two documents.
    """

    def __init__(self) -> None:
        """Initialize an empty citation graph."""
        self._forward: dict[str, list[dict]] = {}
        self._reverse: dict[str, list[dict]] = {}

    def add_citation(
        self,
        source_id: str,
        target_id: str,
        metadata: dict | None = None,
    ) -> None:
        """Add a citation edge from *source_id* to *target_id*.

        Args:
            source_id: The document that contains the citation.
            target_id: The document being cited.
            metadata: Optional metadata to attach to the edge
                (e.g. citation context, section, page number).
        """
        edge_meta = metadata or {}
        forward_entry = {"document_id": target_id, **edge_meta}
        reverse_entry = {"document_id": source_id, **edge_meta}

        self._forward.setdefault(source_id, []).append(forward_entry)
        self._reverse.setdefault(target_id, []).append(reverse_entry)

        logger.debug(
            "Added citation edge: %s -> %s (metadata=%s)",
            source_id,
            target_id,
            edge_meta,
        )

    def get_citations(self, document_id: str) -> list[dict]:
        """Return documents cited by *document_id*.

        Args:
            document_id: The citing document.

        Returns:
            A list of dicts, each containing at least ``"document_id"``
            and any edge metadata.
        """
        return list(self._forward.get(document_id, []))

    def get_cited_by(self, document_id: str) -> list[dict]:
        """Return documents that cite *document_id*.

        Args:
            document_id: The cited document.

        Returns:
            A list of dicts, each containing at least ``"document_id"``
            and any edge metadata.
        """
        return list(self._reverse.get(document_id, []))

    def get_citation_path(
        self,
        source_id: str,
        target_id: str,
    ) -> list[str] | None:
        """Find a shortest citation path from *source_id* to *target_id*.

        Uses breadth-first search over the forward citation edges.

        Args:
            source_id: Starting document.
            target_id: Destination document.

        Returns:
            An ordered list of document IDs forming the path (inclusive of
            both endpoints), or ``None`` if no path exists.
        """
        if source_id == target_id:
            return [source_id]

        visited: set[str] = {source_id}
        queue: deque[list[str]] = deque([[source_id]])

        while queue:
            path = queue.popleft()
            current = path[-1]

            for entry in self._forward.get(current, []):
                neighbor = entry["document_id"]
                if neighbor == target_id:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(path + [neighbor])

        return None
