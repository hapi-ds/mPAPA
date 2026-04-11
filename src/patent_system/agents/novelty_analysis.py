"""Novelty Analysis Agent for the patent drafting pipeline.

Compares the invention disclosure against prior art using the RAG engine
to identify novel aspects, potential conflicts, and suggested claim scope.

Requirements: 4.1, 4.2, 4.4
"""

import logging
import time
from typing import Any, Protocol

from patent_system.agents.state import PatentWorkflowState
from patent_system.db.models import NoveltyAnalysisResult
from patent_system.logging_config import log_agent_invocation

logger = logging.getLogger(__name__)


class RAGQueryable(Protocol):
    """Protocol for objects that support RAG queries."""

    def query(
        self, topic_id: int, query_text: str, top_k: int = 5
    ) -> list[dict]: ...


class _PlaceholderRAG:
    """Placeholder RAG engine returning empty results."""

    def query(
        self, topic_id: int, query_text: str, top_k: int = 5
    ) -> list[dict]:
        return []


def _extract_query_from_disclosure(disclosure: dict | None) -> str:
    """Build a RAG query string from the invention disclosure.

    Combines the technical problem and novel features into a single
    query string suitable for semantic retrieval.

    Args:
        disclosure: The invention disclosure dict, or None.

    Returns:
        A query string. Returns an empty string when disclosure is
        None or has no useful content.
    """
    if not disclosure:
        return ""

    parts: list[str] = []

    technical_problem = disclosure.get("technical_problem", "")
    if technical_problem:
        parts.append(technical_problem)

    novel_features = disclosure.get("novel_features", [])
    if isinstance(novel_features, list):
        for feature in novel_features:
            if feature:
                parts.append(str(feature))

    return " ".join(parts)


def _analyze_prior_art(
    disclosure: dict | None, prior_art_docs: list[dict]
) -> NoveltyAnalysisResult:
    """Produce a structured novelty analysis from disclosure and prior art.

    This is a deterministic placeholder that inspects the disclosure and
    retrieved prior art to build a ``NoveltyAnalysisResult``. In production,
    this would be backed by an LLM call via DSPy.

    Args:
        disclosure: The invention disclosure dict.
        prior_art_docs: Retrieved prior art documents from the RAG engine.

    Returns:
        A ``NoveltyAnalysisResult`` with novel_aspects, potential_conflicts,
        and suggested_claim_scope.
    """
    novel_features = (
        disclosure.get("novel_features", []) if disclosure else []
    )
    if not isinstance(novel_features, list):
        novel_features = []

    # Identify potential conflicts by matching prior art titles/abstracts
    # against novel features (simple keyword overlap as placeholder logic).
    potential_conflicts: list[dict] = []
    for doc in prior_art_docs:
        doc_text = (doc.get("text", "") + " " + doc.get("title", "")).lower()
        for feature in novel_features:
            if feature and str(feature).lower() in doc_text:
                potential_conflicts.append(
                    {
                        "feature": str(feature),
                        "conflicting_document": doc.get("text", "")[:200],
                        "score": doc.get("score", 0.0),
                    }
                )

    # Novel aspects are features with no detected conflicts
    conflicting_features = {c["feature"] for c in potential_conflicts}
    novel_aspects = [
        str(f) for f in novel_features if str(f) not in conflicting_features
    ]

    # If no features at all, note that analysis was inconclusive
    if not novel_features:
        novel_aspects = ["No novel features identified in disclosure"]

    # Build suggested claim scope
    if novel_aspects and novel_aspects[0] != "No novel features identified in disclosure":
        suggested_claim_scope = (
            "Claims should focus on: " + "; ".join(novel_aspects)
        )
    else:
        suggested_claim_scope = "Insufficient data to suggest claim scope"

    return NoveltyAnalysisResult(
        novel_aspects=novel_aspects,
        potential_conflicts=potential_conflicts,
        suggested_claim_scope=suggested_claim_scope,
    )


def novelty_analysis_node(
    state: PatentWorkflowState,
    rag_engine: RAGQueryable | None = None,
) -> dict[str, Any]:
    """Run the Novelty Analysis Agent.

    1. Extracts key features from the invention disclosure.
    2. Queries the RAG engine for relevant prior art documents.
    3. Produces a structured ``NoveltyAnalysisResult``.
    4. Persists analysis results in the returned state dict.
    5. Logs the agent invocation.

    Args:
        state: The current workflow state.
        rag_engine: Optional RAG engine for prior art retrieval.
            Uses a placeholder returning empty results if not provided.

    Returns:
        Dict with ``novelty_analysis`` (serialized result dict) and
        ``current_step`` set to ``"novelty_analysis"``.
    """
    start = time.monotonic()

    if rag_engine is None:
        rag_engine = _PlaceholderRAG()

    disclosure = state.get("invention_disclosure")
    topic_id = state.get("topic_id", 0)

    # Step 1: Extract query from disclosure
    query_text = _extract_query_from_disclosure(disclosure)

    # Step 2: Retrieve relevant prior art via RAG
    prior_art_docs: list[dict] = []
    if query_text:
        prior_art_docs = rag_engine.query(
            topic_id=topic_id, query_text=query_text, top_k=5
        )

    # Step 3: Produce structured analysis
    analysis = _analyze_prior_art(disclosure, prior_art_docs)

    duration_ms = (time.monotonic() - start) * 1000

    # Step 5: Log agent invocation
    log_agent_invocation(
        logger=logger,
        name="NoveltyAnalysisAgent",
        input_summary=(
            f"topic_id={topic_id}, "
            f"query_length={len(query_text)}, "
            f"prior_art_count={len(prior_art_docs)}"
        ),
        output_summary=(
            f"novel_aspects={len(analysis.novel_aspects)}, "
            f"conflicts={len(analysis.potential_conflicts)}"
        ),
        duration_ms=duration_ms,
    )

    # Step 4: Return analysis in state (persistence)
    return {
        "novelty_analysis": analysis.model_dump(),
        "current_step": "novelty_analysis",
    }
