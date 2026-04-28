"""Novelty Analysis Agent for the patent drafting pipeline.

Uses the DSPy NoveltyAnalysisModule to compare the invention disclosure
and drafted claims against prior art to identify novel aspects, potential
conflicts, and suggested claim scope.

Requirements: 4.1, 4.2, 4.4, 5.1
"""

import json
import logging
import time
from typing import Any, Protocol

import dspy
import httpx
import litellm.exceptions
import requests.exceptions

from patent_system.agents.domain_profiles import DEFAULT_PROFILE_SLUG
from patent_system.agents.personality import resolve_personality_mode
from patent_system.agents.review_notes import build_review_notes_text
from patent_system.agents.state import PatentWorkflowState
from patent_system.dspy_modules.modules import NoveltyAnalysisModule
from patent_system.exceptions import LLMConnectionError
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


def _prepare_text(value: dict | str | None) -> str:
    """Serialize a state value to a text representation."""
    if not value:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return str(value)


def novelty_analysis_node(
    state: PatentWorkflowState,
    rag_engine: RAGQueryable | None = None,
    review_notes_mode: str = "continue",
) -> dict[str, Any]:
    """Run the Novelty Analysis Agent.

    Uses the LLM via DSPy NoveltyAnalysisModule to produce a detailed
    novelty assessment comparing the invention against prior art.

    Args:
        state: The current workflow state.
        rag_engine: Optional RAG engine (unused in LLM-based analysis
            but kept for interface compatibility).
        review_notes_mode: Either ``"continue"`` (inject upstream notes)
            or ``"rerun"`` (inject own notes only).

    Returns:
        Dict with ``novelty_analysis`` (assessment string) and
        ``current_step`` set to ``"novelty_analysis"``.
    """
    start = time.monotonic()

    mode = resolve_personality_mode(state, "novelty_analysis")
    domain_slug = state.get("domain_profile_slug") or DEFAULT_PROFILE_SLUG

    # Build review notes text
    review_notes = state.get("review_notes") or {}
    notes_text = build_review_notes_text(review_notes, "novelty_analysis", review_notes_mode)

    disclosure = state.get("invention_disclosure")
    claims_text = state.get("claims_text", "")
    prior_art_summary = state.get("prior_art_summary", "")

    disclosure_text = _prepare_text(disclosure)

    module = NoveltyAnalysisModule()
    try:
        prediction = module(
            invention_disclosure=disclosure_text,
            claims_text=claims_text,
            prior_art_summary=prior_art_summary,
            personality_mode=mode.value,
            review_notes_text=notes_text or None,
            domain_profile_slug=domain_slug,
        )
    except (
        requests.exceptions.ConnectionError,
        httpx.ConnectError,
        litellm.exceptions.APIConnectionError,
        ConnectionError,
        OSError,
    ) as exc:
        base_url = (
            dspy.settings.lm.kwargs.get("api_base", "unknown")
            if dspy.settings.lm
            else "unknown"
        )
        raise LLMConnectionError(
            f"LM Studio unreachable at {base_url}: {exc}"
        ) from exc

    novelty_text = prediction.novelty_assessment

    duration_ms = (time.monotonic() - start) * 1000

    log_agent_invocation(
        logger=logger,
        name="NoveltyAnalysisAgent",
        input_summary=(
            f"disclosure_length={len(disclosure_text)}, "
            f"claims_length={len(claims_text)}, "
            f"prior_art_length={len(prior_art_summary)}, "
            f"personality_mode={mode.value}, "
            f"review_notes_length={len(notes_text)}, "
            f"domain_profile={domain_slug}"
        ),
        output_summary=f"assessment_length={len(novelty_text)}",
        duration_ms=duration_ms,
    )

    return {
        "novelty_analysis": novelty_text,
        "current_step": "novelty_analysis",
        "personality_mode_used": mode.value,
    }
