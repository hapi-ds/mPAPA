"""Disclosure Summary Agent for the patent drafting pipeline.

Uses the DSPy DisclosureSummaryModule to generate a concise summary
of all preceding workflow steps: initial idea, claims, prior art,
novelty analysis, consistency review, market assessment, and legal
assessment.

Requirements: 9.1, 9.5
"""

import json
import logging
import time
from typing import Any

import dspy
import httpx
import litellm.exceptions
import requests.exceptions

from patent_system.agents.domain_profiles import DEFAULT_PROFILE_SLUG
from patent_system.agents.personality import resolve_personality_mode
from patent_system.agents.review_notes import build_review_notes_text
from patent_system.agents.state import PatentWorkflowState
from patent_system.dspy_modules.modules import DisclosureSummaryModule
from patent_system.exceptions import LLMConnectionError
from patent_system.logging_config import log_agent_invocation

logger = logging.getLogger(__name__)


def _prepare_text(value: dict | str | None) -> str:
    """Serialize a state value to a text representation.

    Args:
        value: A dict, string, or None from the workflow state.

    Returns:
        A string representation suitable for the DSPy module input.
        Returns an empty string when value is None.
    """
    if not value:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return str(value)


def disclosure_summary_node(
    state: PatentWorkflowState,
    review_notes_mode: str = "continue",
) -> dict[str, Any]:
    """Run the Disclosure Summary Agent.

    1. Extracts all seven preceding step fields from state:
       ``invention_disclosure`` (as initial_idea), ``claims_text``,
       ``prior_art_summary``, ``novelty_analysis``, ``review_feedback``
       (as consistency_review), ``market_assessment``, ``legal_assessment``.
    2. Calls ``DisclosureSummaryModule`` to generate a summary.
    3. Logs agent invocation.
    4. Returns dict with ``disclosure_summary`` and ``current_step``.

    Args:
        state: The current workflow state.
        review_notes_mode: Either ``"continue"`` (inject upstream notes)
            or ``"rerun"`` (inject own notes only).

    Returns:
        Dict with ``disclosure_summary`` (summary string) and
        ``current_step`` set to ``"disclosure_summary"``.
    """
    start = time.monotonic()

    mode = resolve_personality_mode(state, "disclosure_summary")
    domain_slug = state.get("domain_profile_slug") or DEFAULT_PROFILE_SLUG

    # Build review notes text
    review_notes = state.get("review_notes") or {}
    notes_text = build_review_notes_text(review_notes, "disclosure_summary", review_notes_mode)

    # Extract all seven preceding step fields
    disclosure = state.get("invention_disclosure")
    claims_text = state.get("claims_text", "")
    prior_art_summary = state.get("prior_art_summary", "")
    novelty = state.get("novelty_analysis")
    consistency_review = state.get("review_feedback", "")
    market_assessment = state.get("market_assessment", "")
    legal_assessment = state.get("legal_assessment", "")

    # Convert dict values to strings
    initial_idea_text = _prepare_text(disclosure)
    novelty_text = _prepare_text(novelty)

    module = DisclosureSummaryModule()
    try:
        prediction = module(
            initial_idea=initial_idea_text,
            claims_text=claims_text,
            prior_art_summary=prior_art_summary,
            novelty_analysis=novelty_text,
            consistency_review=consistency_review,
            market_assessment=market_assessment,
            legal_assessment=legal_assessment,
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

    disclosure_summary = prediction.disclosure_summary

    duration_ms = (time.monotonic() - start) * 1000

    log_agent_invocation(
        logger=logger,
        name="DisclosureSummaryAgent",
        input_summary=(
            f"initial_idea_length={len(initial_idea_text)}, "
            f"claims_length={len(claims_text)}, "
            f"prior_art_length={len(prior_art_summary)}, "
            f"novelty_length={len(novelty_text)}, "
            f"consistency_length={len(consistency_review)}, "
            f"market_length={len(market_assessment)}, "
            f"legal_length={len(legal_assessment)}, "
            f"personality_mode={mode.value}, "
            f"review_notes_length={len(notes_text)}, "
            f"domain_profile={domain_slug}"
        ),
        output_summary=f"summary_length={len(disclosure_summary)}",
        duration_ms=duration_ms,
    )

    return {
        "disclosure_summary": disclosure_summary,
        "current_step": "disclosure_summary",
        "personality_mode_used": mode.value,
    }
