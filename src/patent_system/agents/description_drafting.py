"""Description Drafter Agent for the patent drafting pipeline.

Uses the DSPy DraftDescriptionModule to generate a full patent
specification from approved claims and prior art results.

Generates sections: Technical Field, Background Art, Summary of
Invention, Detailed Description, Drawing Descriptions, and
Industrial Applicability.

Requirements: 7.1, 7.2, 7.4
"""

import json
import logging
import time
from typing import Any

import dspy
import httpx
import litellm.exceptions
import requests.exceptions

from patent_system.agents.personality import resolve_personality_mode
from patent_system.agents.state import PatentWorkflowState
from patent_system.dspy_modules.modules import DraftDescriptionModule, RefineClaimsModule
from patent_system.exceptions import LLMConnectionError
from patent_system.logging_config import log_agent_invocation

logger = logging.getLogger(__name__)


def _prepare_claims_text(state: PatentWorkflowState) -> str:
    """Extract the approved claims text from state.

    Args:
        state: The current workflow state.

    Returns:
        The claims string, or empty string if absent.
    """
    return state.get("claims_text", "") or ""


def _prepare_prior_art_summary(state: PatentWorkflowState) -> str:
    """Build a prior art summary string from the state's prior_art_results.

    Args:
        state: The current workflow state.

    Returns:
        A text summary of prior art results suitable for the DSPy module.
        Returns an empty string when no results are available.
    """
    results = state.get("prior_art_results") or []
    if not results:
        return ""
    try:
        return json.dumps(results, default=str)
    except (TypeError, ValueError):
        return str(results)


def _prepare_disclosure_text(state: PatentWorkflowState) -> str:
    """Serialize the invention disclosure dict from state.

    Args:
        state: The current workflow state.

    Returns:
        A string representation of the disclosure, or empty string.
    """
    disclosure = state.get("invention_disclosure")
    if not disclosure:
        return ""
    try:
        return json.dumps(disclosure, default=str)
    except (TypeError, ValueError):
        return str(disclosure)


def _has_analysis_feedback(state: PatentWorkflowState) -> bool:
    """Return True if the state contains feedback from analysis steps.

    Args:
        state: The current workflow state.

    Returns:
        True if at least one analysis field (novelty, consistency review,
        market, or legal) has non-empty content.
    """
    novelty = state.get("novelty_analysis")
    if novelty:
        if isinstance(novelty, str) and novelty.strip():
            return True
        if isinstance(novelty, dict):
            return True
    if (state.get("review_feedback") or "").strip():
        return True
    if (state.get("market_assessment") or "").strip():
        return True
    if (state.get("legal_assessment") or "").strip():
        return True
    return False


def _prepare_novelty_text(state: PatentWorkflowState) -> str:
    """Serialize the novelty analysis from state to a string.

    Args:
        state: The current workflow state.

    Returns:
        A string representation of the novelty analysis, or empty string.
    """
    novelty = state.get("novelty_analysis")
    if not novelty:
        return ""
    if isinstance(novelty, str):
        return novelty
    try:
        return json.dumps(novelty, indent=2, default=str)
    except (TypeError, ValueError):
        return str(novelty)


def description_drafting_node(state: PatentWorkflowState) -> dict[str, Any]:
    """Run the Description Drafter Agent.

    When analysis feedback is available (novelty analysis, consistency
    review, market potential, legal assessment), first refines the claims
    using a dedicated ``RefineClaimsModule``, then generates the full
    patent specification based on the refined claims.

    When no analysis feedback exists (e.g. first run), uses the original
    claims directly.

    1. Optionally refines claims via ``RefineClaimsModule`` when feedback
       is available from prior analysis steps.
    2. Uses DSPy ``DraftDescriptionModule`` to generate the full patent
       specification.
    3. Generates sections: Technical Field, Background Art, Summary of
       Invention, Detailed Description, Drawing Descriptions, Industrial
       Applicability.
    4. Logs agent invocation.
    5. Returns dict with ``description_text``, ``claims_text``,
       ``current_step``.

    Args:
        state: The current workflow state.

    Returns:
        Dict with ``description_text`` (generated specification string),
        ``claims_text`` (refined or original claims), and
        ``current_step`` set to ``"description_drafting"``.
    """
    start = time.monotonic()

    mode = resolve_personality_mode(state, "patent_draft")

    original_claims = _prepare_claims_text(state)
    prior_art_summary = _prepare_prior_art_summary(state)
    disclosure_text = _prepare_disclosure_text(state)

    # --- Step 1: Refine claims if analysis feedback is available ---
    claims = original_claims
    claims_refined = False

    if original_claims and _has_analysis_feedback(state):
        try:
            refine_module = RefineClaimsModule()
            refine_prediction = refine_module(
                original_claims=original_claims,
                invention_disclosure=disclosure_text,
                novelty_analysis=_prepare_novelty_text(state),
                consistency_review=state.get("review_feedback", "") or "",
                market_assessment=state.get("market_assessment", "") or "",
                legal_assessment=state.get("legal_assessment", "") or "",
                personality_mode=mode.value,
            )
            refined = refine_prediction.refined_claims
            if refined and refined.strip():
                claims = refined
                claims_refined = True
                logger.info(
                    "Claims refined for topic %d (original=%d chars, refined=%d chars)",
                    state.get("topic_id", 0),
                    len(original_claims),
                    len(claims),
                )
        except (
            requests.exceptions.ConnectionError,
            httpx.ConnectError,
            litellm.exceptions.APIConnectionError,
            ConnectionError,
            OSError,
        ) as exc:
            base_url = dspy.settings.lm.kwargs.get("api_base", "unknown") if dspy.settings.lm else "unknown"
            raise LLMConnectionError(
                f"LM Studio unreachable at {base_url}: {exc}"
            ) from exc
        except Exception:
            logger.warning(
                "Claims refinement failed for topic %d, using original claims",
                state.get("topic_id", 0),
                exc_info=True,
            )

    # --- Step 2: Generate description based on (refined) claims ---
    draft_module = DraftDescriptionModule()
    try:
        prediction = draft_module(
            claims=claims,
            prior_art_summary=prior_art_summary,
            invention_disclosure=disclosure_text,
            personality_mode=mode.value,
        )
    except (
        requests.exceptions.ConnectionError,
        httpx.ConnectError,
        litellm.exceptions.APIConnectionError,
        ConnectionError,
        OSError,
    ) as exc:
        base_url = dspy.settings.lm.kwargs.get("api_base", "unknown") if dspy.settings.lm else "unknown"
        raise LLMConnectionError(
            f"LM Studio unreachable at {base_url}: {exc}"
        ) from exc
    description_text = prediction.description_text

    duration_ms = (time.monotonic() - start) * 1000

    # Log agent invocation
    log_agent_invocation(
        logger=logger,
        name="DescriptionDrafterAgent",
        input_summary=(
            f"claims_length={len(claims)}, "
            f"claims_refined={claims_refined}, "
            f"prior_art_count={len(state.get('prior_art_results') or [])}, "
            f"disclosure_length={len(disclosure_text)}, "
            f"personality_mode={mode.value}"
        ),
        output_summary=f"description_length={len(description_text)}",
        duration_ms=duration_ms,
    )

    return {
        "description_text": description_text,
        "claims_text": claims,
        "current_step": "description_drafting",
        "personality_mode_used": mode.value,
    }
