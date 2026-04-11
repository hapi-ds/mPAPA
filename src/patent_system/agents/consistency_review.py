"""Consistency Reviewer Agent for the patent drafting pipeline.

Uses the DSPy ReviewConsistencyModule to check drafted claims against
the patent description for terminology consistency, completeness of
claim element descriptions, absence of contradictions, and proper
antecedent basis.

Requirements: 6.1
"""

import logging
import time
from typing import Any

import dspy
import httpx
import litellm.exceptions
import requests.exceptions

from patent_system.agents.state import PatentWorkflowState
from patent_system.dspy_modules.modules import ReviewConsistencyModule
from patent_system.exceptions import LLMConnectionError
from patent_system.logging_config import log_agent_invocation

logger = logging.getLogger(__name__)


def consistency_review_node(state: PatentWorkflowState) -> dict[str, Any]:
    """Run the Consistency Reviewer Agent.

    1. Uses DSPy ``ReviewConsistencyModule`` to check claims against
       the description.
    2. Checks for terminology consistency, completeness, contradictions,
       and antecedent basis.
    3. Returns feedback and approved/not-approved status.
    4. Logs agent invocation.
    5. Returns dict with ``review_feedback``, ``review_approved``,
       ``current_step``.

    Args:
        state: The current workflow state.

    Returns:
        Dict with ``review_feedback`` (feedback string from the
        reviewer), ``review_approved`` (bool indicating approval),
        and ``current_step`` set to ``"consistency_review"``.
    """
    start = time.monotonic()

    claims = state.get("claims_text", "")
    description = state.get("description_text", "")

    # Run the DSPy review module
    review_module = ReviewConsistencyModule()
    try:
        prediction = review_module(claims=claims, description=description)
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

    feedback = prediction.feedback
    approved = prediction.approved

    # Normalise approved to a bool (DSPy may return a string)
    if isinstance(approved, str):
        approved = approved.lower().strip() in ("true", "yes", "1")

    duration_ms = (time.monotonic() - start) * 1000

    # Log agent invocation
    log_agent_invocation(
        logger=logger,
        name="ConsistencyReviewerAgent",
        input_summary=(
            f"claims_length={len(claims)}, "
            f"description_length={len(description)}"
        ),
        output_summary=f"approved={approved}, feedback_length={len(feedback)}",
        duration_ms=duration_ms,
    )

    return {
        "review_feedback": feedback,
        "review_approved": approved,
        "current_step": "consistency_review",
    }
