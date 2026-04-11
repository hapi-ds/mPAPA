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

from patent_system.agents.state import PatentWorkflowState
from patent_system.dspy_modules.modules import DraftDescriptionModule
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


def description_drafting_node(state: PatentWorkflowState) -> dict[str, Any]:
    """Run the Description Drafter Agent.

    1. Uses DSPy ``DraftDescriptionModule`` to generate the full patent
       specification.
    2. Generates sections: Technical Field, Background Art, Summary of
       Invention, Detailed Description, Drawing Descriptions, Industrial
       Applicability.
    3. Based on approved claims and prior art results.
    4. Logs agent invocation.
    5. Returns dict with ``description_text``, ``current_step``.

    Args:
        state: The current workflow state.

    Returns:
        Dict with ``description_text`` (generated specification string)
        and ``current_step`` set to ``"description_drafting"``.
    """
    start = time.monotonic()

    claims = _prepare_claims_text(state)
    prior_art_summary = _prepare_prior_art_summary(state)
    disclosure_text = _prepare_disclosure_text(state)

    # Generate description via DSPy
    draft_module = DraftDescriptionModule()
    try:
        prediction = draft_module(
            claims=claims,
            prior_art_summary=prior_art_summary,
            invention_disclosure=disclosure_text,
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
            f"prior_art_count={len(state.get('prior_art_results') or [])}, "
            f"disclosure_length={len(disclosure_text)}"
        ),
        output_summary=f"description_length={len(description_text)}",
        duration_ms=duration_ms,
    )

    return {
        "description_text": description_text,
        "current_step": "description_drafting",
    }
