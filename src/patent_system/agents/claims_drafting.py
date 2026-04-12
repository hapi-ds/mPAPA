"""Claims Drafting Agent for the patent drafting pipeline.

Uses the DSPy DraftClaimsModule to generate patent claims in
European patent format (English) from the invention disclosure and
novelty analysis.

Requirements: 5.1, 5.2, 5.4
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
from patent_system.dspy_modules.modules import DraftClaimsModule
from patent_system.exceptions import LLMConnectionError
from patent_system.logging_config import log_agent_invocation

logger = logging.getLogger(__name__)


def _prepare_disclosure_text(disclosure: dict | None) -> str:
    """Serialize the invention disclosure dict to a text representation.

    Args:
        disclosure: The invention disclosure dict, or None.

    Returns:
        A string representation suitable for the DSPy module input.
        Returns an empty string when disclosure is None.
    """
    if not disclosure:
        return ""
    try:
        return json.dumps(disclosure, default=str)
    except (TypeError, ValueError):
        return str(disclosure)


def _prepare_novelty_text(novelty: dict | None) -> str:
    """Serialize the novelty analysis dict to a text representation.

    Args:
        novelty: The novelty analysis dict, or None.

    Returns:
        A string representation suitable for the DSPy module input.
        Returns an empty string when novelty is None.
    """
    if not novelty:
        return ""
    try:
        return json.dumps(novelty, default=str)
    except (TypeError, ValueError):
        return str(novelty)


def claims_drafting_node(state: PatentWorkflowState) -> dict[str, Any]:
    """Run the Claims Drafting Agent.

    1. Uses DSPy ``DraftClaimsModule`` to generate claims in
       European patent style (English).
    2. Takes ``invention_disclosure`` and ``novelty_analysis`` from
       state as input.
    3. Increments ``iteration_count``.
    4. Logs agent invocation.
    5. Returns dict with ``claims_text``, ``iteration_count``,
       ``current_step``.

    Args:
        state: The current workflow state.

    Returns:
        Dict with ``claims_text`` (generated claims string),
        ``iteration_count`` (incremented), and ``current_step``
        set to ``"claims_drafting"``.
    """
    start = time.monotonic()

    disclosure = state.get("invention_disclosure")
    novelty = state.get("novelty_analysis")
    iteration_count = state.get("iteration_count", 0)

    # Prepare text inputs for the DSPy module
    disclosure_text = _prepare_disclosure_text(disclosure)
    novelty_text = _prepare_novelty_text(novelty)

    # Generate claims via DSPy
    draft_module = DraftClaimsModule()
    try:
        prediction = draft_module(
            invention_disclosure=disclosure_text,
            novelty_analysis=novelty_text,
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
    claims_text = prediction.claims_text

    # Increment iteration count
    new_iteration_count = iteration_count + 1

    duration_ms = (time.monotonic() - start) * 1000

    # Log agent invocation
    log_agent_invocation(
        logger=logger,
        name="ClaimsDraftingAgent",
        input_summary=(
            f"disclosure_length={len(disclosure_text)}, "
            f"novelty_length={len(novelty_text)}, "
            f"iteration={new_iteration_count}"
        ),
        output_summary=f"claims_length={len(claims_text)}",
        duration_ms=duration_ms,
    )

    return {
        "claims_text": claims_text,
        "iteration_count": new_iteration_count,
        "current_step": "claims_drafting",
    }
