"""Legal Clarification Agent for the patent drafting pipeline.

Uses the DSPy LegalClarificationModule to assess IP ownership,
employment agreements, and prior art conflicts from the invention
disclosure, claims text, prior art summary, and novelty analysis.

Requirements: 8.1, 8.5
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
from patent_system.dspy_modules.modules import LegalClarificationModule
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


def legal_clarification_node(state: PatentWorkflowState) -> dict[str, Any]:
    """Run the Legal Clarification Agent.

    1. Extracts ``invention_disclosure``, ``claims_text``,
       ``prior_art_summary``, and ``novelty_analysis`` from state.
    2. Calls ``LegalClarificationModule`` to assess legal aspects.
    3. Logs agent invocation.
    4. Returns dict with ``legal_assessment`` and ``current_step``.

    Args:
        state: The current workflow state.

    Returns:
        Dict with ``legal_assessment`` (assessment string) and
        ``current_step`` set to ``"legal_clarification"``.
    """
    start = time.monotonic()

    disclosure = state.get("invention_disclosure")
    claims_text = state.get("claims_text", "")
    prior_art_summary = state.get("prior_art_summary", "")
    novelty = state.get("novelty_analysis")

    disclosure_text = _prepare_text(disclosure)
    novelty_text = _prepare_text(novelty)

    module = LegalClarificationModule()
    try:
        prediction = module(
            invention_disclosure=disclosure_text,
            claims_text=claims_text,
            prior_art_summary=prior_art_summary,
            novelty_analysis=novelty_text,
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

    legal_assessment = prediction.legal_assessment

    duration_ms = (time.monotonic() - start) * 1000

    log_agent_invocation(
        logger=logger,
        name="LegalClarificationAgent",
        input_summary=(
            f"disclosure_length={len(disclosure_text)}, "
            f"claims_length={len(claims_text)}, "
            f"prior_art_length={len(prior_art_summary)}, "
            f"novelty_length={len(novelty_text)}"
        ),
        output_summary=f"assessment_length={len(legal_assessment)}",
        duration_ms=duration_ms,
    )

    return {
        "legal_assessment": legal_assessment,
        "current_step": "legal_clarification",
    }
