"""Market Potential Agent for the patent drafting pipeline.

Uses the DSPy MarketPotentialModule to assess economic viability
and market potential of an invention from the invention disclosure,
claims text, and novelty analysis.

Requirements: 7.1, 7.5
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
from patent_system.dspy_modules.modules import MarketPotentialModule
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


def market_potential_node(state: PatentWorkflowState) -> dict[str, Any]:
    """Run the Market Potential Agent.

    1. Extracts ``invention_disclosure``, ``claims_text``, and
       ``novelty_analysis`` from state.
    2. Calls ``MarketPotentialModule`` to assess market potential.
    3. Logs agent invocation.
    4. Returns dict with ``market_assessment`` and ``current_step``.

    Args:
        state: The current workflow state.

    Returns:
        Dict with ``market_assessment`` (assessment string) and
        ``current_step`` set to ``"market_potential"``.
    """
    start = time.monotonic()

    mode = resolve_personality_mode(state, "market_potential")

    disclosure = state.get("invention_disclosure")
    claims_text = state.get("claims_text", "")
    novelty = state.get("novelty_analysis")

    disclosure_text = _prepare_text(disclosure)
    novelty_text = _prepare_text(novelty)

    module = MarketPotentialModule()
    try:
        prediction = module(
            invention_disclosure=disclosure_text,
            claims_text=claims_text,
            novelty_analysis=novelty_text,
            personality_mode=mode.value,
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

    market_assessment = prediction.market_assessment

    duration_ms = (time.monotonic() - start) * 1000

    log_agent_invocation(
        logger=logger,
        name="MarketPotentialAgent",
        input_summary=(
            f"disclosure_length={len(disclosure_text)}, "
            f"claims_length={len(claims_text)}, "
            f"novelty_length={len(novelty_text)}, "
            f"personality_mode={mode.value}"
        ),
        output_summary=f"assessment_length={len(market_assessment)}",
        duration_ms=duration_ms,
    )

    return {
        "market_assessment": market_assessment,
        "current_step": "market_potential",
        "personality_mode_used": mode.value,
    }
