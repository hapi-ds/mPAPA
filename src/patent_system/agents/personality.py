"""Personality mode definitions and prefix generation for agent nodes.

Provides the ``PersonalityMode`` enum, per-agent default mappings, prompt
prefix generation, prefix round-trip parsing, and state-based mode resolution.

Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 2.1, 2.2, 2.3, 2.4, 2.5, 3.2, 5.1
"""

import logging
import re
from enum import StrEnum

logger = logging.getLogger(__name__)


class PersonalityMode(StrEnum):
    """Predefined behavioural profiles controlling agent tone and rigor.

    Exactly three modes are supported.  Using ``StrEnum`` ensures values
    are plain strings and invalid values raise ``ValueError`` on construction.
    """

    CRITICAL = "critical"
    NEUTRAL = "neutral"
    INNOVATION_FRIENDLY = "innovation_friendly"


# ---------------------------------------------------------------------------
# Per-agent default personality modes
# ---------------------------------------------------------------------------

AGENT_PERSONALITY_DEFAULTS: dict[str, PersonalityMode] = {
    "initial_idea": PersonalityMode.NEUTRAL,
    "claims_drafting": PersonalityMode.NEUTRAL,
    "prior_art_search": PersonalityMode.NEUTRAL,
    "novelty_analysis": PersonalityMode.CRITICAL,
    "consistency_review": PersonalityMode.CRITICAL,
    "market_potential": PersonalityMode.NEUTRAL,
    "legal_clarification": PersonalityMode.CRITICAL,
    "disclosure_summary": PersonalityMode.NEUTRAL,
    "patent_draft": PersonalityMode.NEUTRAL,
}

# ---------------------------------------------------------------------------
# Prompt prefix strings (each >= 20 chars, tagged with [Personality: <mode>])
# ---------------------------------------------------------------------------

_PERSONALITY_PREFIXES: dict[PersonalityMode, str] = {
    PersonalityMode.CRITICAL: (
        "[Personality: critical] You are a skeptical, rigorous analyst. "
        "Question every assumption. Highlight weaknesses and gaps. "
        "Demand strong evidence before accepting any positive claim. "
        "Do not sugarcoat findings — be direct about problems and risks."
    ),
    PersonalityMode.NEUTRAL: (
        "[Personality: neutral] You are a balanced, objective analyst. "
        "Provide fair assessment without bias toward positive or negative framing. "
        "Present evidence on all sides and let the facts speak for themselves."
    ),
    PersonalityMode.INNOVATION_FRIENDLY: (
        "[Personality: innovation_friendly] You are a constructive, "
        "opportunity-focused analyst. "
        "Emphasize novel aspects, creative possibilities, and paths forward. "
        "Still note material risks and weaknesses, but frame them as challenges "
        "to address rather than reasons to abandon the approach."
    ),
}

# Regex for extracting the mode name from a ``[Personality: <mode>]`` tag.
_TAG_RE = re.compile(r"\[Personality:\s*(\w+)\]")


def generate_personality_prefix(mode: str | PersonalityMode) -> str:
    """Return the prompt prefix for the given personality mode.

    Args:
        mode: A ``PersonalityMode`` value or its string representation.
            Invalid values fall back to ``CRITICAL`` and log a warning.

    Returns:
        A non-empty prefix string (>= 20 characters).
    """
    try:
        resolved = PersonalityMode(mode)
    except ValueError:
        logger.warning(
            "Invalid personality mode %r — falling back to CRITICAL", mode
        )
        resolved = PersonalityMode.CRITICAL

    return _PERSONALITY_PREFIXES[resolved]


def parse_mode_from_prefix(prefix: str) -> PersonalityMode:
    """Extract the personality mode from a ``[Personality: ...]`` tag.

    Args:
        prefix: A string containing a ``[Personality: <mode_name>]`` tag.

    Returns:
        The corresponding ``PersonalityMode``.

    Raises:
        ValueError: If no valid tag is found in *prefix*.
    """
    match = _TAG_RE.search(prefix)
    if match is None:
        raise ValueError(f"No [Personality: ...] tag found in prefix: {prefix!r}")
    return PersonalityMode(match.group(1))


def resolve_personality_mode(
    state: dict,
    agent_name: str,
) -> PersonalityMode:
    """Resolve the active personality mode for an agent from workflow state.

    Resolution order:
    1. ``state["personality_modes"][agent_name]``
    2. ``AGENT_PERSONALITY_DEFAULTS[agent_name]``
    3. ``PersonalityMode.CRITICAL``

    Args:
        state: The ``PatentWorkflowState`` dict (or any mapping).
        agent_name: The canonical agent node name (e.g. ``"novelty_analysis"``).

    Returns:
        The resolved ``PersonalityMode``.
    """
    modes = state.get("personality_modes")
    if isinstance(modes, dict):
        raw = modes.get(agent_name)
        if raw is not None:
            try:
                return PersonalityMode(raw)
            except ValueError:
                logger.warning(
                    "Invalid personality mode %r for agent %r in state "
                    "— falling back to defaults",
                    raw,
                    agent_name,
                )

    # Fall back to per-agent defaults, then to CRITICAL.
    return AGENT_PERSONALITY_DEFAULTS.get(agent_name, PersonalityMode.CRITICAL)
