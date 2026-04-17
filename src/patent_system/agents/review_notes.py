"""Review notes helper functions for building formatted review notes text.

Provides pure functions for formatting individual review notes and assembling
them into a single text block for injection into DSPy module inputs.

Requirements: 5.1, 5.2, 5.3, 5.5, 7.1, 7.2, 7.4, 8.1, 8.2
"""

import logging

from patent_system.db.repository import WORKFLOW_STEP_ORDER

logger = logging.getLogger(__name__)

# Human-readable display names for each workflow step key.
_STEP_DISPLAY_NAMES: dict[str, str] = {
    "initial_idea": "Initial Idea",
    "claims_drafting": "Claims Drafting",
    "prior_art_search": "Prior Art Search",
    "novelty_analysis": "Novelty Analysis",
    "consistency_review": "Consistency Review",
    "market_potential": "Market Potential",
    "legal_clarification": "Legal Clarification",
    "disclosure_summary": "Disclosure Summary",
    "patent_draft": "Patent Draft",
}


def format_single_note(step_key: str, notes_text: str) -> str:
    """Format a single step's review notes with a labeled header.

    Args:
        step_key: The canonical step key (e.g. "claims_drafting").
        notes_text: The user's review notes text.

    Returns:
        Formatted string: "[User Review Notes from <display_name>]: <notes_text>"
    """
    display_name = _STEP_DISPLAY_NAMES.get(step_key, step_key)
    return f"[User Review Notes from {display_name}]: {notes_text}"


def build_review_notes_text(
    review_notes: dict[str, str],
    current_step_key: str,
    mode: str,
) -> str:
    """Build the review notes text to inject into a DSPy module input.

    Args:
        review_notes: Dict mapping step_key → notes string from state.
        current_step_key: The step key being executed.
        mode: Either "rerun" (inject own notes) or "continue" (inject upstream notes).

    Returns:
        Formatted review notes string, or empty string if no applicable notes.
    """
    if mode == "rerun":
        notes = review_notes.get(current_step_key, "")
        if notes:
            return format_single_note(current_step_key, notes)
        return ""

    if mode != "continue":
        logger.warning(
            "Invalid review notes mode %r, treating as 'continue'",
            mode,
        )

    # "continue" mode (or invalid mode fallback): collect upstream notes
    parts: list[str] = []
    for step_key in WORKFLOW_STEP_ORDER:
        if step_key == current_step_key:
            break
        notes = review_notes.get(step_key, "")
        if notes:
            parts.append(format_single_note(step_key, notes))

    return "\n\n".join(parts)
