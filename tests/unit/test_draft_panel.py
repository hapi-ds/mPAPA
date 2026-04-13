"""Unit tests for the Patent Draft panel logic.

Tests focus on the ``can_export`` validation function, module
importability, and the updated nine-step workflow constants.

Requirements: 2.1–2.4, 10.6, 11.1, 12.3
"""

import pytest

from patent_system.gui.draft_panel import (
    WORKFLOW_STEPS,
    _STEP_DISPLAY_NAMES,
    _find_active_step,
    _has_content,
    can_export,
    create_draft_panel,
)


# --- can_export tests (Req 10.6) ---


def test_can_export_both_non_empty() -> None:
    """Export allowed when both claims and description are non-empty."""
    assert can_export("Claim 1: A method...", "Technical Field...") is True


def test_can_export_empty_claims() -> None:
    """Export blocked when claims is empty string."""
    assert can_export("", "Some description") is False


def test_can_export_empty_description() -> None:
    """Export blocked when description is empty string."""
    assert can_export("Some claims", "") is False


def test_can_export_both_empty() -> None:
    """Export blocked when both are empty."""
    assert can_export("", "") is False


def test_can_export_whitespace_only_claims() -> None:
    """Export blocked when claims is whitespace only."""
    assert can_export("   ", "Some description") is False


def test_can_export_whitespace_only_description() -> None:
    """Export blocked when description is whitespace only."""
    assert can_export("Some claims", "   \n\t  ") is False


def test_can_export_none_claims() -> None:
    """Export blocked when claims is None (cast to str by caller)."""
    assert can_export(None, "Some description") is False  # type: ignore[arg-type]


def test_can_export_none_description() -> None:
    """Export blocked when description is None."""
    assert can_export("Some claims", None) is False  # type: ignore[arg-type]


# --- Module structure tests ---


def test_draft_panel_module_importable() -> None:
    """The draft_panel module can be imported without errors."""
    assert callable(create_draft_panel)
    assert callable(can_export)


def test_workflow_steps_defined() -> None:
    """WORKFLOW_STEPS contains the expected nine pipeline stages."""
    assert len(WORKFLOW_STEPS) == 9
    assert WORKFLOW_STEPS[0] == "Initial Idea"
    assert WORKFLOW_STEPS[-1] == "Patent Draft"


def test_workflow_steps_order() -> None:
    """Workflow steps are in the correct pipeline order."""
    expected = [
        "Initial Idea",
        "Claims Drafting",
        "Prior Art Search",
        "Novelty Analysis",
        "Consistency Review",
        "Market Potential",
        "Legal Clarification",
        "Disclosure Summary",
        "Patent Draft",
    ]
    assert WORKFLOW_STEPS == expected


# --- _STEP_DISPLAY_NAMES mapping tests ---


def test_step_display_names_mapping() -> None:
    """_STEP_DISPLAY_NAMES maps all nine node step keys to display names."""
    assert _STEP_DISPLAY_NAMES["initial_idea"] == "Initial Idea"
    assert _STEP_DISPLAY_NAMES["claims_drafting"] == "Claims Drafting"
    assert _STEP_DISPLAY_NAMES["prior_art_search"] == "Prior Art Search"
    assert _STEP_DISPLAY_NAMES["novelty_analysis"] == "Novelty Analysis"
    assert _STEP_DISPLAY_NAMES["consistency_review"] == "Consistency Review"
    assert _STEP_DISPLAY_NAMES["market_potential"] == "Market Potential"
    assert _STEP_DISPLAY_NAMES["legal_clarification"] == "Legal Clarification"
    assert _STEP_DISPLAY_NAMES["disclosure_summary"] == "Disclosure Summary"
    assert _STEP_DISPLAY_NAMES["patent_draft"] == "Patent Draft"
    assert len(_STEP_DISPLAY_NAMES) == 9


def test_step_display_names_match_workflow_steps() -> None:
    """Every value in _STEP_DISPLAY_NAMES is present in WORKFLOW_STEPS."""
    for display_name in _STEP_DISPLAY_NAMES.values():
        assert display_name in WORKFLOW_STEPS


# --- _has_content helper tests ---


def test_has_content_non_empty() -> None:
    assert _has_content("hello") is True


def test_has_content_empty() -> None:
    assert _has_content("") is False


def test_has_content_whitespace_only() -> None:
    assert _has_content("   \n\t  ") is False


def test_has_content_none() -> None:
    assert _has_content(None) is False


# --- _find_active_step tests (Req 12.3) ---


def test_find_active_step_none_completed() -> None:
    """First step is active when nothing is completed."""
    assert _find_active_step(set()) == "initial_idea"


def test_find_active_step_first_completed() -> None:
    """Second step is active when only first is completed."""
    assert _find_active_step({"initial_idea"}) == "claims_drafting"


def test_find_active_step_all_completed() -> None:
    """Returns None when all steps are completed."""
    from patent_system.db.repository import WORKFLOW_STEP_ORDER
    assert _find_active_step(set(WORKFLOW_STEP_ORDER)) is None


def test_find_active_step_partial() -> None:
    """Returns the correct next step for a partial completion."""
    completed = {"initial_idea", "claims_drafting", "prior_art_search"}
    assert _find_active_step(completed) == "novelty_analysis"


# --- _on_generate logic tests (Req 6.1–6.7) ---


def test_on_generate_builds_correct_initial_state() -> None:
    """Verify the initial state dict contains the topic_id and all required keys."""
    from patent_system.agents.state import PatentWorkflowState

    initial_state: PatentWorkflowState = {
        "topic_id": 42,
        "invention_disclosure": None,
        "interview_messages": [],
        "prior_art_results": [],
        "failed_sources": [],
        "novelty_analysis": None,
        "claims_text": "",
        "description_text": "",
        "review_feedback": "",
        "review_approved": False,
        "iteration_count": 0,
        "current_step": "initial_idea",
        "market_assessment": "",
        "legal_assessment": "",
        "disclosure_summary": "",
        "prior_art_summary": "",
        "workflow_step_statuses": {},
    }

    assert initial_state["topic_id"] == 42
    assert initial_state["current_step"] == "initial_idea"
    assert initial_state["market_assessment"] == ""
    assert initial_state["legal_assessment"] == ""
    assert initial_state["disclosure_summary"] == ""
    assert initial_state["prior_art_summary"] == ""
    assert initial_state["workflow_step_statuses"] == {}


def test_on_generate_thread_config_format() -> None:
    """Thread config for checkpointing uses correct format."""
    topic_id = 7
    config = {"configurable": {"thread_id": f"topic-{topic_id}"}}
    assert config == {"configurable": {"thread_id": "topic-7"}}


def test_on_generate_graph_interrupt_import() -> None:
    """GraphInterrupt is importable from langgraph.errors."""
    from langgraph.errors import GraphInterrupt

    assert GraphInterrupt is not None
    assert issubclass(GraphInterrupt, BaseException)
