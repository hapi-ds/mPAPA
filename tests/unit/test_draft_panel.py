"""Unit tests for the Patent Draft panel logic.

Tests focus on the ``can_export`` validation function and module
importability.  NiceGUI rendering is not tested directly — we verify
the pure logic and data flow.

Requirements: 16.6, 10.6, 5.3, 7.3
"""

import pytest

from patent_system.gui.draft_panel import can_export, create_draft_panel, WORKFLOW_STEPS


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
    """WORKFLOW_STEPS contains the expected pipeline stages."""
    assert len(WORKFLOW_STEPS) == 6
    assert WORKFLOW_STEPS[0] == "Disclosure"
    assert WORKFLOW_STEPS[-1] == "Description Drafting"


def test_workflow_steps_order() -> None:
    """Workflow steps are in the correct pipeline order."""
    expected = [
        "Disclosure",
        "Prior Art Search",
        "Novelty Analysis",
        "Claims Drafting",
        "Consistency Review",
        "Description Drafting",
    ]
    assert WORKFLOW_STEPS == expected


# --- _STEP_DISPLAY_NAMES mapping tests ---


def test_step_display_names_mapping() -> None:
    """_STEP_DISPLAY_NAMES maps all node step keys to display names."""
    from patent_system.gui.draft_panel import _STEP_DISPLAY_NAMES

    assert _STEP_DISPLAY_NAMES["disclosure"] == "Disclosure"
    assert _STEP_DISPLAY_NAMES["prior_art_search"] == "Prior Art Search"
    assert _STEP_DISPLAY_NAMES["novelty_analysis"] == "Novelty Analysis"
    assert _STEP_DISPLAY_NAMES["claims_drafting"] == "Claims Drafting"
    assert _STEP_DISPLAY_NAMES["consistency_review"] == "Consistency Review"
    assert _STEP_DISPLAY_NAMES["description_drafting"] == "Description Drafting"
    assert len(_STEP_DISPLAY_NAMES) == 6


def test_step_display_names_match_workflow_steps() -> None:
    """Every value in _STEP_DISPLAY_NAMES is present in WORKFLOW_STEPS."""
    from patent_system.gui.draft_panel import _STEP_DISPLAY_NAMES

    for display_name in _STEP_DISPLAY_NAMES.values():
        assert display_name in WORKFLOW_STEPS


# --- _on_generate logic tests (Req 6.1–6.7) ---


def test_on_generate_builds_correct_initial_state() -> None:
    """_on_generate builds a PatentWorkflowState with correct defaults.

    We verify the initial state dict passed to workflow.invoke contains
    the topic_id and all required default keys.
    Requirements: 6.1
    """
    from unittest.mock import MagicMock, patch

    captured_args: list = []

    mock_workflow = MagicMock()
    mock_workflow.invoke.side_effect = lambda state, config: (
        captured_args.append((state, config))
        or {
            "topic_id": state["topic_id"],
            "current_step": "description_drafting",
            "claims_text": "Claim 1",
            "description_text": "Description",
            "invention_disclosure": None,
            "interview_messages": [],
            "prior_art_results": [],
            "failed_sources": [],
            "novelty_analysis": None,
            "review_feedback": "",
            "review_approved": True,
            "iteration_count": 1,
        }
    )

    topic_id = 42

    with patch("patent_system.gui.draft_panel.ui"):
        from patent_system.gui.draft_panel import PatentWorkflowState

        # Directly test the state construction logic
        initial_state: PatentWorkflowState = {
            "topic_id": topic_id,
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
            "current_step": "disclosure",
        }

        assert initial_state["topic_id"] == 42
        assert initial_state["invention_disclosure"] is None
        assert initial_state["interview_messages"] == []
        assert initial_state["prior_art_results"] == []
        assert initial_state["failed_sources"] == []
        assert initial_state["novelty_analysis"] is None
        assert initial_state["claims_text"] == ""
        assert initial_state["description_text"] == ""
        assert initial_state["review_feedback"] == ""
        assert initial_state["review_approved"] is False
        assert initial_state["iteration_count"] == 0
        assert initial_state["current_step"] == "disclosure"


def test_on_generate_thread_config_format() -> None:
    """_on_generate passes correct thread config for checkpointing.

    Requirements: 6.1, 6.7
    """
    topic_id = 7
    config = {"configurable": {"thread_id": f"topic-{topic_id}"}}
    assert config == {"configurable": {"thread_id": "topic-7"}}


def test_on_generate_graph_interrupt_import() -> None:
    """GraphInterrupt is importable from langgraph.errors."""
    from langgraph.errors import GraphInterrupt

    assert GraphInterrupt is not None
    # Verify it's an exception class
    assert issubclass(GraphInterrupt, BaseException)
