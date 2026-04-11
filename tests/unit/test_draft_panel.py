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
