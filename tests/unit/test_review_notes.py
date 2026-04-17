"""Unit tests for the review notes helper module.

Requirements: 5.1–5.6, 7.1–7.4, 8.1–8.4
"""

from patent_system.agents.review_notes import (
    _STEP_DISPLAY_NAMES,
    build_review_notes_text,
    format_single_note,
)
from patent_system.db.repository import WORKFLOW_STEP_ORDER


class TestFormatSingleNote:
    """Tests for format_single_note with each known step_key."""

    def test_each_known_step_key(self) -> None:
        """format_single_note should produce the correct label for every step."""
        for step_key, display_name in _STEP_DISPLAY_NAMES.items():
            result = format_single_note(step_key, "some feedback")
            assert result == f"[User Review Notes from {display_name}]: some feedback"

    def test_unknown_step_key_falls_back_to_raw_key(self) -> None:
        """format_single_note should use the raw key when display name is missing."""
        result = format_single_note("unknown_step", "feedback text")
        assert result == "[User Review Notes from unknown_step]: feedback text"

    def test_empty_notes_text(self) -> None:
        """format_single_note should work with empty notes text."""
        result = format_single_note("claims_drafting", "")
        assert result == "[User Review Notes from Claims Drafting]: "


class TestBuildReviewNotesTextRerunMode:
    """Tests for build_review_notes_text in rerun mode."""

    def test_rerun_with_non_empty_notes_for_current_step(self) -> None:
        """Rerun mode should return formatted note for the current step."""
        notes = {"claims_drafting": "Please revise claim 3"}
        result = build_review_notes_text(notes, "claims_drafting", "rerun")
        assert result == "[User Review Notes from Claims Drafting]: Please revise claim 3"

    def test_rerun_with_empty_notes_for_current_step(self) -> None:
        """Rerun mode should return empty string when current step has no notes."""
        notes = {"claims_drafting": ""}
        result = build_review_notes_text(notes, "claims_drafting", "rerun")
        assert result == ""

    def test_rerun_with_missing_current_step_key(self) -> None:
        """Rerun mode should return empty string when current step is not in dict."""
        notes = {"novelty_analysis": "some notes"}
        result = build_review_notes_text(notes, "claims_drafting", "rerun")
        assert result == ""

    def test_rerun_ignores_other_steps(self) -> None:
        """Rerun mode should not include notes from other steps."""
        notes = {
            "claims_drafting": "note on claims",
            "novelty_analysis": "note on novelty",
            "prior_art_search": "note on prior art",
        }
        result = build_review_notes_text(notes, "claims_drafting", "rerun")
        assert "Claims Drafting" in result
        assert "Novelty Analysis" not in result
        assert "Prior Art Search" not in result


class TestBuildReviewNotesTextContinueMode:
    """Tests for build_review_notes_text in continue mode."""

    def test_continue_with_multiple_upstream_notes(self) -> None:
        """Continue mode should return all upstream notes in canonical order."""
        notes = {
            "initial_idea": "refine the idea",
            "claims_drafting": "revise claim 3",
            "prior_art_search": "add more references",
        }
        result = build_review_notes_text(notes, "novelty_analysis", "continue")
        expected = (
            "[User Review Notes from Initial Idea]: refine the idea\n\n"
            "[User Review Notes from Claims Drafting]: revise claim 3\n\n"
            "[User Review Notes from Prior Art Search]: add more references"
        )
        assert result == expected

    def test_continue_with_no_upstream_notes(self) -> None:
        """Continue mode should return empty string when no upstream notes exist."""
        notes = {"novelty_analysis": "note on current step"}
        result = build_review_notes_text(notes, "novelty_analysis", "continue")
        assert result == ""

    def test_continue_excludes_current_step(self) -> None:
        """Continue mode should not include the current step's notes."""
        notes = {
            "claims_drafting": "upstream note",
            "prior_art_search": "current step note",
        }
        result = build_review_notes_text(notes, "prior_art_search", "continue")
        assert "Claims Drafting" in result
        assert "Prior Art Search" not in result

    def test_continue_excludes_downstream_steps(self) -> None:
        """Continue mode should not include notes from steps after current."""
        notes = {
            "initial_idea": "upstream",
            "patent_draft": "downstream note",
        }
        result = build_review_notes_text(notes, "claims_drafting", "continue")
        assert "Initial Idea" in result
        assert "Patent Draft" not in result

    def test_continue_skips_empty_upstream_notes(self) -> None:
        """Continue mode should skip upstream steps with empty notes."""
        notes = {
            "initial_idea": "",
            "claims_drafting": "has content",
            "prior_art_search": "",
        }
        result = build_review_notes_text(notes, "novelty_analysis", "continue")
        assert result == "[User Review Notes from Claims Drafting]: has content"

    def test_continue_for_first_step_returns_empty(self) -> None:
        """Continue mode for the first step should return empty (no upstream)."""
        notes = {"initial_idea": "some note"}
        result = build_review_notes_text(notes, "initial_idea", "continue")
        assert result == ""

    def test_continue_for_last_step_collects_all(self) -> None:
        """Continue mode for patent_draft should collect all prior steps."""
        notes = {
            "initial_idea": "idea note",
            "claims_drafting": "claims note",
            "prior_art_search": "prior art note",
            "novelty_analysis": "novelty note",
            "consistency_review": "consistency note",
            "market_potential": "market note",
            "legal_clarification": "legal note",
            "disclosure_summary": "summary note",
        }
        result = build_review_notes_text(notes, "patent_draft", "continue")
        # All 8 upstream steps should appear
        for key in WORKFLOW_STEP_ORDER[:-1]:
            display = _STEP_DISPLAY_NAMES[key]
            assert f"[User Review Notes from {display}]:" in result


class TestBuildReviewNotesTextEdgeCases:
    """Tests for edge cases in build_review_notes_text."""

    def test_unknown_step_key_in_rerun(self) -> None:
        """Rerun mode with unknown step_key should return empty if not in dict."""
        notes = {"claims_drafting": "some note"}
        result = build_review_notes_text(notes, "unknown_step", "rerun")
        assert result == ""

    def test_unknown_step_key_in_continue(self) -> None:
        """Continue mode with unknown step_key collects nothing (not in WORKFLOW_STEP_ORDER)."""
        # If the key isn't in WORKFLOW_STEP_ORDER, the loop never hits `break`,
        # so all steps are collected as "upstream".
        notes = {"claims_drafting": "note"}
        result = build_review_notes_text(notes, "unknown_step", "continue")
        # Since "unknown_step" is never found in WORKFLOW_STEP_ORDER,
        # the for loop completes without breaking, collecting all steps.
        assert "Claims Drafting" in result

    def test_invalid_mode_treats_as_continue(self) -> None:
        """Invalid mode should be treated as continue mode."""
        notes = {
            "initial_idea": "upstream note",
            "claims_drafting": "current note",
        }
        result = build_review_notes_text(notes, "claims_drafting", "invalid_mode")
        # Should behave like continue: only upstream notes
        assert "Initial Idea" in result
        assert "Claims Drafting" not in result

    def test_empty_dict_returns_empty_string(self) -> None:
        """Empty review_notes dict should always return empty string."""
        assert build_review_notes_text({}, "claims_drafting", "rerun") == ""
        assert build_review_notes_text({}, "claims_drafting", "continue") == ""
