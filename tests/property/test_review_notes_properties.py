"""Property-based tests for review notes persistence.

Feature: step-review-notes
"""

import sqlite3

from hypothesis import given, settings
from hypothesis import strategies as st

from patent_system.db.repository import VALID_STEP_KEYS, WorkflowStepRepository
from patent_system.db.schema import init_schema


# Strategy for valid step keys
step_key_strategy = st.sampled_from(sorted(VALID_STEP_KEYS))

# Strategy for arbitrary text content (including empty strings and unicode)
text_strategy = st.text(min_size=0, max_size=500)


def _fresh_db() -> sqlite3.Connection:
    """Create a fresh in-memory SQLite connection with full schema."""
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    init_schema(conn)
    return conn


class TestReviewNotesPersistenceRoundTrip:
    """Property 1: Review notes persistence round-trip.

    For any valid step_key, content string, and review_notes string,
    upsert then get_step returns the same review_notes.

    **Validates: Requirements 1.2, 1.3**
    """

    @given(
        step_key=step_key_strategy,
        content=text_strategy,
        review_notes=text_strategy,
    )
    @settings(max_examples=100)
    def test_upsert_then_get_step_preserves_review_notes(
        self,
        step_key: str,
        content: str,
        review_notes: str,
    ) -> None:
        """Upsert a step with review_notes, then get_step returns the same value."""
        conn = _fresh_db()
        try:
            cursor = conn.execute(
                "INSERT INTO topics (name) VALUES (?)", ("test-topic",)
            )
            conn.commit()
            topic_id: int = cursor.lastrowid  # type: ignore[assignment]

            repo = WorkflowStepRepository(conn)

            repo.upsert(
                topic_id=topic_id,
                step_key=step_key,
                content=content,
                status="completed",
                review_notes=review_notes,
            )

            result = repo.get_step(topic_id, step_key)
            assert result is not None, "get_step should return a dict after upsert"
            assert result["review_notes"] == review_notes, (
                f"Expected review_notes={review_notes!r}, got {result['review_notes']!r}"
            )
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Properties 2–6: Review notes helper functions
# ---------------------------------------------------------------------------

from patent_system.agents.review_notes import (
    _STEP_DISPLAY_NAMES,
    build_review_notes_text,
    format_single_note,
)
from patent_system.db.repository import WORKFLOW_STEP_ORDER

# Strategy for non-empty text (used where empty strings are excluded)
non_empty_text_strategy = st.text(min_size=1, max_size=500)

# Strategy for review_notes dicts: maps valid step keys to arbitrary text
review_notes_strategy = st.dictionaries(
    keys=step_key_strategy,
    values=text_strategy,
    min_size=0,
    max_size=len(WORKFLOW_STEP_ORDER),
)

# Strategy for mode values (rerun or continue)
mode_strategy = st.sampled_from(["rerun", "continue"])


class TestReviewNotesFormattingContainsStepLabel:
    """Property 2: Review notes formatting contains step label.

    For any valid step_key and non-empty notes string, format_single_note
    returns a string containing [User Review Notes from <display_name>]:

    **Validates: Requirements 5.3, 7.2**
    """

    @given(
        step_key=step_key_strategy,
        notes=non_empty_text_strategy,
    )
    @settings(max_examples=100)
    def test_format_single_note_contains_step_label(
        self,
        step_key: str,
        notes: str,
    ) -> None:
        result = format_single_note(step_key, notes)
        display_name = _STEP_DISPLAY_NAMES[step_key]
        expected_label = f"[User Review Notes from {display_name}]:"
        assert expected_label in result


class TestRerunModeInjectsOnlyCurrentStepNotes:
    """Property 3: Rerun mode injects only current step's notes.

    For any review_notes dict and valid current_step_key,
    build_review_notes_text in rerun mode returns only the current step's
    notes and no other step's notes.

    **Validates: Requirements 5.1, 8.1, 8.3**
    """

    @given(
        review_notes=review_notes_strategy,
        current_step_key=step_key_strategy,
    )
    @settings(max_examples=100)
    def test_rerun_mode_only_current_step(
        self,
        review_notes: dict[str, str],
        current_step_key: str,
    ) -> None:
        result = build_review_notes_text(review_notes, current_step_key, "rerun")

        current_notes = review_notes.get(current_step_key, "")
        if current_notes:
            # Result should contain the current step's display name
            display_name = _STEP_DISPLAY_NAMES[current_step_key]
            assert f"[User Review Notes from {display_name}]:" in result
            assert current_notes in result
        else:
            assert result == ""

        # Result must NOT contain any other step's display name
        for other_key in WORKFLOW_STEP_ORDER:
            if other_key == current_step_key:
                continue
            other_display = _STEP_DISPLAY_NAMES[other_key]
            assert f"[User Review Notes from {other_display}]:" not in result


class TestContinueModeInjectsOnlyUpstreamNotes:
    """Property 4: Continue mode injects only upstream notes.

    For any review_notes dict and valid current_step_key,
    build_review_notes_text in continue mode returns only upstream notes
    in canonical order, not the current step's notes.

    **Validates: Requirements 5.2, 7.1, 7.4, 8.2, 8.4**
    """

    @given(
        review_notes=review_notes_strategy,
        current_step_key=step_key_strategy,
    )
    @settings(max_examples=100)
    def test_continue_mode_only_upstream_notes(
        self,
        review_notes: dict[str, str],
        current_step_key: str,
    ) -> None:
        result = build_review_notes_text(review_notes, current_step_key, "continue")

        current_idx = WORKFLOW_STEP_ORDER.index(current_step_key)
        upstream_keys = WORKFLOW_STEP_ORDER[:current_idx]

        # Current step's display name must NOT appear
        current_display = _STEP_DISPLAY_NAMES[current_step_key]
        assert f"[User Review Notes from {current_display}]:" not in result

        # Steps after current must NOT appear
        for after_key in WORKFLOW_STEP_ORDER[current_idx + 1 :]:
            after_display = _STEP_DISPLAY_NAMES[after_key]
            assert f"[User Review Notes from {after_display}]:" not in result

        # Upstream steps with non-empty notes should appear in order
        expected_parts: list[str] = []
        for uk in upstream_keys:
            notes = review_notes.get(uk, "")
            if notes:
                expected_parts.append(
                    f"[User Review Notes from {_STEP_DISPLAY_NAMES[uk]}]: {notes}"
                )

        if expected_parts:
            assert result == "\n\n".join(expected_parts)
        else:
            assert result == ""


class TestEmptyReviewNotesProduceNoInjection:
    """Property 5: Empty review notes produce no injection.

    For any review_notes dict where all values are empty strings,
    build_review_notes_text returns an empty string.

    **Validates: Requirements 5.6, 6.3, 7.3**
    """

    @given(
        step_key=step_key_strategy,
        mode=mode_strategy,
    )
    @settings(max_examples=100)
    def test_all_empty_notes_produce_empty_string(
        self,
        step_key: str,
        mode: str,
    ) -> None:
        # Build a dict with all keys mapped to empty strings
        empty_notes = {k: "" for k in WORKFLOW_STEP_ORDER}
        result = build_review_notes_text(empty_notes, step_key, mode)
        assert result == ""


class TestDeterministicFormatting:
    """Property 6: Review notes formatting is deterministic.

    For any review_notes dict, step_key, and mode, calling
    build_review_notes_text twice returns identical strings.

    **Validates: Requirements 6.4**
    """

    @given(
        review_notes=review_notes_strategy,
        step_key=step_key_strategy,
        mode=mode_strategy,
    )
    @settings(max_examples=100)
    def test_deterministic_output(
        self,
        review_notes: dict[str, str],
        step_key: str,
        mode: str,
    ) -> None:
        result1 = build_review_notes_text(review_notes, step_key, mode)
        result2 = build_review_notes_text(review_notes, step_key, mode)
        assert result1 == result2
