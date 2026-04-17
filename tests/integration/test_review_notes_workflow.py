"""Integration tests for review notes flow through agent workflow nodes.

Verifies that review notes set in state are correctly read by agent
nodes, passed to DSPy modules via build_review_notes_text, and that
the correct mode behavior (continue vs rerun) is applied.

Requirements: 5.1, 5.2, 7.1, 8.1–8.4
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import dspy
import pytest

from patent_system.agents.claims_drafting import claims_drafting_node
from patent_system.agents.consistency_review import consistency_review_node
from patent_system.agents.description_drafting import description_drafting_node
from patent_system.agents.disclosure_summary import disclosure_summary_node
from patent_system.agents.legal_clarification import legal_clarification_node
from patent_system.agents.market_potential import market_potential_node
from patent_system.agents.novelty_analysis import novelty_analysis_node


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(**overrides) -> dict:
    """Create a minimal PatentWorkflowState dict for testing."""
    base: dict = {
        "topic_id": 1,
        "invention_disclosure": {
            "technical_problem": "Slow processing",
            "novel_features": ["Feature A"],
            "implementation_details": "Uses GPU",
            "potential_variations": ["CPU fallback"],
        },
        "interview_messages": [],
        "prior_art_results": [],
        "prior_art_summary": "Some prior art summary",
        "failed_sources": [],
        "novelty_analysis": "Novel: Feature A",
        "claims_text": "Claim 1: A method for fast processing.",
        "description_text": "Technical Field: Processing systems.",
        "review_feedback": "Looks consistent",
        "review_approved": False,
        "iteration_count": 0,
        "current_step": "",
        "personality_modes": {},
        "market_assessment": "High potential",
        "legal_assessment": "No issues found",
        "disclosure_summary": "",
        "workflow_step_statuses": {},
        "review_notes": {},
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Test: Agent node with non-empty review_notes in continue mode
# ---------------------------------------------------------------------------


class TestAgentNodesWithReviewNotesContinueMode:
    """Verify that agent nodes pass upstream review notes to DSPy modules
    when review_notes are present in state (default continue mode)."""

    @patch("patent_system.agents.claims_drafting.DraftClaimsModule")
    def test_claims_drafting_with_review_notes(self, mock_cls):
        """Claims drafting receives upstream notes (initial_idea) in continue mode."""
        mock_instance = MagicMock()
        mock_instance.return_value = dspy.Prediction(claims_text="mocked claims")
        mock_cls.return_value = mock_instance

        state = _make_state(
            review_notes={
                "initial_idea": "refine the idea",
                "claims_drafting": "revise claim 3",
            },
        )
        result = claims_drafting_node(state)

        call_kwargs = mock_instance.call_args.kwargs
        # In continue mode, should have upstream notes (initial_idea) but NOT own notes
        assert call_kwargs.get("review_notes_text") is not None
        assert "Initial Idea" in call_kwargs["review_notes_text"]
        assert "refine the idea" in call_kwargs["review_notes_text"]
        # Own notes should NOT be included in continue mode
        assert "revise claim 3" not in call_kwargs["review_notes_text"]
        assert result["claims_text"] == "mocked claims"

    @patch("patent_system.agents.novelty_analysis.NoveltyAnalysisModule")
    def test_novelty_analysis_with_review_notes(self, mock_cls):
        """Novelty analysis receives upstream notes in continue mode."""
        mock_instance = MagicMock()
        mock_instance.return_value = dspy.Prediction(novelty_assessment="mocked assessment")
        mock_cls.return_value = mock_instance

        state = _make_state(
            review_notes={
                "initial_idea": "focus on GPU aspect",
                "claims_drafting": "strengthen claim 1",
                "prior_art_search": "check patent X",
            },
        )
        result = novelty_analysis_node(state)

        call_kwargs = mock_instance.call_args.kwargs
        notes_text = call_kwargs.get("review_notes_text")
        assert notes_text is not None
        # Should include upstream notes
        assert "Initial Idea" in notes_text
        assert "Claims Drafting" in notes_text
        assert "Prior Art Search" in notes_text
        assert result["novelty_analysis"] == "mocked assessment"

    @patch("patent_system.agents.consistency_review.ReviewConsistencyModule")
    def test_consistency_review_with_review_notes(self, mock_cls):
        """Consistency review receives upstream notes in continue mode."""
        mock_instance = MagicMock()
        mock_instance.return_value = dspy.Prediction(feedback="OK", approved=True)
        mock_cls.return_value = mock_instance

        state = _make_state(
            review_notes={
                "claims_drafting": "check dependent claims",
                "novelty_analysis": "novel aspect confirmed",
            },
        )
        result = consistency_review_node(state)

        call_kwargs = mock_instance.call_args.kwargs
        notes_text = call_kwargs.get("review_notes_text")
        assert notes_text is not None
        assert "Claims Drafting" in notes_text
        assert "Novelty Analysis" in notes_text

    @patch("patent_system.agents.market_potential.MarketPotentialModule")
    def test_market_potential_with_review_notes(self, mock_cls):
        """Market potential receives upstream notes in continue mode."""
        mock_instance = MagicMock()
        mock_instance.return_value = dspy.Prediction(market_assessment="mocked market")
        mock_cls.return_value = mock_instance

        state = _make_state(
            review_notes={"claims_drafting": "focus on industrial use"},
        )
        result = market_potential_node(state)

        call_kwargs = mock_instance.call_args.kwargs
        notes_text = call_kwargs.get("review_notes_text")
        assert notes_text is not None
        assert "Claims Drafting" in notes_text
        assert "focus on industrial use" in notes_text

    @patch("patent_system.agents.legal_clarification.LegalClarificationModule")
    def test_legal_clarification_with_review_notes(self, mock_cls):
        """Legal clarification receives upstream notes in continue mode."""
        mock_instance = MagicMock()
        mock_instance.return_value = dspy.Prediction(legal_assessment="mocked legal")
        mock_cls.return_value = mock_instance

        state = _make_state(
            review_notes={"novelty_analysis": "check prior art conflict"},
        )
        result = legal_clarification_node(state)

        call_kwargs = mock_instance.call_args.kwargs
        notes_text = call_kwargs.get("review_notes_text")
        assert notes_text is not None
        assert "Novelty Analysis" in notes_text

    @patch("patent_system.agents.disclosure_summary.DisclosureSummaryModule")
    def test_disclosure_summary_with_review_notes(self, mock_cls):
        """Disclosure summary receives upstream notes in continue mode."""
        mock_instance = MagicMock()
        mock_instance.return_value = dspy.Prediction(disclosure_summary="mocked summary")
        mock_cls.return_value = mock_instance

        state = _make_state(
            review_notes={
                "claims_drafting": "revise claim 3",
                "legal_clarification": "IP ownership clear",
            },
        )
        result = disclosure_summary_node(state)

        call_kwargs = mock_instance.call_args.kwargs
        notes_text = call_kwargs.get("review_notes_text")
        assert notes_text is not None
        assert "Claims Drafting" in notes_text
        assert "Legal Clarification" in notes_text


# ---------------------------------------------------------------------------
# Test: Agent node with empty review_notes in state
# ---------------------------------------------------------------------------


class TestAgentNodesWithEmptyReviewNotes:
    """Verify that agent nodes pass None for review_notes_text when
    review_notes dict is empty or absent."""

    @patch("patent_system.agents.claims_drafting.DraftClaimsModule")
    def test_claims_drafting_empty_review_notes(self, mock_cls):
        mock_instance = MagicMock()
        mock_instance.return_value = dspy.Prediction(claims_text="claims")
        mock_cls.return_value = mock_instance

        state = _make_state(review_notes={})
        claims_drafting_node(state)

        call_kwargs = mock_instance.call_args.kwargs
        assert call_kwargs.get("review_notes_text") is None

    @patch("patent_system.agents.novelty_analysis.NoveltyAnalysisModule")
    def test_novelty_analysis_empty_review_notes(self, mock_cls):
        mock_instance = MagicMock()
        mock_instance.return_value = dspy.Prediction(novelty_assessment="assessment")
        mock_cls.return_value = mock_instance

        state = _make_state(review_notes={})
        novelty_analysis_node(state)

        call_kwargs = mock_instance.call_args.kwargs
        assert call_kwargs.get("review_notes_text") is None

    @patch("patent_system.agents.claims_drafting.DraftClaimsModule")
    def test_claims_drafting_missing_review_notes_key(self, mock_cls):
        """When review_notes key is absent from state, should default to empty."""
        mock_instance = MagicMock()
        mock_instance.return_value = dspy.Prediction(claims_text="claims")
        mock_cls.return_value = mock_instance

        state = _make_state()
        # Remove review_notes key entirely
        del state["review_notes"]
        claims_drafting_node(state)

        call_kwargs = mock_instance.call_args.kwargs
        assert call_kwargs.get("review_notes_text") is None


# ---------------------------------------------------------------------------
# Test: Agent node with review_notes_mode="rerun"
# ---------------------------------------------------------------------------


class TestAgentNodesRerunMode:
    """Verify that agent nodes in rerun mode pass only the current step's
    own notes to the DSPy module."""

    @patch("patent_system.agents.claims_drafting.DraftClaimsModule")
    def test_claims_drafting_rerun_mode(self, mock_cls):
        """In rerun mode, claims_drafting should receive its own notes only."""
        mock_instance = MagicMock()
        mock_instance.return_value = dspy.Prediction(claims_text="revised claims")
        mock_cls.return_value = mock_instance

        state = _make_state(
            review_notes={
                "initial_idea": "refine the idea",
                "claims_drafting": "revise claim 3",
            },
        )
        result = claims_drafting_node(state, review_notes_mode="rerun")

        call_kwargs = mock_instance.call_args.kwargs
        notes_text = call_kwargs.get("review_notes_text")
        assert notes_text is not None
        # Should contain own notes
        assert "Claims Drafting" in notes_text
        assert "revise claim 3" in notes_text
        # Should NOT contain upstream notes
        assert "Initial Idea" not in notes_text
        assert "refine the idea" not in notes_text

    @patch("patent_system.agents.consistency_review.ReviewConsistencyModule")
    def test_consistency_review_rerun_mode(self, mock_cls):
        """In rerun mode, consistency_review should receive its own notes only."""
        mock_instance = MagicMock()
        mock_instance.return_value = dspy.Prediction(feedback="revised feedback", approved=True)
        mock_cls.return_value = mock_instance

        state = _make_state(
            review_notes={
                "claims_drafting": "upstream note",
                "consistency_review": "check terminology again",
            },
        )
        result = consistency_review_node(state, review_notes_mode="rerun")

        call_kwargs = mock_instance.call_args.kwargs
        notes_text = call_kwargs.get("review_notes_text")
        assert notes_text is not None
        assert "Consistency Review" in notes_text
        assert "check terminology again" in notes_text
        assert "upstream note" not in notes_text

    @patch("patent_system.agents.claims_drafting.DraftClaimsModule")
    def test_rerun_mode_with_no_own_notes(self, mock_cls):
        """In rerun mode with no own notes, review_notes_text should be None."""
        mock_instance = MagicMock()
        mock_instance.return_value = dspy.Prediction(claims_text="claims")
        mock_cls.return_value = mock_instance

        state = _make_state(
            review_notes={"initial_idea": "some upstream note"},
        )
        claims_drafting_node(state, review_notes_mode="rerun")

        call_kwargs = mock_instance.call_args.kwargs
        assert call_kwargs.get("review_notes_text") is None


# ---------------------------------------------------------------------------
# Test: description_drafting_node accumulates all upstream notes
# ---------------------------------------------------------------------------


class TestDescriptionDraftingAccumulatesNotes:
    """Verify that description_drafting_node collects ALL upstream notes
    from steps 1–8 in continue mode."""

    @patch("patent_system.agents.description_drafting.DraftDescriptionModule")
    @patch("patent_system.agents.description_drafting.RefineClaimsModule")
    def test_accumulates_all_upstream_notes(self, mock_refine_cls, mock_draft_cls):
        """Patent draft step should see all non-empty upstream review notes."""
        mock_refine = MagicMock()
        mock_refine.return_value = dspy.Prediction(refined_claims="refined claims")
        mock_refine_cls.return_value = mock_refine

        mock_draft = MagicMock()
        mock_draft.return_value = dspy.Prediction(description_text="mocked description")
        mock_draft_cls.return_value = mock_draft

        state = _make_state(
            review_notes={
                "initial_idea": "focus on GPU",
                "claims_drafting": "revise claim 3",
                "prior_art_search": "check patent X",
                "novelty_analysis": "novel aspect confirmed",
                "consistency_review": "terminology OK",
                "market_potential": "high demand",
                "legal_clarification": "IP clear",
                "disclosure_summary": "summary looks good",
            },
        )
        result = description_drafting_node(state)

        # Both refine and draft modules should receive the accumulated notes
        refine_kwargs = mock_refine.call_args.kwargs
        draft_kwargs = mock_draft.call_args.kwargs

        for kwargs in [refine_kwargs, draft_kwargs]:
            notes_text = kwargs.get("review_notes_text")
            assert notes_text is not None
            # All 8 upstream steps should be present
            assert "Initial Idea" in notes_text
            assert "Claims Drafting" in notes_text
            assert "Prior Art Search" in notes_text
            assert "Novelty Analysis" in notes_text
            assert "Consistency Review" in notes_text
            assert "Market Potential" in notes_text
            assert "Legal Clarification" in notes_text
            assert "Disclosure Summary" in notes_text

    @patch("patent_system.agents.description_drafting.DraftDescriptionModule")
    def test_no_upstream_notes(self, mock_draft_cls):
        """Patent draft with empty review notes should pass None."""
        mock_draft = MagicMock()
        mock_draft.return_value = dspy.Prediction(description_text="description")
        mock_draft_cls.return_value = mock_draft

        # No analysis feedback → no refine step, goes straight to draft
        state = _make_state(
            review_notes={},
            novelty_analysis=None,
            review_feedback="",
            market_assessment="",
            legal_assessment="",
        )
        result = description_drafting_node(state)

        draft_kwargs = mock_draft.call_args.kwargs
        assert draft_kwargs.get("review_notes_text") is None

    @patch("patent_system.agents.description_drafting.DraftDescriptionModule")
    @patch("patent_system.agents.description_drafting.RefineClaimsModule")
    def test_partial_upstream_notes(self, mock_refine_cls, mock_draft_cls):
        """Patent draft with some upstream notes should include only non-empty ones."""
        mock_refine = MagicMock()
        mock_refine.return_value = dspy.Prediction(refined_claims="refined")
        mock_refine_cls.return_value = mock_refine

        mock_draft = MagicMock()
        mock_draft.return_value = dspy.Prediction(description_text="description")
        mock_draft_cls.return_value = mock_draft

        state = _make_state(
            review_notes={
                "initial_idea": "focus on GPU",
                "claims_drafting": "",  # empty — should be excluded
                "novelty_analysis": "novel aspect confirmed",
            },
        )
        result = description_drafting_node(state)

        draft_kwargs = mock_draft.call_args.kwargs
        notes_text = draft_kwargs.get("review_notes_text")
        assert notes_text is not None
        assert "Initial Idea" in notes_text
        assert "Novelty Analysis" in notes_text
        # Empty claims_drafting notes should not appear
        assert "Claims Drafting" not in notes_text


# ---------------------------------------------------------------------------
# Task 12.1: End-to-end integration tests for review notes flow
# Requirements: 1.4, 5.1, 5.2, 7.1, 8.1, 8.2
# ---------------------------------------------------------------------------


class TestReviewNotesEndToEndFlow:
    """End-to-end tests: save review notes to DB, load them back, and verify
    they flow correctly through agent nodes and DSPy modules."""

    def test_save_and_load_review_notes_round_trip(self, in_memory_db):
        """Save review notes to DB via repository, load them back, verify match."""
        from patent_system.db.repository import WorkflowStepRepository

        repo = WorkflowStepRepository(in_memory_db)

        # First, create a topic so FK constraint is satisfied
        in_memory_db.execute("INSERT INTO topics (name) VALUES ('test-topic')")
        in_memory_db.commit()

        # Save a step with review notes
        repo.upsert(
            topic_id=1,
            step_key="claims_drafting",
            content="Claim 1: A method...",
            status="completed",
            personality_mode="critical",
            review_notes="Please revise claim 3 for clarity",
        )

        # Load back via get_step
        step = repo.get_step(1, "claims_drafting")
        assert step is not None
        assert step["review_notes"] == "Please revise claim 3 for clarity"
        assert step["content"] == "Claim 1: A method..."

        # Load back via get_by_topic
        steps = repo.get_by_topic(1)
        assert len(steps) == 1
        assert steps[0]["review_notes"] == "Please revise claim 3 for clarity"

    @patch("patent_system.agents.claims_drafting.DraftClaimsModule")
    def test_rerun_step_with_review_notes_injects_own_notes(self, mock_cls):
        """Rerun step with review notes — verify DSPy input contains own notes."""
        mock_instance = MagicMock()
        mock_instance.return_value = dspy.Prediction(claims_text="revised claims")
        mock_cls.return_value = mock_instance

        state = _make_state(
            review_notes={
                "initial_idea": "upstream note",
                "claims_drafting": "revise claim 3 for clarity",
            },
        )
        result = claims_drafting_node(state, review_notes_mode="rerun")

        call_kwargs = mock_instance.call_args.kwargs
        notes_text = call_kwargs.get("review_notes_text")
        assert notes_text is not None
        assert "Claims Drafting" in notes_text
        assert "revise claim 3 for clarity" in notes_text
        # Upstream notes should NOT be present in rerun mode
        assert "upstream note" not in notes_text
        assert result["claims_text"] == "revised claims"

    @patch("patent_system.agents.novelty_analysis.NoveltyAnalysisModule")
    def test_continue_to_next_step_injects_upstream_notes(self, mock_cls):
        """Continue to next step with review notes — verify DSPy input contains upstream notes."""
        mock_instance = MagicMock()
        mock_instance.return_value = dspy.Prediction(novelty_assessment="assessment")
        mock_cls.return_value = mock_instance

        state = _make_state(
            review_notes={
                "initial_idea": "focus on GPU aspect",
                "claims_drafting": "strengthen claim 1",
                "prior_art_search": "check patent X",
            },
        )
        # Default mode is "continue" — simulates "Continue to Next Step"
        result = novelty_analysis_node(state)

        call_kwargs = mock_instance.call_args.kwargs
        notes_text = call_kwargs.get("review_notes_text")
        assert notes_text is not None
        # All upstream notes should be present
        assert "Initial Idea" in notes_text
        assert "focus on GPU aspect" in notes_text
        assert "Claims Drafting" in notes_text
        assert "strengthen claim 1" in notes_text
        assert "Prior Art Search" in notes_text
        assert "check patent X" in notes_text
        # Own notes (novelty_analysis) should NOT be present
        assert "Novelty Analysis" not in notes_text

    @patch("patent_system.agents.description_drafting.DraftDescriptionModule")
    @patch("patent_system.agents.description_drafting.RefineClaimsModule")
    def test_description_drafting_accumulates_all_upstream_notes(
        self, mock_refine_cls, mock_draft_cls
    ):
        """description_drafting_node with review notes from multiple steps — verify all accumulated."""
        mock_refine = MagicMock()
        mock_refine.return_value = dspy.Prediction(refined_claims="refined claims")
        mock_refine_cls.return_value = mock_refine

        mock_draft = MagicMock()
        mock_draft.return_value = dspy.Prediction(description_text="full description")
        mock_draft_cls.return_value = mock_draft

        state = _make_state(
            review_notes={
                "initial_idea": "focus on GPU",
                "claims_drafting": "revise claim 3",
                "prior_art_search": "check patent X",
                "novelty_analysis": "novel aspect confirmed",
                "consistency_review": "terminology OK",
                "market_potential": "high demand",
                "legal_clarification": "IP clear",
                "disclosure_summary": "summary looks good",
            },
        )
        result = description_drafting_node(state)

        # Verify both refine and draft modules received all 8 upstream notes
        for mock_obj in [mock_refine, mock_draft]:
            call_kwargs = mock_obj.call_args.kwargs
            notes_text = call_kwargs.get("review_notes_text")
            assert notes_text is not None
            for label in [
                "Initial Idea",
                "Claims Drafting",
                "Prior Art Search",
                "Novelty Analysis",
                "Consistency Review",
                "Market Potential",
                "Legal Clarification",
                "Disclosure Summary",
            ]:
                assert label in notes_text

        assert result["description_text"] == "full description"

    @patch("patent_system.agents.claims_drafting.DraftClaimsModule")
    def test_empty_review_notes_no_injection(self, mock_cls):
        """Empty review notes — verify no injection into DSPy input."""
        mock_instance = MagicMock()
        mock_instance.return_value = dspy.Prediction(claims_text="claims output")
        mock_cls.return_value = mock_instance

        state = _make_state(review_notes={})
        claims_drafting_node(state)

        call_kwargs = mock_instance.call_args.kwargs
        assert call_kwargs.get("review_notes_text") is None

    @patch("patent_system.agents.novelty_analysis.NoveltyAnalysisModule")
    def test_empty_review_notes_no_injection_novelty(self, mock_cls):
        """Empty review notes on novelty analysis — verify no injection."""
        mock_instance = MagicMock()
        mock_instance.return_value = dspy.Prediction(novelty_assessment="assessment")
        mock_cls.return_value = mock_instance

        state = _make_state(review_notes={})
        novelty_analysis_node(state)

        call_kwargs = mock_instance.call_args.kwargs
        assert call_kwargs.get("review_notes_text") is None


# ---------------------------------------------------------------------------
# Task 12.2: Backward compatibility tests
# Requirements: 5.6, 6.3, 9.4
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """Verify backward compatibility: existing workflows, DSPy modules, and
    agent nodes continue to work without review_notes."""

    def test_migration_adds_review_notes_column_to_existing_table(self, in_memory_db):
        """Create a workflow_steps table WITHOUT review_notes column, run migration,
        verify existing rows get empty string default."""
        # Drop the existing table and recreate WITHOUT review_notes column
        in_memory_db.execute("DROP TABLE IF EXISTS workflow_steps")
        in_memory_db.execute(
            """CREATE TABLE workflow_steps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic_id INTEGER NOT NULL,
                step_key TEXT NOT NULL,
                content TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'pending',
                personality_mode TEXT NOT NULL DEFAULT '',
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (topic_id) REFERENCES topics(id),
                UNIQUE(topic_id, step_key)
            )"""
        )
        # Insert a topic and a pre-existing row without review_notes
        in_memory_db.execute("INSERT OR IGNORE INTO topics (id, name) VALUES (1, 'test-topic')")
        in_memory_db.execute(
            """INSERT INTO workflow_steps (topic_id, step_key, content, status)
               VALUES (1, 'claims_drafting', 'old claims', 'completed')"""
        )
        in_memory_db.commit()

        # Verify review_notes column does NOT exist yet
        cursor = in_memory_db.execute("PRAGMA table_info(workflow_steps)")
        col_names = {row[1] for row in cursor.fetchall()}
        assert "review_notes" not in col_names

        # Run the migration
        from patent_system.db.schema import _migrate_workflow_steps_review_notes

        _migrate_workflow_steps_review_notes(in_memory_db)

        # Verify column now exists
        cursor = in_memory_db.execute("PRAGMA table_info(workflow_steps)")
        col_names = {row[1] for row in cursor.fetchall()}
        assert "review_notes" in col_names

        # Verify existing row has empty string default
        row = in_memory_db.execute(
            "SELECT review_notes FROM workflow_steps WHERE topic_id = 1 AND step_key = 'claims_drafting'"
        ).fetchone()
        assert row is not None
        assert row[0] == ""

    def test_existing_steps_load_correctly_after_migration(self, in_memory_db):
        """Existing workflow steps without review_notes column still load correctly after migration."""
        from patent_system.db.repository import WorkflowStepRepository

        # Insert a topic and step (schema already has review_notes via init_schema)
        in_memory_db.execute("INSERT INTO topics (name) VALUES ('compat-topic')")
        in_memory_db.commit()

        repo = WorkflowStepRepository(in_memory_db)
        # Upsert without explicit review_notes (uses default empty string)
        repo.upsert(
            topic_id=1,
            step_key="novelty_analysis",
            content="Some novelty analysis",
            status="completed",
        )

        # Load and verify
        step = repo.get_step(1, "novelty_analysis")
        assert step is not None
        assert step["content"] == "Some novelty analysis"
        assert step["review_notes"] == ""

        steps = repo.get_by_topic(1)
        assert len(steps) == 1
        assert steps[0]["review_notes"] == ""

    @patch("patent_system.agents.claims_drafting.DraftClaimsModule")
    def test_dspy_module_forward_without_review_notes_text(self, mock_cls):
        """Existing DSPy module calls without review_notes_text parameter still work (default None)."""
        mock_instance = MagicMock()
        mock_instance.return_value = dspy.Prediction(claims_text="claims output")
        mock_cls.return_value = mock_instance

        state = _make_state(review_notes={})
        result = claims_drafting_node(state)

        call_kwargs = mock_instance.call_args.kwargs
        # review_notes_text should be None (no notes to inject)
        assert call_kwargs.get("review_notes_text") is None
        # Module should still produce valid output
        assert result["claims_text"] == "claims output"

    @patch("patent_system.agents.novelty_analysis.NoveltyAnalysisModule")
    def test_dspy_module_forward_without_review_notes_text_novelty(self, mock_cls):
        """NoveltyAnalysisModule forward() without review_notes_text still works."""
        mock_instance = MagicMock()
        mock_instance.return_value = dspy.Prediction(novelty_assessment="assessment")
        mock_cls.return_value = mock_instance

        state = _make_state(review_notes={})
        result = novelty_analysis_node(state)

        call_kwargs = mock_instance.call_args.kwargs
        assert call_kwargs.get("review_notes_text") is None
        assert result["novelty_analysis"] == "assessment"

    @patch("patent_system.agents.claims_drafting.DraftClaimsModule")
    def test_agent_node_without_review_notes_in_state(self, mock_cls):
        """Agent nodes without review_notes in state still work (default empty dict)."""
        mock_instance = MagicMock()
        mock_instance.return_value = dspy.Prediction(claims_text="claims")
        mock_cls.return_value = mock_instance

        state = _make_state()
        # Remove review_notes key entirely to simulate old state
        del state["review_notes"]
        result = claims_drafting_node(state)

        call_kwargs = mock_instance.call_args.kwargs
        # Should default to no notes injection
        assert call_kwargs.get("review_notes_text") is None
        assert result["claims_text"] == "claims"

    @patch("patent_system.agents.novelty_analysis.NoveltyAnalysisModule")
    def test_agent_node_without_review_notes_in_state_novelty(self, mock_cls):
        """Novelty analysis node without review_notes in state still works."""
        mock_instance = MagicMock()
        mock_instance.return_value = dspy.Prediction(novelty_assessment="assessment")
        mock_cls.return_value = mock_instance

        state = _make_state()
        del state["review_notes"]
        result = novelty_analysis_node(state)

        call_kwargs = mock_instance.call_args.kwargs
        assert call_kwargs.get("review_notes_text") is None
        assert result["novelty_analysis"] == "assessment"

    @patch("patent_system.agents.description_drafting.DraftDescriptionModule")
    def test_description_drafting_without_review_notes_in_state(self, mock_cls):
        """description_drafting_node without review_notes in state still works."""
        mock_draft = MagicMock()
        mock_draft.return_value = dspy.Prediction(description_text="description")
        mock_cls.return_value = mock_draft

        state = _make_state(
            novelty_analysis=None,
            review_feedback="",
            market_assessment="",
            legal_assessment="",
        )
        del state["review_notes"]
        result = description_drafting_node(state)

        call_kwargs = mock_draft.call_args.kwargs
        assert call_kwargs.get("review_notes_text") is None
        assert result["description_text"] == "description"
