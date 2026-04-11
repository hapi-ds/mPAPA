"""Integration tests for end-to-end workflow execution.

Tests the LangGraph workflow graph structure, routing logic, and
state TypedDict completeness without requiring a real LLM.

Requirements: 9.1, 9.2, 9.3
"""

from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock, patch

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END

from patent_system.agents.graph import (
    build_patent_workflow,
    should_revise_or_proceed,
)
from patent_system.agents.state import PatentWorkflowState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(**overrides) -> dict:
    """Create a minimal PatentWorkflowState dict for testing."""
    base: dict = {
        "topic_id": 1,
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
        "current_step": "",
    }
    base.update(overrides)
    return base


@pytest.fixture
def checkpointer():
    """Provide a SqliteSaver backed by an in-memory SQLite connection."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    yield saver
    conn.close()


# ---------------------------------------------------------------------------
# should_revise_or_proceed routing
# ---------------------------------------------------------------------------


class TestShouldReviseOrProceed:
    """Test the conditional routing function with various states."""

    def test_approved_routes_to_description(self):
        state = _make_state(review_approved=True, iteration_count=0)
        assert should_revise_or_proceed(state) == "description_drafting"

    def test_approved_routes_to_description_regardless_of_iteration(self):
        state = _make_state(review_approved=True, iteration_count=5)
        assert should_revise_or_proceed(state) == "description_drafting"

    def test_not_approved_low_iteration_routes_to_claims(self):
        state = _make_state(review_approved=False, iteration_count=0)
        assert should_revise_or_proceed(state) == "claims_drafting"

    def test_not_approved_iteration_1_routes_to_claims(self):
        state = _make_state(review_approved=False, iteration_count=1)
        assert should_revise_or_proceed(state) == "claims_drafting"

    def test_not_approved_iteration_2_routes_to_claims(self):
        state = _make_state(review_approved=False, iteration_count=2)
        assert should_revise_or_proceed(state) == "claims_drafting"

    def test_not_approved_iteration_3_routes_to_human_review(self):
        state = _make_state(review_approved=False, iteration_count=3)
        assert should_revise_or_proceed(state) == "human_review"

    def test_not_approved_iteration_above_3_routes_to_human_review(self):
        state = _make_state(review_approved=False, iteration_count=10)
        assert should_revise_or_proceed(state) == "human_review"


# ---------------------------------------------------------------------------
# Graph structure
# ---------------------------------------------------------------------------


class TestGraphStructure:
    """Test that the compiled graph has the expected nodes and edges."""

    def test_graph_has_expected_nodes(self, checkpointer):
        compiled = build_patent_workflow(checkpointer)
        node_names = set(compiled.get_graph().nodes.keys())

        expected = {
            "__start__",
            "__end__",
            "disclosure",
            "prior_art_search",
            "novelty_analysis",
            "claims_drafting",
            "consistency_review",
            "human_review",
            "description_drafting",
        }
        assert expected.issubset(node_names)

    def test_entry_point_is_disclosure(self, checkpointer):
        compiled = build_patent_workflow(checkpointer)
        graph = compiled.get_graph()
        # The __start__ node should have an edge to disclosure
        start_edges = [
            e.target for e in graph.edges if e.source == "__start__"
        ]
        assert "disclosure" in start_edges

    def test_description_drafting_leads_to_end(self, checkpointer):
        compiled = build_patent_workflow(checkpointer)
        graph = compiled.get_graph()
        desc_edges = [
            e.target for e in graph.edges if e.source == "description_drafting"
        ]
        assert "__end__" in desc_edges

    def test_human_review_loops_to_claims(self, checkpointer):
        compiled = build_patent_workflow(checkpointer)
        graph = compiled.get_graph()
        hr_edges = [
            e.target for e in graph.edges if e.source == "human_review"
        ]
        assert "claims_drafting" in hr_edges


# ---------------------------------------------------------------------------
# PatentWorkflowState completeness
# ---------------------------------------------------------------------------


class TestWorkflowStateFields:
    """Test that PatentWorkflowState TypedDict has all required fields."""

    def test_has_all_required_fields(self):
        expected_fields = {
            "topic_id",
            "invention_disclosure",
            "interview_messages",
            "prior_art_results",
            "failed_sources",
            "novelty_analysis",
            "claims_text",
            "description_text",
            "review_feedback",
            "review_approved",
            "iteration_count",
            "current_step",
        }
        actual_fields = set(PatentWorkflowState.__annotations__.keys())
        assert expected_fields == actual_fields

    def test_state_dict_is_valid(self):
        """A fully populated state dict should contain all expected keys."""
        state = _make_state()
        for field in PatentWorkflowState.__annotations__:
            assert field in state
