"""Integration tests for end-to-end workflow execution.

Tests the LangGraph workflow graph structure, routing logic, and
state TypedDict completeness without requiring a real LLM.

Requirements: 15.1, 15.2, 15.3, 15.4, 15.5
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
        "market_assessment": "",
        "legal_assessment": "",
        "disclosure_summary": "",
        "prior_art_summary": "",
        "workflow_step_statuses": {},
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
# should_revise_or_proceed routing (legacy, kept for backward compat)
# ---------------------------------------------------------------------------


class TestShouldReviseOrProceed:
    """Test the legacy conditional routing function with various states."""

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
# Graph structure — linear 9-node chain
# ---------------------------------------------------------------------------

# The expected linear node order (Requirements 15.1, 15.2)
EXPECTED_NODE_ORDER = [
    "initial_idea",
    "claims_drafting",
    "prior_art_search",
    "novelty_analysis",
    "consistency_review",
    "market_potential",
    "legal_clarification",
    "disclosure_summary",
    "patent_draft",
]


class TestGraphStructure:
    """Test that the compiled graph has the expected 9-node linear chain."""

    def test_graph_has_expected_nodes(self, checkpointer):
        compiled = build_patent_workflow(checkpointer)
        node_names = set(compiled.get_graph().nodes.keys())

        expected = {"__start__", "__end__"} | set(EXPECTED_NODE_ORDER)
        assert expected.issubset(node_names)

    def test_graph_has_no_old_nodes(self, checkpointer):
        """Old nodes (disclosure, human_review) should not be present."""
        compiled = build_patent_workflow(checkpointer)
        node_names = set(compiled.get_graph().nodes.keys())

        assert "disclosure" not in node_names
        assert "human_review" not in node_names

    def test_entry_point_is_initial_idea(self, checkpointer):
        compiled = build_patent_workflow(checkpointer)
        graph = compiled.get_graph()
        start_edges = [
            e.target for e in graph.edges if e.source == "__start__"
        ]
        assert "initial_idea" in start_edges

    def test_patent_draft_leads_to_end(self, checkpointer):
        compiled = build_patent_workflow(checkpointer)
        graph = compiled.get_graph()
        draft_edges = [
            e.target for e in graph.edges if e.source == "patent_draft"
        ]
        assert "__end__" in draft_edges

    def test_linear_edge_order(self, checkpointer):
        """Verify the linear chain follows the expected order."""
        compiled = build_patent_workflow(checkpointer)
        graph = compiled.get_graph()

        for i in range(len(EXPECTED_NODE_ORDER) - 1):
            source = EXPECTED_NODE_ORDER[i]
            expected_target = EXPECTED_NODE_ORDER[i + 1]
            targets = [e.target for e in graph.edges if e.source == source]
            assert expected_target in targets, (
                f"Expected edge {source} -> {expected_target}, "
                f"but {source} targets are {targets}"
            )

    def test_no_conditional_edges(self, checkpointer):
        """The new graph should have no conditional edges (no review loop)."""
        compiled = build_patent_workflow(checkpointer)
        graph = compiled.get_graph()
        # Every non-start/end node should have exactly one outgoing edge
        for node_key in EXPECTED_NODE_ORDER:
            targets = [e.target for e in graph.edges if e.source == node_key]
            assert len(targets) == 1, (
                f"Node {node_key} has {len(targets)} outgoing edges: {targets}"
            )


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
            "market_assessment",
            "legal_assessment",
            "disclosure_summary",
            "prior_art_summary",
            "workflow_step_statuses",
        }
        actual_fields = set(PatentWorkflowState.__annotations__.keys())
        assert expected_fields == actual_fields

    def test_state_dict_is_valid(self):
        """A fully populated state dict should contain all expected keys."""
        state = _make_state()
        for field in PatentWorkflowState.__annotations__:
            assert field in state
