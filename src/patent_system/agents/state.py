"""Workflow state definition for the patent drafting pipeline.

Defines the TypedDict that flows through every node in the LangGraph
StateGraph, carrying invention data, search results, drafts, and
review loop counters.
"""

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class PatentWorkflowState(TypedDict):
    """Full state carried through the patent drafting workflow graph."""

    topic_id: int
    invention_disclosure: dict | None
    interview_messages: Annotated[list, add_messages]
    prior_art_results: list[dict]
    failed_sources: list[str]
    novelty_analysis: dict | None
    claims_text: str
    description_text: str
    review_feedback: str
    review_approved: bool
    iteration_count: int
    current_step: str

    # New fields for interactive workflow steps
    market_assessment: str
    legal_assessment: str
    disclosure_summary: str
    prior_art_summary: str
    workflow_step_statuses: dict  # step_key -> "pending" | "completed"
