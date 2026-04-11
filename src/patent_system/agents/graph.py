"""LangGraph workflow definition for the patent drafting pipeline.

Builds a StateGraph wiring the agent sequence:
  disclosure → prior_art_search → novelty_analysis → claims_drafting
  → consistency_review → (conditional: loop / human_review / description_drafting)
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from patent_system.agents.claims_drafting import claims_drafting_node
from patent_system.agents.consistency_review import consistency_review_node
from patent_system.agents.description_drafting import description_drafting_node
from patent_system.agents.disclosure import disclosure_node
from patent_system.agents.novelty_analysis import novelty_analysis_node
from patent_system.agents.prior_art_search import prior_art_search_node
from patent_system.agents.state import PatentWorkflowState

if TYPE_CHECKING:
    from patent_system.agents.novelty_analysis import RAGQueryable


# ---------------------------------------------------------------------------
# Conditional routing
# ---------------------------------------------------------------------------

def should_revise_or_proceed(state: PatentWorkflowState) -> str:
    """Conditional edge after consistency_review.

    Returns:
        ``"description_drafting"`` when the review is approved,
        ``"claims_drafting"`` when not approved and fewer than 3 iterations,
        ``"human_review"`` when not approved and 3+ iterations.
    """
    if state["review_approved"]:
        return "description_drafting"
    if state["iteration_count"] < 3:
        return "claims_drafting"
    return "human_review"


def _human_review_node(state: PatentWorkflowState) -> dict:
    """Human-in-the-loop review node.

    Uses LangGraph ``interrupt`` to pause execution so the user can
    inspect unresolved feedback and manually edit claims before the
    workflow resumes.
    """
    interrupt("Review required: unresolved consistency feedback after 3 iterations.")
    return {"current_step": "human_review"}


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_patent_workflow(checkpointer, rag_engine: RAGQueryable | None = None):
    """Build and compile the patent drafting workflow graph.

    Args:
        checkpointer: A LangGraph checkpointer instance (e.g.
            ``SqliteSaver``) used to persist workflow state between
            steps.
        rag_engine: Optional RAG engine passed to the novelty analysis
            node via ``functools.partial``.  When *None*, the node
            falls back to its internal placeholder.

    Returns:
        A compiled LangGraph ``CompiledStateGraph`` ready for
        invocation.
    """
    graph = StateGraph(PatentWorkflowState)

    # Bind rag_engine to novelty_analysis_node via partial
    bound_novelty_node = functools.partial(
        novelty_analysis_node, rag_engine=rag_engine
    )

    # Bind rag_engine to prior_art_search_node via partial
    bound_prior_art_node = functools.partial(
        prior_art_search_node, rag_engine=rag_engine
    )

    # Register nodes
    graph.add_node("disclosure", disclosure_node)
    graph.add_node("prior_art_search", bound_prior_art_node)
    graph.add_node("novelty_analysis", bound_novelty_node)
    graph.add_node("claims_drafting", claims_drafting_node)
    graph.add_node("consistency_review", consistency_review_node)
    graph.add_node("human_review", _human_review_node)
    graph.add_node("description_drafting", description_drafting_node)

    # Linear edges: disclosure → prior_art_search → novelty_analysis
    #               → claims_drafting → consistency_review
    graph.set_entry_point("disclosure")
    graph.add_edge("disclosure", "prior_art_search")
    graph.add_edge("prior_art_search", "novelty_analysis")
    graph.add_edge("novelty_analysis", "claims_drafting")
    graph.add_edge("claims_drafting", "consistency_review")

    # Conditional edge after consistency_review
    graph.add_conditional_edges(
        "consistency_review",
        should_revise_or_proceed,
        {
            "description_drafting": "description_drafting",
            "claims_drafting": "claims_drafting",
            "human_review": "human_review",
        },
    )

    # human_review loops back to claims_drafting after user edits
    graph.add_edge("human_review", "claims_drafting")

    # description_drafting is the final step
    graph.add_edge("description_drafting", END)

    return graph.compile(checkpointer=checkpointer)
