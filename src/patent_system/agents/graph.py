"""LangGraph workflow definition for the patent drafting pipeline.

Builds a StateGraph wiring the agent sequence:
  disclosure → prior_art_search → novelty_analysis → claims_drafting
  → consistency_review → (conditional: loop / human_review / description_drafting)

Placeholder node functions are used for each agent step; real
implementations will be wired in later tasks.
"""

from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from patent_system.agents.state import PatentWorkflowState


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


# ---------------------------------------------------------------------------
# Placeholder agent node functions (replaced by real agents in later tasks)
# ---------------------------------------------------------------------------

def _disclosure_node(state: PatentWorkflowState) -> dict:
    """Placeholder for the Invention Disclosure Agent."""
    return {"current_step": "disclosure"}


def _prior_art_search_node(state: PatentWorkflowState) -> dict:
    """Placeholder for the Prior Art Search Agent."""
    return {"current_step": "prior_art_search"}


def _novelty_analysis_node(state: PatentWorkflowState) -> dict:
    """Placeholder for the Novelty Analysis Agent."""
    return {"current_step": "novelty_analysis"}


def _claims_drafting_node(state: PatentWorkflowState) -> dict:
    """Placeholder for the Claims Drafting Agent."""
    return {"current_step": "claims_drafting"}


def _consistency_review_node(state: PatentWorkflowState) -> dict:
    """Placeholder for the Consistency Reviewer Agent."""
    return {"current_step": "consistency_review"}


def _human_review_node(state: PatentWorkflowState) -> dict:
    """Human-in-the-loop review node.

    Uses LangGraph ``interrupt`` to pause execution so the user can
    inspect unresolved feedback and manually edit claims before the
    workflow resumes.
    """
    interrupt("Review required: unresolved consistency feedback after 3 iterations.")
    return {"current_step": "human_review"}


def _description_drafting_node(state: PatentWorkflowState) -> dict:
    """Placeholder for the Description Drafter Agent."""
    return {"current_step": "description_drafting"}


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_patent_workflow(checkpointer):
    """Build and compile the patent drafting workflow graph.

    Args:
        checkpointer: A LangGraph checkpointer instance (e.g.
            ``SqliteSaver``) used to persist workflow state between
            steps.

    Returns:
        A compiled LangGraph ``CompiledStateGraph`` ready for
        invocation.
    """
    graph = StateGraph(PatentWorkflowState)

    # Register nodes
    graph.add_node("disclosure", _disclosure_node)
    graph.add_node("prior_art_search", _prior_art_search_node)
    graph.add_node("novelty_analysis", _novelty_analysis_node)
    graph.add_node("claims_drafting", _claims_drafting_node)
    graph.add_node("consistency_review", _consistency_review_node)
    graph.add_node("human_review", _human_review_node)
    graph.add_node("description_drafting", _description_drafting_node)

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
