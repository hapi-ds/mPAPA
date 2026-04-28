"""LangGraph workflow definition for the patent drafting pipeline.

Builds a linear 9-node StateGraph for the interactive patent draft workflow:
  initial_idea → claims_drafting → prior_art_search → novelty_analysis
  → consistency_review → market_potential → legal_clarification
  → disclosure_summary → patent_draft

Each node (except the last) is wrapped with an interrupt so the UI can
pause for user review between steps.

Requirements: 15.1, 15.2, 15.3, 15.4, 15.5
"""

from __future__ import annotations

import functools
import json
from typing import TYPE_CHECKING, Any

from langgraph.graph import END, StateGraph

import dspy
import httpx
import litellm.exceptions
import requests.exceptions

from patent_system.agents.claims_drafting import claims_drafting_node
from patent_system.agents.consistency_review import consistency_review_node
from patent_system.agents.description_drafting import description_drafting_node
from patent_system.agents.disclosure_summary import disclosure_summary_node
from patent_system.agents.domain_profiles import DEFAULT_PROFILE_SLUG
from patent_system.agents.legal_clarification import legal_clarification_node
from patent_system.agents.market_potential import market_potential_node
from patent_system.agents.novelty_analysis import novelty_analysis_node
from patent_system.agents.personality import resolve_personality_mode
from patent_system.agents.review_notes import build_review_notes_text
from patent_system.agents.state import PatentWorkflowState
from patent_system.dspy_modules.modules import PriorArtSummaryModule
from patent_system.exceptions import LLMConnectionError
from patent_system.logging_config import log_agent_invocation

if TYPE_CHECKING:
    from patent_system.agents.novelty_analysis import RAGQueryable


# ---------------------------------------------------------------------------
# Legacy routing (kept for backward compatibility with existing tests)
# ---------------------------------------------------------------------------

def should_revise_or_proceed(state: PatentWorkflowState) -> str:
    """Conditional edge after consistency_review (legacy).

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
# Node helpers
# ---------------------------------------------------------------------------

def _make_interrupt_wrapper(node_fn, step_key):
    """Thin wrapper that just calls the node function.

    Interrupts are handled via ``interrupt_after`` at compile time,
    not inside the node. This wrapper exists only to preserve the
    node registration pattern.
    """
    def wrapped(state):
        return node_fn(state)
    return wrapped


def _initial_idea_node(state: PatentWorkflowState) -> dict[str, Any]:
    """Load the invention disclosure from state and return it as a string.

    This is a lightweight node with no LLM call — it simply extracts
    the ``invention_disclosure`` from state and serializes it so
    downstream nodes can consume it as text.

    Args:
        state: The current workflow state.

    Returns:
        Dict with ``current_step`` set to ``"initial_idea"``.
    """
    disclosure = state.get("invention_disclosure")
    if not disclosure:
        disclosure_text = ""
    elif isinstance(disclosure, str):
        disclosure_text = disclosure
    else:
        try:
            disclosure_text = json.dumps(disclosure, default=str)
        except (TypeError, ValueError):
            disclosure_text = str(disclosure)

    return {
        "current_step": "initial_idea",
        "invention_disclosure": disclosure_text,
    }


def _local_prior_art_summary_node(
    state: PatentWorkflowState,
    review_notes_mode: str = "continue",
) -> dict[str, Any]:
    """Summarize locally stored prior art references using the LLM.

    Reads ``prior_art_results`` from state (pre-loaded from the local DB
    by the Draft Panel) and uses the LLM to produce a comprehensive
    analytical summary. Makes NO external network requests (Req 4.2).

    Args:
        state: The current workflow state.
        review_notes_mode: Either ``"continue"`` (inject upstream notes)
            or ``"rerun"`` (inject own notes only).

    Returns:
        Dict with ``prior_art_summary`` and ``current_step``.
    """
    import logging
    import time

    _logger = logging.getLogger(__name__)
    start = time.monotonic()

    mode = resolve_personality_mode(state, "prior_art_search")
    domain_slug = state.get("domain_profile_slug") or DEFAULT_PROFILE_SLUG

    # Build review notes text
    review_notes = state.get("review_notes") or {}
    notes_text = build_review_notes_text(review_notes, "prior_art_search", review_notes_mode)

    results = state.get("prior_art_results") or []
    disclosure_text = state.get("invention_disclosure", "") or ""
    if isinstance(disclosure_text, dict):
        disclosure_text = json.dumps(disclosure_text, default=str)
    claims_text = state.get("claims_text", "")

    # Build a comprehensive text of ALL references for the LLM
    ref_parts: list[str] = []
    patent_count = 0
    paper_count = 0
    for i, rec in enumerate(results, 1):
        ref_type = rec.get("type", "unknown")
        title = rec.get("title", "Untitled")
        source = rec.get("source", "")
        abstract = rec.get("abstract", "") or ""
        patent_num = rec.get("patent_number", "")

        if ref_type == "patent" or patent_num:
            patent_count += 1
            entry = f"[Patent {patent_count}] {title}"
            if patent_num:
                entry += f" ({patent_num})"
        else:
            paper_count += 1
            entry = f"[Paper {paper_count}] {title}"

        if source:
            entry += f" — Source: {source}"
        if abstract:
            entry += f"\n  Abstract: {abstract}"
        ref_parts.append(entry)

    if not ref_parts:
        summary = (
            "No prior art references found in the local database. "
            "Consider running a search in the Research tab first."
        )
        return {
            "prior_art_summary": summary,
            "current_step": "prior_art_search",
        }

    references_text = (
        f"Total: {len(results)} references ({patent_count} patents, {paper_count} scientific papers)\n\n"
        + "\n\n".join(ref_parts)
    )

    # Use LLM to produce an analytical summary
    module = PriorArtSummaryModule()
    try:
        prediction = module(
            invention_disclosure=disclosure_text,
            claims_text=claims_text,
            prior_art_references=references_text,
            personality_mode=mode.value,
            review_notes_text=notes_text or None,
            domain_profile_slug=domain_slug,
        )
        summary = prediction.prior_art_summary
    except (
        requests.exceptions.ConnectionError,
        httpx.ConnectError,
        litellm.exceptions.APIConnectionError,
        ConnectionError,
        OSError,
    ) as exc:
        base_url = (
            dspy.settings.lm.kwargs.get("api_base", "unknown")
            if dspy.settings.lm
            else "unknown"
        )
        raise LLMConnectionError(
            f"LM Studio unreachable at {base_url}: {exc}"
        ) from exc

    duration_ms = (time.monotonic() - start) * 1000
    log_agent_invocation(
        logger=_logger,
        name="PriorArtSummaryAgent",
        input_summary=f"references={len(results)} ({patent_count} patents, {paper_count} papers), personality_mode={mode.value}, review_notes_length={len(notes_text)}, domain_profile={domain_slug}",
        output_summary=f"summary_length={len(summary)}",
        duration_ms=duration_ms,
    )

    return {
        "prior_art_summary": summary,
        "current_step": "prior_art_search",
        "personality_mode_used": mode.value,
    }


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_patent_workflow(checkpointer, rag_engine: RAGQueryable | None = None):
    """Build and compile the patent drafting workflow graph.

    Constructs a linear 9-node chain with interrupt wrappers between
    each step (except the last) so the UI can pause for user review.

    Node order:
        initial_idea → claims_drafting → prior_art_search →
        novelty_analysis → consistency_review → market_potential →
        legal_clarification → disclosure_summary → patent_draft

    Args:
        checkpointer: A LangGraph checkpointer instance (e.g.
            ``SqliteSaver``) used to persist workflow state between
            steps.
        rag_engine: Optional RAG engine passed to the novelty analysis
            and prior art search nodes via ``functools.partial``.
            When *None*, nodes fall back to their internal placeholders.

    Returns:
        A compiled LangGraph ``CompiledStateGraph`` ready for
        invocation.
    """
    graph = StateGraph(PatentWorkflowState)

    # Bind rag_engine to nodes that accept it
    bound_novelty_node = functools.partial(
        novelty_analysis_node, rag_engine=rag_engine
    )

    # Define the linear step sequence (node_key, node_fn) pairs
    # NOTE: prior_art_search uses _local_prior_art_summary_node which
    # reads from state.prior_art_results (pre-loaded from local DB by
    # the Draft Panel). It makes NO external network requests (Req 4.2).
    steps = [
        ("initial_idea", _initial_idea_node),
        ("claims_drafting", claims_drafting_node),
        ("prior_art_search", _local_prior_art_summary_node),
        ("novelty_analysis", bound_novelty_node),
        ("consistency_review", consistency_review_node),
        ("market_potential", market_potential_node),
        ("legal_clarification", legal_clarification_node),
        ("disclosure_summary", disclosure_summary_node),
        ("patent_draft", description_drafting_node),
    ]

    # Register nodes:
    # - initial_idea: NO interrupt (read-only passthrough, user reviews on the UI before starting)
    # - steps 2–8: interrupt AFTER (node runs, output applied to state, then graph pauses)
    # - patent_draft (last): NO interrupt (workflow ends)
    interrupt_after_nodes: list[str] = []
    for i, (step_key, node_fn) in enumerate(steps):
        if i == 0:
            # initial_idea — no interrupt, flows straight into claims_drafting
            graph.add_node(step_key, node_fn)
        elif i < len(steps) - 1:
            graph.add_node(step_key, node_fn)
            interrupt_after_nodes.append(step_key)
        else:
            graph.add_node(step_key, node_fn)

    # Wire linear edges
    graph.set_entry_point("initial_idea")
    for i in range(len(steps) - 1):
        graph.add_edge(steps[i][0], steps[i + 1][0])
    graph.add_edge(steps[-1][0], END)

    return graph.compile(
        checkpointer=checkpointer,
        interrupt_after=interrupt_after_nodes,
    )
