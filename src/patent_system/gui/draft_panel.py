"""Patent Draft panel UI for the Patent Analysis & Drafting System.

Provides a nine-step interactive workflow with progress indicator,
expandable step sections, "Continue to Next Step" / "Rerun this Step"
buttons, and DOCX export.

Requirements: 2.1–2.4, 3.1–3.5, 4.1–4.4, 5.1–5.5, 6.1–6.4,
              7.1–7.5, 8.1–8.5, 9.1–9.5, 10.1–10.4, 11.1–11.5,
              12.1–12.4, 12a.1–12a.5, 13.1–13.3
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from typing import TYPE_CHECKING, Any

from langgraph.errors import GraphInterrupt
from nicegui import ui

from patent_system.agents.state import PatentWorkflowState
from patent_system.agents.personality import AGENT_PERSONALITY_DEFAULTS
from patent_system.db.repository import (
    InventionDisclosureRepository,
    PatentDraftRepository,
    PatentRepository,
    PersonalityPreferenceRepository,
    ResearchSessionRepository,
    ScientificPaperRepository,
    WorkflowStepRepository,
    WORKFLOW_STEP_ORDER,
)

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)

# Nine workflow steps displayed in the progress indicator (Req 11.1)
WORKFLOW_STEPS: list[str] = [
    "Initial Idea",
    "Claims Drafting",
    "Prior Art Search",
    "Novelty Analysis",
    "Consistency Review",
    "Market Potential",
    "Legal Clarification",
    "Disclosure Summary",
    "Patent Draft",
]

# Map node current_step values to WORKFLOW_STEPS display names
_STEP_DISPLAY_NAMES: dict[str, str] = {
    "initial_idea": "Initial Idea",
    "claims_drafting": "Claims Drafting",
    "prior_art_search": "Prior Art Search",
    "novelty_analysis": "Novelty Analysis",
    "consistency_review": "Consistency Review",
    "market_potential": "Market Potential",
    "legal_clarification": "Legal Clarification",
    "disclosure_summary": "Disclosure Summary",
    "patent_draft": "Patent Draft",
}

# Reverse mapping: display name → step key
_DISPLAY_NAME_TO_KEY: dict[str, str] = {v: k for k, v in _STEP_DISPLAY_NAMES.items()}

# Badge colors for personality mode indicators (Req 8.2)
_MODE_BADGE_COLORS: dict[str, str] = {
    "critical": "red",
    "neutral": "grey",
    "innovation_friendly": "green",
}

# Human-readable labels for personality modes (Req 8.1)
_MODE_BADGE_LABELS: dict[str, str] = {
    "critical": "Critical",
    "neutral": "Neutral",
    "innovation_friendly": "Innovation-Friendly",
}


def can_export(claims: str, description: str) -> bool:
    """Check whether claims and description are non-empty and suitable for export.

    Args:
        claims: The patent claims text.
        description: The patent description text.

    Returns:
        True if both claims and description are non-empty strings,
        False otherwise.
    """
    if not claims or not isinstance(claims, str) or not claims.strip():
        return False
    if not description or not isinstance(description, str) or not description.strip():
        return False
    return True


def _has_content(text: str | None) -> bool:
    """Return True if *text* contains at least one non-whitespace character."""
    return bool(text and text.strip())


def _find_active_step(completed_keys: set[str]) -> str | None:
    """Return the first step key not in *completed_keys*, or None if all done.

    Validates: Requirement 12.3
    """
    for key in WORKFLOW_STEP_ORDER:
        if key not in completed_keys:
            return key
    return None


def create_draft_panel(
    container: Any,
    topic_id: int,
    *,
    workflow: CompiledStateGraph | None = None,
    conn: sqlite3.Connection | None = None,
    disclosure_repo: InventionDisclosureRepository | None = None,
    workflow_step_repo: WorkflowStepRepository | None = None,
    progress_bar_container: Any | None = None,
    personality_pref_repo: PersonalityPreferenceRepository | None = None,
) -> None:
    """Populate *container* with the nine-step interactive Patent Draft UI.

    Args:
        container: A NiceGUI container element (e.g. ``ui.column``).
        topic_id: The active topic ID.
        workflow: Compiled LangGraph workflow (9-node chain with interrupts).
        conn: SQLite connection for loading prior art / drafts.
        disclosure_repo: Repository for invention disclosures.
        workflow_step_repo: Repository for per-step persistence.
        progress_bar_container: Optional external container (e.g. in the
            header) where progress chips and status label are rendered.
            When provided, the in-panel sticky footer is not created.
        personality_pref_repo: Repository for per-topic personality preferences.
    """
    container.clear()

    # ------------------------------------------------------------------
    # Panel-local mutable state
    # ------------------------------------------------------------------
    panel_state: dict[str, Any] = {
        "claims": "",
        "description": "",
        "step_contents": {},      # step_key -> str
        "completed_keys": set(),  # step keys with status "completed"
        "active_key": None,       # current active step key
        "running": False,         # True while an LLM step is executing
        "step_modes": {},         # step_key -> personality mode string (Req 8.1)
        "step_review_notes": {},  # step_key -> review notes string (Req 3.1)
    }

    # ------------------------------------------------------------------
    # Load saved workflow steps from DB (Req 12.1)
    # ------------------------------------------------------------------
    if workflow_step_repo is not None:
        try:
            saved_steps = workflow_step_repo.get_by_topic(topic_id)
            for step in saved_steps:
                key = step["step_key"]
                panel_state["step_contents"][key] = step["content"]
                if step["status"] == "completed":
                    panel_state["completed_keys"].add(key)
                # Populate personality mode from DB (Req 8.4)
                if step.get("personality_mode"):
                    panel_state["step_modes"][key] = step["personality_mode"]
                # Populate review notes from DB (Req 1.4)
                panel_state["step_review_notes"][key] = step["review_notes"]
        except Exception:
            logger.exception("Failed to load workflow steps for topic %d", topic_id)

    # Load saved draft for claims/description
    draft_repo: PatentDraftRepository | None = None
    if conn is not None:
        draft_repo = PatentDraftRepository(conn)
        try:
            saved_draft = draft_repo.get_by_topic(topic_id)
            if saved_draft:
                panel_state["claims"] = saved_draft["claims_text"]
                panel_state["description"] = saved_draft["description_text"]
        except Exception:
            logger.exception("Failed to load saved draft for topic %d", topic_id)

    # Determine active step
    panel_state["active_key"] = _find_active_step(panel_state["completed_keys"])

    # ------------------------------------------------------------------
    # Build UI
    # ------------------------------------------------------------------
    # panel_state["spinner"] is set later when the progress bar is built.

    with container:
        ui.label("Patent Draft").classes("text-h6 q-mb-sm")

        # ------------------------------------------------------------------
        # Helper: load local prior art references from DB (no network)
        # ------------------------------------------------------------------
        def _load_local_prior_art() -> list[dict]:
            """Load patent and paper references from the local SQLite DB.

            Returns a list of dicts with title, abstract, source, etc.
            Makes NO external network requests (Req 4.2).
            """
            results: list[dict] = []
            if conn is None:
                return results
            try:
                session_repo = ResearchSessionRepository(conn)
                patent_repo = PatentRepository(conn)
                paper_repo = ScientificPaperRepository(conn)
                sessions = session_repo.get_by_topic(topic_id)
                for session in sessions:
                    for rec in patent_repo.get_by_session(session["id"]):
                        results.append({
                            "title": rec.title,
                            "abstract": rec.abstract or "",
                            "source": rec.source,
                            "patent_number": rec.patent_number,
                            "type": "patent",
                        })
                    for rec in paper_repo.get_by_session(session["id"]):
                        results.append({
                            "title": rec.title,
                            "abstract": rec.abstract or "",
                            "source": rec.source,
                            "doi": rec.doi,
                            "type": "paper",
                        })
            except Exception:
                logger.exception("Failed to load local prior art for topic %d", topic_id)
            return results

        # ------------------------------------------------------------------
        # Helper: build initial LangGraph state from prior step contents
        # ------------------------------------------------------------------
        def _build_initial_state(from_step_key: str | None = None) -> PatentWorkflowState:
            """Build a PatentWorkflowState pre-populated from saved step contents.

            If *from_step_key* is given, only steps before it are loaded.
            """
            sc = panel_state["step_contents"]

            # Build disclosure dict from initial_idea text
            idea_text = sc.get("initial_idea", "")
            disclosure: dict[str, Any] | str
            if idea_text:
                disclosure = idea_text
            else:
                disclosure = None  # type: ignore[assignment]

            # Load prior art from DB for implementation_details
            if conn is not None and disclosure is None:
                if disclosure_repo is not None:
                    try:
                        saved = disclosure_repo.get_by_topic(topic_id)
                        if saved:
                            disclosure = json.dumps({
                                "technical_problem": saved["primary_description"],
                                "novel_features": saved.get("search_terms", []),
                                "implementation_details": "",
                                "potential_variations": [],
                            })
                    except Exception:
                        pass

            # Load personality preferences for this topic, falling back to defaults
            personality_modes: dict[str, str] = dict(AGENT_PERSONALITY_DEFAULTS)
            if personality_pref_repo is not None:
                try:
                    saved_prefs = personality_pref_repo.get_by_topic(topic_id)
                    if saved_prefs:
                        personality_modes.update(saved_prefs)
                except Exception:
                    logger.exception(
                        "Failed to load personality preferences for topic %d — using defaults",
                        topic_id,
                    )

            # Cache the resolved modes so _stream_once can look up the
            # correct mode per step (LangGraph strips personality_mode_used
            # from node output because it's not in PatentWorkflowState).
            panel_state["personality_modes"] = {
                k: (v.value if hasattr(v, "value") else str(v))
                for k, v in personality_modes.items()
            }

            state: PatentWorkflowState = {
                "topic_id": topic_id,
                "invention_disclosure": disclosure,
                "interview_messages": [],
                "prior_art_results": _load_local_prior_art(),
                "failed_sources": [],
                "novelty_analysis": sc.get("novelty_analysis"),
                "claims_text": sc.get("claims_drafting", "") or panel_state["claims"],
                "description_text": sc.get("patent_draft", "") or panel_state["description"],
                "review_feedback": sc.get("consistency_review", ""),
                "review_approved": False,
                "iteration_count": 0,
                "current_step": from_step_key or "initial_idea",
                "market_assessment": sc.get("market_potential", ""),
                "legal_assessment": sc.get("legal_clarification", ""),
                "disclosure_summary": sc.get("disclosure_summary", ""),
                "prior_art_summary": sc.get("prior_art_search", ""),
                "workflow_step_statuses": {},
                "personality_modes": personality_modes,
                "review_notes": dict(panel_state.get("step_review_notes", {})),
            }
            return state

        # ------------------------------------------------------------------
        # Helper: persist step content to DB
        # ------------------------------------------------------------------
        def _persist_step(step_key: str, content: str, status: str = "completed", personality_mode: str = "", review_notes: str = "") -> None:
            """Save step content via WorkflowStepRepository. Shows notification on failure."""
            if workflow_step_repo is None:
                return
            try:
                workflow_step_repo.upsert(topic_id, step_key, content, status, personality_mode=personality_mode, review_notes=review_notes)
            except Exception:
                logger.exception("Failed to save step %s for topic %d", step_key, topic_id)
                ui.notify(
                    f"Could not save step '{_STEP_DISPLAY_NAMES.get(step_key, step_key)}' — your edits are preserved in memory.",
                    type="warning",
                )

        def _save_draft() -> None:
            """Persist current claims and description to the patent_drafts table."""
            if draft_repo is not None:
                try:
                    draft_repo.upsert(topic_id, panel_state["claims"], panel_state["description"])
                except Exception:
                    logger.debug("Failed to auto-save draft for topic %d", topic_id, exc_info=True)

        # ------------------------------------------------------------------
        # Core: stream one graph segment and process its events
        # ------------------------------------------------------------------
        async def _stream_once(stream_input, config: dict) -> None:
            """Run one workflow.stream() call and process events until it ends."""
            import queue as _queue_mod

            event_queue: _queue_mod.Queue = _queue_mod.Queue()
            stream_error: list[Exception] = []

            def _run_stream() -> None:
                try:
                    for event in workflow.stream(stream_input, config):  # type: ignore[union-attr]
                        event_queue.put(event)
                except Exception as exc:
                    stream_error.append(exc)
                finally:
                    event_queue.put(None)

            loop = asyncio.get_event_loop()
            loop.run_in_executor(None, _run_stream)

            while True:
                event = await asyncio.to_thread(event_queue.get)
                if event is None:
                    break

                for node_name, node_output in event.items():
                    if node_name == "__end__":
                        continue
                    if not isinstance(node_output, dict):
                        continue

                    step_key = node_output.get("current_step", node_name)
                    # Map legacy node names to canonical step keys
                    if step_key == "description_drafting":
                        step_key = "patent_draft"
                    display_name = _STEP_DISPLAY_NAMES.get(step_key, step_key)

                    # Extract content produced by this node
                    content = ""
                    if step_key == "initial_idea":
                        disc = node_output.get("invention_disclosure", "")
                        content = disc if isinstance(disc, str) else json.dumps(disc, default=str)
                    elif step_key == "claims_drafting":
                        content = node_output.get("claims_text", "")
                    elif step_key == "prior_art_search":
                        # prior_art_search_node returns prior_art_summary (string)
                        content = node_output.get("prior_art_summary", "")
                        if not content:
                            # Fallback: build summary from prior_art_results list
                            results = node_output.get("prior_art_results", [])
                            if results:
                                content = f"Found {len(results)} references.\n"
                                for idx, r in enumerate(results[:20], 1):
                                    content += f"[{idx}] {r.get('title', 'Untitled')}\n"
                    elif step_key == "novelty_analysis":
                        na = node_output.get("novelty_analysis")
                        if isinstance(na, str):
                            content = na
                        elif isinstance(na, dict):
                            content = json.dumps(na, indent=2, default=str)
                        elif na:
                            content = str(na)
                    elif step_key == "consistency_review":
                        content = node_output.get("review_feedback", "")
                    elif step_key == "market_potential":
                        content = node_output.get("market_assessment", "")
                    elif step_key == "legal_clarification":
                        content = node_output.get("legal_assessment", "")
                    elif step_key == "disclosure_summary":
                        content = node_output.get("disclosure_summary", "")
                    elif step_key == "patent_draft":
                        content = node_output.get("description_text", "")
                        claims = node_output.get("claims_text", "")
                        if claims:
                            panel_state["claims"] = claims
                            if claims_textarea_ref.get("el") is not None:
                                claims_textarea_ref["el"].value = claims
                                claims_textarea_ref["el"].update()

                    # Store in panel state (content only — do NOT overwrite
                    # a step that's already marked completed)
                    if content:
                        panel_state["step_contents"][step_key] = content
                        # Look up the mode from the resolved personality_modes
                        # (node_output won't contain personality_mode_used because
                        # LangGraph strips keys not in PatentWorkflowState).
                        personality_mode_used = panel_state.get("personality_modes", {}).get(step_key, "")
                        # Update step_modes for badge display (Req 8.3)
                        panel_state["step_modes"][step_key] = personality_mode_used
                        if step_key not in panel_state["completed_keys"]:
                            _persist_step(step_key, content, "pending", personality_mode=personality_mode_used)

                    # Update the textarea for this step
                    ta = step_textareas.get(step_key)
                    if ta is not None and content:
                        ta.value = content
                        ta.update()

                    _refresh_chips()

            if stream_error:
                raise stream_error[0]

        # ------------------------------------------------------------------
        # Core workflow execution: run from a given step key
        # ------------------------------------------------------------------
        async def _run_workflow_from(start_key: str, *, force_fresh: bool = False) -> None:
            """Execute the LangGraph workflow starting from *start_key*.

            Streams events, updates step textareas, persists results.

            Args:
                start_key: The step key to start from.
                force_fresh: If True, always start a new graph run
                    (ignore any existing checkpoint).
            """
            if workflow is None:
                ui.notify("Workflow not available — please restart the application.", type="negative")
                return

            if panel_state["running"]:
                return
            panel_state["running"] = True

            panel_state["spinner"].set_visibility(True)
            _refresh_chips()  # Shows "Running: ..." in footer

            config = {"configurable": {"thread_id": f"topic-{topic_id}"}}

            if force_fresh:
                # Discard any stale checkpoint so we start clean
                stream_input: Any = _build_initial_state(start_key)
            else:
                # Check for existing interrupted checkpoint to resume
                existing_snapshot = await asyncio.to_thread(workflow.get_state, config)
                is_resuming = (
                    existing_snapshot is not None
                    and hasattr(existing_snapshot, "next")
                    and existing_snapshot.next
                )

                if is_resuming:
                    # Inject user edits into the checkpoint state before resuming.
                    # This ensures downstream nodes see the user's changes.
                    sc = panel_state["step_contents"]
                    state_updates: dict[str, Any] = {}
                    if sc.get("initial_idea"):
                        state_updates["invention_disclosure"] = sc["initial_idea"]
                    if sc.get("claims_drafting"):
                        state_updates["claims_text"] = sc["claims_drafting"]
                    if sc.get("prior_art_search"):
                        state_updates["prior_art_summary"] = sc["prior_art_search"]
                    if sc.get("novelty_analysis"):
                        state_updates["novelty_analysis"] = sc["novelty_analysis"]
                    if sc.get("consistency_review"):
                        state_updates["review_feedback"] = sc["consistency_review"]
                    if sc.get("market_potential"):
                        state_updates["market_assessment"] = sc["market_potential"]
                    if sc.get("legal_clarification"):
                        state_updates["legal_assessment"] = sc["legal_clarification"]
                    if sc.get("disclosure_summary"):
                        state_updates["disclosure_summary"] = sc["disclosure_summary"]
                    if state_updates:
                        await asyncio.to_thread(
                            workflow.update_state, config, state_updates
                        )
                    stream_input = None  # resume from checkpoint
                else:
                    stream_input = _build_initial_state(start_key)

            try:
                # First stream call
                await _stream_once(stream_input, config)

                # Check if interrupted
                snapshot = await asyncio.to_thread(workflow.get_state, config)
                is_interrupted = (
                    snapshot is not None
                    and hasattr(snapshot, "next")
                    and snapshot.next
                )

                if is_interrupted:
                    # Auto-open the expansion for the newly active step
                    panel_state["active_key"] = _find_active_step(panel_state["completed_keys"])
                    active = panel_state["active_key"]
                    if active and active in step_expansions:
                        step_expansions[active].open()
                else:
                    # Graph completed — mark all steps with content as completed
                    for sk in WORKFLOW_STEP_ORDER:
                        if _has_content(panel_state["step_contents"].get(sk, "")):
                            panel_state["completed_keys"].add(sk)
                            _persist_step(
                                sk,
                                panel_state["step_contents"][sk],
                                "completed",
                                personality_mode=panel_state["step_modes"].get(sk, "critical"),
                            )
                    _save_draft()

            except GraphInterrupt:
                pass  # Status handled by _refresh_chips in finally

            except Exception as exc:
                from patent_system.exceptions import LLMConnectionError
                failed_display = _STEP_DISPLAY_NAMES.get(
                    panel_state.get("active_key", ""), "unknown"
                )
                if isinstance(exc, LLMConnectionError):
                    ui.notify(f"LLM connection failed at '{failed_display}': {exc}", type="negative")
                else:
                    ui.notify(f"Workflow failed at '{failed_display}': {exc}", type="negative")
                logger.exception("Workflow failed for topic %d", topic_id)

            finally:
                panel_state["running"] = False
                panel_state["spinner"].set_visibility(False)
                # Recalculate active step
                panel_state["active_key"] = _find_active_step(panel_state["completed_keys"])
                _refresh_chips()
                _refresh_step_sections()

        # ------------------------------------------------------------------
        # Re-run a single step by calling its node function directly
        # ------------------------------------------------------------------
        async def _rerun_single_step(step_key: str) -> None:
            """Re-run a single step: clear → call LLM → show new content → confirm.

            1. Saves the old content for potential revert
            2. Clears the textarea
            3. Calls the node function directly (LLM regenerates)
            4. Shows the new content
            5. Shows Keep/Revert buttons for the user to decide
            """
            if panel_state["running"]:
                return

            # Save old content for revert
            old_content = panel_state["step_contents"].get(step_key, "")

            # Clear the textarea immediately
            ta = step_textareas.get(step_key)
            if ta is not None:
                ta.value = ""

            panel_state["running"] = True
            panel_state["spinner"].set_visibility(True)
            _refresh_chips()

            new_content = ""
            try:
                # Build state from current panel contents
                state = _build_initial_state(step_key)

                # Persist review notes before rerunning
                rn_ta = step_review_textareas.get(step_key)
                if rn_ta is not None and hasattr(rn_ta, 'value'):
                    panel_state["step_review_notes"][step_key] = rn_ta.value or ""
                    # Update state with latest review notes
                    state["review_notes"] = dict(panel_state.get("step_review_notes", {}))

                from patent_system.agents.graph import _local_prior_art_summary_node
                from patent_system.agents.claims_drafting import claims_drafting_node
                from patent_system.agents.novelty_analysis import novelty_analysis_node
                from patent_system.agents.consistency_review import consistency_review_node
                from patent_system.agents.market_potential import market_potential_node
                from patent_system.agents.legal_clarification import legal_clarification_node
                from patent_system.agents.disclosure_summary import disclosure_summary_node
                from patent_system.agents.description_drafting import description_drafting_node

                node_map = {
                    "claims_drafting": claims_drafting_node,
                    "prior_art_search": _local_prior_art_summary_node,
                    "novelty_analysis": novelty_analysis_node,
                    "consistency_review": consistency_review_node,
                    "market_potential": market_potential_node,
                    "legal_clarification": legal_clarification_node,
                    "disclosure_summary": disclosure_summary_node,
                    "patent_draft": description_drafting_node,
                }

                node_fn = node_map.get(step_key)
                if node_fn is None:
                    ui.notify(f"Cannot re-run step '{step_key}'", type="warning")
                    return

                result = await asyncio.to_thread(node_fn, state, review_notes_mode="rerun")

                # Extract personality mode used by this node
                # Direct node calls return personality_mode_used; fall back
                # to the resolved preferences if missing.
                rerun_personality_mode = result.get(
                    "personality_mode_used",
                    panel_state.get("personality_modes", {}).get(step_key, ""),
                )
                # Update step_modes for badge display (Req 8.3)
                panel_state["step_modes"][step_key] = rerun_personality_mode

                # Extract content from the result
                if step_key == "claims_drafting":
                    new_content = result.get("claims_text", "")
                elif step_key == "prior_art_search":
                    new_content = result.get("prior_art_summary", "")
                elif step_key == "novelty_analysis":
                    na = result.get("novelty_analysis")
                    new_content = na if isinstance(na, str) else json.dumps(na, indent=2, default=str) if na else ""
                elif step_key == "consistency_review":
                    new_content = result.get("review_feedback", "")
                elif step_key == "market_potential":
                    new_content = result.get("market_assessment", "")
                elif step_key == "legal_clarification":
                    new_content = result.get("legal_assessment", "")
                elif step_key == "disclosure_summary":
                    new_content = result.get("disclosure_summary", "")
                elif step_key == "patent_draft":
                    new_content = result.get("description_text", "")
                    new_claims = result.get("claims_text", "")
                    if new_claims:
                        panel_state["claims"] = new_claims
                        panel_state["step_contents"]["claims_drafting"] = new_claims
                        if claims_textarea_ref.get("el") is not None:
                            claims_textarea_ref["el"].value = new_claims
                        # Also update the claims_drafting textarea if visible
                        claims_ta = step_textareas.get("claims_drafting")
                        if claims_ta is not None:
                            claims_ta.value = new_claims
                        _persist_step("claims_drafting", new_claims, "completed", personality_mode=rerun_personality_mode)
                        _save_draft()

            except Exception as exc:
                from patent_system.exceptions import LLMConnectionError
                display = _STEP_DISPLAY_NAMES.get(step_key, step_key)
                if isinstance(exc, LLMConnectionError):
                    ui.notify(f"LLM connection failed at '{display}': {exc}", type="negative")
                else:
                    ui.notify(f"Re-run failed at '{display}': {exc}", type="negative")
                logger.exception("Re-run failed for step %s topic %d", step_key, topic_id)
                # Restore old content on failure
                if ta is not None:
                    ta.value = old_content
                panel_state["step_contents"][step_key] = old_content
                return

            finally:
                panel_state["running"] = False
                panel_state["spinner"].set_visibility(False)
                _refresh_chips()

            # Show new content in textarea
            if ta is not None and new_content:
                ta.value = new_content
            panel_state["step_contents"][step_key] = new_content or old_content

            # Open the expansion
            if step_key in step_expansions:
                step_expansions[step_key].open()

            # Show Keep / Revert confirmation
            display = _STEP_DISPLAY_NAMES.get(step_key, step_key)

            def _on_keep() -> None:
                panel_state["step_contents"][step_key] = new_content
                _persist_step(step_key, new_content, "completed", personality_mode=rerun_personality_mode)
                panel_state["completed_keys"].add(step_key)
                ui.notify(f"{display}: new content saved.", type="positive")
                _refresh_step_sections()

            def _on_revert() -> None:
                panel_state["step_contents"][step_key] = old_content
                _persist_step(step_key, old_content, "completed", personality_mode=rerun_personality_mode)
                panel_state["completed_keys"].add(step_key)
                if ta is not None:
                    ta.value = old_content
                ui.notify(f"{display}: reverted to previous content.", type="info")
                _refresh_step_sections()

            if new_content and new_content != old_content:
                with ui.dialog() as dlg, ui.card():
                    ui.label(f"New content generated for {display}.").classes("text-subtitle1")
                    ui.label("Keep the new version or revert to the previous one?").classes("text-body2 q-mb-md")
                    with ui.row().classes("justify-end gap-2"):
                        ui.button("Revert", on_click=lambda: (dlg.close(), _on_revert()), icon="undo").props("flat color=warning")
                        ui.button("Keep New", on_click=lambda: (dlg.close(), _on_keep()), icon="check").props("color=primary")
                dlg.open()
            elif new_content:
                # Same content — save and mark completed so the step
                # returns to its normal state and the user gets feedback.
                _persist_step(step_key, new_content, "completed", personality_mode=rerun_personality_mode)
                panel_state["completed_keys"].add(step_key)
                ui.notify(f"{display}: re-run complete (no changes).", type="info")

            _refresh_step_sections()

        # ------------------------------------------------------------------
        # Step sections container
        # ------------------------------------------------------------------
        step_textareas: dict[str, Any] = {}
        step_review_textareas: dict[str, Any] = {}  # step_key -> review notes textarea (Req 3.1)
        step_continue_btns: dict[str, Any] = {}
        step_rerun_btns: dict[str, Any] = {}
        step_save_btns: dict[str, Any] = {}
        step_expansions: dict[str, Any] = {}
        step_badge_containers: dict[str, Any] = {}  # step_key -> badge container element (Req 8.1)
        claims_textarea_ref: dict[str, Any] = {"el": None}
        description_textarea_ref: dict[str, Any] = {"el": None}

        steps_container = ui.column().classes("w-full")

        def _render_badge(badge_container: Any, step_key: str) -> None:
            """Render a personality mode badge inside *badge_container* for *step_key*.

            Clears the container first, then adds a ``ui.badge`` if a mode
            is recorded in ``panel_state["step_modes"]``.

            Validates: Requirements 8.1, 8.2, 8.3
            """
            badge_container.clear()
            mode = panel_state["step_modes"].get(step_key)
            if mode:
                label = _MODE_BADGE_LABELS.get(mode, mode)
                color = _MODE_BADGE_COLORS.get(mode, "grey")
                with badge_container:
                    ui.badge(label, color=color).props("dense")

        def _refresh_step_sections() -> None:
            """Update visibility/enabled state of buttons and textareas."""
            active_key = panel_state["active_key"]
            is_running = panel_state.get("running", False)
            for key in WORKFLOW_STEP_ORDER:
                ta = step_textareas.get(key)
                rn_ta = step_review_textareas.get(key)
                cont_btn = step_continue_btns.get(key)
                rerun_btn = step_rerun_btns.get(key)
                save_btn = step_save_btns.get(key)

                is_completed = key in panel_state["completed_keys"]
                is_active = key == active_key
                has_text = _has_content(panel_state["step_contents"].get(key, ""))

                # Agent output textarea: ALWAYS readonly for steps 2–8 (Req 2.1)
                # Step 9 (patent_draft) textareas remain editable (claims/description)
                # Step 1 (initial_idea) has no textarea (None)
                if ta is not None:
                    if key not in ("initial_idea", "patent_draft"):
                        # Steps 2–8: always readonly
                        ta.props("readonly")
                    else:
                        # Step 9: only readonly while running
                        if is_running:
                            ta.props("readonly")
                        else:
                            ta.props(remove="readonly")

                # Review notes textarea: readonly while running, editable otherwise (Req 3.4, 3.5)
                if rn_ta is not None:
                    if is_running:
                        rn_ta.props("readonly")
                    else:
                        rn_ta.props(remove="readonly")

                # Continue button: visible only during initial run-through
                if cont_btn is not None:
                    cont_btn.set_visibility(is_active and not is_completed)
                    if has_text:
                        cont_btn.enable()
                    else:
                        cont_btn.disable()

                # Rerun button: visible on any step with content, hidden while running
                if rerun_btn is not None:
                    rerun_btn.set_visibility(has_text and not is_running)

                # Save button: visible on any step with content, hidden while running
                if save_btn is not None:
                    save_btn.set_visibility(has_text and not is_running)

                # Refresh personality mode badge (Req 8.3)
                badge_container = step_badge_containers.get(key)
                if badge_container is not None:
                    _render_badge(badge_container, key)

            _update_export_state()

        # ------------------------------------------------------------------
        # Render each step section
        # ------------------------------------------------------------------
        with steps_container:
            for step_idx, step_key in enumerate(WORKFLOW_STEP_ORDER):
                step_num = step_idx + 1
                display_name = _STEP_DISPLAY_NAMES[step_key]
                saved_content = panel_state["step_contents"].get(step_key, "")

                # --- Special handling for Step 1: Initial Idea (read-only) ---
                if step_key == "initial_idea":
                    with ui.expansion(
                        f"Step {step_num}: {display_name}",
                        icon="lightbulb",
                    ).classes("w-full q-mb-sm") as exp:
                        step_expansions[step_key] = exp

                        # Personality mode badge container (Req 8.1)
                        badge_ctr = ui.row().classes("gap-1 items-center q-mb-xs")
                        step_badge_containers[step_key] = badge_ctr
                        _render_badge(badge_ctr, step_key)

                        # Container for disclosure content (re-rendered on rerun)
                        idea_content_container = ui.column().classes("w-full")

                        def _render_initial_idea() -> None:
                            """Load disclosure from DB and render inside the container."""
                            idea_content_container.clear()
                            _disc_data: dict | None = None
                            if disclosure_repo is not None:
                                try:
                                    _disc_data = disclosure_repo.get_by_topic(topic_id)
                                except Exception:
                                    pass

                            with idea_content_container:
                                if _disc_data is not None:
                                    ui.label("Primary Description").classes("text-subtitle2 q-mt-sm")
                                    ui.label(_disc_data["primary_description"]).classes(
                                        "text-body2 q-pa-sm bg-grey-1 rounded"
                                    ).style("white-space: pre-wrap;")

                                    _terms = _disc_data.get("search_terms", [])
                                    if _terms:
                                        ui.label("Search Terms").classes("text-subtitle2 q-mt-sm")
                                        with ui.row().classes("gap-1 flex-wrap"):
                                            for term in _terms:
                                                ui.chip(term, color="primary").props("outline dense")

                                    _idea_parts = [_disc_data["primary_description"]]
                                    if _terms:
                                        _idea_parts.append("\nSearch Terms: " + ", ".join(_terms))
                                    content_str = "\n".join(_idea_parts)
                                    panel_state["step_contents"][step_key] = content_str
                                else:
                                    ui.label(
                                        "No invention disclosure saved yet. "
                                        "Go to the Research tab to enter your invention description first."
                                    ).classes("text-grey q-pa-sm")
                                    panel_state["step_contents"][step_key] = ""

                        _render_initial_idea()

                        # Hidden textarea ref for consistency with other steps
                        # (not displayed, but keeps step_textareas dict complete)
                        step_textareas[step_key] = None  # type: ignore[assignment]

                        # Review Notes textarea for step 1 (Req 3.1, 3.7)
                        review_ta_initial = ui.textarea(
                            label="Your Review Notes",
                            placeholder="Add your review notes here…",
                            value=panel_state["step_review_notes"].get(step_key, ""),
                        ).classes("w-full").props(
                            'outlined '
                            'input-style="height: 150px; overflow-y: auto;" '
                            'bg-color="amber-1"'
                        )
                        step_review_textareas[step_key] = review_ta_initial

                        def _on_review_change_initial(e: Any) -> None:
                            panel_state["step_review_notes"]["initial_idea"] = e.value if e.value else ""

                        review_ta_initial.on("change", _on_review_change_initial)

                        # Continue button — marks step 1 done, kicks off the
                        # graph which will run initial_idea node + interrupt,
                        # then immediately resumes to execute claims_drafting.
                        async def _on_continue_initial() -> None:
                            # Re-read disclosure from DB to pick up any changes
                            _render_initial_idea()
                            content = panel_state["step_contents"].get("initial_idea", "")
                            if not _has_content(content):
                                ui.notify(
                                    "No invention disclosure found. "
                                    "Please complete the Research tab first.",
                                    type="warning",
                                )
                                return
                            # Read and persist review notes (Req 4.2)
                            rn_ta = step_review_textareas.get("initial_idea")
                            if rn_ta is not None and hasattr(rn_ta, 'value'):
                                panel_state["step_review_notes"]["initial_idea"] = rn_ta.value or ""
                            review_notes = panel_state["step_review_notes"].get("initial_idea", "")
                            _persist_step("initial_idea", content, "completed", personality_mode="", review_notes=review_notes)
                            panel_state["completed_keys"].add("initial_idea")
                            panel_state["active_key"] = _find_active_step(panel_state["completed_keys"])
                            _refresh_chips()
                            _refresh_step_sections()
                            # Start graph fresh — initial_idea has no
                            # interrupt so the graph flows straight into
                            # claims_drafting, which pauses for review.
                            await _run_workflow_from("initial_idea", force_fresh=True)

                        cont_btn = ui.button(
                            "Continue to Next Step",
                            on_click=_on_continue_initial,
                            icon="arrow_forward",
                        ).props("color=primary")
                        step_continue_btns[step_key] = cont_btn

                        # Re-run button — reloads disclosure from DB and resets all steps
                        async def _on_rerun_initial() -> None:
                            if workflow_step_repo is not None:
                                try:
                                    workflow_step_repo.reset_from_step(topic_id, "initial_idea")
                                except Exception:
                                    logger.exception("Failed to reset from initial_idea")
                            panel_state["completed_keys"] -= set(WORKFLOW_STEP_ORDER)
                            panel_state["active_key"] = "initial_idea"
                            # Clear step_modes for all reset steps so stale badges don't show
                            for k in WORKFLOW_STEP_ORDER:
                                panel_state["step_modes"].pop(k, None)
                            # Reload disclosure from DB to pick up changes
                            _render_initial_idea()
                            _refresh_chips()
                            _refresh_step_sections()

                        rerun_btn = ui.button(
                            "Rerun this Step",
                            on_click=_on_rerun_initial,
                            icon="replay",
                        ).props("color=warning flat")
                        step_rerun_btns[step_key] = rerun_btn

                # --- Special handling for Step 9: Patent Draft ---
                elif step_key == "patent_draft":
                    with ui.expansion(
                        f"Step {step_num}: {display_name}",
                        icon="description",
                    ).classes("w-full q-mb-sm") as exp:
                        step_expansions[step_key] = exp

                        # Personality mode badge container (Req 8.1)
                        badge_ctr = ui.row().classes("gap-1 items-center q-mb-xs")
                        step_badge_containers[step_key] = badge_ctr
                        _render_badge(badge_ctr, step_key)

                        # Claims Editor
                        claims_ta = ui.textarea(
                            label="Patent Claims",
                            placeholder="Claims will be generated here…",
                            value=panel_state["claims"],
                        ).classes("w-full").props(
                            'outlined '
                            'input-style="height: 300px; overflow-y: auto;"'
                        )
                        claims_textarea_ref["el"] = claims_ta

                        def _on_claims_change(e: Any) -> None:
                            panel_state["claims"] = e.value if e.value else ""
                            _update_export_state()
                            _save_draft()

                        claims_ta.on("change", _on_claims_change)

                        # Description Editor
                        desc_ta = ui.textarea(
                            label="Patent Description",
                            placeholder="Description will be generated here…",
                            value=panel_state["description"],
                        ).classes("w-full").props(
                            'outlined '
                            'input-style="height: 400px; overflow-y: auto;"'
                        )
                        description_textarea_ref["el"] = desc_ta

                        def _on_desc_change(e: Any) -> None:
                            panel_state["description"] = e.value if e.value else ""
                            _update_export_state()
                            _save_draft()

                        desc_ta.on("change", _on_desc_change)

                        # We also keep a textarea for the step content (hidden, for consistency)
                        step_textareas[step_key] = desc_ta

                        # Review Notes textarea for step 9 (Req 3.1, 3.2)
                        review_ta_patent = ui.textarea(
                            label="Your Review Notes",
                            placeholder="Add your review notes here…",
                            value=panel_state["step_review_notes"].get(step_key, ""),
                        ).classes("w-full").props(
                            'outlined '
                            'input-style="height: 150px; overflow-y: auto;" '
                            'bg-color="amber-1"'
                        )
                        step_review_textareas[step_key] = review_ta_patent

                        def _on_review_change_patent(e: Any) -> None:
                            panel_state["step_review_notes"]["patent_draft"] = e.value if e.value else ""

                        review_ta_patent.on("change", _on_review_change_patent)

                        # Continue button (not applicable for last step, but kept for consistency)
                        cont_btn = ui.button(
                            "Continue to Next Step",
                            on_click=lambda: None,
                            icon="arrow_forward",
                        ).props("color=primary")
                        cont_btn.set_visibility(False)
                        step_continue_btns[step_key] = cont_btn

                        # Re-run button
                        async def _on_rerun_patent_draft() -> None:
                            if workflow_step_repo is not None:
                                try:
                                    workflow_step_repo.reset_from_step(topic_id, "patent_draft")
                                except Exception:
                                    logger.exception("Failed to reset from patent_draft")
                            panel_state["completed_keys"].discard("patent_draft")
                            panel_state["active_key"] = "patent_draft"
                            _refresh_chips()
                            _refresh_step_sections()
                            await _rerun_single_step("patent_draft")

                        rerun_btn = ui.button(
                            "Rerun this Step",
                            on_click=_on_rerun_patent_draft,
                            icon="replay",
                        ).props("color=warning flat")
                        step_rerun_btns[step_key] = rerun_btn

                        # Save button for step 9
                        def _on_save_patent_draft() -> None:
                            # Read current textarea values
                            if claims_textarea_ref.get("el") is not None:
                                panel_state["claims"] = claims_textarea_ref["el"].value or ""
                            if description_textarea_ref.get("el") is not None:
                                panel_state["description"] = description_textarea_ref["el"].value or ""

                            claims_val = panel_state["claims"]
                            desc_val = panel_state["description"]

                            # Save to patent_drafts table
                            _save_draft()

                            # Also persist to workflow_steps table
                            if desc_val.strip():
                                _persist_step("patent_draft", desc_val, "completed", personality_mode=panel_state["step_modes"].get("patent_draft", "critical"))
                                panel_state["step_contents"]["patent_draft"] = desc_val
                                panel_state["completed_keys"].add("patent_draft")
                            if claims_val.strip():
                                _persist_step("claims_drafting", claims_val, "completed", personality_mode=panel_state["step_modes"].get("claims_drafting", "critical"))
                                panel_state["step_contents"]["claims_drafting"] = claims_val

                            ui.notify("Patent draft saved.", type="positive")
                            _refresh_chips()
                            _update_export_state()

                        save_btn = ui.button(
                            "Save Edits",
                            on_click=_on_save_patent_draft,
                            icon="save",
                        ).props("flat color=primary")
                        step_save_btns[step_key] = save_btn

                # --- Generic steps 2–8 ---
                else:
                    _icon_map = {
                        "claims_drafting": "gavel",
                        "prior_art_search": "search",
                        "novelty_analysis": "science",
                        "consistency_review": "fact_check",
                        "market_potential": "trending_up",
                        "legal_clarification": "balance",
                        "disclosure_summary": "summarize",
                    }
                    with ui.expansion(
                        f"Step {step_num}: {display_name}",
                        icon=_icon_map.get(step_key, "article"),
                    ).classes("w-full q-mb-sm") as exp:
                        step_expansions[step_key] = exp

                        # Personality mode badge container (Req 8.1)
                        badge_ctr = ui.row().classes("gap-1 items-center q-mb-xs")
                        step_badge_containers[step_key] = badge_ctr
                        _render_badge(badge_ctr, step_key)

                        ta = ui.textarea(
                            label=display_name,
                            placeholder=f"{display_name} content will appear here…",
                            value=saved_content,
                        ).classes("w-full").props(
                            'outlined readonly '
                            'input-style="height: 300px; overflow-y: auto;"'
                        ).style('background-color: #f5f5f5;')
                        step_textareas[step_key] = ta

                        def _make_change_handler(sk: str):
                            def handler(e: Any) -> None:
                                panel_state["step_contents"][sk] = e.value if e.value else ""
                                cont = step_continue_btns.get(sk)
                                if cont is not None:
                                    if _has_content(e.value):
                                        cont.enable()
                                    else:
                                        cont.disable()
                            return handler

                        ta.on("change", _make_change_handler(step_key))

                        # Review Notes textarea (Req 3.1–3.6)
                        review_ta = ui.textarea(
                            label="Your Review Notes",
                            placeholder="Add your review notes here…",
                            value=panel_state["step_review_notes"].get(step_key, ""),
                        ).classes("w-full").props(
                            'outlined '
                            'input-style="height: 150px; overflow-y: auto;" '
                            'bg-color="amber-1"'
                        )
                        step_review_textareas[step_key] = review_ta

                        def _make_review_change_handler(sk: str):
                            def handler(e: Any) -> None:
                                panel_state["step_review_notes"][sk] = e.value if e.value else ""
                            return handler

                        review_ta.on("change", _make_review_change_handler(step_key))

                        # Continue button
                        def _make_continue_handler(sk: str):
                            async def handler() -> None:
                                # Read current value directly from textarea
                                # (change event may not have fired yet)
                                ta_el = step_textareas.get(sk)
                                if ta_el is not None and hasattr(ta_el, 'value'):
                                    panel_state["step_contents"][sk] = ta_el.value or ""
                                # Read current review notes from textarea
                                rn_ta = step_review_textareas.get(sk)
                                if rn_ta is not None and hasattr(rn_ta, 'value'):
                                    panel_state["step_review_notes"][sk] = rn_ta.value or ""
                                content = panel_state["step_contents"].get(sk, "")
                                if not _has_content(content):
                                    return
                                review_notes = panel_state["step_review_notes"].get(sk, "")
                                _persist_step(sk, content, "completed", personality_mode=panel_state["step_modes"].get(sk, "critical"), review_notes=review_notes)
                                panel_state["completed_keys"].add(sk)
                                panel_state["active_key"] = _find_active_step(panel_state["completed_keys"])
                                _refresh_chips()
                                _refresh_step_sections()
                                # Resume workflow from checkpoint
                                next_key = panel_state["active_key"]
                                if next_key is not None:
                                    await _run_workflow_from(sk)
                            return handler

                        cont_btn = ui.button(
                            "Continue to Next Step",
                            on_click=_make_continue_handler(step_key),
                            icon="arrow_forward",
                        ).props("color=primary")
                        step_continue_btns[step_key] = cont_btn

                        # Re-run button
                        def _make_rerun_handler(sk: str):
                            async def handler() -> None:
                                # Read current textarea value before resetting
                                ta_el = step_textareas.get(sk)
                                if ta_el is not None and hasattr(ta_el, 'value'):
                                    panel_state["step_contents"][sk] = ta_el.value or ""

                                if workflow_step_repo is not None:
                                    try:
                                        workflow_step_repo.reset_from_step(topic_id, sk)
                                    except Exception:
                                        logger.exception("Failed to reset from %s", sk)
                                # Remove this step and all subsequent from completed
                                idx = WORKFLOW_STEP_ORDER.index(sk)
                                for k in WORKFLOW_STEP_ORDER[idx:]:
                                    panel_state["completed_keys"].discard(k)
                                panel_state["active_key"] = sk
                                _refresh_chips()
                                _refresh_step_sections()
                                # Re-run this single step directly (calls the LLM)
                                await _rerun_single_step(sk)
                            return handler

                        rerun_btn = ui.button(
                            "Rerun this Step",
                            on_click=_make_rerun_handler(step_key),
                            icon="replay",
                        ).props("color=warning flat")
                        step_rerun_btns[step_key] = rerun_btn

                        # Save notes button (Req 4.1, 4.4)
                        def _make_save_handler(sk: str):
                            def handler() -> None:
                                rn_ta = step_review_textareas.get(sk)
                                if rn_ta is not None and hasattr(rn_ta, 'value'):
                                    panel_state["step_review_notes"][sk] = rn_ta.value or ""
                                content = panel_state["step_contents"].get(sk, "")
                                review_notes = panel_state["step_review_notes"].get(sk, "")
                                if _has_content(content):
                                    _persist_step(sk, content, "completed", personality_mode=panel_state["step_modes"].get(sk, "critical"), review_notes=review_notes)
                                    panel_state["completed_keys"].add(sk)
                                    display = _STEP_DISPLAY_NAMES.get(sk, sk)
                                    ui.notify(f"{display}: notes saved.", type="positive")
                            return handler

                        save_btn = ui.button(
                            "Save Notes",
                            on_click=_make_save_handler(step_key),
                            icon="note_add",
                        ).props("flat color=primary")
                        step_save_btns[step_key] = save_btn

        # ------------------------------------------------------------------
        # Export section (Req 10.4, 13.1–13.3)
        # ------------------------------------------------------------------
        export_warning = ui.label(
            "Export disabled: complete all workflow steps first."
        ).classes("text-warning text-caption q-mt-sm")

        def _on_export() -> None:
            """Generate a DOCX file with all workflow step content."""
            claims = panel_state["claims"]
            description = panel_state["description"]

            if not can_export(claims, description):
                ui.notify("Claims and description must not be empty.", type="warning")
                return

            from datetime import date
            from pathlib import Path
            from re import sub as re_sub

            from patent_system.export.docx_exporter import DOCXExporter

            try:
                template_dir = Path("src/patent_system/export/templates")
                template_name = None
                try:
                    from patent_system.config import AppSettings
                    _settings = AppSettings()
                    template_dir = _settings.docx_template_dir
                    template_name = _settings.docx_template_name
                except Exception:
                    pass

                topic_name = f"topic_{topic_id}"
                if conn is not None:
                    try:
                        from patent_system.db.repository import TopicRepository
                        topic = TopicRepository(conn).get_by_id(topic_id)
                        if topic:
                            topic_name = re_sub(r'[^\w\s-]', '', topic.name).strip().replace(' ', '_')
                    except Exception:
                        pass

                # Load references
                references: list[dict] = []
                from patent_system.gui.research_panel import _SOURCE_URLS
                if conn is not None:
                    try:
                        session_repo = ResearchSessionRepository(conn)
                        patent_repo = PatentRepository(conn)
                        paper_repo = ScientificPaperRepository(conn)
                        sessions = session_repo.get_by_topic(topic_id)
                        for session in sessions:
                            for rec in patent_repo.get_by_session(session["id"]):
                                record_id = rec.patent_number or ""
                                url_tpl = _SOURCE_URLS.get(rec.source, "")
                                url = url_tpl.format(id=record_id) if url_tpl and record_id and record_id != "UNKNOWN" else ""
                                references.append({
                                    "title": rec.title,
                                    "abstract": rec.abstract or "",
                                    "source": rec.source,
                                    "patent_number": rec.patent_number,
                                    "has_full_text": bool(rec.full_text),
                                    "relevance_score": rec.relevance_score,
                                    "url": url,
                                })
                            for rec in paper_repo.get_by_session(session["id"]):
                                record_id = rec.doi or ""
                                url_tpl = _SOURCE_URLS.get(rec.source, "")
                                url = url_tpl.format(id=record_id) if url_tpl and record_id else ""
                                references.append({
                                    "title": rec.title,
                                    "abstract": rec.abstract or "",
                                    "source": rec.source,
                                    "doi": rec.doi,
                                    "has_full_text": bool(rec.full_text),
                                    "relevance_score": rec.relevance_score,
                                    "url": url,
                                })
                    except Exception:
                        logger.exception("Failed to load references for export")

                # Include local documents
                if conn is not None:
                    try:
                        from patent_system.db.repository import LocalDocumentRepository
                        local_doc_repo = LocalDocumentRepository(conn)
                        for doc in local_doc_repo.get_by_topic(topic_id):
                            content = doc["content"]
                            abstract = content[:500].strip()
                            if len(content) > 500:
                                abstract += "…"
                            references.append({
                                "title": doc["filename"],
                                "abstract": abstract,
                                "source": "Local Document",
                                "has_full_text": True,
                                "url": "",
                            })
                    except Exception:
                        logger.exception("Failed to load local documents for export")

                # Load chat history
                chat_messages: list[dict] = []
                if conn is not None:
                    try:
                        from patent_system.db.repository import ChatHistoryRepository
                        chat_repo = ChatHistoryRepository(conn)
                        for msg in chat_repo.get_by_topic(topic_id):
                            chat_messages.append({"role": msg.role, "message": msg.message})
                    except Exception:
                        logger.exception("Failed to load chat history for export")

                today = date.today().isoformat()
                filename = f"{topic_name}_{today}.docx"
                output_path = Path(f"data/export/{filename}")

                # Build workflow_steps dict for export (Req 13.1–13.3)
                workflow_steps_dict: dict[str, str] = {}
                for sk in WORKFLOW_STEP_ORDER:
                    c = panel_state["step_contents"].get(sk, "")
                    if c:
                        workflow_steps_dict[sk] = c

                exporter = DOCXExporter(template_dir, template_name)
                exporter.export(
                    claims, description, output_path,
                    references=references,
                    chat_history=chat_messages if chat_messages else None,
                    workflow_steps=workflow_steps_dict if workflow_steps_dict else None,
                )

                ui.download(str(output_path))
                ui.notify("DOCX exported successfully.", type="positive")
                logger.info("Exported DOCX for topic %d to %s", topic_id, output_path)

            except Exception as exc:
                logger.exception("Failed to export DOCX for topic %d", topic_id)
                ui.notify(f"Export failed: {exc}", type="negative")

        export_button = ui.button(
            "Export to DOCX",
            on_click=_on_export,
            icon="download",
        ).props("color=secondary").classes("q-mt-sm")

        def _update_export_state() -> None:
            """Enable/disable export button based on content (Req 10.4)."""
            # Export is available when all steps have content and claims+description exist.
            # We check step_contents (not completed_keys) because the last step
            # has no Continue button and may not be in completed_keys.
            all_have_content = all(
                _has_content(panel_state["step_contents"].get(sk, ""))
                for sk in WORKFLOW_STEP_ORDER
            )
            exportable = all_have_content and can_export(
                panel_state["claims"], panel_state["description"]
            )
            if exportable:
                export_button.enable()
                export_warning.set_visibility(False)
            else:
                export_button.disable()
                export_warning.set_visibility(True)

        _update_export_state()

        # ------------------------------------------------------------------
        # Progress chips + spinner + status label (Req 11.1–11.5)
        # Rendered into the header's progress_bar_container when provided,
        # otherwise falls back to an in-panel element.
        # ------------------------------------------------------------------
        if progress_bar_container is not None:
            progress_bar_container.clear()
            _chips_parent = progress_bar_container
            _in_header = True
        else:
            _chips_parent = ui.element("div").classes(
                "w-full q-pa-sm bg-white"
            ).style(
                "position: sticky; bottom: 0; z-index: 10; "
                "border-top: 1px solid #e0e0e0;"
            )
            _in_header = False

        step_chip_elements: dict[str, Any] = {}

        with _chips_parent:
            with ui.column().classes("w-full gap-0 items-center"):
                with ui.row().classes("w-full gap-1 flex-wrap justify-center items-center"):
                    for step_name in WORKFLOW_STEPS:
                        chip = ui.chip(
                            step_name, icon="radio_button_unchecked", color="grey-4"
                        ).props("outline dense")
                        step_chip_elements[step_name] = chip

                    # Spinner sits next to the chips
                    spinner = ui.spinner(
                        "dots", size="sm",
                        color="white" if _in_header else "primary",
                    )
                    spinner.set_visibility(False)
                    panel_state["spinner"] = spinner

                progress_label = ui.label("").classes(
                    "text-caption text-center w-full"
                    + (" text-white" if _in_header else " text-grey")
                )

        def _refresh_chips() -> None:
            """Update chip icons/colours and status text to reflect current state."""
            active_key = panel_state["active_key"]
            completed = len(panel_state["completed_keys"])
            total = len(WORKFLOW_STEP_ORDER)

            for key in WORKFLOW_STEP_ORDER:
                display = _STEP_DISPLAY_NAMES[key]
                chip = step_chip_elements.get(display)
                if chip is None:
                    continue
                if key in panel_state["completed_keys"]:
                    chip._props["icon"] = "check_circle"
                    chip._props["color"] = "positive"
                elif key == active_key:
                    chip._props["icon"] = "hourglass_top"
                    chip._props["color"] = "primary"
                else:
                    chip._props["icon"] = "radio_button_unchecked"
                    chip._props["color"] = "grey-4"
                chip.update()

            # Update status text to reflect the CURRENT active step
            if panel_state.get("running"):
                display = _STEP_DISPLAY_NAMES.get(active_key or "", "")
                progress_label.set_text(f"⏳ Running: {display}…")
            elif completed == total:
                progress_label.set_text(f"✓ All {total} steps complete — ready to export")
            elif active_key:
                display = _STEP_DISPLAY_NAMES.get(active_key, active_key)
                has_content = _has_content(panel_state["step_contents"].get(active_key, ""))
                if has_content:
                    progress_label.set_text(f"Step {completed + 1}/{total}: Review {display} and continue")
                else:
                    progress_label.set_text(f"Step {completed + 1}/{total}: {display}")
            else:
                progress_label.set_text("")

        _refresh_chips()

        # Initial refresh of step sections
        _refresh_step_sections()

        # Expose helpers on the container for external wiring
        def set_claims(text: str) -> None:
            panel_state["claims"] = text
            if claims_textarea_ref.get("el") is not None:
                claims_textarea_ref["el"].value = text
            _update_export_state()
            _save_draft()

        def set_description(text: str) -> None:
            panel_state["description"] = text
            if description_textarea_ref.get("el") is not None:
                description_textarea_ref["el"].value = text
            _update_export_state()
            _save_draft()

        def set_current_step(step_name: str) -> None:
            panel_state["active_key"] = step_name

        container.set_claims = set_claims  # type: ignore[attr-defined]
        container.set_description = set_description  # type: ignore[attr-defined]
        container.set_current_step = set_current_step  # type: ignore[attr-defined]
