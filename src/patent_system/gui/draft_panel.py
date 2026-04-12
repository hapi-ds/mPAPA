"""Patent Draft panel UI for the Patent Analysis & Drafting System.

Provides the "Generate Patent Draft" button, expandable sections for
claims and description editors, an "Export to DOCX" button with
validation, a workflow step indicator, and an invention disclosure
review/edit placeholder.

Requirements: 16.6, 5.3, 5.4, 7.3, 7.4, 10.1, 10.6, 9.4, 2.5, 4.3
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from typing import TYPE_CHECKING, Any

from langgraph.errors import GraphInterrupt
from nicegui import ui

from patent_system.agents.state import PatentWorkflowState
from patent_system.db.repository import PatentRepository, ResearchSessionRepository

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)

# Workflow steps displayed in the step indicator (Req 9.4)
WORKFLOW_STEPS: list[str] = [
    "Disclosure",
    "Prior Art Search",
    "Novelty Analysis",
    "Claims Drafting",
    "Consistency Review",
    "Description Drafting",
]

# Map node current_step values to WORKFLOW_STEPS display names
_STEP_DISPLAY_NAMES: dict[str, str] = {
    "disclosure": "Disclosure",
    "prior_art_search": "Prior Art Search",
    "novelty_analysis": "Novelty Analysis",
    "claims_drafting": "Claims Drafting",
    "consistency_review": "Consistency Review",
    "description_drafting": "Description Drafting",
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


def create_draft_panel(
    container: Any,
    topic_id: int,
    *,
    workflow: CompiledStateGraph | None = None,
    conn: sqlite3.Connection | None = None,
) -> None:
    """Populate *container* with the Patent Draft panel UI components.

    The panel contains:
    - A workflow step indicator showing the current pipeline stage (Req 9.4)
    - An invention disclosure review/edit placeholder (Req 2.5)
    - A "Generate Patent Draft" button (Req 16.6)
    - Expandable sections for claims editor and description editor
      (Req 5.3, 7.3)
    - An "Export to DOCX" button with validation (Req 10.1, 10.6)

    The actual agent invocations and database persistence will be wired
    in task 12.1.  For now the generate button logs the action and the
    editors hold local state.

    Args:
        container: A NiceGUI container element (e.g. ``ui.column``) to
            populate.  The container is cleared before adding content.
        topic_id: The active topic ID whose patent draft is shown.
    """
    container.clear()

    # Panel-local state
    panel_state: dict[str, Any] = {
        "claims": "",
        "description": "",
        "current_step": "Disclosure",
    }

    with container:
        ui.label("Patent Draft").classes("text-h6 q-mb-sm")

        # --- Workflow step indicator (Req 9.4) ---
        ui.label("Workflow Progress").classes("text-subtitle2 q-mt-sm")
        with ui.stepper().props("flat").classes("w-full q-mb-md") as stepper:
            for step_name in WORKFLOW_STEPS:
                ui.step(step_name)

        # --- Invention disclosure review/edit placeholder (Req 2.5, 4.3) ---
        with ui.expansion(
            "Invention Disclosure Review",
            icon="description",
        ).classes("w-full q-mb-sm"):
            ui.label(
                "The invention disclosure details will appear here "
                "once the disclosure interview is complete. "
                "You will be able to review and edit the extracted "
                "details before proceeding."
            ).classes("text-grey q-pa-sm")

        # --- Generate Patent Draft button (Req 16.6) ---
        async def _on_generate() -> None:
            """Invoke the compiled workflow and update the UI with results.

            Builds an initial PatentWorkflowState, invokes the workflow
            with a thread config for checkpointing, updates the stepper
            UI, and populates the claims/description textareas.

            Handles GraphInterrupt (human_review) by notifying the user
            and enabling the claims editor, and general exceptions by
            showing an error notification with the failed step name.

            Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7
            """
            if workflow is None:
                ui.notify(
                    "Workflow not available — please restart the application.",
                    type="negative",
                )
                logger.warning(
                    "Generate clicked but workflow is None for topic %d",
                    topic_id,
                )
                return

            logger.info(
                "Generate Patent Draft clicked for topic %d", topic_id
            )

            ui.notify(
                "Generating patent draft… this may take a moment.",
                type="info",
                spinner=True,
                timeout=0,
                close_button=True,
            )

            # Build invention disclosure from saved research data
            disclosure: dict[str, Any] | None = None
            if conn is not None:
                try:
                    session_repo = ResearchSessionRepository(conn)
                    patent_repo = PatentRepository(conn)
                    sessions = session_repo.get_by_topic(topic_id)
                    queries = [s["query"] for s in sessions if s.get("query")]
                    abstracts: list[str] = []
                    for session in sessions:
                        for rec in patent_repo.get_by_session(session["id"]):
                            if rec.abstract:
                                abstracts.append(rec.abstract)
                    if queries or abstracts:
                        disclosure = {
                            "technical_problem": "; ".join(queries) if queries else "Not specified",
                            "novel_features": queries[:5],
                            "implementation_details": "\n".join(abstracts[:10]),
                            "potential_variations": [],
                        }
                except Exception:
                    logger.exception("Failed to load research data for topic %d", topic_id)

            # Build initial state with all required keys (Req 6.1)
            initial_state: PatentWorkflowState = {
                "topic_id": topic_id,
                "invention_disclosure": disclosure,
                "interview_messages": [],
                "prior_art_results": [],
                "failed_sources": [],
                "novelty_analysis": None,
                "claims_text": "",
                "description_text": "",
                "review_feedback": "",
                "review_approved": False,
                "iteration_count": 0,
                "current_step": "disclosure",
            }

            config = {"configurable": {"thread_id": f"topic-{topic_id}"}}

            try:
                result = await asyncio.to_thread(workflow.invoke, initial_state, config)

                # Update stepper UI from current_step (Req 6.2)
                step_key = result.get("current_step", "")
                display_name = _STEP_DISPLAY_NAMES.get(step_key, "")
                if display_name:
                    step_index = WORKFLOW_STEPS.index(display_name)
                    set_current_step(display_name)
                    # Advance stepper to the completed step
                    for _i in range(step_index + 1):
                        stepper.next()

                # Populate claims textarea (Req 6.3)
                claims_result = result.get("claims_text", "")
                if claims_result:
                    set_claims(claims_result)

                # Populate description textarea (Req 6.4)
                description_result = result.get("description_text", "")
                if description_result:
                    set_description(description_result)

                # Open the editor sections so the user sees the results
                claims_expansion.open()
                description_expansion.open()

                ui.notify(
                    "Patent draft generated successfully.",
                    type="positive",
                    close_button=True,
                )

                logger.info(
                    "Workflow completed for topic %d at step '%s'",
                    topic_id,
                    step_key,
                )

            except GraphInterrupt:
                # Human review interrupt (Req 6.6)
                logger.info(
                    "Workflow interrupted for human review on topic %d",
                    topic_id,
                )
                ui.notify(
                    "Human review required: please review and edit the "
                    "claims before resuming the workflow.",
                    type="warning",
                    close_button=True,
                )
                # Populate claims from the latest checkpoint state
                try:
                    snapshot = workflow.get_state(config)
                    if snapshot and snapshot.values:
                        checkpoint_claims = snapshot.values.get(
                            "claims_text", ""
                        )
                        if checkpoint_claims:
                            set_claims(checkpoint_claims)
                        # Update stepper to consistency review (last
                        # completed step before human_review)
                        cs = snapshot.values.get(
                            "current_step", "consistency_review"
                        )
                        display = _STEP_DISPLAY_NAMES.get(
                            cs, "Consistency Review"
                        )
                        set_current_step(display)
                except Exception:
                    logger.debug(
                        "Could not read checkpoint state after interrupt",
                        exc_info=True,
                    )
                # Enable claims editor for manual editing
                if claims_textarea is not None:
                    claims_textarea.enabled = True
                claims_expansion.open()

            except Exception as exc:
                # General error handling (Req 6.5)
                failed_step = panel_state.get("current_step", "unknown")
                logger.exception(
                    "Workflow failed for topic %d at step '%s': %s",
                    topic_id,
                    failed_step,
                    exc,
                )
                ui.notify(
                    f"Workflow failed at step '{failed_step}': {exc}",
                    type="negative",
                    close_button=True,
                )

        ui.button(
            "Generate Patent Draft",
            on_click=_on_generate,
            icon="auto_fix_high",
        ).props("color=primary").classes("q-mb-md")

        # --- Claims editor (Req 5.3, 5.4) ---
        claims_textarea: ui.textarea | None = None
        with ui.expansion(
            "Claims Editor",
            icon="gavel",
        ).classes("w-full q-mb-sm") as claims_expansion:
            claims_textarea = ui.textarea(
                label="Patent Claims",
                placeholder="Claims will be generated here…",
                value=panel_state["claims"],
            ).classes("w-full").props("outlined autogrow")

            def _on_claims_change(e: Any) -> None:
                panel_state["claims"] = e.value if e.value else ""
                _update_export_state()
                logger.debug(
                    "Claims updated for topic %d (%d chars)",
                    topic_id,
                    len(panel_state["claims"]),
                )

            claims_textarea.on("change", _on_claims_change)

        # --- Description editor (Req 7.3, 7.4) ---
        description_textarea: ui.textarea | None = None
        with ui.expansion(
            "Description Editor",
            icon="article",
        ).classes("w-full q-mb-sm") as description_expansion:
            description_textarea = ui.textarea(
                label="Patent Description",
                placeholder="Description will be generated here…",
                value=panel_state["description"],
            ).classes("w-full").props("outlined autogrow")

            def _on_description_change(e: Any) -> None:
                panel_state["description"] = e.value if e.value else ""
                _update_export_state()
                logger.debug(
                    "Description updated for topic %d (%d chars)",
                    topic_id,
                    len(panel_state["description"]),
                )

            description_textarea.on("change", _on_description_change)

        # --- Export warning label (Req 10.6) ---
        export_warning = ui.label(
            "Export disabled: claims and description must not be empty."
        ).classes("text-warning text-caption q-mt-sm")

        # --- Export to DOCX button (Req 10.1, 10.6) ---
        def _on_export() -> None:
            """Placeholder handler — actual export wired in task 12.1."""
            logger.info(
                "Export to DOCX clicked for topic %d", topic_id
            )

        export_button = ui.button(
            "Export to DOCX",
            on_click=_on_export,
            icon="download",
        ).props("color=secondary").classes("q-mt-sm")

        def _update_export_state() -> None:
            """Enable/disable export button based on content (Req 10.6)."""
            exportable = can_export(
                panel_state["claims"],
                panel_state["description"],
            )
            if exportable:
                export_button.enable()
                export_warning.set_visibility(False)
            else:
                export_button.disable()
                export_warning.set_visibility(True)

        # Initial state: export disabled (no content yet)
        _update_export_state()

        # Expose helpers on the container for external wiring
        def set_claims(text: str) -> None:
            """Set claims text programmatically."""
            panel_state["claims"] = text
            if claims_textarea is not None:
                claims_textarea.value = text
            _update_export_state()

        def set_description(text: str) -> None:
            """Set description text programmatically."""
            panel_state["description"] = text
            if description_textarea is not None:
                description_textarea.value = text
            _update_export_state()

        def set_current_step(step_name: str) -> None:
            """Update the workflow step indicator."""
            panel_state["current_step"] = step_name

        container.set_claims = set_claims  # type: ignore[attr-defined]
        container.set_description = set_description  # type: ignore[attr-defined]
        container.set_current_step = set_current_step  # type: ignore[attr-defined]
