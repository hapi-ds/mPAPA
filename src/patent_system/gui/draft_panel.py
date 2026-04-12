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
from patent_system.db.repository import (
    InventionDisclosureRepository,
    PatentDraftRepository,
    PatentRepository,
    ResearchSessionRepository,
    ScientificPaperRepository,
)

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
    disclosure_repo: InventionDisclosureRepository | None = None,
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

    # Load saved draft from DB
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

    with container:
        ui.label("Patent Draft").classes("text-h6 q-mb-sm")

        # --- Workflow step indicator (Req 9.4) ---
        ui.label("Workflow Progress").classes("text-subtitle2 q-mt-sm")

        # Progress status label — shows current step in real time
        progress_label = ui.label("").classes("text-caption text-grey q-mb-xs")
        progress_bar = ui.linear_progress(value=0, show_value=False).classes("q-mb-sm")
        progress_bar.set_visibility(False)
        spinner = ui.spinner("dots", size="lg", color="primary").classes("q-mb-sm")
        spinner.set_visibility(False)

        # Step completion chips — one per workflow step
        step_chips_container = ui.row().classes("w-full gap-1 q-mb-md flex-wrap")
        step_chip_elements: dict[str, Any] = {}
        with step_chips_container:
            for step_name in WORKFLOW_STEPS:
                chip = ui.chip(step_name, icon="radio_button_unchecked", color="grey-4").props("outline")
                step_chip_elements[step_name] = chip

        def _mark_step_done(step_display_name: str) -> None:
            """Mark a step chip as completed."""
            chip = step_chip_elements.get(step_display_name)
            if chip is not None:
                chip._props["icon"] = "check_circle"
                chip._props["color"] = "positive"
                chip.update()

        def _mark_step_active(step_display_name: str) -> None:
            """Mark a step chip as currently running."""
            chip = step_chip_elements.get(step_display_name)
            if chip is not None:
                chip._props["icon"] = "hourglass_top"
                chip._props["color"] = "primary"
                chip.update()

        # --- Instruction banner (hidden by default) ---
        instruction_card = ui.card().classes("w-full q-mb-sm bg-yellow-1")
        instruction_card.set_visibility(False)
        with instruction_card:
            with ui.row().classes("items-center gap-2"):
                ui.icon("info", color="warning").classes("text-h5")
                instruction_text = ui.label("").classes("text-body2")

        # --- Invention disclosure review (Req 2.5, 4.3) ---
        with ui.expansion(
            "Invention Disclosure Review",
            icon="description",
        ).classes("w-full q-mb-sm") as disclosure_expansion:
            # Load and display actual disclosure data
            _disc = None
            if disclosure_repo is not None:
                try:
                    _disc = disclosure_repo.get_by_topic(topic_id)
                except Exception:
                    pass

            if _disc is not None:
                ui.label("Primary Description").classes("text-subtitle2 q-mt-sm")
                ui.label(_disc["primary_description"]).classes(
                    "text-body2 q-pa-sm bg-grey-1 rounded"
                ).style("white-space: pre-wrap;")
                terms = _disc.get("search_terms", [])
                if terms:
                    ui.label("Search Terms").classes("text-subtitle2 q-mt-sm")
                    with ui.row().classes("gap-1 flex-wrap"):
                        for term in terms:
                            ui.chip(term, color="primary").props("outline dense")
            else:
                ui.label(
                    "No invention disclosure saved yet. "
                    "Go to the Research tab to enter your invention description first."
                ).classes("text-grey q-pa-sm")

        # --- Generate Patent Draft button (Req 16.6) ---
        async def _on_generate() -> None:
            """Invoke the compiled workflow with streaming progress updates."""
            if workflow is None:
                ui.notify(
                    "Workflow not available — please restart the application.",
                    type="negative",
                )
                return

            logger.info("Generate Patent Draft clicked for topic %d", topic_id)

            # Hide any previous instructions
            instruction_card.set_visibility(False)

            # Load stored disclosure (Req 7.1, 7.2, 7.4)
            saved_disclosure: dict | None = None
            if disclosure_repo is not None:
                try:
                    saved_disclosure = disclosure_repo.get_by_topic(topic_id)
                except Exception:
                    logger.exception("Failed to load disclosure for topic %d", topic_id)

            if saved_disclosure is None:
                ui.notify(
                    "No invention disclosure found. Please complete the Research tab first.",
                    type="warning",
                    close_button=True,
                )
                return

            # Show progress
            progress_label.set_text("Starting patent generation…")
            progress_bar.set_visibility(True)
            progress_bar.set_value(0)
            spinner.set_visibility(True)
            generate_button.disable()

            # Build invention disclosure from stored data
            disclosure: dict[str, Any] = {
                "technical_problem": saved_disclosure["primary_description"],
                "novel_features": saved_disclosure.get("search_terms", []),
                "implementation_details": "",
                "potential_variations": [],
            }

            # Concatenate abstracts and full texts from both tables (Req 7.3)
            if conn is not None:
                try:
                    session_repo = ResearchSessionRepository(conn)
                    patent_repo = PatentRepository(conn)
                    paper_repo = ScientificPaperRepository(conn)
                    from patent_system.db.repository import LocalDocumentRepository
                    local_doc_repo = LocalDocumentRepository(conn)
                    sessions = session_repo.get_by_topic(topic_id)
                    text_parts: list[str] = []
                    for session in sessions:
                        for rec in patent_repo.get_by_session(session["id"]):
                            if rec.abstract:
                                text_parts.append(rec.abstract)
                            if rec.full_text:
                                text_parts.append(rec.full_text)
                        for rec in paper_repo.get_by_session(session["id"]):
                            if rec.abstract:
                                text_parts.append(rec.abstract)
                            if rec.full_text:
                                text_parts.append(rec.full_text)
                    # Include local documents
                    for doc in local_doc_repo.get_by_topic(topic_id):
                        if doc["content"]:
                            text_parts.append(doc["content"][:5000])
                    if text_parts:
                        disclosure["implementation_details"] = "\n".join(text_parts)
                except Exception:
                    logger.exception("Failed to load prior art for topic %d", topic_id)

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

            # Check if there's an existing interrupted checkpoint to resume
            existing_snapshot = await asyncio.to_thread(workflow.get_state, config)
            is_resuming = (
                existing_snapshot is not None
                and hasattr(existing_snapshot, "next")
                and existing_snapshot.next
            )

            if is_resuming:
                # Resume from interrupt — update claims from editor if edited
                logger.info("Resuming workflow from interrupt for topic %d", topic_id)
                progress_label.set_text("Resuming workflow…")

                # Inject edited claims into the checkpoint before resuming
                if panel_state["claims"]:
                    await asyncio.to_thread(
                        workflow.update_state,
                        config,
                        {"claims_text": panel_state["claims"], "review_approved": True},
                    )

                stream_input = None  # None = resume from checkpoint
            else:
                stream_input = initial_state

            total_steps = len(WORKFLOW_STEPS)

            try:
                import queue

                event_queue: queue.Queue = queue.Queue()
                stream_error: list[Exception] = []

                def _run_stream() -> None:
                    """Run workflow.stream in a thread, pushing events to queue."""
                    try:
                        for event in workflow.stream(stream_input, config):
                            event_queue.put(event)
                    except Exception as exc:
                        stream_error.append(exc)
                    finally:
                        event_queue.put(None)  # sentinel

                # Start streaming in background
                loop = asyncio.get_event_loop()
                stream_future = loop.run_in_executor(None, _run_stream)

                # Process events as they arrive, updating UI in real time
                while True:
                    # Poll the queue without blocking the event loop
                    event = await asyncio.to_thread(event_queue.get)
                    if event is None:
                        break  # stream finished

                    for node_name, node_output in event.items():
                        if node_name == "__end__":
                            continue
                        step_key = node_output.get("current_step", node_name) if isinstance(node_output, dict) else node_name
                        display_name = _STEP_DISPLAY_NAMES.get(step_key, step_key)
                        progress_label.set_text(f"Running: {display_name}…")

                        if display_name in WORKFLOW_STEPS:
                            step_idx = WORKFLOW_STEPS.index(display_name)
                            progress_bar.set_value((step_idx + 1) / total_steps)
                            # Mark previous steps as done, current as active
                            for i, sn in enumerate(WORKFLOW_STEPS):
                                if i < step_idx:
                                    _mark_step_done(sn)
                                elif i == step_idx:
                                    _mark_step_done(sn)

                        # Extract claims/description as they become available
                        if isinstance(node_output, dict):
                            if node_output.get("claims_text"):
                                set_claims(node_output["claims_text"])
                                claims_expansion.open()
                            if node_output.get("description_text"):
                                set_description(node_output["description_text"])
                                description_expansion.open()

                # Wait for the thread to finish
                await stream_future

                # Re-raise any error from the stream thread
                if stream_error:
                    raise stream_error[0]

                # Check if the workflow was interrupted for human review
                # (stream() doesn't raise GraphInterrupt — it just stops)
                snapshot = await asyncio.to_thread(workflow.get_state, config)
                is_interrupted = (
                    snapshot is not None
                    and hasattr(snapshot, "next")
                    and snapshot.next
                )

                if is_interrupted:
                    # Human review needed
                    logger.info("Workflow paused for human review on topic %d", topic_id)
                    spinner.set_visibility(False)

                    if snapshot.values:
                        checkpoint_claims = snapshot.values.get("claims_text", "")
                        if checkpoint_claims:
                            set_claims(checkpoint_claims)
                            claims_expansion.open()

                    progress_label.set_text("⏸ Paused — your review is needed")
                    progress_bar.set_value(0.7)

                    # Mark completed steps
                    for sn in ["Disclosure", "Prior Art Search", "Novelty Analysis", "Claims Drafting"]:
                        _mark_step_done(sn)
                    _mark_step_active("Consistency Review")

                    instruction_text.set_text(
                        "The AI drafted claims and ran consistency checks, but needs your input. "
                        "Please review and edit the claims in the Claims Editor below, then click "
                        "'Generate Patent Draft' again to continue generating the full description."
                    )
                    instruction_card.set_visibility(True)

                    if claims_textarea is not None:
                        claims_textarea.enabled = True

                    ui.notify(
                        "Review needed — please check the claims and click Generate again.",
                        type="warning",
                        close_button=True,
                    )
                else:
                    # Workflow completed fully
                    spinner.set_visibility(False)
                    progress_label.set_text("✓ Patent draft complete")
                    progress_bar.set_value(1.0)
                    for sn in WORKFLOW_STEPS:
                        _mark_step_done(sn)
                    claims_expansion.open()
                    description_expansion.open()

                    ui.notify("Patent draft generated successfully.", type="positive", close_button=True)
                    logger.info("Workflow completed for topic %d", topic_id)

            except GraphInterrupt:
                spinner.set_visibility(False)
                logger.info("GraphInterrupt caught for topic %d", topic_id)
                progress_label.set_text("⏸ Paused — your review is needed")
                instruction_text.set_text(
                    "The AI needs your input. Please review and edit the claims below, "
                    "then click 'Generate Patent Draft' again to continue."
                )
                instruction_card.set_visibility(True)
                if claims_textarea is not None:
                    claims_textarea.enabled = True
                claims_expansion.open()

            except Exception as exc:
                spinner.set_visibility(False)
                failed_step = panel_state.get("current_step", "unknown")
                logger.exception("Workflow failed for topic %d at step '%s': %s", topic_id, failed_step, exc)
                progress_label.set_text(f"✗ Failed at {failed_step}")
                ui.notify(f"Workflow failed: {exc}", type="negative", close_button=True)

            finally:
                generate_button.enable()

        generate_button = ui.button(
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
            ).classes("w-full").props('outlined autogrow input-style="min-height: 300px"')

            def _on_claims_change(e: Any) -> None:
                panel_state["claims"] = e.value if e.value else ""
                _update_export_state()
                _save_draft()
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
            ).classes("w-full").props('outlined autogrow input-style="min-height: 400px"')

            def _on_description_change(e: Any) -> None:
                panel_state["description"] = e.value if e.value else ""
                _update_export_state()
                _save_draft()
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
            """Generate a DOCX file with references and trigger browser download."""
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
                # Use settings if available, otherwise defaults
                template_dir = Path("src/patent_system/export/templates")
                template_name = None
                try:
                    from patent_system.config import AppSettings
                    _settings = AppSettings()
                    template_dir = _settings.docx_template_dir
                    template_name = _settings.docx_template_name
                except Exception:
                    pass

                # Load topic name for filename
                topic_name = f"topic_{topic_id}"
                if conn is not None:
                    try:
                        from patent_system.db.repository import TopicRepository
                        topic = TopicRepository(conn).get_by_id(topic_id)
                        if topic:
                            # Sanitize for filename
                            topic_name = re_sub(r'[^\w\s-]', '', topic.name).strip().replace(' ', '_')
                    except Exception:
                        pass

                # Load references from both patent and paper tables
                references: list[dict] = []
                # URL templates for clickable links
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
                                    "url": url,
                                })
                    except Exception:
                        logger.exception("Failed to load references for export")

                # Include local documents as references
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

                exporter = DOCXExporter(template_dir, template_name)
                exporter.export(
                    claims, description, output_path,
                    references=references,
                    chat_history=chat_messages if chat_messages else None,
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
        def _save_draft() -> None:
            """Persist current claims and description to DB."""
            if draft_repo is not None:
                try:
                    draft_repo.upsert(
                        topic_id,
                        panel_state["claims"],
                        panel_state["description"],
                    )
                except Exception:
                    logger.debug("Failed to auto-save draft for topic %d", topic_id, exc_info=True)

        def set_claims(text: str) -> None:
            """Set claims text programmatically."""
            panel_state["claims"] = text
            if claims_textarea is not None:
                claims_textarea.value = text
            _update_export_state()
            _save_draft()

        def set_description(text: str) -> None:
            """Set description text programmatically."""
            panel_state["description"] = text
            if description_textarea is not None:
                description_textarea.value = text
            _update_export_state()
            _save_draft()

        def set_current_step(step_name: str) -> None:
            """Update the workflow step indicator."""
            panel_state["current_step"] = step_name

        container.set_claims = set_claims  # type: ignore[attr-defined]
        container.set_description = set_description  # type: ignore[attr-defined]
        container.set_current_step = set_current_step  # type: ignore[attr-defined]
