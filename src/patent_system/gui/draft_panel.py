"""Patent Draft panel UI for the Patent Analysis & Drafting System.

Provides the "Generate Patent Draft" button, expandable sections for
claims and description editors, an "Export to DOCX" button with
validation, a workflow step indicator, and an invention disclosure
review/edit placeholder.

Requirements: 16.6, 5.3, 5.4, 7.3, 7.4, 10.1, 10.6, 9.4, 2.5, 4.3
"""

from __future__ import annotations

import logging
from typing import Any

from nicegui import ui

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


def create_draft_panel(container: Any, topic_id: int) -> None:
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
        def _on_generate() -> None:
            """Placeholder handler — actual generation wired in task 12.1."""
            logger.info(
                "Generate Patent Draft clicked for topic %d", topic_id
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
