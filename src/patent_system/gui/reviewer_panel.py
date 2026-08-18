"""Patent Review panel — independent patent reviewer with Rule Card system.

Provides a NiceGUI tab for reviewing patents against official examination
guidelines. Completely independent of the workflow — can review any patent
text (imported, pasted, or taken from the current draft).

Requirements: REQ-GUI-1 through REQ-GUI-10
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from nicegui import ui

from patent_system.config import AppSettings, get_base_dir
from patent_system.reviewer.card_registry import CardRegistry
from patent_system.reviewer.engine import ReviewEngine, ReviewReport
from patent_system.reviewer.report import render_markdown
from patent_system.reviewer.rule_card import RuleCard

logger = logging.getLogger(__name__)

# Jurisdiction display labels
_JURISDICTIONS = {
    "EP": "🇪🇺 Europe (EPO)",
    "US": "🇺🇸 United States (USPTO)",
    "PCT": "🌐 PCT (WIPO)",
    "CN": "🇨🇳 China (CNIPA)",
    "JP": "🇯🇵 Japan (JPO)",
    "KR": "🇰🇷 South Korea (KIPO)",
    "IN": "🇮🇳 India (IPO)",
    "DE": "🇩🇪 Germany (DPMA)",
}


def create_reviewer_panel(
    container: ui.column,
    *,
    settings: AppSettings | None = None,
    topic_id: int | None = None,
    claims_text: str = "",
    description_text: str = "",
) -> None:
    """Create the Patent Review panel inside the given container.

    Args:
        container: NiceGUI column to render into.
        settings: App settings for LLM connection URL and model name.
        topic_id: Optional current topic ID (for linking reviews).
        claims_text: Pre-filled claims text from the draft workflow.
        description_text: Pre-filled description text from the draft workflow.
    """
    container.clear()

    app_settings = settings or AppSettings()

    # Determine card cache directory
    card_cache_dir = get_base_dir() / "data" / "rule_cards"
    card_cache_dir.mkdir(parents=True, exist_ok=True)

    # Also check the committed rule-cards directory in the project
    project_cards_dir = get_base_dir() / "rule-cards"

    # Initialize registry — prefer project cards, fall back to cache
    registry = CardRegistry(
        cache_dir=project_cards_dir if project_cards_dir.exists() else card_cache_dir,
        repo_url="https://raw.githubusercontent.com/hapi-ds/mPAPA/main/rule-cards",
    )

    # State
    state: dict[str, Any] = {
        "patent_text": "",
        "selected_cards": [],
        "report": None,
        "is_running": False,
    }

    with container:
        ui.label("Patent Review").classes("text-h5 q-mb-md")
        ui.label(
            "Review a patent against official examination guidelines. "
            "Select jurisdiction and task, then start the review."
        ).classes("text-body2 text-grey-7 q-mb-md")

        # === INPUT SECTION ===
        with ui.card().classes("w-full q-mb-md"):
            ui.label("Patent Text").classes("text-subtitle1 font-bold")

            with ui.row().classes("w-full gap-2 q-mb-sm"):
                paste_btn = ui.button("Paste Text", icon="content_paste")
                import_btn = ui.button("Import File", icon="upload_file")
                if claims_text or description_text:
                    use_draft_btn = ui.button(
                        "Use Current Draft", icon="description"
                    ).props("color=primary")
                else:
                    use_draft_btn = ui.button(
                        "Use Current Draft", icon="description"
                    ).props("disable")

            patent_input = ui.textarea(
                label="Patent text (claims + description)",
                placeholder="Paste your patent claims and description here, or import a file...",
            ).classes("w-full").props('rows=12 outlined')

            # Pre-fill if draft text available
            if claims_text or description_text:
                combined = ""
                if claims_text:
                    combined += f"=== CLAIMS ===\n\n{claims_text}\n\n"
                if description_text:
                    combined += f"=== DESCRIPTION ===\n\n{description_text}"
                patent_input.value = combined
                state["patent_text"] = combined

            def _on_patent_text_change(e):
                state["patent_text"] = e.value

            patent_input.on("change", _on_patent_text_change)

            # Use Draft button handler
            def _use_draft():
                combined = ""
                if claims_text:
                    combined += f"=== CLAIMS ===\n\n{claims_text}\n\n"
                if description_text:
                    combined += f"=== DESCRIPTION ===\n\n{description_text}"
                patent_input.value = combined
                state["patent_text"] = combined
                ui.notify("Draft text loaded", type="positive")

            use_draft_btn.on_click(_use_draft)

            # Import file handler
            async def _handle_upload(e):
                content = e.content.read()
                try:
                    text = content.decode("utf-8")
                except UnicodeDecodeError:
                    # Try to extract from PDF/DOCX via pymupdf
                    import tempfile
                    import pymupdf
                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                        f.write(content)
                        tmp_path = f.name
                    try:
                        doc = pymupdf.open(tmp_path)
                        text = "\n\n".join(
                            page.get_text("text") for page in doc
                        )
                        doc.close()
                    except Exception:
                        ui.notify("Could not read file", type="negative")
                        return
                    finally:
                        Path(tmp_path).unlink(missing_ok=True)

                patent_input.value = text
                state["patent_text"] = text
                ui.notify(f"Loaded {len(text)} characters", type="positive")

            upload = ui.upload(
                on_upload=_handle_upload,
                label="Drop file here (PDF, TXT, MD)",
                auto_upload=True,
            ).classes("w-full q-mt-sm").props('accept=".pdf,.txt,.md,.docx"')
            upload.set_visibility(False)

            def _toggle_upload():
                upload.set_visibility(not upload.visible)

            import_btn.on_click(_toggle_upload)

        # === CARD SELECTION SECTION ===
        with ui.card().classes("w-full q-mb-md"):
            ui.label("Rule Cards").classes("text-subtitle1 font-bold")

            # Discover available cards
            available_cards = registry.list_local()
            cards_by_jurisdiction: dict[str, list[RuleCard]] = {}
            for card in available_cards:
                cards_by_jurisdiction.setdefault(card.jurisdiction, []).append(card)

            if not available_cards:
                ui.label(
                    "No rule cards found. Place card JSON files in rule-cards/ directory."
                ).classes("text-warning")
            else:
                ui.label(
                    f"{len(available_cards)} cards available across "
                    f"{len(cards_by_jurisdiction)} jurisdiction(s)"
                ).classes("text-caption text-grey-6 q-mb-sm")

                # Card checkboxes grouped by jurisdiction
                card_checkboxes: dict[str, ui.checkbox] = {}

                for jurisdiction, cards in sorted(cards_by_jurisdiction.items()):
                    label = _JURISDICTIONS.get(jurisdiction, jurisdiction)
                    with ui.expansion(label, icon="gavel").classes("w-full"):
                        for card in sorted(cards, key=lambda c: c.task_label):
                            cb = ui.checkbox(
                                f"{card.task_label} ({len(card.rules)} rules)",
                                value=False,
                            )
                            card_checkboxes[card.card_id] = cb

                def _get_selected_card_ids() -> list[str]:
                    return [
                        card_id
                        for card_id, cb in card_checkboxes.items()
                        if cb.value
                    ]

                # Quick-select buttons
                with ui.row().classes("q-mt-sm gap-2"):
                    def _select_all_ep():
                        for card_id, cb in card_checkboxes.items():
                            if card_id.startswith("epo_"):
                                cb.value = True

                    def _select_all_us():
                        for card_id, cb in card_checkboxes.items():
                            if card_id.startswith("uspto_"):
                                cb.value = True

                    def _deselect_all():
                        for cb in card_checkboxes.values():
                            cb.value = False

                    ui.button("All EP", on_click=_select_all_ep, icon="eu").props("flat dense")
                    ui.button("All US", on_click=_select_all_us, icon="flag").props("flat dense")
                    ui.button("Clear", on_click=_deselect_all, icon="clear").props("flat dense")

        # === RUN REVIEW BUTTON ===
        with ui.row().classes("w-full justify-center q-mb-md"):
            run_spinner = ui.spinner("bars", size="md")
            run_spinner.set_visibility(False)
            run_status = ui.label("")
            run_status.set_visibility(False)

            async def _run_review():
                # Validate inputs
                text = patent_input.value.strip()
                if not text:
                    ui.notify("Please enter patent text first", type="warning")
                    return

                selected_ids = _get_selected_card_ids()
                if not selected_ids:
                    ui.notify("Please select at least one rule card", type="warning")
                    return

                # Disable UI during review
                state["is_running"] = True
                run_btn.props("disable loading")
                run_spinner.set_visibility(True)
                run_status.set_visibility(True)
                run_status.set_text(f"Reviewing against {len(selected_ids)} card(s)...")

                try:
                    engine = ReviewEngine(
                        lm_studio_base_url=app_settings.lm_studio_base_url,
                        model_name="local-model",
                        card_registry=registry,
                    )

                    report = await engine.review(
                        patent_text=text,
                        card_ids=selected_ids,
                    )
                    state["report"] = report
                    _display_report(report)
                    ui.notify(
                        f"Review complete: {len(report.findings)} findings",
                        type="positive",
                    )
                except Exception as exc:
                    logger.exception("Review failed")
                    ui.notify(f"Review failed: {exc}", type="negative")
                finally:
                    state["is_running"] = False
                    run_btn.props(remove="disable loading")
                    run_spinner.set_visibility(False)
                    run_status.set_visibility(False)

            run_btn = ui.button(
                "▶ Start Review",
                on_click=_run_review,
                icon="play_arrow",
            ).props("color=primary size=lg")

        # === RESULTS SECTION ===
        results_container = ui.column().classes("w-full")

        def _display_report(report: ReviewReport) -> None:
            """Render the review report in the results container."""
            results_container.clear()
            with results_container:
                # Summary bar
                stats = report.statistics
                critical = stats.get("critical", 0)
                major = stats.get("major", 0)
                minor = stats.get("minor", 0)
                observation = stats.get("observation", 0)

                with ui.card().classes("w-full q-mb-md"):
                    ui.label("Results").classes("text-subtitle1 font-bold")
                    with ui.row().classes("gap-4 q-my-sm"):
                        if critical:
                            ui.badge(f"{critical} Critical", color="red").props("rounded")
                        if major:
                            ui.badge(f"{major} Major", color="orange").props("rounded")
                        if minor:
                            ui.badge(f"{minor} Minor", color="yellow-8").props("rounded")
                        if observation:
                            ui.badge(f"{observation} Observations", color="blue").props("rounded")

                    if report.overall_summary:
                        ui.label(report.overall_summary).classes(
                            "text-body2 q-mt-sm"
                        )

                # Individual findings
                if report.findings:
                    _severity_colors = {
                        "critical": "red-1",
                        "major": "orange-1",
                        "minor": "yellow-1",
                        "observation": "blue-1",
                    }
                    _severity_icons = {
                        "critical": "error",
                        "major": "warning",
                        "minor": "info",
                        "observation": "lightbulb",
                    }

                    for i, finding in enumerate(report.findings, 1):
                        severity = finding.severity.lower()
                        bg = _severity_colors.get(severity, "grey-1")
                        icon = _severity_icons.get(severity, "help")

                        with ui.card().classes(f"w-full q-mb-sm bg-{bg}"):
                            with ui.row().classes("items-center gap-2"):
                                ui.icon(icon).classes("text-lg")
                                ui.label(
                                    f"[{finding.rule_id}] {finding.severity.upper()}"
                                ).classes("text-subtitle2 font-bold")
                                if finding.location:
                                    ui.badge(finding.location).props("outline")

                            ui.label(finding.finding).classes("text-body2 q-mt-xs")

                            if finding.suggestion:
                                with ui.row().classes("q-mt-xs items-start gap-1"):
                                    ui.icon("tips_and_updates", size="xs").classes("text-grey-7")
                                    ui.label(finding.suggestion).classes(
                                        "text-body2 text-grey-8"
                                    )

                            if finding.reference:
                                ui.label(finding.reference).classes(
                                    "text-caption text-grey-6 q-mt-xs"
                                )
                else:
                    ui.label("No findings — patent appears compliant.").classes(
                        "text-positive text-body1"
                    )

                # Export buttons
                with ui.row().classes("q-mt-md gap-2"):
                    def _export_markdown():
                        md = render_markdown(report)
                        ui.download(
                            md.encode("utf-8"),
                            filename="review_report.md",
                        )

                    ui.button(
                        "Export Markdown", on_click=_export_markdown, icon="description"
                    ).props("flat")

                    # DOCX export
                    def _export_docx():
                        from patent_system.reviewer.report import render_docx
                        import tempfile
                        out_path = Path(tempfile.mktemp(suffix=".docx"))
                        render_docx(report, out_path)
                        ui.download(
                            out_path.read_bytes(),
                            filename="review_report.docx",
                        )
                        out_path.unlink(missing_ok=True)

                    ui.button(
                        "Export DOCX", on_click=_export_docx, icon="file_download"
                    ).props("flat")
