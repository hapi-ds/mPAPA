"""Settings panel UI for the Patent Analysis & Drafting System.

Provides the Settings tab with two sections:
1. Agent Personality Configuration — per-agent personality mode selectors
   with save/load persistence via PersonalityPreferenceRepository.
2. System Configuration (read-only) — displays all AppSettings values
   with sensitive fields masked.

Requirements: 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 9.1, 9.4
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Any

from nicegui import ui

from patent_system.agents.personality import (
    AGENT_PERSONALITY_DEFAULTS,
    PersonalityMode,
)
from patent_system.config import AppSettings
from patent_system.db.repository import PersonalityPreferenceRepository

logger = logging.getLogger(__name__)

# Human-readable labels for agent node names.
_AGENT_DISPLAY_NAMES: dict[str, str] = {
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

# Dropdown options: PersonalityMode value → human-readable label.
_MODE_OPTIONS: dict[str, str] = {
    "critical": "Critical",
    "neutral": "Neutral",
    "innovation_friendly": "Innovation-Friendly",
}

# Brief descriptions shown above the selectors (Req 7.8).
_MODE_DESCRIPTIONS: dict[str, str] = {
    "critical": (
        "Skeptical and rigorous — questions assumptions, highlights "
        "weaknesses, demands strong evidence."
    ),
    "neutral": (
        "Balanced and objective — presents evidence on all sides "
        "without bias toward positive or negative framing."
    ),
    "innovation_friendly": (
        "Constructive and opportunity-focused — emphasizes novel aspects "
        "and paths forward while still noting material risks."
    ),
}

# Fields in AppSettings that contain sensitive values and should be masked.
_SENSITIVE_FIELDS: frozenset[str] = frozenset({
    "lm_studio_api_key",
    "epo_ops_key",
    "epo_ops_secret",
})

# Logical grouping of AppSettings fields for the read-only display.
_SETTINGS_GROUPS: list[tuple[str, list[str]]] = [
    ("LM Studio", ["lm_studio_base_url", "lm_studio_api_key"]),
    (
        "Models",
        [
            "model_disclosure",
            "model_search",
            "model_claims",
            "model_description",
            "model_review",
            "model_chat",
            "embedding_model_name",
        ],
    ),
    ("Database", ["database_path"]),
    ("DOCX Export", ["docx_template_dir", "docx_template_name"]),
    ("Monitoring", ["monitoring_interval_hours"]),
    (
        "Search",
        [
            "search_max_results_per_source",
            "search_request_delay_seconds",
            "search_relevance_top_k",
        ],
    ),
    ("EPO Open Patent Services", ["epo_ops_key", "epo_ops_secret"]),
    ("Web Server", ["nicegui_port", "nicegui_reload"]),
    ("Logging", ["log_file_path", "log_level"]),
    (
        "Personality Defaults",
        ["default_personality_mode", "agent_personality_overrides"],
    ),
]


def _mask_value(field_name: str, value: Any) -> str:
    """Return a display string, masking sensitive fields.

    Non-empty sensitive values are replaced with ``"***"``.
    """
    if field_name in _SENSITIVE_FIELDS:
        str_val = str(value)
        return "***" if str_val else ""
    return str(value)


def _format_field_name(name: str) -> str:
    """Convert a snake_case field name to a human-readable label."""
    return name.replace("_", " ").title()


def create_settings_panel(
    container: Any,
    topic_id: int,
    conn: sqlite3.Connection,
    settings: AppSettings,
    personality_pref_repo: PersonalityPreferenceRepository,
) -> None:
    """Populate *container* with the Settings panel UI.

    The panel has two sections:

    1. **Agent Personality Configuration** — one dropdown per agent with
       a Save button that persists selections via *personality_pref_repo*.
    2. **System Configuration** (read-only) — displays all ``AppSettings``
       values with sensitive fields masked.

    Args:
        container: A NiceGUI container element (e.g. ``ui.column``) that
            will be cleared and repopulated.
        topic_id: The currently selected topic ID.
        conn: SQLite connection (unused directly but available for future
            extensions).
        settings: The application settings instance.
        personality_pref_repo: Repository for reading/writing per-topic
            personality preferences.
    """
    container.clear()

    # Load saved preferences, falling back to defaults (Req 7.6, 9.2, 9.3).
    saved_prefs = personality_pref_repo.get_by_topic(topic_id)

    # Build the effective preference dict: saved value → agent default.
    effective: dict[str, str] = {}
    for agent_name in AGENT_PERSONALITY_DEFAULTS:
        if saved_prefs and agent_name in saved_prefs:
            effective[agent_name] = saved_prefs[agent_name]
        else:
            effective[agent_name] = AGENT_PERSONALITY_DEFAULTS[agent_name].value

    # Dict to hold references to ui.select elements keyed by agent name.
    selectors: dict[str, ui.select] = {}

    with container:
        # ── Section 1: Agent Personality Configuration (Req 7.2, 7.8) ──
        ui.label("Agent Personality Configuration").classes(
            "text-h5 q-mt-md q-mb-sm"
        )
        ui.separator()

        # Mode descriptions (Req 7.8).
        with ui.column().classes("q-mb-md q-gutter-xs"):
            for mode_value, description in _MODE_DESCRIPTIONS.items():
                with ui.row().classes("items-start q-gutter-xs"):
                    ui.label(f"{_MODE_OPTIONS[mode_value]}:").classes(
                        "text-bold text-caption"
                    )
                    ui.label(description).classes("text-caption text-grey-8")

        # Per-agent selectors (Req 7.2, 7.3, 7.6).
        with ui.column().classes("q-gutter-sm w-full"):
            for agent_name in AGENT_PERSONALITY_DEFAULTS:
                display_name = _AGENT_DISPLAY_NAMES.get(
                    agent_name, _format_field_name(agent_name)
                )
                with ui.row().classes("items-center q-gutter-sm"):
                    selector = ui.select(
                        options=_MODE_OPTIONS,
                        value=effective[agent_name],
                        label=display_name,
                    ).classes("min-w-[220px]")
                    selectors[agent_name] = selector

        # Save button (Req 7.3, 7.4, 9.1, 9.4).
        async def _save_preferences() -> None:
            prefs_to_save: dict[str, str] = {
                name: sel.value for name, sel in selectors.items()
            }
            try:
                personality_pref_repo.save(topic_id, prefs_to_save)
                ui.notify(
                    "Personality preferences saved.",
                    type="positive",
                )
            except (sqlite3.Error, Exception) as exc:
                logger.error(
                    "Failed to save personality preferences for topic %d: %s",
                    topic_id,
                    exc,
                )
                ui.notify(
                    "Failed to save preferences. See logs for details.",
                    type="negative",
                )

        ui.button("Save", on_click=_save_preferences).classes("q-mt-sm").props(
            "color=primary"
        )

        # ── Section 2: System Configuration (read-only) (Req 7.5) ──
        ui.label("System Configuration").classes("text-h5 q-mt-lg q-mb-sm")
        ui.separator()

        settings_dict = settings.model_dump()

        for group_label, field_names in _SETTINGS_GROUPS:
            ui.label(group_label).classes("text-subtitle1 text-bold q-mt-sm")
            with ui.column().classes("q-gutter-xs q-pl-md"):
                for field_name in field_names:
                    if field_name not in settings_dict:
                        continue
                    raw_value = settings_dict[field_name]
                    display_value = _mask_value(field_name, raw_value)
                    display_label = _format_field_name(field_name)
                    ui.label(f"{display_label}: {display_value}").classes(
                        "text-caption"
                    )
