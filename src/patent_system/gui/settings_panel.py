"""Settings panel UI for the Patent Analysis & Drafting System.

Provides the Settings tab with three sections:
1. Domain Profile — profile selector with preview and reload button.
2. Agent Personality Configuration — per-agent personality mode selectors
   with save/load persistence via PersonalityPreferenceRepository.
3. System Configuration (read-only) — displays all AppSettings values
   with sensitive fields masked.

Requirements: 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 9.1, 9.4, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 6.1, 6.2, 6.3, 6.5
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Any

from nicegui import ui

from patent_system.agents.domain_profiles import (
    DEFAULT_PROFILE_SLUG,
    ProfileLoader,
)
from patent_system.agents.personality import (
    AGENT_PERSONALITY_DEFAULTS,
    PersonalityMode,
)
from patent_system.config import AppSettings
from patent_system.db.repository import (
    PersonalityPreferenceRepository,
    TopicDomainProfileRepository,
)

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
    profile_loader: ProfileLoader | None = None,
    domain_profile_repo: TopicDomainProfileRepository | None = None,
) -> None:
    """Populate *container* with the Settings panel UI.

    The panel has three sections:

    1. **Domain Profile** — profile selector dropdown with read-only
       previews, reload button, and directory path note.
    2. **Agent Personality Configuration** — one dropdown per agent with
       a Save button that persists selections via *personality_pref_repo*.
    3. **System Configuration** (read-only) — displays all ``AppSettings``
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
        profile_loader: ProfileLoader instance for loading domain profiles.
            If None, the domain profile section is skipped.
        domain_profile_repo: Repository for persisting per-topic domain
            profile selections. If None, the domain profile section is skipped.
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

    # Domain profile selector reference (used by save button).
    profile_selector: ui.select | None = None

    with container:
        # ── Section 0: Domain Profile (Req 10.1–10.6) ──
        if profile_loader is not None and domain_profile_repo is not None:
            ui.label("Domain Profile").classes("text-h5 q-mt-md q-mb-sm")
            ui.separator()

            # Determine the active profile slug for this topic.
            saved_slug = domain_profile_repo.get_by_topic(topic_id)
            available_profiles = profile_loader.get_all()
            profile_options = {
                p.slug: p.domain_label for p in available_profiles
            }

            # Default to DEFAULT_PROFILE_SLUG if no saved selection or
            # saved slug is not in loaded profiles (Req 6.3, 6.5).
            if saved_slug and saved_slug in profile_options:
                active_slug = saved_slug
            else:
                active_slug = DEFAULT_PROFILE_SLUG

            # Build a lookup dict for preview content.
            profiles_by_slug = {p.slug: p for p in available_profiles}

            # Profile selector dropdown (Req 10.2).
            profile_selector = ui.select(
                options=profile_options,
                value=active_slug,
                label="Active Domain Profile",
            ).classes("min-w-[300px]")

            # Read-only preview areas (Req 10.3).
            role_prompt_preview = ui.textarea(
                label="Role Prompt (read-only preview)",
                value=profiles_by_slug.get(active_slug, profiles_by_slug.get(DEFAULT_PROFILE_SLUG, None)).role_prompt if profiles_by_slug else "",
            ).classes("w-full").props("readonly outlined")

            guidance_preview = ui.textarea(
                label="Content Structure Guidance (read-only preview)",
                value=profiles_by_slug.get(active_slug, profiles_by_slug.get(DEFAULT_PROFILE_SLUG, None)).content_structure_guidance if profiles_by_slug else "",
            ).classes("w-full").props("readonly outlined")

            # Update previews when dropdown selection changes.
            def _on_profile_change(e: Any) -> None:
                selected_slug = e.value if hasattr(e, "value") else e
                profile = profiles_by_slug.get(selected_slug)
                if profile:
                    role_prompt_preview.set_value(profile.role_prompt)
                    guidance_preview.set_value(profile.content_structure_guidance)

            profile_selector.on_value_change(_on_profile_change)

            # Reload Profiles button (Req 10.5).
            def _reload_profiles() -> None:
                nonlocal profiles_by_slug
                profile_loader.reload()
                refreshed_profiles = profile_loader.get_all()
                new_options = {
                    p.slug: p.domain_label for p in refreshed_profiles
                }
                profiles_by_slug = {p.slug: p for p in refreshed_profiles}
                profile_selector.options = new_options  # type: ignore[union-attr]
                profile_selector.update()  # type: ignore[union-attr]
                ui.notify("Profiles reloaded.", type="info")

            ui.button(
                "Reload Profiles", on_click=_reload_profiles
            ).classes("q-mt-sm").props("flat color=secondary")

            # Informational note (Req 10.6).
            with ui.column().classes("q-mt-sm q-pa-sm bg-blue-1 rounded"):
                ui.label(
                    f"ℹ Profiles are YAML files in: {profile_loader.profiles_dir}"
                ).classes("text-caption")
                ui.label(
                    "Add or edit .yaml files to manage profiles. "
                    "See existing files as examples."
                ).classes("text-caption text-grey-8")

        # ── Section 1: Agent Personality Configuration (Req 7.2, 7.8) ──
        ui.label("Agent Personality Configuration").classes(
            "text-h5 q-mt-lg q-mb-sm"
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

        # Save button — persists both personality preferences and domain
        # profile selection in a single action (Req 7.3, 7.4, 9.1, 9.4, 10.4).
        # Capture profile_selector in closure for save.
        _profile_selector_ref = profile_selector
        _domain_profile_repo_ref = domain_profile_repo

        async def _save_preferences() -> None:
            prefs_to_save: dict[str, str] = {
                name: sel.value for name, sel in selectors.items()
            }
            try:
                personality_pref_repo.save(topic_id, prefs_to_save)

                # Also persist domain profile selection if available.
                if _profile_selector_ref is not None and _domain_profile_repo_ref is not None:
                    selected_slug = _profile_selector_ref.value
                    _domain_profile_repo_ref.save(topic_id, selected_slug)

                ui.notify(
                    "Settings saved.",
                    type="positive",
                )
            except (sqlite3.Error, Exception) as exc:
                logger.error(
                    "Failed to save settings for topic %d: %s",
                    topic_id,
                    exc,
                )
                ui.notify(
                    "Failed to save settings. See logs for details.",
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
