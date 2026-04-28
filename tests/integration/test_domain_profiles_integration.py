"""Integration tests for agent node domain awareness.

Verifies that agent nodes read ``domain_profile_slug`` from state,
pass it to DSPy module calls, and include it in log output.
Also tests workflow state population from saved topic preference.

Requirements: 9.1, 9.2, 9.3, 7.3
"""

from __future__ import annotations

import logging
import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from patent_system.agents.domain_profiles import DEFAULT_PROFILE_SLUG, ProfileLoader
from patent_system.db.repository import TopicDomainProfileRepository
from patent_system.db.schema import init_schema


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(**overrides) -> dict:
    """Create a minimal PatentWorkflowState dict for testing."""
    base: dict = {
        "topic_id": 1,
        "invention_disclosure": {
            "technical_problem": "Slow processing",
            "novel_features": ["Feature A"],
            "implementation_details": "Uses GPU",
            "potential_variations": ["CPU fallback"],
        },
        "interview_messages": [],
        "prior_art_results": [],
        "prior_art_summary": "Some prior art summary",
        "failed_sources": [],
        "novelty_analysis": "Novel: Feature A",
        "claims_text": "Claim 1: A method for fast processing.",
        "description_text": "Technical Field: Processing systems.",
        "review_feedback": "",
        "review_approved": False,
        "iteration_count": 0,
        "current_step": "",
        "market_assessment": "",
        "legal_assessment": "",
        "disclosure_summary": "",
        "workflow_step_statuses": {},
        "personality_modes": {},
        "review_notes": {},
        "domain_profile_slug": DEFAULT_PROFILE_SLUG,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Test: Agent nodes read domain_profile_slug from state and pass to DSPy
# ---------------------------------------------------------------------------


class TestAgentNodesReadDomainProfileSlug:
    """Verify agent nodes read domain_profile_slug from state (Req 9.1)."""

    @patch("patent_system.agents.claims_drafting.DraftClaimsModule")
    def test_claims_drafting_reads_slug_from_state(self, mock_cls):
        """claims_drafting_node reads domain_profile_slug from state."""
        mock_instance = MagicMock()
        mock_instance.return_value.claims_text = "Claim 1: A method..."
        mock_cls.return_value = mock_instance

        from patent_system.agents.claims_drafting import claims_drafting_node

        state = _make_state(domain_profile_slug="pharma-chemistry")
        claims_drafting_node(state)

        # The DSPy module must have been called with the slug from state
        mock_instance.assert_called_once()
        call_kwargs = mock_instance.call_args.kwargs
        assert call_kwargs.get("domain_profile_slug") == "pharma-chemistry"

    @patch("patent_system.agents.market_potential.MarketPotentialModule")
    def test_market_potential_reads_slug_from_state(self, mock_cls):
        """market_potential_node reads domain_profile_slug from state."""
        mock_instance = MagicMock()
        mock_instance.return_value.market_assessment = "High potential"
        mock_cls.return_value = mock_instance

        from patent_system.agents.market_potential import market_potential_node

        state = _make_state(domain_profile_slug="software-ai")
        market_potential_node(state)

        mock_instance.assert_called_once()
        call_kwargs = mock_instance.call_args.kwargs
        assert call_kwargs.get("domain_profile_slug") == "software-ai"


# ---------------------------------------------------------------------------
# Test: Agent nodes pass domain_profile_slug to DSPy module calls
# ---------------------------------------------------------------------------


class TestAgentNodesPassSlugToDSPy:
    """Verify agent nodes pass domain_profile_slug to DSPy modules (Req 9.2)."""

    @patch("patent_system.agents.claims_drafting.DraftClaimsModule")
    def test_claims_drafting_passes_slug_to_dspy(self, mock_cls):
        """claims_drafting_node passes domain_profile_slug kwarg to DraftClaimsModule."""
        mock_instance = MagicMock()
        mock_instance.return_value.claims_text = "Claim 1"
        mock_cls.return_value = mock_instance

        from patent_system.agents.claims_drafting import claims_drafting_node

        state = _make_state(domain_profile_slug="medtech-mechanical-engineering")
        claims_drafting_node(state)

        mock_instance.assert_called_once()
        call_kwargs = mock_instance.call_args.kwargs
        assert "domain_profile_slug" in call_kwargs
        assert call_kwargs["domain_profile_slug"] == "medtech-mechanical-engineering"

    @patch("patent_system.agents.market_potential.MarketPotentialModule")
    def test_market_potential_passes_slug_to_dspy(self, mock_cls):
        """market_potential_node passes domain_profile_slug kwarg to MarketPotentialModule."""
        mock_instance = MagicMock()
        mock_instance.return_value.market_assessment = "Assessment text"
        mock_cls.return_value = mock_instance

        from patent_system.agents.market_potential import market_potential_node

        state = _make_state(domain_profile_slug="electrical-engineering-semiconductors")
        market_potential_node(state)

        mock_instance.assert_called_once()
        call_kwargs = mock_instance.call_args.kwargs
        assert "domain_profile_slug" in call_kwargs
        assert call_kwargs["domain_profile_slug"] == "electrical-engineering-semiconductors"

    @patch("patent_system.agents.claims_drafting.DraftClaimsModule")
    def test_claims_drafting_uses_default_when_slug_absent(self, mock_cls):
        """When domain_profile_slug is empty, claims_drafting_node uses default."""
        mock_instance = MagicMock()
        mock_instance.return_value.claims_text = "Claim 1"
        mock_cls.return_value = mock_instance

        from patent_system.agents.claims_drafting import claims_drafting_node

        state = _make_state(domain_profile_slug="")
        claims_drafting_node(state)

        mock_instance.assert_called_once()
        call_kwargs = mock_instance.call_args.kwargs
        assert call_kwargs["domain_profile_slug"] == DEFAULT_PROFILE_SLUG

    @patch("patent_system.agents.market_potential.MarketPotentialModule")
    def test_market_potential_uses_default_when_slug_absent(self, mock_cls):
        """When domain_profile_slug is empty, market_potential_node uses default."""
        mock_instance = MagicMock()
        mock_instance.return_value.market_assessment = "Assessment"
        mock_cls.return_value = mock_instance

        from patent_system.agents.market_potential import market_potential_node

        state = _make_state(domain_profile_slug="")
        market_potential_node(state)

        mock_instance.assert_called_once()
        call_kwargs = mock_instance.call_args.kwargs
        assert call_kwargs["domain_profile_slug"] == DEFAULT_PROFILE_SLUG


# ---------------------------------------------------------------------------
# Test: Agent nodes include domain_profile_slug in log output
# ---------------------------------------------------------------------------


class TestAgentNodesLogDomainProfile:
    """Verify agent nodes include domain_profile=<slug> in log output (Req 9.3)."""

    @patch("patent_system.agents.claims_drafting.DraftClaimsModule")
    def test_claims_drafting_logs_domain_profile(self, mock_cls, caplog):
        """claims_drafting_node includes domain_profile=<slug> in log output."""
        mock_instance = MagicMock()
        mock_instance.return_value.claims_text = "Claim 1"
        mock_cls.return_value = mock_instance

        from patent_system.agents.claims_drafting import claims_drafting_node

        state = _make_state(domain_profile_slug="pharma-chemistry")

        with caplog.at_level(logging.INFO, logger="patent_system.agents.claims_drafting"):
            claims_drafting_node(state)

        # The input_summary is stored in extra_fields on the log record
        found = False
        for record in caplog.records:
            extra = getattr(record, "extra_fields", None)
            if extra and "domain_profile=pharma-chemistry" in extra.get("input", ""):
                found = True
                break
        assert found, "domain_profile=pharma-chemistry not found in log extra_fields"

    @patch("patent_system.agents.market_potential.MarketPotentialModule")
    def test_market_potential_logs_domain_profile(self, mock_cls, caplog):
        """market_potential_node includes domain_profile=<slug> in log output."""
        mock_instance = MagicMock()
        mock_instance.return_value.market_assessment = "High potential"
        mock_cls.return_value = mock_instance

        from patent_system.agents.market_potential import market_potential_node

        state = _make_state(domain_profile_slug="software-ai")

        with caplog.at_level(logging.INFO, logger="patent_system.agents.market_potential"):
            market_potential_node(state)

        found = False
        for record in caplog.records:
            extra = getattr(record, "extra_fields", None)
            if extra and "domain_profile=software-ai" in extra.get("input", ""):
                found = True
                break
        assert found, "domain_profile=software-ai not found in log extra_fields"

    @patch("patent_system.agents.claims_drafting.DraftClaimsModule")
    def test_claims_drafting_logs_default_when_slug_empty(self, mock_cls, caplog):
        """When slug is empty, log shows domain_profile=general-patent-drafting."""
        mock_instance = MagicMock()
        mock_instance.return_value.claims_text = "Claim 1"
        mock_cls.return_value = mock_instance

        from patent_system.agents.claims_drafting import claims_drafting_node

        state = _make_state(domain_profile_slug="")

        with caplog.at_level(logging.INFO, logger="patent_system.agents.claims_drafting"):
            claims_drafting_node(state)

        found = False
        for record in caplog.records:
            extra = getattr(record, "extra_fields", None)
            if extra and f"domain_profile={DEFAULT_PROFILE_SLUG}" in extra.get("input", ""):
                found = True
                break
        assert found, f"domain_profile={DEFAULT_PROFILE_SLUG} not found in log extra_fields"


# ---------------------------------------------------------------------------
# Test: Workflow state population from saved topic preference
# ---------------------------------------------------------------------------


class TestWorkflowStatePopulationFromSavedPreference:
    """Verify workflow state is populated from saved topic preference (Req 7.3)."""

    @pytest.fixture
    def db_conn(self) -> sqlite3.Connection:
        """Provide a fresh in-memory SQLite connection with full schema."""
        conn = sqlite3.connect(":memory:")
        conn.execute("PRAGMA foreign_keys = ON")
        init_schema(conn)
        yield conn
        conn.close()

    @pytest.fixture
    def topic_id(self, db_conn: sqlite3.Connection) -> int:
        """Insert a test topic and return its ID."""
        cursor = db_conn.execute(
            "INSERT INTO topics (name) VALUES (?)", ("Test Topic",)
        )
        db_conn.commit()
        return cursor.lastrowid

    def test_saved_preference_is_read_correctly(
        self, db_conn: sqlite3.Connection, topic_id: int
    ):
        """A saved domain profile slug is correctly retrieved for state population."""
        repo = TopicDomainProfileRepository(db_conn)
        repo.save(topic_id, "biotechnology-life-sciences")

        # Simulate what the draft panel does: read saved slug
        saved_slug = repo.get_by_topic(topic_id)
        domain_profile_slug = saved_slug or DEFAULT_PROFILE_SLUG

        assert domain_profile_slug == "biotechnology-life-sciences"

    def test_no_saved_preference_falls_back_to_default(
        self, db_conn: sqlite3.Connection, topic_id: int
    ):
        """When no preference is saved, falls back to DEFAULT_PROFILE_SLUG."""
        repo = TopicDomainProfileRepository(db_conn)

        saved_slug = repo.get_by_topic(topic_id)
        domain_profile_slug = saved_slug or DEFAULT_PROFILE_SLUG

        assert domain_profile_slug == DEFAULT_PROFILE_SLUG

    def test_saved_preference_populates_workflow_state(
        self, db_conn: sqlite3.Connection, topic_id: int
    ):
        """Saved preference is used to populate the domain_profile_slug in state."""
        repo = TopicDomainProfileRepository(db_conn)
        repo.save(topic_id, "telecommunications-standards")

        # Simulate workflow state construction (as done in draft_panel.py)
        saved_slug = repo.get_by_topic(topic_id)
        domain_profile_slug = saved_slug or DEFAULT_PROFILE_SLUG

        state = _make_state(
            topic_id=topic_id,
            domain_profile_slug=domain_profile_slug,
        )

        assert state["domain_profile_slug"] == "telecommunications-standards"

    def test_overwritten_preference_uses_latest(
        self, db_conn: sqlite3.Connection, topic_id: int
    ):
        """When preference is overwritten, the latest value is used."""
        repo = TopicDomainProfileRepository(db_conn)
        repo.save(topic_id, "pharma-chemistry")
        repo.save(topic_id, "materials-science-nanotechnology")

        saved_slug = repo.get_by_topic(topic_id)
        domain_profile_slug = saved_slug or DEFAULT_PROFILE_SLUG

        assert domain_profile_slug == "materials-science-nanotechnology"


# ---------------------------------------------------------------------------
# Test: Settings panel domain profile UI logic (Req 10.1–10.6)
# ---------------------------------------------------------------------------


class TestSettingsPanelDomainProfileUI:
    """Verify settings panel domain profile section logic (Req 10.1–10.6).

    NiceGUI rendering is complex to test directly, so these tests verify:
    - create_settings_panel() accepts the new parameters without error
    - The underlying logic for profile selection, preview, save, and reload
    - Fallback behavior when no saved selection exists
    """

    @pytest.fixture
    def db_conn(self) -> sqlite3.Connection:
        """Provide a fresh in-memory SQLite connection with full schema."""
        conn = sqlite3.connect(":memory:")
        conn.execute("PRAGMA foreign_keys = ON")
        init_schema(conn)
        yield conn
        conn.close()

    @pytest.fixture
    def topic_id(self, db_conn: sqlite3.Connection) -> int:
        """Insert a test topic and return its ID."""
        cursor = db_conn.execute(
            "INSERT INTO topics (name) VALUES (?)", ("Test Topic",)
        )
        db_conn.commit()
        return cursor.lastrowid

    @pytest.fixture
    def profile_loader(self, tmp_path) -> ProfileLoader:
        """Provide a ProfileLoader with built-in profiles in a temp directory."""
        return ProfileLoader(tmp_path / "profiles")

    @pytest.fixture
    def domain_profile_repo(self, db_conn: sqlite3.Connection) -> TopicDomainProfileRepository:
        """Provide a TopicDomainProfileRepository instance."""
        return TopicDomainProfileRepository(db_conn)

    @pytest.fixture
    def personality_pref_repo(self, db_conn: sqlite3.Connection):
        """Provide a PersonalityPreferenceRepository instance."""
        from patent_system.db.repository import PersonalityPreferenceRepository

        return PersonalityPreferenceRepository(db_conn)

    @patch("patent_system.gui.settings_panel.ui")
    def test_create_settings_panel_accepts_domain_profile_params(
        self,
        mock_ui,
        db_conn,
        topic_id,
        profile_loader,
        domain_profile_repo,
        personality_pref_repo,
    ):
        """create_settings_panel() accepts profile_loader and domain_profile_repo params."""
        from patent_system.config import AppSettings
        from patent_system.gui.settings_panel import create_settings_panel

        # Mock the container
        mock_container = MagicMock()
        mock_container.__enter__ = MagicMock(return_value=mock_container)
        mock_container.__exit__ = MagicMock(return_value=False)

        # Mock ui elements to return context managers
        mock_ui.label.return_value = MagicMock()
        mock_ui.separator.return_value = MagicMock()
        mock_ui.select.return_value = MagicMock(value=DEFAULT_PROFILE_SLUG, on_value_change=MagicMock())
        mock_ui.textarea.return_value = MagicMock(classes=MagicMock(return_value=MagicMock(props=MagicMock(return_value=MagicMock()))))
        mock_ui.button.return_value = MagicMock(classes=MagicMock(return_value=MagicMock(props=MagicMock(return_value=MagicMock()))))
        mock_ui.column.return_value = MagicMock(__enter__=MagicMock(), __exit__=MagicMock(return_value=False))
        mock_ui.row.return_value = MagicMock(__enter__=MagicMock(), __exit__=MagicMock(return_value=False))

        settings = AppSettings(
            lm_studio_base_url="http://localhost:1234",
            database_path=":memory:",
        )

        # Should not raise — verifies the function signature accepts the new params
        create_settings_panel(
            container=mock_container,
            topic_id=topic_id,
            conn=db_conn,
            settings=settings,
            personality_pref_repo=personality_pref_repo,
            profile_loader=profile_loader,
            domain_profile_repo=domain_profile_repo,
        )

    def test_profile_selector_logic_uses_saved_selection(
        self,
        db_conn,
        topic_id,
        profile_loader,
        domain_profile_repo,
    ):
        """Profile selector logic pre-selects the saved profile for the topic."""
        # Save a profile selection
        domain_profile_repo.save(topic_id, "pharma-chemistry")

        # Simulate the logic from create_settings_panel
        saved_slug = domain_profile_repo.get_by_topic(topic_id)
        available_profiles = profile_loader.get_all()
        profile_options = {p.slug: p.domain_label for p in available_profiles}

        if saved_slug and saved_slug in profile_options:
            active_slug = saved_slug
        else:
            active_slug = DEFAULT_PROFILE_SLUG

        assert active_slug == "pharma-chemistry"

    def test_profile_selector_logic_defaults_when_no_saved_selection(
        self,
        db_conn,
        topic_id,
        profile_loader,
        domain_profile_repo,
    ):
        """Profile selector defaults to DEFAULT_PROFILE_SLUG when no saved selection."""
        # No saved selection
        saved_slug = domain_profile_repo.get_by_topic(topic_id)
        available_profiles = profile_loader.get_all()
        profile_options = {p.slug: p.domain_label for p in available_profiles}

        if saved_slug and saved_slug in profile_options:
            active_slug = saved_slug
        else:
            active_slug = DEFAULT_PROFILE_SLUG

        assert active_slug == DEFAULT_PROFILE_SLUG

    def test_profile_selector_logic_defaults_when_saved_slug_not_in_profiles(
        self,
        db_conn,
        topic_id,
        profile_loader,
        domain_profile_repo,
    ):
        """Profile selector defaults when saved slug references a deleted profile (Req 6.5)."""
        # Save a slug that doesn't exist in loaded profiles
        domain_profile_repo.save(topic_id, "nonexistent-profile")

        saved_slug = domain_profile_repo.get_by_topic(topic_id)
        available_profiles = profile_loader.get_all()
        profile_options = {p.slug: p.domain_label for p in available_profiles}

        if saved_slug and saved_slug in profile_options:
            active_slug = saved_slug
        else:
            active_slug = DEFAULT_PROFILE_SLUG

        assert active_slug == DEFAULT_PROFILE_SLUG

    def test_profile_preview_updates_on_selection_change(
        self,
        profile_loader,
    ):
        """Selecting a profile provides the correct preview content (Req 10.3)."""
        available_profiles = profile_loader.get_all()
        profiles_by_slug = {p.slug: p for p in available_profiles}

        # Simulate selecting pharma-chemistry
        selected_slug = "pharma-chemistry"
        profile = profiles_by_slug.get(selected_slug)

        assert profile is not None
        assert "Pharmacy" in profile.role_prompt or "pharma" in profile.role_prompt.lower()
        assert "formulation" in profile.content_structure_guidance.lower()

        # Simulate selecting software-ai
        selected_slug = "software-ai"
        profile = profiles_by_slug.get(selected_slug)

        assert profile is not None
        assert "software" in profile.role_prompt.lower() or "artificial intelligence" in profile.role_prompt.lower()
        assert "technical effect" in profile.content_structure_guidance.lower()

    def test_save_persists_domain_profile_selection(
        self,
        db_conn,
        topic_id,
        profile_loader,
        domain_profile_repo,
    ):
        """Saving settings persists the domain profile selection (Req 10.4)."""
        # Simulate the save action from the settings panel
        selected_slug = "biotechnology-life-sciences"
        domain_profile_repo.save(topic_id, selected_slug)

        # Verify persistence
        retrieved_slug = domain_profile_repo.get_by_topic(topic_id)
        assert retrieved_slug == "biotechnology-life-sciences"

        # Simulate changing and saving again
        domain_profile_repo.save(topic_id, "software-ai")
        retrieved_slug = domain_profile_repo.get_by_topic(topic_id)
        assert retrieved_slug == "software-ai"

    def test_reload_refreshes_dropdown_options(
        self,
        tmp_path,
    ):
        """Reload button triggers ProfileLoader.reload() and refreshes options (Req 10.5)."""
        profiles_dir = tmp_path / "profiles"
        loader = ProfileLoader(profiles_dir)

        # Initial load should have 9 built-in profiles
        initial_profiles = loader.get_all()
        initial_count = len(initial_profiles)
        assert initial_count == 9

        # Add a new YAML file to the directory
        new_profile_yaml = """\
slug: custom-test-profile
domain_label: "Custom Test Profile"
role_prompt: |
  You are a custom test patent attorney.
content_structure_guidance: |
  Custom guidance for testing purposes.
"""
        (profiles_dir / "custom-test-profile.yaml").write_text(new_profile_yaml)

        # Before reload, the new profile is not in the registry
        assert loader.get_by_slug("custom-test-profile") is None

        # Reload picks up the new file
        loader.reload()
        refreshed_profiles = loader.get_all()
        assert len(refreshed_profiles) == initial_count + 1
        assert loader.get_by_slug("custom-test-profile") is not None

        # Simulate refreshing dropdown options (as the UI does)
        new_options = {p.slug: p.domain_label for p in refreshed_profiles}
        assert "custom-test-profile" in new_options
        assert new_options["custom-test-profile"] == "Custom Test Profile"

    def test_directory_note_shows_profiles_path(
        self,
        tmp_path,
    ):
        """The informational note shows the correct profiles directory path (Req 10.6)."""
        profiles_dir = tmp_path / "profiles"
        loader = ProfileLoader(profiles_dir)

        # The settings panel displays loader.profiles_dir
        assert loader.profiles_dir == profiles_dir
        assert profiles_dir.exists()


# ---------------------------------------------------------------------------
# Test: Draft panel domain profile indicator logic (Req 11.1–11.4)
# ---------------------------------------------------------------------------


class TestDraftPanelDomainProfileIndicator:
    """Verify draft panel domain profile indicator logic (Req 11.1–11.4).

    NiceGUI rendering is complex to test directly, so these tests verify:
    - WorkflowStepRepository persists domain_profile_slug alongside step content
    - ProfileLoader.get_by_slug() returns the correct domain_label for display
    - Re-running a step with a different profile updates the persisted slug
    """

    @pytest.fixture
    def db_conn(self) -> sqlite3.Connection:
        """Provide a fresh in-memory SQLite connection with full schema."""
        conn = sqlite3.connect(":memory:")
        conn.execute("PRAGMA foreign_keys = ON")
        init_schema(conn)
        yield conn
        conn.close()

    @pytest.fixture
    def topic_id(self, db_conn: sqlite3.Connection) -> int:
        """Insert a test topic and return its ID."""
        cursor = db_conn.execute(
            "INSERT INTO topics (name) VALUES (?)", ("Test Topic",)
        )
        db_conn.commit()
        return cursor.lastrowid

    @pytest.fixture
    def profile_loader(self, tmp_path) -> ProfileLoader:
        """Provide a ProfileLoader with built-in profiles in a temp directory."""
        return ProfileLoader(tmp_path / "profiles")

    def test_step_persists_domain_profile_slug(
        self, db_conn: sqlite3.Connection, topic_id: int
    ):
        """WorkflowStepRepository persists domain_profile_slug alongside step content (Req 11.4)."""
        from patent_system.db.repository import WorkflowStepRepository

        repo = WorkflowStepRepository(db_conn)
        repo.upsert(
            topic_id=topic_id,
            step_key="claims_drafting",
            content="Claim 1: A method for processing data.",
            status="completed",
            personality_mode="critical",
            domain_profile_slug="pharma-chemistry",
        )

        step = repo.get_step(topic_id, "claims_drafting")
        assert step is not None
        assert step["domain_profile_slug"] == "pharma-chemistry"
        assert step["content"] == "Claim 1: A method for processing data."

    def test_step_domain_profile_slug_round_trip_via_get_by_topic(
        self, db_conn: sqlite3.Connection, topic_id: int
    ):
        """domain_profile_slug survives round-trip through get_by_topic (Req 11.4)."""
        from patent_system.db.repository import WorkflowStepRepository

        repo = WorkflowStepRepository(db_conn)
        repo.upsert(
            topic_id=topic_id,
            step_key="prior_art_search",
            content="Prior art summary text.",
            status="completed",
            domain_profile_slug="software-ai",
        )

        steps = repo.get_by_topic(topic_id)
        assert len(steps) == 1
        assert steps[0]["step_key"] == "prior_art_search"
        assert steps[0]["domain_profile_slug"] == "software-ai"

    def test_profile_loader_returns_correct_domain_label(
        self, profile_loader: ProfileLoader
    ):
        """ProfileLoader.get_by_slug() returns the correct domain_label for display (Req 11.2)."""
        profile = profile_loader.get_by_slug("pharma-chemistry")
        assert profile is not None
        assert profile.domain_label != ""
        assert "Pharma" in profile.domain_label or "Chemistry" in profile.domain_label

        profile = profile_loader.get_by_slug("software-ai")
        assert profile is not None
        assert profile.domain_label != ""
        assert "Software" in profile.domain_label or "AI" in profile.domain_label

    def test_profile_loader_domain_label_matches_slug(
        self, profile_loader: ProfileLoader
    ):
        """Each built-in profile has a non-empty domain_label suitable for UI display (Req 11.1)."""
        all_profiles = profile_loader.get_all()
        assert len(all_profiles) >= 9

        for profile in all_profiles:
            assert profile.domain_label.strip() != ""
            # domain_label should be human-readable (contains spaces or uppercase)
            assert any(c.isupper() or c == " " for c in profile.domain_label)

    def test_rerun_step_with_different_profile_updates_slug(
        self, db_conn: sqlite3.Connection, topic_id: int
    ):
        """Re-running a step with a different profile updates the persisted slug (Req 11.3)."""
        from patent_system.db.repository import WorkflowStepRepository

        repo = WorkflowStepRepository(db_conn)

        # First run with pharma-chemistry profile
        repo.upsert(
            topic_id=topic_id,
            step_key="novelty_analysis",
            content="Novelty analysis v1.",
            status="completed",
            domain_profile_slug="pharma-chemistry",
        )

        step = repo.get_step(topic_id, "novelty_analysis")
        assert step is not None
        assert step["domain_profile_slug"] == "pharma-chemistry"

        # Re-run with software-ai profile (simulates user changing profile and re-running)
        repo.upsert(
            topic_id=topic_id,
            step_key="novelty_analysis",
            content="Novelty analysis v2 with software focus.",
            status="completed",
            domain_profile_slug="software-ai",
        )

        step = repo.get_step(topic_id, "novelty_analysis")
        assert step is not None
        assert step["domain_profile_slug"] == "software-ai"
        assert step["content"] == "Novelty analysis v2 with software focus."

    def test_step_with_empty_slug_defaults_to_empty_string(
        self, db_conn: sqlite3.Connection, topic_id: int
    ):
        """When no domain_profile_slug is provided, it defaults to empty string (Req 11.4)."""
        from patent_system.db.repository import WorkflowStepRepository

        repo = WorkflowStepRepository(db_conn)
        repo.upsert(
            topic_id=topic_id,
            step_key="market_potential",
            content="Market assessment text.",
            status="completed",
        )

        step = repo.get_step(topic_id, "market_potential")
        assert step is not None
        assert step["domain_profile_slug"] == ""

    def test_domain_label_lookup_for_display(
        self, db_conn: sqlite3.Connection, topic_id: int, profile_loader: ProfileLoader
    ):
        """Integration: persisted slug can be used to look up domain_label for display (Req 11.1, 11.2)."""
        from patent_system.db.repository import WorkflowStepRepository

        repo = WorkflowStepRepository(db_conn)
        repo.upsert(
            topic_id=topic_id,
            step_key="claims_drafting",
            content="Claims text.",
            status="completed",
            domain_profile_slug="biotechnology-life-sciences",
        )

        # Simulate what the draft panel does: read step, look up label
        step = repo.get_step(topic_id, "claims_drafting")
        assert step is not None

        slug = step["domain_profile_slug"]
        profile = profile_loader.get_by_slug(slug)
        assert profile is not None
        assert "Biotechnology" in profile.domain_label or "Life" in profile.domain_label

    def test_domain_label_fallback_for_unknown_slug(
        self, profile_loader: ProfileLoader
    ):
        """When a persisted slug is not found, get_by_slug returns None for fallback handling (Req 11.1)."""
        profile = profile_loader.get_by_slug("nonexistent-deleted-profile")
        assert profile is None

        # The draft panel should fall back to default profile for display
        default_profile = profile_loader.get_by_slug(DEFAULT_PROFILE_SLUG)
        assert default_profile is not None
        assert default_profile.domain_label != ""
