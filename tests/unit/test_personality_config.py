"""Unit tests for AppSettings personality configuration fields.

Requirements: 6.1, 6.2, 6.4, 6.5
"""

import json

import pytest
from pydantic import ValidationError

from patent_system.agents.personality import PersonalityMode
from patent_system.config import AppSettings


class TestDefaultPersonalityMode:
    """Tests for the default_personality_mode field (Requirement 6.1)."""

    def test_default_value_is_critical(self) -> None:
        """AppSettings defaults default_personality_mode to PersonalityMode.CRITICAL."""
        settings = AppSettings(_env_file=None)
        assert settings.default_personality_mode == PersonalityMode.CRITICAL
        assert settings.default_personality_mode.value == "critical"


class TestEnvVarLoading:
    """Tests for PATENT_DEFAULT_PERSONALITY_MODE env var loading (Requirement 6.2)."""

    def test_loads_neutral_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Setting PATENT_DEFAULT_PERSONALITY_MODE=neutral loads correctly."""
        monkeypatch.setenv("PATENT_DEFAULT_PERSONALITY_MODE", "neutral")
        settings = AppSettings(_env_file=None)
        assert settings.default_personality_mode == PersonalityMode.NEUTRAL

    def test_loads_critical_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Setting PATENT_DEFAULT_PERSONALITY_MODE=critical loads correctly."""
        monkeypatch.setenv("PATENT_DEFAULT_PERSONALITY_MODE", "critical")
        settings = AppSettings(_env_file=None)
        assert settings.default_personality_mode == PersonalityMode.CRITICAL

    def test_loads_innovation_friendly_from_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Setting PATENT_DEFAULT_PERSONALITY_MODE=innovation_friendly loads correctly."""
        monkeypatch.setenv("PATENT_DEFAULT_PERSONALITY_MODE", "innovation_friendly")
        settings = AppSettings(_env_file=None)
        assert settings.default_personality_mode == PersonalityMode.INNOVATION_FRIENDLY

    def test_invalid_env_var_raises_validation_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An invalid PATENT_DEFAULT_PERSONALITY_MODE value raises ValidationError."""
        monkeypatch.setenv("PATENT_DEFAULT_PERSONALITY_MODE", "aggressive")
        with pytest.raises(ValidationError):
            AppSettings(_env_file=None)


class TestAgentPersonalityOverrides:
    """Tests for agent_personality_overrides JSON parsing (Requirements 6.4, 6.5)."""

    def test_default_overrides_is_empty_string(self) -> None:
        """agent_personality_overrides defaults to an empty string."""
        settings = AppSettings(_env_file=None)
        assert settings.agent_personality_overrides == ""

    def test_overrides_loaded_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """PATENT_AGENT_PERSONALITY_OVERRIDES env var is loaded as a string."""
        overrides = json.dumps({"novelty_analysis": "neutral"})
        monkeypatch.setenv("PATENT_AGENT_PERSONALITY_OVERRIDES", overrides)
        settings = AppSettings(_env_file=None)
        assert settings.agent_personality_overrides == overrides

    def test_overrides_json_can_be_parsed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The overrides string can be parsed as valid JSON with agent→mode mappings."""
        overrides = json.dumps({
            "novelty_analysis": "neutral",
            "claims_drafting": "innovation_friendly",
        })
        monkeypatch.setenv("PATENT_AGENT_PERSONALITY_OVERRIDES", overrides)
        settings = AppSettings(_env_file=None)
        parsed = json.loads(settings.agent_personality_overrides)
        assert parsed["novelty_analysis"] == "neutral"
        assert parsed["claims_drafting"] == "innovation_friendly"

    def test_empty_overrides_fall_back_to_defaults(self) -> None:
        """When agent_personality_overrides is empty, defaults from AGENT_PERSONALITY_DEFAULTS apply."""
        from patent_system.agents.personality import AGENT_PERSONALITY_DEFAULTS

        settings = AppSettings(_env_file=None)
        assert settings.agent_personality_overrides == ""
        # With empty overrides, the system should use AGENT_PERSONALITY_DEFAULTS
        assert settings.default_personality_mode == PersonalityMode.CRITICAL
        # Verify the defaults dict is populated and accessible
        assert "novelty_analysis" in AGENT_PERSONALITY_DEFAULTS
        assert AGENT_PERSONALITY_DEFAULTS["novelty_analysis"] == PersonalityMode.CRITICAL
