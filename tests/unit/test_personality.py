"""Unit tests for the personality module.

Covers PersonalityMode enum, AGENT_PERSONALITY_DEFAULTS, generate_personality_prefix,
and resolve_personality_mode with various state shapes.

Requirements: 1.1–1.6, 2.1–2.5
"""

import pytest

from patent_system.agents.personality import (
    AGENT_PERSONALITY_DEFAULTS,
    PersonalityMode,
    generate_personality_prefix,
    resolve_personality_mode,
)


class TestPersonalityModeEnum:
    """Verify PersonalityMode enum has exactly 3 values with correct strings."""

    def test_enum_has_exactly_three_members(self) -> None:
        assert len(PersonalityMode) == 3

    def test_critical_value(self) -> None:
        assert PersonalityMode.CRITICAL == "critical"
        assert PersonalityMode.CRITICAL.value == "critical"

    def test_neutral_value(self) -> None:
        assert PersonalityMode.NEUTRAL == "neutral"
        assert PersonalityMode.NEUTRAL.value == "neutral"

    def test_innovation_friendly_value(self) -> None:
        assert PersonalityMode.INNOVATION_FRIENDLY == "innovation_friendly"
        assert PersonalityMode.INNOVATION_FRIENDLY.value == "innovation_friendly"

    def test_construction_from_valid_strings(self) -> None:
        assert PersonalityMode("critical") is PersonalityMode.CRITICAL
        assert PersonalityMode("neutral") is PersonalityMode.NEUTRAL
        assert PersonalityMode("innovation_friendly") is PersonalityMode.INNOVATION_FRIENDLY

    def test_invalid_string_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            PersonalityMode("aggressive")


class TestAgentPersonalityDefaults:
    """Verify AGENT_PERSONALITY_DEFAULTS contains all expected agent keys."""

    EXPECTED_AGENTS = {
        "initial_idea",
        "claims_drafting",
        "prior_art_search",
        "novelty_analysis",
        "consistency_review",
        "market_potential",
        "legal_clarification",
        "disclosure_summary",
        "patent_draft",
    }

    def test_contains_all_expected_agents(self) -> None:
        assert set(AGENT_PERSONALITY_DEFAULTS.keys()) == self.EXPECTED_AGENTS

    def test_critical_agents(self) -> None:
        """novelty_analysis, consistency_review, legal_clarification default to CRITICAL."""
        critical_agents = ["novelty_analysis", "consistency_review", "legal_clarification"]
        for agent in critical_agents:
            assert AGENT_PERSONALITY_DEFAULTS[agent] == PersonalityMode.CRITICAL, (
                f"{agent} should default to CRITICAL"
            )

    def test_neutral_agents(self) -> None:
        """Remaining agents default to NEUTRAL."""
        neutral_agents = [
            "initial_idea",
            "claims_drafting",
            "prior_art_search",
            "market_potential",
            "disclosure_summary",
            "patent_draft",
        ]
        for agent in neutral_agents:
            assert AGENT_PERSONALITY_DEFAULTS[agent] == PersonalityMode.NEUTRAL, (
                f"{agent} should default to NEUTRAL"
            )

    def test_all_values_are_personality_modes(self) -> None:
        for agent, mode in AGENT_PERSONALITY_DEFAULTS.items():
            assert isinstance(mode, PersonalityMode), (
                f"{agent} has non-PersonalityMode value: {mode!r}"
            )


class TestGeneratePersonalityPrefix:
    """Verify generate_personality_prefix returns strings with expected keywords."""

    def test_critical_prefix_contains_skeptical(self) -> None:
        prefix = generate_personality_prefix(PersonalityMode.CRITICAL)
        assert "skeptical" in prefix.lower()

    def test_critical_prefix_contains_weaknesses(self) -> None:
        prefix = generate_personality_prefix(PersonalityMode.CRITICAL)
        assert "weaknesses" in prefix.lower()

    def test_neutral_prefix_contains_balanced(self) -> None:
        prefix = generate_personality_prefix(PersonalityMode.NEUTRAL)
        assert "balanced" in prefix.lower()

    def test_neutral_prefix_contains_objective(self) -> None:
        prefix = generate_personality_prefix(PersonalityMode.NEUTRAL)
        assert "objective" in prefix.lower()

    def test_innovation_friendly_prefix_contains_constructive(self) -> None:
        prefix = generate_personality_prefix(PersonalityMode.INNOVATION_FRIENDLY)
        assert "constructive" in prefix.lower()

    def test_innovation_friendly_prefix_contains_novel(self) -> None:
        prefix = generate_personality_prefix(PersonalityMode.INNOVATION_FRIENDLY)
        assert "novel" in prefix.lower()

    def test_accepts_string_input(self) -> None:
        prefix = generate_personality_prefix("neutral")
        assert "balanced" in prefix.lower()

    def test_invalid_mode_falls_back_to_critical(self) -> None:
        prefix = generate_personality_prefix("invalid_mode")
        critical_prefix = generate_personality_prefix(PersonalityMode.CRITICAL)
        assert prefix == critical_prefix

    def test_each_prefix_has_personality_tag(self) -> None:
        for mode in PersonalityMode:
            prefix = generate_personality_prefix(mode)
            assert f"[Personality: {mode.value}]" in prefix


class TestResolvePersonalityMode:
    """Verify resolve_personality_mode with various state shapes."""

    def test_full_dict_returns_specified_mode(self) -> None:
        state = {
            "personality_modes": {
                "novelty_analysis": "innovation_friendly",
            }
        }
        result = resolve_personality_mode(state, "novelty_analysis")
        assert result == PersonalityMode.INNOVATION_FRIENDLY

    def test_partial_dict_returns_specified_for_present_agent(self) -> None:
        state = {
            "personality_modes": {
                "claims_drafting": "critical",
            }
        }
        result = resolve_personality_mode(state, "claims_drafting")
        assert result == PersonalityMode.CRITICAL

    def test_partial_dict_falls_back_to_default_for_missing_agent(self) -> None:
        state = {
            "personality_modes": {
                "claims_drafting": "critical",
            }
        }
        result = resolve_personality_mode(state, "novelty_analysis")
        assert result == AGENT_PERSONALITY_DEFAULTS["novelty_analysis"]

    def test_empty_personality_modes_dict(self) -> None:
        state = {"personality_modes": {}}
        result = resolve_personality_mode(state, "novelty_analysis")
        assert result == AGENT_PERSONALITY_DEFAULTS["novelty_analysis"]

    def test_missing_personality_modes_key(self) -> None:
        state = {}
        result = resolve_personality_mode(state, "novelty_analysis")
        assert result == AGENT_PERSONALITY_DEFAULTS["novelty_analysis"]

    def test_personality_modes_is_none(self) -> None:
        state = {"personality_modes": None}
        result = resolve_personality_mode(state, "novelty_analysis")
        assert result == AGENT_PERSONALITY_DEFAULTS["novelty_analysis"]

    def test_unknown_agent_falls_back_to_critical(self) -> None:
        state = {}
        result = resolve_personality_mode(state, "unknown_agent")
        assert result == PersonalityMode.CRITICAL

    def test_invalid_mode_in_state_falls_back_to_default(self) -> None:
        state = {
            "personality_modes": {
                "novelty_analysis": "bogus_mode",
            }
        }
        result = resolve_personality_mode(state, "novelty_analysis")
        assert result == AGENT_PERSONALITY_DEFAULTS["novelty_analysis"]

    def test_all_known_agents_resolve_from_defaults(self) -> None:
        """With empty state, every known agent resolves to its default."""
        state = {}
        for agent, expected_mode in AGENT_PERSONALITY_DEFAULTS.items():
            result = resolve_personality_mode(state, agent)
            assert result == expected_mode, (
                f"{agent}: expected {expected_mode}, got {result}"
            )
