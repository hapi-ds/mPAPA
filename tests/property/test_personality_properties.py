"""Property-based tests for Agent Personality Modes.

Validates: Requirements 1.1, 1.6, 2.4, 2.5, 3.2, 4.1, 4.4
"""

import logging
import sqlite3
from contextlib import contextmanager
from typing import Generator

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from patent_system.agents.personality import (
    AGENT_PERSONALITY_DEFAULTS,
    PersonalityMode,
    generate_personality_prefix,
    parse_mode_from_prefix,
    resolve_personality_mode,
)


@contextmanager
def unittest_mock_patch_log(
    logger: logging.Logger,
) -> Generator[list[logging.LogRecord], None, None]:
    """Context manager that captures log records from a specific logger."""
    records: list[logging.LogRecord] = []

    class _Handler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Handler()
    logger.addHandler(handler)
    old_level = logger.level
    logger.setLevel(logging.DEBUG)
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# The three valid mode strings
_VALID_MODES = ["critical", "neutral", "innovation_friendly"]

# Strategy that produces only valid PersonalityMode values
_valid_mode = st.sampled_from(list(PersonalityMode))

# Strategy that produces arbitrary strings guaranteed NOT to be valid modes
_invalid_mode_string = st.text(min_size=1, max_size=50).filter(
    lambda s: s not in _VALID_MODES
)

# Known agent names from AGENT_PERSONALITY_DEFAULTS
_known_agent = st.sampled_from(list(AGENT_PERSONALITY_DEFAULTS.keys()))

# Agent names guaranteed NOT in AGENT_PERSONALITY_DEFAULTS
_unknown_agent = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="_"),
    min_size=1,
    max_size=40,
).filter(lambda s: s not in AGENT_PERSONALITY_DEFAULTS)


# ---------------------------------------------------------------------------
# Property 1: Personality mode enum domain
# Feature: agent-personality-modes, Property 1: Personality mode enum domain
# ---------------------------------------------------------------------------


class TestPersonalityModeEnumDomain:
    """Property 1: Personality mode enum domain.

    For any string s, constructing PersonalityMode(s) succeeds iff s is one
    of {"critical", "neutral", "innovation_friendly"}. All other strings
    raise ValueError.

    **Validates: Requirements 1.1, 1.6**
    """

    @given(mode=_valid_mode)
    @settings(max_examples=100)
    def test_valid_mode_construction_succeeds(self, mode: PersonalityMode) -> None:
        """For any valid mode string, PersonalityMode(s) succeeds and round-trips."""
        result = PersonalityMode(mode.value)
        assert result == mode
        assert result.value in _VALID_MODES

    @given(s=_invalid_mode_string)
    @settings(max_examples=100)
    def test_invalid_mode_construction_raises(self, s: str) -> None:
        """For any string not in the valid set, PersonalityMode(s) raises ValueError."""
        with pytest.raises(ValueError):
            PersonalityMode(s)


# ---------------------------------------------------------------------------
# Property 2: Prefix length invariant
# Feature: agent-personality-modes, Property 2: Prefix length invariant
# ---------------------------------------------------------------------------


class TestPrefixLengthInvariant:
    """Property 2: Prefix length invariant.

    For any valid PersonalityMode, generate_personality_prefix(m) returns
    a non-empty string of at least 20 characters.

    **Validates: Requirements 2.4**
    """

    @given(mode=_valid_mode)
    @settings(max_examples=100)
    def test_prefix_is_nonempty_and_at_least_20_chars(
        self, mode: PersonalityMode
    ) -> None:
        """For any valid mode, the generated prefix is non-empty and >= 20 chars."""
        prefix = generate_personality_prefix(mode)
        assert isinstance(prefix, str)
        assert len(prefix) > 0
        assert len(prefix) >= 20


# ---------------------------------------------------------------------------
# Property 3: Prefix generation round-trip
# Feature: agent-personality-modes, Property 3: Prefix generation round-trip
# ---------------------------------------------------------------------------


class TestPrefixGenerationRoundTrip:
    """Property 3: Prefix generation round-trip.

    For any valid PersonalityMode m, generating a prefix, parsing the mode
    from it, then generating a prefix again produces an identical string.

    **Validates: Requirements 2.5**
    """

    @given(mode=_valid_mode)
    @settings(max_examples=100)
    def test_generate_parse_generate_is_idempotent(
        self, mode: PersonalityMode
    ) -> None:
        """generate(parse(generate(m))) == generate(m) for any valid mode."""
        first_prefix = generate_personality_prefix(mode)
        parsed_mode = parse_mode_from_prefix(first_prefix)
        second_prefix = generate_personality_prefix(parsed_mode)
        assert second_prefix == first_prefix


# ---------------------------------------------------------------------------
# Property 4: Default mode resolution from absent state
# Feature: agent-personality-modes, Property 4: Default mode resolution from absent state
# ---------------------------------------------------------------------------


class TestDefaultModeResolutionFromAbsentState:
    """Property 4: Default mode resolution from absent state.

    For any state where personality_modes is absent, empty, or missing the
    agent, resolve_personality_mode returns the default from
    AGENT_PERSONALITY_DEFAULTS or CRITICAL for unknown agents.

    **Validates: Requirements 3.2**
    """

    @given(agent=_known_agent)
    @settings(max_examples=100)
    def test_absent_state_returns_agent_default(self, agent: str) -> None:
        """When personality_modes is absent, known agents get their default."""
        # State with no personality_modes key at all
        state_missing_key: dict = {}
        result = resolve_personality_mode(state_missing_key, agent)
        assert result == AGENT_PERSONALITY_DEFAULTS[agent]

    @given(agent=_known_agent)
    @settings(max_examples=100)
    def test_empty_dict_returns_agent_default(self, agent: str) -> None:
        """When personality_modes is an empty dict, known agents get their default."""
        state_empty: dict = {"personality_modes": {}}
        result = resolve_personality_mode(state_empty, agent)
        assert result == AGENT_PERSONALITY_DEFAULTS[agent]

    @given(agent=_unknown_agent)
    @settings(max_examples=100)
    def test_unknown_agent_absent_state_returns_critical(self, agent: str) -> None:
        """When personality_modes is absent and agent is unknown, returns CRITICAL."""
        state: dict = {}
        result = resolve_personality_mode(state, agent)
        assert result == PersonalityMode.CRITICAL

    @given(agent=_unknown_agent)
    @settings(max_examples=100)
    def test_unknown_agent_empty_dict_returns_critical(self, agent: str) -> None:
        """When personality_modes is empty and agent is unknown, returns CRITICAL."""
        state: dict = {"personality_modes": {}}
        result = resolve_personality_mode(state, agent)
        assert result == PersonalityMode.CRITICAL


# ---------------------------------------------------------------------------
# Property 7: Invalid env var validation
# Feature: agent-personality-modes, Property 7: Invalid env var validation
# ---------------------------------------------------------------------------


class TestInvalidEnvVarValidation:
    """Property 7: Invalid env var validation.

    For any string not in {"critical", "neutral", "innovation_friendly"},
    setting PATENT_DEFAULT_PERSONALITY_MODE and constructing AppSettings()
    raises ValidationError.

    **Validates: Requirements 6.3**
    """

    # Env-safe invalid mode strings: no null bytes (OS rejects them)
    _env_safe_invalid_mode = st.text(
        alphabet=st.characters(
            whitelist_categories=("L", "N", "P", "S", "Z"),
            blacklist_characters="\x00",
        ),
        min_size=1,
        max_size=50,
    ).filter(lambda s: s not in _VALID_MODES)

    @given(s=_env_safe_invalid_mode)
    @settings(max_examples=100)
    def test_invalid_env_var_raises_validation_error(self, s: str) -> None:
        """For any invalid mode string in the env var, AppSettings raises ValidationError."""
        import os

        from pydantic import ValidationError

        from patent_system.config import AppSettings

        env_key = "PATENT_DEFAULT_PERSONALITY_MODE"
        original = os.environ.get(env_key)
        try:
            os.environ[env_key] = s
            with pytest.raises(ValidationError):
                AppSettings(_env_file=None)
        finally:
            if original is None:
                os.environ.pop(env_key, None)
            else:
                os.environ[env_key] = original


# ---------------------------------------------------------------------------
# Property 8: Workflow step personality persistence round-trip
# Feature: agent-personality-modes, Property 8: Workflow step personality persistence round-trip
# ---------------------------------------------------------------------------


class TestWorkflowStepPersonalityPersistenceRoundTrip:
    """Property 8: Workflow step personality persistence round-trip.

    For any valid step_key, content, and PersonalityMode, upsert then
    get_step returns the same personality_mode.

    **Validates: Requirements 8.4**
    """

    # Strategy for valid step keys
    _valid_step_key = st.sampled_from(
        [
            "initial_idea",
            "claims_drafting",
            "prior_art_search",
            "novelty_analysis",
            "consistency_review",
            "market_potential",
            "legal_clarification",
            "disclosure_summary",
            "patent_draft",
        ]
    )

    # Safe content text
    _safe_content = st.text(
        alphabet=st.characters(
            whitelist_categories=("L", "N", "P", "S"),
            whitelist_characters=" /-_:.",
        ),
        min_size=0,
        max_size=200,
    )

    @given(
        step_key=_valid_step_key,
        content=_safe_content,
        mode=_valid_mode,
    )
    @settings(max_examples=100)
    def test_upsert_then_get_step_preserves_personality_mode(
        self,
        step_key: str,
        content: str,
        mode: PersonalityMode,
    ) -> None:
        """For any valid step_key, content, and mode, upsert then get_step returns the same mode."""
        from patent_system.db.repository import WorkflowStepRepository
        from patent_system.db.schema import init_schema

        conn = sqlite3.connect(":memory:")
        conn.execute("PRAGMA foreign_keys = ON")
        init_schema(conn)
        try:
            conn.execute("INSERT INTO topics (name) VALUES (?)", ("test",))
            conn.commit()
            topic_id = conn.execute(
                "SELECT id FROM topics WHERE name = 'test'"
            ).fetchone()[0]

            repo = WorkflowStepRepository(conn)
            repo.upsert(
                topic_id=topic_id,
                step_key=step_key,
                content=content,
                status="completed",
                personality_mode=mode.value,
            )

            result = repo.get_step(topic_id, step_key)
            assert result is not None
            assert result["personality_mode"] == mode.value
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Property 9: Per-topic personality config persistence round-trip
# Feature: agent-personality-modes, Property 9: Per-topic personality config persistence round-trip
# ---------------------------------------------------------------------------


class TestPerTopicPersonalityConfigPersistenceRoundTrip:
    """Property 9: Per-topic personality config persistence round-trip.

    For any valid mapping of agent names to PersonalityMode values,
    save then get_by_topic returns the same mapping.

    **Validates: Requirements 9.1, 9.2**
    """

    # Strategy for agent-name-to-mode mappings (non-empty)
    _agent_mode_mapping = st.dictionaries(
        keys=st.sampled_from(list(AGENT_PERSONALITY_DEFAULTS.keys())),
        values=_valid_mode.map(lambda m: m.value),
        min_size=1,
        max_size=len(AGENT_PERSONALITY_DEFAULTS),
    )

    @given(mapping=_agent_mode_mapping)
    @settings(max_examples=100)
    def test_save_then_get_by_topic_returns_same_mapping(
        self,
        mapping: dict[str, str],
    ) -> None:
        """For any valid agent→mode mapping, save then get_by_topic returns the same mapping."""
        from patent_system.db.repository import PersonalityPreferenceRepository
        from patent_system.db.schema import init_schema

        conn = sqlite3.connect(":memory:")
        conn.execute("PRAGMA foreign_keys = ON")
        init_schema(conn)
        try:
            conn.execute("INSERT INTO topics (name) VALUES (?)", ("test",))
            conn.commit()
            topic_id = conn.execute(
                "SELECT id FROM topics WHERE name = 'test'"
            ).fetchone()[0]

            repo = PersonalityPreferenceRepository(conn)
            repo.save(topic_id, mapping)

            result = repo.get_by_topic(topic_id)
            assert result == mapping
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Property 5: DSPy prefix prepending
# Feature: agent-personality-modes, Property 5: DSPy prefix prepending
# ---------------------------------------------------------------------------


class TestDSPyPrefixPrepending:
    """Property 5: DSPy prefix prepending.

    For any valid PersonalityMode and non-empty input, calling forward()
    with personality_mode results in the primary input passed to
    self.predict() starting with the prefix.

    **Validates: Requirements 4.1**
    """

    # Non-empty input strings for the primary input field
    _nonempty_input = st.text(min_size=1, max_size=200).filter(lambda s: s.strip())

    @given(mode=_valid_mode, disclosure=_nonempty_input, novelty=_nonempty_input)
    @settings(max_examples=100)
    def test_forward_prepends_prefix_to_primary_input(
        self,
        mode: PersonalityMode,
        disclosure: str,
        novelty: str,
    ) -> None:
        """For any valid mode and non-empty input, predict receives prefixed primary input."""
        from unittest.mock import MagicMock

        import dspy

        from patent_system.dspy_modules.modules import DraftClaimsModule

        module = DraftClaimsModule()
        module.predict = MagicMock(return_value=dspy.Prediction(claims_text="test"))

        module.forward(
            invention_disclosure=disclosure,
            novelty_analysis=novelty,
            personality_mode=mode.value,
        )

        expected_prefix = generate_personality_prefix(mode)
        _args, kwargs = module.predict.call_args
        assert kwargs["invention_disclosure"].startswith(expected_prefix)


# ---------------------------------------------------------------------------
# Property 6: Invalid mode fallback in DSPy modules
# Feature: agent-personality-modes, Property 6: Invalid mode fallback in DSPy modules
# ---------------------------------------------------------------------------


class TestInvalidModeFallbackInDSPyModules:
    """Property 6: Invalid mode fallback in DSPy modules.

    For any invalid personality_mode string, DSPy module forward() falls
    back to CRITICAL prefix and logs a warning.

    **Validates: Requirements 4.4**
    """

    @given(invalid_mode=_invalid_mode_string)
    @settings(max_examples=100)
    def test_invalid_mode_falls_back_to_critical_prefix(
        self,
        invalid_mode: str,
    ) -> None:
        """For any invalid mode string, predict receives the CRITICAL prefix."""
        from unittest.mock import MagicMock

        import dspy

        from patent_system.dspy_modules.modules import DraftClaimsModule

        module = DraftClaimsModule()
        module.predict = MagicMock(return_value=dspy.Prediction(claims_text="test"))

        module.forward(
            invention_disclosure="test disclosure",
            novelty_analysis="test novelty",
            personality_mode=invalid_mode,
        )

        critical_prefix = generate_personality_prefix(PersonalityMode.CRITICAL)
        _args, kwargs = module.predict.call_args
        assert kwargs["invention_disclosure"].startswith(critical_prefix)

    @given(invalid_mode=_invalid_mode_string)
    @settings(max_examples=100)
    def test_invalid_mode_logs_warning(
        self,
        invalid_mode: str,
    ) -> None:
        """For any invalid mode string, a warning is logged."""
        import logging
        from unittest.mock import MagicMock

        import dspy

        from patent_system.dspy_modules.modules import DraftClaimsModule

        module = DraftClaimsModule()
        module.predict = MagicMock(return_value=dspy.Prediction(claims_text="test"))

        logger = logging.getLogger("patent_system.agents.personality")
        with unittest_mock_patch_log(logger) as log_records:
            module.forward(
                invention_disclosure="test disclosure",
                novelty_analysis="test novelty",
                personality_mode=invalid_mode,
            )

        warning_messages = [
            r.getMessage() for r in log_records if r.levelno == logging.WARNING
        ]
        assert any("Invalid personality mode" in msg for msg in warning_messages)
