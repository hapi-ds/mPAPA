"""Property-based tests for AppSettings validation and environment variable precedence.

Validates: Requirements 19.1, 19.2, 19.3
"""

import os

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from patent_system.config import AppSettings

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# String settings that accept arbitrary non-empty text values
_STRING_SETTINGS: list[tuple[str, str]] = [
    ("lm_studio_base_url", "PATENT_LM_STUDIO_BASE_URL"),
    ("lm_studio_api_key", "PATENT_LM_STUDIO_API_KEY"),
    ("model_disclosure", "PATENT_MODEL_DISCLOSURE"),
    ("model_search", "PATENT_MODEL_SEARCH"),
    ("model_claims", "PATENT_MODEL_CLAIMS"),
    ("model_description", "PATENT_MODEL_DESCRIPTION"),
    ("model_review", "PATENT_MODEL_REVIEW"),
    ("model_chat", "PATENT_MODEL_CHAT"),
    ("embedding_model_name", "PATENT_EMBEDDING_MODEL_NAME"),
    ("log_level", "PATENT_LOG_LEVEL"),
]

# Generate a random non-empty printable string (no newlines/control chars)
_safe_text = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "S"), whitelist_characters=" /-_:."),
    min_size=1,
    max_size=80,
)

# Generate a random positive integer for monitoring_interval_hours
_pos_int = st.integers(min_value=1, max_value=10_000)


# ---------------------------------------------------------------------------
# Property 12: Settings validation
# Feature: patent-analysis-drafting, Property 12: Settings validation
# ---------------------------------------------------------------------------


class TestSettingsValidation:
    """Property 12: Settings validation.

    For any config dict with all required valid values, constructing
    AppSettings succeeds; for any config with an invalid type for a
    validated field, it raises ValidationError.

    **Validates: Requirements 19.1, 19.3**
    """

    @given(
        lm_studio_base_url=_safe_text,
        model_disclosure=_safe_text,
        model_search=_safe_text,
        model_claims=_safe_text,
        model_description=_safe_text,
        model_review=_safe_text,
        model_chat=_safe_text,
        embedding_model_name=_safe_text,
        log_level=_safe_text,
        monitoring_interval_hours=_pos_int,
    )
    @settings(max_examples=100)
    def test_valid_config_constructs_successfully(
        self,
        lm_studio_base_url: str,
        model_disclosure: str,
        model_search: str,
        model_claims: str,
        model_description: str,
        model_review: str,
        model_chat: str,
        embedding_model_name: str,
        log_level: str,
        monitoring_interval_hours: int,
    ) -> None:
        """For any valid values, AppSettings construction succeeds."""
        s = AppSettings(
            lm_studio_base_url=lm_studio_base_url,
            model_disclosure=model_disclosure,
            model_search=model_search,
            model_claims=model_claims,
            model_description=model_description,
            model_review=model_review,
            model_chat=model_chat,
            embedding_model_name=embedding_model_name,
            log_level=log_level,
            monitoring_interval_hours=monitoring_interval_hours,
        )
        assert s.lm_studio_base_url == lm_studio_base_url
        assert s.model_disclosure == model_disclosure
        assert s.monitoring_interval_hours == monitoring_interval_hours

    @given(bad_value=st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=20))
    @settings(max_examples=100)
    def test_invalid_monitoring_interval_raises_validation_error(
        self,
        bad_value: str,
    ) -> None:
        """For any non-integer string for monitoring_interval_hours, ValidationError is raised."""
        env_key = "PATENT_MONITORING_INTERVAL_HOURS"
        original = os.environ.get(env_key)
        try:
            os.environ[env_key] = bad_value
            with pytest.raises(ValidationError):
                AppSettings()
        finally:
            if original is None:
                os.environ.pop(env_key, None)
            else:
                os.environ[env_key] = original


# ---------------------------------------------------------------------------
# Property 13: Environment variable precedence
# Feature: patent-analysis-drafting, Property 13: Environment variable precedence
# ---------------------------------------------------------------------------


class TestEnvVarPrecedence:
    """Property 13: Environment variable precedence.

    For any setting key with distinct values in .env and env var,
    the env var value wins.

    **Validates: Requirements 19.2**
    """

    @given(
        setting_idx=st.integers(min_value=0, max_value=len(_STRING_SETTINGS) - 1),
        env_value=_safe_text,
    )
    @settings(max_examples=100)
    def test_env_var_overrides_default(
        self,
        setting_idx: int,
        env_value: str,
    ) -> None:
        """For any string setting, an env var with PATENT_ prefix overrides the default."""
        field_name, env_name = _STRING_SETTINGS[setting_idx]
        original = os.environ.get(env_name)
        try:
            os.environ[env_name] = env_value
            s = AppSettings()
            assert getattr(s, field_name) == env_value
        finally:
            if original is None:
                os.environ.pop(env_name, None)
            else:
                os.environ[env_name] = original

    @given(monitoring_hours=_pos_int)
    @settings(max_examples=100)
    def test_env_var_overrides_monitoring_interval(
        self,
        monitoring_hours: int,
    ) -> None:
        """For any valid integer, PATENT_MONITORING_INTERVAL_HOURS env var takes precedence."""
        env_key = "PATENT_MONITORING_INTERVAL_HOURS"
        original = os.environ.get(env_key)
        try:
            os.environ[env_key] = str(monitoring_hours)
            s = AppSettings()
            assert s.monitoring_interval_hours == monitoring_hours
        finally:
            if original is None:
                os.environ.pop(env_key, None)
            else:
                os.environ[env_key] = original
