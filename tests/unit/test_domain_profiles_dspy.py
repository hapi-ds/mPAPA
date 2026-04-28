"""Unit tests for DSPy module domain prefix injection.

Tests that DraftClaimsModule.forward() correctly injects domain prefixes
into the primary input field based on the domain_profile_slug parameter.

Validates: Requirements 8.1, 8.2, 8.3, 8.4
"""

import logging
from pathlib import Path
from unittest.mock import MagicMock

import dspy
import pytest

from patent_system.agents.domain_profiles import (
    DEFAULT_PROFILE_SLUG,
    ProfileLoader,
    generate_domain_prefix,
)
from patent_system.agents.personality import (
    PersonalityMode,
    generate_personality_prefix,
)
from patent_system.dspy_modules.modules import (
    DraftClaimsModule,
    set_profile_loader,
    _get_profile_loader,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def profile_loader(tmp_path: Path) -> ProfileLoader:
    """Create a ProfileLoader with built-in profiles in a temp directory."""
    profiles_dir = tmp_path / "domain_profiles"
    loader = ProfileLoader(profiles_dir)
    return loader


@pytest.fixture(autouse=True)
def _set_and_reset_loader(profile_loader: ProfileLoader):
    """Set the module-level profile loader before each test and reset after."""
    set_profile_loader(profile_loader)
    yield
    # Reset to None to avoid test pollution
    import patent_system.dspy_modules.modules as mod
    mod._profile_loader = None


def _make_draft_claims_module() -> DraftClaimsModule:
    """Create a DraftClaimsModule with predict replaced by a MagicMock."""
    module = DraftClaimsModule()
    module.predict = MagicMock(return_value=dspy.Prediction(claims_text="mocked"))
    return module


def _get_primary_input(mock_predict) -> str:
    """Extract the invention_disclosure value from the mocked predict call."""
    _args, kwargs = mock_predict.call_args
    return kwargs["invention_disclosure"]


# ---------------------------------------------------------------------------
# Test: forward() with valid domain_profile_slug includes domain prefix
# ---------------------------------------------------------------------------


class TestDomainPrefixInjectedWithValidSlug:
    """Validates: Requirement 8.1 — domain prefix injected for valid slug."""

    def test_forward_with_valid_slug_includes_domain_tag(
        self, profile_loader: ProfileLoader
    ) -> None:
        """forward() with a valid domain_profile_slug includes [Domain: <slug>] tag."""
        module = _make_draft_claims_module()
        slug = "pharma-chemistry"

        module.forward(
            invention_disclosure="test disclosure",
            novelty_analysis="test novelty",
            personality_mode="critical",
            domain_profile_slug=slug,
        )

        primary_input = _get_primary_input(module.predict)
        assert f"[Domain: {slug}]" in primary_input

    def test_forward_with_valid_slug_includes_role_prompt(
        self, profile_loader: ProfileLoader
    ) -> None:
        """forward() with a valid slug includes the profile's role_prompt content."""
        module = _make_draft_claims_module()
        slug = "pharma-chemistry"
        profile = profile_loader.get_by_slug(slug)
        assert profile is not None

        module.forward(
            invention_disclosure="test disclosure",
            novelty_analysis="test novelty",
            personality_mode="neutral",
            domain_profile_slug=slug,
        )

        primary_input = _get_primary_input(module.predict)
        # The domain prefix includes the role_prompt text
        assert profile.role_prompt.strip().split("\n")[0] in primary_input

    def test_forward_with_default_slug_includes_domain_tag(
        self, profile_loader: ProfileLoader
    ) -> None:
        """forward() with the default slug includes [Domain: general-patent-drafting]."""
        module = _make_draft_claims_module()

        module.forward(
            invention_disclosure="test disclosure",
            novelty_analysis="test novelty",
            personality_mode="critical",
            domain_profile_slug=DEFAULT_PROFILE_SLUG,
        )

        primary_input = _get_primary_input(module.predict)
        assert f"[Domain: {DEFAULT_PROFILE_SLUG}]" in primary_input


# ---------------------------------------------------------------------------
# Test: forward() without domain_profile_slug uses default profile
# ---------------------------------------------------------------------------


class TestDefaultProfileWhenSlugAbsent:
    """Validates: Requirement 8.2 — default profile used when slug is absent."""

    def test_forward_without_slug_uses_default_profile(
        self, profile_loader: ProfileLoader
    ) -> None:
        """forward() without domain_profile_slug uses the default profile prefix."""
        module = _make_draft_claims_module()

        module.forward(
            invention_disclosure="test disclosure",
            novelty_analysis="test novelty",
            personality_mode="critical",
        )

        primary_input = _get_primary_input(module.predict)
        assert f"[Domain: {DEFAULT_PROFILE_SLUG}]" in primary_input

    def test_forward_with_none_slug_uses_default_profile(
        self, profile_loader: ProfileLoader
    ) -> None:
        """forward() with domain_profile_slug=None uses the default profile prefix."""
        module = _make_draft_claims_module()

        module.forward(
            invention_disclosure="test disclosure",
            novelty_analysis="test novelty",
            personality_mode="critical",
            domain_profile_slug=None,
        )

        primary_input = _get_primary_input(module.predict)
        assert f"[Domain: {DEFAULT_PROFILE_SLUG}]" in primary_input


# ---------------------------------------------------------------------------
# Test: personality prefix appears before domain prefix
# ---------------------------------------------------------------------------


class TestPrefixOrdering:
    """Validates: Requirement 8.3 — personality prefix before domain prefix."""

    def test_personality_prefix_before_domain_prefix(
        self, profile_loader: ProfileLoader
    ) -> None:
        """Personality prefix appears before domain prefix in the combined input."""
        module = _make_draft_claims_module()
        slug = "pharma-chemistry"
        personality_prefix = generate_personality_prefix(PersonalityMode.CRITICAL)

        module.forward(
            invention_disclosure="test disclosure",
            novelty_analysis="test novelty",
            personality_mode="critical",
            domain_profile_slug=slug,
        )

        primary_input = _get_primary_input(module.predict)
        personality_pos = primary_input.find("[Personality:")
        domain_pos = primary_input.find("[Domain:")

        assert personality_pos >= 0, "Personality tag should be in the input"
        assert domain_pos >= 0, "Domain tag should be in the input"
        assert personality_pos < domain_pos, (
            f"Personality prefix (pos {personality_pos}) should appear "
            f"before domain prefix (pos {domain_pos})"
        )

    def test_domain_prefix_before_original_input(
        self, profile_loader: ProfileLoader
    ) -> None:
        """Domain prefix appears before the original input text."""
        module = _make_draft_claims_module()
        slug = "medtech-mechanical-engineering"
        original_input = "my unique invention disclosure text"

        module.forward(
            invention_disclosure=original_input,
            novelty_analysis="test novelty",
            personality_mode="neutral",
            domain_profile_slug=slug,
        )

        primary_input = _get_primary_input(module.predict)
        domain_pos = primary_input.find("[Domain:")
        original_pos = primary_input.find(original_input)

        assert domain_pos >= 0, "Domain tag should be in the input"
        assert original_pos >= 0, "Original input should be in the input"
        assert domain_pos < original_pos, (
            f"Domain prefix (pos {domain_pos}) should appear "
            f"before original input (pos {original_pos})"
        )

    def test_full_ordering_personality_domain_input(
        self, profile_loader: ProfileLoader
    ) -> None:
        """Full ordering: personality prefix, then domain prefix, then original input."""
        module = _make_draft_claims_module()
        slug = "software-ai"
        original_input = "unique disclosure content xyz"

        module.forward(
            invention_disclosure=original_input,
            novelty_analysis="test novelty",
            personality_mode="innovation_friendly",
            domain_profile_slug=slug,
        )

        primary_input = _get_primary_input(module.predict)
        personality_pos = primary_input.find("[Personality:")
        domain_pos = primary_input.find("[Domain:")
        original_pos = primary_input.find(original_input)

        assert personality_pos < domain_pos < original_pos, (
            f"Expected ordering: personality ({personality_pos}) < "
            f"domain ({domain_pos}) < original ({original_pos})"
        )


# ---------------------------------------------------------------------------
# Test: invalid slug falls back to default profile with warning logged
# ---------------------------------------------------------------------------


class TestInvalidSlugFallback:
    """Validates: Requirement 8.4 — invalid slug falls back to default with warning."""

    def test_invalid_slug_falls_back_to_default(
        self, profile_loader: ProfileLoader, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An invalid slug falls back to the default profile prefix."""
        module = _make_draft_claims_module()
        invalid_slug = "nonexistent-domain-profile"

        with caplog.at_level(logging.WARNING):
            module.forward(
                invention_disclosure="test disclosure",
                novelty_analysis="test novelty",
                personality_mode="critical",
                domain_profile_slug=invalid_slug,
            )

        primary_input = _get_primary_input(module.predict)
        # Should fall back to default profile
        assert f"[Domain: {DEFAULT_PROFILE_SLUG}]" in primary_input

    def test_invalid_slug_logs_warning(
        self, profile_loader: ProfileLoader, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An invalid slug logs a warning about the fallback."""
        module = _make_draft_claims_module()
        invalid_slug = "totally-bogus-slug"

        with caplog.at_level(logging.WARNING):
            module.forward(
                invention_disclosure="test disclosure",
                novelty_analysis="test novelty",
                personality_mode="critical",
                domain_profile_slug=invalid_slug,
            )

        # Check that a warning was logged mentioning the invalid slug
        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any(invalid_slug in msg for msg in warning_messages), (
            f"Expected a warning mentioning {invalid_slug!r}, "
            f"got: {warning_messages}"
        )

    def test_invalid_slug_still_includes_personality_prefix(
        self, profile_loader: ProfileLoader
    ) -> None:
        """Even with an invalid slug, personality prefix is still present."""
        module = _make_draft_claims_module()
        invalid_slug = "does-not-exist"

        module.forward(
            invention_disclosure="test disclosure",
            novelty_analysis="test novelty",
            personality_mode="neutral",
            domain_profile_slug=invalid_slug,
        )

        primary_input = _get_primary_input(module.predict)
        assert "[Personality: neutral]" in primary_input
