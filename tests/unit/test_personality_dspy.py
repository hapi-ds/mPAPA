"""Unit tests for DSPy module personality integration.

Tests each module's forward() with each valid mode, default behavior when
personality_mode is None, and that Signature definitions are unchanged
after forward() calls.

Requirements: 4.1–4.4
"""

from unittest.mock import MagicMock

import dspy
import pytest

from patent_system.agents.personality import (
    PersonalityMode,
    generate_personality_prefix,
)
from patent_system.dspy_modules.modules import (
    DraftClaimsModule,
    DraftDescriptionModule,
    DisclosureSummaryModule,
    InterviewQuestionModule,
    LegalClarificationModule,
    MarketPotentialModule,
    NoveltyAnalysisModule,
    PriorArtSummaryModule,
    RefineClaimsModule,
    ReviewConsistencyModule,
    StructureDisclosureModule,
    SuggestSearchTermsModule,
)
from patent_system.dspy_modules.signatures import (
    AnalyzeLegalClarification,
    AnalyzeMarketPotential,
    AnalyzeNovelty,
    DraftClaims,
    DraftDescription,
    InventionInterviewQuestion,
    RefineClaims,
    ReviewConsistency,
    StructureDisclosure,
    SuggestSearchTerms,
    SummarizeDisclosure,
    SummarizePriorArt,
)


# ---------------------------------------------------------------------------
# Helpers: module factory with mocked predict and expected call kwargs
# ---------------------------------------------------------------------------

# Each entry: (ModuleClass, output_fields_for_mock, forward_kwargs, primary_input_kwarg, signature_class)
_MODULE_SPECS = [
    (
        DraftClaimsModule,
        {"claims_text": "test"},
        {"invention_disclosure": "disc", "novelty_analysis": "nov"},
        "invention_disclosure",
        DraftClaims,
    ),
    (
        ReviewConsistencyModule,
        {"feedback": "ok", "approved": True},
        {"claims": "claims", "description": "desc"},
        "claims",
        ReviewConsistency,
    ),
    (
        DraftDescriptionModule,
        {"description_text": "test"},
        {
            "claims": "claims",
            "prior_art_summary": "pa",
            "invention_disclosure": "disc",
        },
        "claims",
        DraftDescription,
    ),
    (
        RefineClaimsModule,
        {"refined_claims": "test"},
        {
            "original_claims": "claims",
            "invention_disclosure": "disc",
            "novelty_analysis": "nov",
            "consistency_review": "cr",
            "market_assessment": "ma",
            "legal_assessment": "la",
        },
        "original_claims",
        RefineClaims,
    ),
    (
        MarketPotentialModule,
        {"market_assessment": "test"},
        {
            "invention_disclosure": "disc",
            "claims_text": "claims",
            "novelty_analysis": "nov",
        },
        "invention_disclosure",
        AnalyzeMarketPotential,
    ),
    (
        LegalClarificationModule,
        {"legal_assessment": "test"},
        {
            "invention_disclosure": "disc",
            "claims_text": "claims",
            "prior_art_summary": "pa",
            "novelty_analysis": "nov",
        },
        "invention_disclosure",
        AnalyzeLegalClarification,
    ),
    (
        DisclosureSummaryModule,
        {"disclosure_summary": "test"},
        {
            "initial_idea": "idea",
            "claims_text": "claims",
            "prior_art_summary": "pa",
            "novelty_analysis": "nov",
            "consistency_review": "cr",
            "market_assessment": "ma",
            "legal_assessment": "la",
        },
        "initial_idea",
        SummarizeDisclosure,
    ),
    (
        NoveltyAnalysisModule,
        {"novelty_assessment": "test"},
        {
            "invention_disclosure": "disc",
            "claims_text": "claims",
            "prior_art_summary": "pa",
        },
        "invention_disclosure",
        AnalyzeNovelty,
    ),
    (
        PriorArtSummaryModule,
        {"prior_art_summary": "test"},
        {
            "invention_disclosure": "disc",
            "claims_text": "claims",
            "prior_art_references": "refs",
        },
        "invention_disclosure",
        SummarizePriorArt,
    ),
    (
        InterviewQuestionModule,
        {"next_question": "test"},
        {"conversation_history": "history", "invention_context": "ctx"},
        "conversation_history",
        InventionInterviewQuestion,
    ),
    (
        StructureDisclosureModule,
        {"disclosure_json": "test"},
        {"transcript": "transcript"},
        "transcript",
        StructureDisclosure,
    ),
    (
        SuggestSearchTermsModule,
        {"search_terms": "test"},
        {"invention_description": "desc"},
        "invention_description",
        SuggestSearchTerms,
    ),
]


def _make_module_with_mock(module_cls, output_fields):
    """Create a module instance with predict replaced by a MagicMock."""
    module = module_cls()
    module.predict = MagicMock(return_value=dspy.Prediction(**output_fields))
    return module


# ---------------------------------------------------------------------------
# Test: forward() with each valid mode prepends correct prefix
# ---------------------------------------------------------------------------


class TestForwardWithValidModes:
    """Test each module's forward() with each valid personality mode."""

    @pytest.mark.parametrize(
        "module_cls,output_fields,forward_kwargs,primary_kwarg,sig_cls",
        _MODULE_SPECS,
        ids=[spec[0].__name__ for spec in _MODULE_SPECS],
    )
    @pytest.mark.parametrize("mode", list(PersonalityMode))
    def test_forward_prepends_correct_prefix(
        self,
        module_cls,
        output_fields,
        forward_kwargs,
        primary_kwarg,
        sig_cls,
        mode,
    ) -> None:
        """forward() with a valid mode prepends the expected prefix to the primary input."""
        module = _make_module_with_mock(module_cls, output_fields)
        expected_prefix = generate_personality_prefix(mode)

        module.forward(**forward_kwargs, personality_mode=mode.value)

        _args, kwargs = module.predict.call_args
        assert kwargs[primary_kwarg].startswith(expected_prefix), (
            f"{module_cls.__name__} with mode={mode.value}: "
            f"expected primary input '{primary_kwarg}' to start with prefix"
        )

    @pytest.mark.parametrize(
        "module_cls,output_fields,forward_kwargs,primary_kwarg,sig_cls",
        _MODULE_SPECS,
        ids=[spec[0].__name__ for spec in _MODULE_SPECS],
    )
    @pytest.mark.parametrize("mode", list(PersonalityMode))
    def test_forward_includes_original_input_after_prefix(
        self,
        module_cls,
        output_fields,
        forward_kwargs,
        primary_kwarg,
        sig_cls,
        mode,
    ) -> None:
        """forward() preserves the original input text after the prefix."""
        module = _make_module_with_mock(module_cls, output_fields)
        original_value = forward_kwargs[primary_kwarg]

        module.forward(**forward_kwargs, personality_mode=mode.value)

        _args, kwargs = module.predict.call_args
        assert original_value in kwargs[primary_kwarg], (
            f"{module_cls.__name__}: original input should be present after prefix"
        )


# ---------------------------------------------------------------------------
# Test: default behavior when personality_mode is None
# ---------------------------------------------------------------------------


class TestDefaultBehaviorWhenModeIsNone:
    """Test that forward() defaults to CRITICAL when personality_mode is None."""

    @pytest.mark.parametrize(
        "module_cls,output_fields,forward_kwargs,primary_kwarg,sig_cls",
        _MODULE_SPECS,
        ids=[spec[0].__name__ for spec in _MODULE_SPECS],
    )
    def test_none_mode_defaults_to_critical(
        self,
        module_cls,
        output_fields,
        forward_kwargs,
        primary_kwarg,
        sig_cls,
    ) -> None:
        """When personality_mode is None, forward() uses CRITICAL prefix."""
        module = _make_module_with_mock(module_cls, output_fields)
        critical_prefix = generate_personality_prefix(PersonalityMode.CRITICAL)

        # Call without personality_mode (defaults to None)
        module.forward(**forward_kwargs)

        _args, kwargs = module.predict.call_args
        assert kwargs[primary_kwarg].startswith(critical_prefix), (
            f"{module_cls.__name__}: None mode should default to CRITICAL prefix"
        )


# ---------------------------------------------------------------------------
# Test: Signature definitions are unchanged after forward() call
# ---------------------------------------------------------------------------


class TestSignatureUnchangedAfterForward:
    """Test that DSPy Signature definitions are not modified by forward()."""

    @pytest.mark.parametrize(
        "module_cls,output_fields,forward_kwargs,primary_kwarg,sig_cls",
        _MODULE_SPECS,
        ids=[spec[0].__name__ for spec in _MODULE_SPECS],
    )
    def test_signature_fields_unchanged_after_forward(
        self,
        module_cls,
        output_fields,
        forward_kwargs,
        primary_kwarg,
        sig_cls,
    ) -> None:
        """Signature input/output fields remain unchanged after forward() call."""
        # Capture signature fields before
        input_fields_before = set(sig_cls.input_fields.keys())
        output_fields_before = set(sig_cls.output_fields.keys())

        module = _make_module_with_mock(module_cls, output_fields)
        module.forward(**forward_kwargs, personality_mode="neutral")

        # Verify signature fields after
        input_fields_after = set(sig_cls.input_fields.keys())
        output_fields_after = set(sig_cls.output_fields.keys())

        assert input_fields_after == input_fields_before, (
            f"{sig_cls.__name__} input fields changed after forward()"
        )
        assert output_fields_after == output_fields_before, (
            f"{sig_cls.__name__} output fields changed after forward()"
        )


# ---------------------------------------------------------------------------
# Test: invalid mode falls back to CRITICAL
# ---------------------------------------------------------------------------


class TestInvalidModeFallback:
    """Test that invalid personality_mode values fall back to CRITICAL."""

    @pytest.mark.parametrize("invalid_mode", ["bogus", "CRITICAL", "Neutral", "123", ""])
    def test_invalid_mode_uses_critical_prefix(self, invalid_mode: str) -> None:
        """Invalid mode strings cause fallback to CRITICAL prefix."""
        module = _make_module_with_mock(DraftClaimsModule, {"claims_text": "test"})
        critical_prefix = generate_personality_prefix(PersonalityMode.CRITICAL)

        # Empty string is handled by the `or "critical"` in forward(),
        # so it actually resolves to "critical" directly. Non-empty invalid
        # strings go through generate_personality_prefix fallback.
        module.forward(
            invention_disclosure="disc",
            novelty_analysis="nov",
            personality_mode=invalid_mode if invalid_mode else None,
        )

        _args, kwargs = module.predict.call_args
        assert kwargs["invention_disclosure"].startswith(critical_prefix)
