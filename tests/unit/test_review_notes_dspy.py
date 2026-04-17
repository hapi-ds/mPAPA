"""Unit tests for DSPy module review_notes_text integration.

Tests each module's forward() with review_notes_text parameter to verify:
- Input string contains review notes after personality prefix when provided
- Input is unchanged from current behavior when review_notes_text is None
- Input is unchanged when review_notes_text is empty string
- Ordering: personality prefix appears before review notes in the input

Requirements: 5.7, 6.1, 6.3
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
    LegalClarificationModule,
    MarketPotentialModule,
    NoveltyAnalysisModule,
    PriorArtSummaryModule,
    RefineClaimsModule,
    ReviewConsistencyModule,
)


# ---------------------------------------------------------------------------
# Module specs: (ModuleClass, mock_output_fields, forward_kwargs, primary_input_kwarg)
# ---------------------------------------------------------------------------

_MODULE_SPECS = [
    (
        DraftClaimsModule,
        {"claims_text": "mocked claims"},
        {"invention_disclosure": "disclosure text", "novelty_analysis": "analysis"},
        "invention_disclosure",
    ),
    (
        ReviewConsistencyModule,
        {"feedback": "ok", "approved": True},
        {"claims": "claims text", "description": "description text"},
        "claims",
    ),
    (
        DraftDescriptionModule,
        {"description_text": "mocked description"},
        {
            "claims": "claims text",
            "prior_art_summary": "prior art",
            "invention_disclosure": "disclosure",
        },
        "claims",
    ),
    (
        RefineClaimsModule,
        {"refined_claims": "mocked refined"},
        {
            "original_claims": "original claims",
            "invention_disclosure": "disclosure",
            "novelty_analysis": "novelty",
            "consistency_review": "consistency",
            "market_assessment": "market",
            "legal_assessment": "legal",
        },
        "original_claims",
    ),
    (
        MarketPotentialModule,
        {"market_assessment": "mocked market"},
        {
            "invention_disclosure": "disclosure text",
            "claims_text": "claims",
            "novelty_analysis": "novelty",
        },
        "invention_disclosure",
    ),
    (
        LegalClarificationModule,
        {"legal_assessment": "mocked legal"},
        {
            "invention_disclosure": "disclosure text",
            "claims_text": "claims",
            "prior_art_summary": "prior art",
            "novelty_analysis": "novelty",
        },
        "invention_disclosure",
    ),
    (
        DisclosureSummaryModule,
        {"disclosure_summary": "mocked summary"},
        {
            "initial_idea": "idea text",
            "claims_text": "claims",
            "prior_art_summary": "prior art",
            "novelty_analysis": "novelty",
            "consistency_review": "consistency",
            "market_assessment": "market",
            "legal_assessment": "legal",
        },
        "initial_idea",
    ),
    (
        NoveltyAnalysisModule,
        {"novelty_assessment": "mocked novelty"},
        {
            "invention_disclosure": "disclosure text",
            "claims_text": "claims",
            "prior_art_summary": "prior art",
        },
        "invention_disclosure",
    ),
    (
        PriorArtSummaryModule,
        {"prior_art_summary": "mocked prior art"},
        {
            "invention_disclosure": "disclosure text",
            "claims_text": "claims",
            "prior_art_references": "references",
        },
        "invention_disclosure",
    ),
]


def _make_module_with_mock(module_cls, output_fields):
    """Create a module instance with predict replaced by a MagicMock."""
    module = module_cls()
    module.predict = MagicMock(return_value=dspy.Prediction(**output_fields))
    return module


def _get_primary_input(mock_predict, primary_kwarg: str) -> str:
    """Extract the primary input field value from the mocked predict call."""
    _args, kwargs = mock_predict.call_args
    return kwargs[primary_kwarg]


# ---------------------------------------------------------------------------
# Test: forward() with review_notes_text includes notes in primary input
# ---------------------------------------------------------------------------


class TestReviewNotesInjected:
    """Test that review_notes_text is appended to the primary input field."""

    @pytest.mark.parametrize(
        "module_cls,output_fields,forward_kwargs,primary_kwarg",
        _MODULE_SPECS,
        ids=[spec[0].__name__ for spec in _MODULE_SPECS],
    )
    def test_review_notes_appended_to_primary_input(
        self, module_cls, output_fields, forward_kwargs, primary_kwarg
    ) -> None:
        """forward() with non-empty review_notes_text includes notes in the primary input."""
        module = _make_module_with_mock(module_cls, output_fields)
        review_notes = "[User Review Notes from Claims Drafting]: revise claim 3"

        module.forward(
            **forward_kwargs,
            personality_mode="critical",
            review_notes_text=review_notes,
        )

        primary_input = _get_primary_input(module.predict, primary_kwarg)
        assert review_notes in primary_input, (
            f"{module_cls.__name__}: review notes should be present in primary input"
        )


# ---------------------------------------------------------------------------
# Test: forward() with review_notes_text=None leaves input unchanged
# ---------------------------------------------------------------------------


class TestReviewNotesNoneUnchanged:
    """Test that review_notes_text=None does not modify the input."""

    @pytest.mark.parametrize(
        "module_cls,output_fields,forward_kwargs,primary_kwarg",
        _MODULE_SPECS,
        ids=[spec[0].__name__ for spec in _MODULE_SPECS],
    )
    def test_none_review_notes_leaves_input_unchanged(
        self, module_cls, output_fields, forward_kwargs, primary_kwarg
    ) -> None:
        """forward() with review_notes_text=None produces same input as without it."""
        # Call without review_notes_text
        module_without = _make_module_with_mock(module_cls, output_fields)
        module_without.forward(**forward_kwargs, personality_mode="critical")
        input_without = _get_primary_input(module_without.predict, primary_kwarg)

        # Call with review_notes_text=None
        module_with_none = _make_module_with_mock(module_cls, output_fields)
        module_with_none.forward(
            **forward_kwargs,
            personality_mode="critical",
            review_notes_text=None,
        )
        input_with_none = _get_primary_input(module_with_none.predict, primary_kwarg)

        assert input_without == input_with_none, (
            f"{module_cls.__name__}: None review_notes_text should not change input"
        )


# ---------------------------------------------------------------------------
# Test: forward() with review_notes_text="" leaves input unchanged
# ---------------------------------------------------------------------------


class TestReviewNotesEmptyStringUnchanged:
    """Test that review_notes_text="" does not modify the input."""

    @pytest.mark.parametrize(
        "module_cls,output_fields,forward_kwargs,primary_kwarg",
        _MODULE_SPECS,
        ids=[spec[0].__name__ for spec in _MODULE_SPECS],
    )
    def test_empty_review_notes_leaves_input_unchanged(
        self, module_cls, output_fields, forward_kwargs, primary_kwarg
    ) -> None:
        """forward() with review_notes_text="" produces same input as without it."""
        # Call without review_notes_text
        module_without = _make_module_with_mock(module_cls, output_fields)
        module_without.forward(**forward_kwargs, personality_mode="critical")
        input_without = _get_primary_input(module_without.predict, primary_kwarg)

        # Call with review_notes_text=""
        module_with_empty = _make_module_with_mock(module_cls, output_fields)
        module_with_empty.forward(
            **forward_kwargs,
            personality_mode="critical",
            review_notes_text="",
        )
        input_with_empty = _get_primary_input(module_with_empty.predict, primary_kwarg)

        assert input_without == input_with_empty, (
            f"{module_cls.__name__}: empty review_notes_text should not change input"
        )


# ---------------------------------------------------------------------------
# Test: ordering — personality prefix appears before review notes
# ---------------------------------------------------------------------------


class TestReviewNotesOrdering:
    """Test that personality prefix appears before review notes in the input."""

    @pytest.mark.parametrize(
        "module_cls,output_fields,forward_kwargs,primary_kwarg",
        _MODULE_SPECS,
        ids=[spec[0].__name__ for spec in _MODULE_SPECS],
    )
    def test_prefix_before_review_notes(
        self, module_cls, output_fields, forward_kwargs, primary_kwarg
    ) -> None:
        """Personality prefix appears before review notes in the primary input."""
        module = _make_module_with_mock(module_cls, output_fields)
        review_notes = "[User Review Notes from Test Step]: some feedback"
        prefix = generate_personality_prefix(PersonalityMode.CRITICAL)

        module.forward(
            **forward_kwargs,
            personality_mode="critical",
            review_notes_text=review_notes,
        )

        primary_input = _get_primary_input(module.predict, primary_kwarg)
        prefix_pos = primary_input.find(prefix)
        notes_pos = primary_input.find(review_notes)

        assert prefix_pos >= 0, "Personality prefix should be in the input"
        assert notes_pos >= 0, "Review notes should be in the input"
        assert prefix_pos < notes_pos, (
            f"{module_cls.__name__}: personality prefix (pos {prefix_pos}) "
            f"should appear before review notes (pos {notes_pos})"
        )

    @pytest.mark.parametrize(
        "module_cls,output_fields,forward_kwargs,primary_kwarg",
        _MODULE_SPECS,
        ids=[spec[0].__name__ for spec in _MODULE_SPECS],
    )
    def test_original_input_between_prefix_and_notes(
        self, module_cls, output_fields, forward_kwargs, primary_kwarg
    ) -> None:
        """Original input text appears between personality prefix and review notes."""
        module = _make_module_with_mock(module_cls, output_fields)
        review_notes = "[User Review Notes from Test Step]: some feedback"
        original_value = forward_kwargs[primary_kwarg]

        module.forward(
            **forward_kwargs,
            personality_mode="critical",
            review_notes_text=review_notes,
        )

        primary_input = _get_primary_input(module.predict, primary_kwarg)
        original_pos = primary_input.find(original_value)
        notes_pos = primary_input.find(review_notes)

        assert original_pos >= 0, "Original input should be in the primary input"
        assert notes_pos >= 0, "Review notes should be in the primary input"
        assert original_pos < notes_pos, (
            f"{module_cls.__name__}: original input (pos {original_pos}) "
            f"should appear before review notes (pos {notes_pos})"
        )
