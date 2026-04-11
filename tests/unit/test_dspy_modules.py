"""Unit tests for DSPy module wrappers."""

from unittest.mock import MagicMock, patch

import dspy
import pytest

from patent_system.config import AppSettings
from patent_system.dspy_modules.modules import (
    DraftClaimsModule,
    DraftDescriptionModule,
    InterviewQuestionModule,
    ReviewConsistencyModule,
    StructureDisclosureModule,
    configure_dspy,
)


class TestConfigureDspy:
    """Tests for the configure_dspy function."""

    def test_returns_lm_instance(self) -> None:
        settings = AppSettings()
        lm = configure_dspy(settings)
        assert isinstance(lm, dspy.LM)

    def test_uses_settings_base_url(self) -> None:
        settings = AppSettings(lm_studio_base_url="http://custom:9999/v1")
        lm = configure_dspy(settings)
        assert lm.kwargs["api_base"] == "http://custom:9999/v1"

    def test_uses_settings_api_key(self) -> None:
        settings = AppSettings(lm_studio_api_key="test-key")
        lm = configure_dspy(settings)
        assert lm.kwargs["api_key"] == "test-key"


class TestModuleInstantiation:
    """Tests that all module wrappers instantiate correctly."""

    def test_interview_question_module_is_dspy_module(self) -> None:
        module = InterviewQuestionModule()
        assert isinstance(module, dspy.Module)

    def test_structure_disclosure_module_is_dspy_module(self) -> None:
        module = StructureDisclosureModule()
        assert isinstance(module, dspy.Module)

    def test_draft_claims_module_is_dspy_module(self) -> None:
        module = DraftClaimsModule()
        assert isinstance(module, dspy.Module)

    def test_review_consistency_module_is_dspy_module(self) -> None:
        module = ReviewConsistencyModule()
        assert isinstance(module, dspy.Module)

    def test_draft_description_module_is_dspy_module(self) -> None:
        module = DraftDescriptionModule()
        assert isinstance(module, dspy.Module)


class TestModulePredicateTypes:
    """Tests that modules use the correct DSPy predictor types."""

    def test_interview_uses_chain_of_thought(self) -> None:
        module = InterviewQuestionModule()
        assert isinstance(module.predict, dspy.ChainOfThought)

    def test_structure_disclosure_uses_chain_of_thought(self) -> None:
        module = StructureDisclosureModule()
        assert isinstance(module.predict, dspy.ChainOfThought)

    def test_draft_claims_uses_chain_of_thought(self) -> None:
        module = DraftClaimsModule()
        assert isinstance(module.predict, dspy.ChainOfThought)

    def test_review_consistency_uses_predict(self) -> None:
        module = ReviewConsistencyModule()
        assert isinstance(module.predict, dspy.Predict)
        # ChainOfThought is a subclass of Predict, so check it's NOT CoT
        assert not isinstance(module.predict, dspy.ChainOfThought)

    def test_draft_description_uses_chain_of_thought(self) -> None:
        module = DraftDescriptionModule()
        assert isinstance(module.predict, dspy.ChainOfThought)


class TestModuleModelName:
    """Tests that modules accept an optional model_name parameter."""

    def test_interview_stores_model_name(self) -> None:
        module = InterviewQuestionModule(model_name="test-model")
        assert module.model_name == "test-model"

    def test_default_model_name_is_none(self) -> None:
        module = InterviewQuestionModule()
        assert module.model_name is None
