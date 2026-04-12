"""DSPy Module wrappers for patent analysis agent tasks.

Each module wraps a DSPy Signature with a forward method that calls
dspy.Predict or dspy.ChainOfThought. The configure_dspy function sets up
DSPy to use LM Studio via the OpenAI-compatible API endpoint.
"""

import dspy

from patent_system.config import AppSettings
from patent_system.dspy_modules.signatures import (
    DraftClaims,
    DraftDescription,
    InventionInterviewQuestion,
    ReviewConsistency,
    StructureDisclosure,
)


def configure_dspy(settings: AppSettings) -> dspy.LM:
    """Configure DSPy to use LM Studio via the OpenAI-compatible API.

    Args:
        settings: Application settings containing LM Studio connection details.

    Returns:
        The configured dspy.LM instance.
    """
    lm = dspy.LM(
        model=f"openai/{settings.model_disclosure}",
        api_base=settings.lm_studio_base_url,
        api_key=settings.lm_studio_api_key,
    )
    dspy.configure(lm=lm)
    return lm


class InterviewQuestionModule(dspy.Module):
    """Generate the next interview question based on conversation history."""

    def __init__(self, model_name: str | None = None) -> None:
        super().__init__()
        self.predict = dspy.ChainOfThought(InventionInterviewQuestion)
        self.model_name = model_name

    def forward(
        self, conversation_history: str, invention_context: str
    ) -> dspy.Prediction:
        """Generate the next interview question.

        Args:
            conversation_history: The conversation so far.
            invention_context: Context about the invention being discussed.

        Returns:
            A DSPy Prediction with a next_question field.
        """
        return self.predict(
            conversation_history=conversation_history,
            invention_context=invention_context,
        )


class StructureDisclosureModule(dspy.Module):
    """Extract structured invention disclosure from interview transcript."""

    def __init__(self, model_name: str | None = None) -> None:
        super().__init__()
        self.predict = dspy.ChainOfThought(StructureDisclosure)
        self.model_name = model_name

    def forward(self, transcript: str) -> dspy.Prediction:
        """Extract structured disclosure from a transcript.

        Args:
            transcript: The full interview transcript.

        Returns:
            A DSPy Prediction with a disclosure_json field.
        """
        return self.predict(transcript=transcript)


class DraftClaimsModule(dspy.Module):
    """Draft patent claims in European patent format (English)."""

    def __init__(self, model_name: str | None = None) -> None:
        super().__init__()
        self.predict = dspy.ChainOfThought(DraftClaims)
        self.model_name = model_name

    def forward(
        self, invention_disclosure: str, novelty_analysis: str
    ) -> dspy.Prediction:
        """Draft patent claims.

        Args:
            invention_disclosure: Structured invention disclosure text.
            novelty_analysis: Novelty analysis results.

        Returns:
            A DSPy Prediction with a claims_text field.
        """
        return self.predict(
            invention_disclosure=invention_disclosure,
            novelty_analysis=novelty_analysis,
        )


class ReviewConsistencyModule(dspy.Module):
    """Review claims against description for consistency issues."""

    def __init__(self, model_name: str | None = None) -> None:
        super().__init__()
        self.predict = dspy.Predict(ReviewConsistency)
        self.model_name = model_name

    def forward(self, claims: str, description: str) -> dspy.Prediction:
        """Review claims for consistency with the description.

        Args:
            claims: The drafted patent claims.
            description: The patent description text.

        Returns:
            A DSPy Prediction with feedback and approved fields.
        """
        return self.predict(claims=claims, description=description)


class DraftDescriptionModule(dspy.Module):
    """Generate full patent specification from approved claims and prior art."""

    def __init__(self, model_name: str | None = None) -> None:
        super().__init__()
        self.predict = dspy.ChainOfThought(DraftDescription)
        self.model_name = model_name

    def forward(
        self,
        claims: str,
        prior_art_summary: str,
        invention_disclosure: str,
    ) -> dspy.Prediction:
        """Generate the full patent description.

        Args:
            claims: Approved patent claims.
            prior_art_summary: Summary of relevant prior art.
            invention_disclosure: Structured invention disclosure.

        Returns:
            A DSPy Prediction with a description_text field.
        """
        return self.predict(
            claims=claims,
            prior_art_summary=prior_art_summary,
            invention_disclosure=invention_disclosure,
        )
