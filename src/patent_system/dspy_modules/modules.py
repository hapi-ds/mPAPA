"""DSPy Module wrappers for patent analysis agent tasks.

Each module wraps a DSPy Signature with a forward method that calls
dspy.Predict or dspy.ChainOfThought. The configure_dspy function sets up
DSPy to use LM Studio via the OpenAI-compatible API endpoint.
"""

import dspy

from patent_system.agents.personality import generate_personality_prefix
from patent_system.config import AppSettings
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
        self,
        conversation_history: str,
        invention_context: str,
        personality_mode: str | None = None,
    ) -> dspy.Prediction:
        """Generate the next interview question.

        Args:
            conversation_history: The conversation so far.
            invention_context: Context about the invention being discussed.
            personality_mode: Optional personality mode string. Defaults to
                ``"critical"`` when *None* or invalid.

        Returns:
            A DSPy Prediction with a next_question field.
        """
        prefix = generate_personality_prefix(personality_mode or "critical")
        prefixed_conversation_history = f"{prefix}\n\n{conversation_history}"
        return self.predict(
            conversation_history=prefixed_conversation_history,
            invention_context=invention_context,
        )


class StructureDisclosureModule(dspy.Module):
    """Extract structured invention disclosure from interview transcript."""

    def __init__(self, model_name: str | None = None) -> None:
        super().__init__()
        self.predict = dspy.ChainOfThought(StructureDisclosure)
        self.model_name = model_name

    def forward(
        self,
        transcript: str,
        personality_mode: str | None = None,
    ) -> dspy.Prediction:
        """Extract structured disclosure from a transcript.

        Args:
            transcript: The full interview transcript.
            personality_mode: Optional personality mode string. Defaults to
                ``"critical"`` when *None* or invalid.

        Returns:
            A DSPy Prediction with a disclosure_json field.
        """
        prefix = generate_personality_prefix(personality_mode or "critical")
        prefixed_transcript = f"{prefix}\n\n{transcript}"
        return self.predict(transcript=prefixed_transcript)


class SuggestSearchTermsModule(dspy.Module):
    """Suggest prior art search terms from an invention description."""

    def __init__(self, model_name: str | None = None) -> None:
        super().__init__()
        self.predict = dspy.ChainOfThought(SuggestSearchTerms)
        self.model_name = model_name

    def forward(
        self,
        invention_description: str,
        personality_mode: str | None = None,
    ) -> dspy.Prediction:
        """Generate search term suggestions.

        Args:
            invention_description: The primary invention description text.
            personality_mode: Optional personality mode string. Defaults to
                ``"critical"`` when *None* or invalid.

        Returns:
            A DSPy Prediction with a search_terms field (one term per line).
        """
        prefix = generate_personality_prefix(personality_mode or "critical")
        prefixed_invention_description = f"{prefix}\n\n{invention_description}"
        return self.predict(invention_description=prefixed_invention_description)


class DraftClaimsModule(dspy.Module):
    """Draft patent claims in European patent format (English)."""

    def __init__(self, model_name: str | None = None) -> None:
        super().__init__()
        self.predict = dspy.ChainOfThought(DraftClaims)
        self.model_name = model_name

    def forward(
        self,
        invention_disclosure: str,
        novelty_analysis: str,
        personality_mode: str | None = None,
        review_notes_text: str | None = None,
    ) -> dspy.Prediction:
        """Draft patent claims.

        Args:
            invention_disclosure: Structured invention disclosure text.
            novelty_analysis: Novelty analysis results.
            personality_mode: Optional personality mode string. Defaults to
                ``"critical"`` when *None* or invalid.
            review_notes_text: Optional formatted review notes to append
                to the primary input after the personality prefix.

        Returns:
            A DSPy Prediction with a claims_text field.
        """
        prefix = generate_personality_prefix(personality_mode or "critical")
        prefixed_invention_disclosure = f"{prefix}\n\n{invention_disclosure}"
        if review_notes_text:
            prefixed_invention_disclosure = f"{prefixed_invention_disclosure}\n\n{review_notes_text}"
        return self.predict(
            invention_disclosure=prefixed_invention_disclosure,
            novelty_analysis=novelty_analysis,
        )


class ReviewConsistencyModule(dspy.Module):
    """Review claims against description for consistency issues."""

    def __init__(self, model_name: str | None = None) -> None:
        super().__init__()
        self.predict = dspy.Predict(ReviewConsistency)
        self.model_name = model_name

    def forward(
        self,
        claims: str,
        description: str,
        personality_mode: str | None = None,
        review_notes_text: str | None = None,
    ) -> dspy.Prediction:
        """Review claims for consistency with the description.

        Args:
            claims: The drafted patent claims.
            description: The patent description text.
            personality_mode: Optional personality mode string. Defaults to
                ``"critical"`` when *None* or invalid.
            review_notes_text: Optional formatted review notes to append
                to the primary input after the personality prefix.

        Returns:
            A DSPy Prediction with feedback and approved fields.
        """
        prefix = generate_personality_prefix(personality_mode or "critical")
        prefixed_claims = f"{prefix}\n\n{claims}"
        if review_notes_text:
            prefixed_claims = f"{prefixed_claims}\n\n{review_notes_text}"
        return self.predict(claims=prefixed_claims, description=description)


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
        personality_mode: str | None = None,
        review_notes_text: str | None = None,
    ) -> dspy.Prediction:
        """Generate the full patent description.

        Args:
            claims: Approved patent claims.
            prior_art_summary: Summary of relevant prior art.
            invention_disclosure: Structured invention disclosure.
            personality_mode: Optional personality mode string. Defaults to
                ``"critical"`` when *None* or invalid.
            review_notes_text: Optional formatted review notes to append
                to the primary input after the personality prefix.

        Returns:
            A DSPy Prediction with a description_text field.
        """
        prefix = generate_personality_prefix(personality_mode or "critical")
        prefixed_claims = f"{prefix}\n\n{claims}"
        if review_notes_text:
            prefixed_claims = f"{prefixed_claims}\n\n{review_notes_text}"
        return self.predict(
            claims=prefixed_claims,
            prior_art_summary=prior_art_summary,
            invention_disclosure=invention_disclosure,
        )


class RefineClaimsModule(dspy.Module):
    """Refine patent claims based on accumulated analysis feedback."""

    def __init__(self, model_name: str | None = None) -> None:
        super().__init__()
        self.predict = dspy.ChainOfThought(RefineClaims)
        self.model_name = model_name

    def forward(
        self,
        original_claims: str,
        invention_disclosure: str,
        novelty_analysis: str,
        consistency_review: str,
        market_assessment: str,
        legal_assessment: str,
        personality_mode: str | None = None,
        review_notes_text: str | None = None,
    ) -> dspy.Prediction:
        """Refine claims using feedback from analysis steps.

        Args:
            original_claims: The current patent claims text.
            invention_disclosure: Structured invention disclosure.
            novelty_analysis: Novelty analysis findings.
            consistency_review: Consistency review feedback.
            market_assessment: Market potential assessment.
            legal_assessment: Legal and IP assessment.
            personality_mode: Optional personality mode string. Defaults to
                ``"critical"`` when *None* or invalid.
            review_notes_text: Optional formatted review notes to append
                to the primary input after the personality prefix.

        Returns:
            A DSPy Prediction with a refined_claims field.
        """
        prefix = generate_personality_prefix(personality_mode or "critical")
        prefixed_original_claims = f"{prefix}\n\n{original_claims}"
        if review_notes_text:
            prefixed_original_claims = f"{prefixed_original_claims}\n\n{review_notes_text}"
        return self.predict(
            original_claims=prefixed_original_claims,
            invention_disclosure=invention_disclosure,
            novelty_analysis=novelty_analysis,
            consistency_review=consistency_review,
            market_assessment=market_assessment,
            legal_assessment=legal_assessment,
        )


class MarketPotentialModule(dspy.Module):
    """Assess economic viability and market potential of an invention."""

    def __init__(self, model_name: str | None = None) -> None:
        super().__init__()
        self.predict = dspy.ChainOfThought(AnalyzeMarketPotential)
        self.model_name = model_name

    def forward(
        self,
        invention_disclosure: str,
        claims_text: str,
        novelty_analysis: str,
        personality_mode: str | None = None,
        review_notes_text: str | None = None,
    ) -> dspy.Prediction:
        """Assess market potential of the invention.

        Args:
            invention_disclosure: Structured invention disclosure text.
            claims_text: Drafted patent claims.
            novelty_analysis: Novelty analysis against prior art.
            personality_mode: Optional personality mode string. Defaults to
                ``"critical"`` when *None* or invalid.
            review_notes_text: Optional formatted review notes to append
                to the primary input after the personality prefix.

        Returns:
            A DSPy Prediction with a market_assessment field.
        """
        prefix = generate_personality_prefix(personality_mode or "critical")
        prefixed_invention_disclosure = f"{prefix}\n\n{invention_disclosure}"
        if review_notes_text:
            prefixed_invention_disclosure = f"{prefixed_invention_disclosure}\n\n{review_notes_text}"
        return self.predict(
            invention_disclosure=prefixed_invention_disclosure,
            claims_text=claims_text,
            novelty_analysis=novelty_analysis,
        )


class LegalClarificationModule(dspy.Module):
    """Assess IP ownership, employment agreements, and prior art conflicts."""

    def __init__(self, model_name: str | None = None) -> None:
        super().__init__()
        self.predict = dspy.ChainOfThought(AnalyzeLegalClarification)
        self.model_name = model_name

    def forward(
        self,
        invention_disclosure: str,
        claims_text: str,
        prior_art_summary: str,
        novelty_analysis: str,
        personality_mode: str | None = None,
        review_notes_text: str | None = None,
    ) -> dspy.Prediction:
        """Assess legal and IP ownership aspects.

        Args:
            invention_disclosure: Structured invention disclosure text.
            claims_text: Drafted patent claims.
            prior_art_summary: Summary of prior art search results.
            novelty_analysis: Novelty analysis against prior art.
            personality_mode: Optional personality mode string. Defaults to
                ``"critical"`` when *None* or invalid.
            review_notes_text: Optional formatted review notes to append
                to the primary input after the personality prefix.

        Returns:
            A DSPy Prediction with a legal_assessment field.
        """
        prefix = generate_personality_prefix(personality_mode or "critical")
        prefixed_invention_disclosure = f"{prefix}\n\n{invention_disclosure}"
        if review_notes_text:
            prefixed_invention_disclosure = f"{prefixed_invention_disclosure}\n\n{review_notes_text}"
        return self.predict(
            invention_disclosure=prefixed_invention_disclosure,
            claims_text=claims_text,
            prior_art_summary=prior_art_summary,
            novelty_analysis=novelty_analysis,
        )


class DisclosureSummaryModule(dspy.Module):
    """Generate a concise summary of all preceding workflow steps."""

    def __init__(self, model_name: str | None = None) -> None:
        super().__init__()
        self.predict = dspy.ChainOfThought(SummarizeDisclosure)
        self.model_name = model_name

    def forward(
        self,
        initial_idea: str,
        claims_text: str,
        prior_art_summary: str,
        novelty_analysis: str,
        consistency_review: str,
        market_assessment: str,
        legal_assessment: str,
        personality_mode: str | None = None,
        review_notes_text: str | None = None,
    ) -> dspy.Prediction:
        """Generate a summary of all preceding workflow steps.

        Args:
            initial_idea: Initial invention idea text.
            claims_text: Drafted patent claims.
            prior_art_summary: Summary of prior art search results.
            novelty_analysis: Novelty analysis against prior art.
            consistency_review: Claims consistency review feedback.
            market_assessment: Market potential assessment.
            legal_assessment: Legal and IP ownership assessment.
            personality_mode: Optional personality mode string. Defaults to
                ``"critical"`` when *None* or invalid.
            review_notes_text: Optional formatted review notes to append
                to the primary input after the personality prefix.

        Returns:
            A DSPy Prediction with a disclosure_summary field.
        """
        prefix = generate_personality_prefix(personality_mode or "critical")
        prefixed_initial_idea = f"{prefix}\n\n{initial_idea}"
        if review_notes_text:
            prefixed_initial_idea = f"{prefixed_initial_idea}\n\n{review_notes_text}"
        return self.predict(
            initial_idea=prefixed_initial_idea,
            claims_text=claims_text,
            prior_art_summary=prior_art_summary,
            novelty_analysis=novelty_analysis,
            consistency_review=consistency_review,
            market_assessment=market_assessment,
            legal_assessment=legal_assessment,
        )


class NoveltyAnalysisModule(dspy.Module):
    """Analyze invention novelty against prior art using LLM."""

    def __init__(self, model_name: str | None = None) -> None:
        super().__init__()
        self.predict = dspy.ChainOfThought(AnalyzeNovelty)
        self.model_name = model_name

    def forward(
        self,
        invention_disclosure: str,
        claims_text: str,
        prior_art_summary: str,
        personality_mode: str | None = None,
        review_notes_text: str | None = None,
    ) -> dspy.Prediction:
        """Analyze novelty of the invention.

        Args:
            invention_disclosure: Structured invention disclosure text.
            claims_text: Drafted patent claims.
            prior_art_summary: Summary of prior art references.
            personality_mode: Optional personality mode string. Defaults to
                ``"critical"`` when *None* or invalid.
            review_notes_text: Optional formatted review notes to append
                to the primary input after the personality prefix.

        Returns:
            A DSPy Prediction with a novelty_assessment field.
        """
        prefix = generate_personality_prefix(personality_mode or "critical")
        prefixed_invention_disclosure = f"{prefix}\n\n{invention_disclosure}"
        if review_notes_text:
            prefixed_invention_disclosure = f"{prefixed_invention_disclosure}\n\n{review_notes_text}"
        return self.predict(
            invention_disclosure=prefixed_invention_disclosure,
            claims_text=claims_text,
            prior_art_summary=prior_art_summary,
        )


class PriorArtSummaryModule(dspy.Module):
    """Produce an analytical summary of prior art references using LLM."""

    def __init__(self, model_name: str | None = None) -> None:
        super().__init__()
        self.predict = dspy.ChainOfThought(SummarizePriorArt)
        self.model_name = model_name

    def forward(
        self,
        invention_disclosure: str,
        claims_text: str,
        prior_art_references: str,
        personality_mode: str | None = None,
        review_notes_text: str | None = None,
    ) -> dspy.Prediction:
        """Summarize prior art references.

        Args:
            invention_disclosure: The invention disclosure text.
            claims_text: Drafted patent claims for context.
            prior_art_references: All references with titles and abstracts.
            personality_mode: Optional personality mode string. Defaults to
                ``"critical"`` when *None* or invalid.
            review_notes_text: Optional formatted review notes to append
                to the primary input after the personality prefix.

        Returns:
            A DSPy Prediction with a prior_art_summary field.
        """
        prefix = generate_personality_prefix(personality_mode or "critical")
        prefixed_invention_disclosure = f"{prefix}\n\n{invention_disclosure}"
        if review_notes_text:
            prefixed_invention_disclosure = f"{prefixed_invention_disclosure}\n\n{review_notes_text}"
        return self.predict(
            invention_disclosure=prefixed_invention_disclosure,
            claims_text=claims_text,
            prior_art_references=prior_art_references,
        )
