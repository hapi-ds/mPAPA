"""DSPy Signature definitions for patent analysis agent tasks."""

import dspy


class InventionInterviewQuestion(dspy.Signature):
    """Generate the next interview question based on conversation history."""

    conversation_history: str = dspy.InputField()
    invention_context: str = dspy.InputField()
    next_question: str = dspy.OutputField()


class StructureDisclosure(dspy.Signature):
    """Extract structured invention disclosure from interview transcript."""

    transcript: str = dspy.InputField()
    disclosure_json: str = dspy.OutputField()


class DraftClaims(dspy.Signature):
    """Draft patent claims in German/European legal format."""

    invention_disclosure: str = dspy.InputField()
    novelty_analysis: str = dspy.InputField()
    claims_text: str = dspy.OutputField()


class ReviewConsistency(dspy.Signature):
    """Review claims against description for consistency issues."""

    claims: str = dspy.InputField()
    description: str = dspy.InputField()
    feedback: str = dspy.OutputField()
    approved: bool = dspy.OutputField()


class DraftDescription(dspy.Signature):
    """Generate full patent specification from approved claims and prior art."""

    claims: str = dspy.InputField()
    prior_art_summary: str = dspy.InputField()
    invention_disclosure: str = dspy.InputField()
    description_text: str = dspy.OutputField()
