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
    """Draft patent claims in European patent format. Write in English."""

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


class RefineClaims(dspy.Signature):
    """Refine patent claims based on feedback from analysis steps.

    Incorporate insights from novelty analysis, consistency review,
    market potential, and legal assessment to improve the claims.
    Preserve the European patent format (preamble, characterizing
    portion, hierarchical dependents). Only make changes that are
    justified by the feedback — do not rewrite claims that need no
    improvement.
    """

    original_claims: str = dspy.InputField(desc="Current patent claims text")
    invention_disclosure: str = dspy.InputField(desc="Structured invention disclosure text")
    novelty_analysis: str = dspy.InputField(desc="Novelty analysis findings and suggested claim scope")
    consistency_review: str = dspy.InputField(desc="Consistency review feedback on the claims")
    market_assessment: str = dspy.InputField(desc="Market potential assessment")
    legal_assessment: str = dspy.InputField(desc="Legal and IP assessment with freedom-to-operate findings")
    refined_claims: str = dspy.OutputField(
        desc="Improved patent claims incorporating feedback. European patent format."
    )


class AnalyzeMarketPotential(dspy.Signature):
    """Assess economic viability and market potential of an invention."""

    invention_disclosure: str = dspy.InputField(desc="Structured invention disclosure text")
    claims_text: str = dspy.InputField(desc="Drafted patent claims")
    novelty_analysis: str = dspy.InputField(desc="Novelty analysis against prior art")
    market_assessment: str = dspy.OutputField(desc="Market potential and economic viability assessment")


class AnalyzeLegalClarification(dspy.Signature):
    """Assess IP ownership, employment agreement implications, and prior art conflicts.

    Analyze both patent references and scientific paper references to identify
    potential IP conflicts, freedom-to-operate issues, and licensing considerations.
    Consider patent claims overlap, publication dates, and jurisdictional aspects.
    """

    invention_disclosure: str = dspy.InputField(desc="Structured invention disclosure text")
    claims_text: str = dspy.InputField(desc="Drafted patent claims")
    prior_art_summary: str = dspy.InputField(
        desc="Comprehensive summary of all prior art references including patents and scientific papers"
    )
    novelty_analysis: str = dspy.InputField(desc="Novelty analysis against prior art")
    legal_assessment: str = dspy.OutputField(
        desc="Legal and IP assessment covering: patent landscape conflicts, freedom-to-operate analysis, "
        "employment/assignment considerations, licensing risks, and recommended actions"
    )


class SummarizeDisclosure(dspy.Signature):
    """Generate a comprehensive summary of all preceding workflow steps.

    Synthesize the invention idea, drafted claims, prior art analysis
    (including all patent and scientific paper references), novelty findings,
    consistency review, market potential, and legal assessment into a
    cohesive disclosure summary suitable for patent filing preparation.
    """

    initial_idea: str = dspy.InputField(desc="Initial invention idea text")
    claims_text: str = dspy.InputField(desc="Drafted patent claims")
    prior_art_summary: str = dspy.InputField(
        desc="Comprehensive summary of all prior art including patents and scientific papers"
    )
    novelty_analysis: str = dspy.InputField(desc="Novelty analysis against prior art")
    consistency_review: str = dspy.InputField(desc="Claims consistency review feedback")
    market_assessment: str = dspy.InputField(desc="Market potential assessment")
    legal_assessment: str = dspy.InputField(desc="Legal and IP ownership assessment")
    disclosure_summary: str = dspy.OutputField(
        desc="Comprehensive disclosure summary covering: invention overview, key claims, "
        "prior art landscape (all patents and papers), novelty position, consistency status, "
        "market viability, legal considerations, and recommended next steps"
    )


class AnalyzeNovelty(dspy.Signature):
    """Analyze the novelty of an invention against prior art references.

    Identify which aspects of the invention are novel, which overlap with
    existing prior art, and suggest how to scope the patent claims.
    """

    invention_disclosure: str = dspy.InputField(desc="Structured invention disclosure text")
    claims_text: str = dspy.InputField(desc="Drafted patent claims")
    prior_art_summary: str = dspy.InputField(desc="Summary of prior art references found")
    novelty_assessment: str = dspy.OutputField(
        desc="Detailed novelty analysis: novel aspects, potential conflicts with prior art, and suggested claim scope"
    )


class SummarizePriorArt(dspy.Signature):
    """Produce a comprehensive analytical summary of prior art references.

    Analyze ALL provided patent and scientific paper references to identify
    key themes, technological trends, gaps in the prior art, and how they
    relate to the invention. Do not simply list references — synthesize
    findings into a coherent narrative covering the state of the art.
    """

    invention_disclosure: str = dspy.InputField(desc="The invention disclosure text")
    claims_text: str = dspy.InputField(desc="Drafted patent claims for context")
    prior_art_references: str = dspy.InputField(
        desc="All prior art references (patents and scientific papers) with titles and abstracts"
    )
    prior_art_summary: str = dspy.OutputField(
        desc="Comprehensive analytical summary of the prior art landscape: key themes, "
        "technological trends, closest prior art to the invention, identified gaps, "
        "and how the references relate to the claimed invention. Cover ALL references."
    )
