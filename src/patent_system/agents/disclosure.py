"""Invention Disclosure Agent for the patent drafting pipeline.

Conducts an interactive interview using DSPy modules to extract
structured invention details, then produces a JSON InventionDisclosure.

Requirements: 2.1, 2.2, 2.3, 2.4
"""

import json
import logging
import time
from typing import Any

from patent_system.agents.state import PatentWorkflowState
from patent_system.dspy_modules.modules import (
    InterviewQuestionModule,
    StructureDisclosureModule,
)
from patent_system.logging_config import log_agent_invocation

logger = logging.getLogger(__name__)

# Interview topic areas matching Requirement 2.1
_INTERVIEW_TOPICS: list[str] = [
    "technical_problem",
    "novel_features",
    "implementation_details",
    "potential_variations",
]


def disclosure_node(state: PatentWorkflowState) -> dict[str, Any]:
    """Run the Invention Disclosure Agent.

    1. Uses ``InterviewQuestionModule`` to generate interview questions
       about each topic area (technical problem, novel features,
       implementation details, potential variations).
    2. Uses ``StructureDisclosureModule`` to produce a structured JSON
       ``InventionDisclosure`` from the accumulated transcript.
    3. Logs the agent invocation with name, input summary, output
       summary, and duration.
    4. Returns a dict with updated ``invention_disclosure`` and
       ``current_step`` fields.

    Args:
        state: The current workflow state.

    Returns:
        Dict with ``invention_disclosure`` (parsed JSON dict) and
        ``current_step`` set to ``"disclosure"``.
    """
    start = time.monotonic()

    interview_module = InterviewQuestionModule()
    structure_module = StructureDisclosureModule()

    # Build conversation context from existing interview messages
    messages = state.get("interview_messages", [])
    conversation_history = "\n".join(
        str(m) for m in messages
    ) if messages else ""

    invention_context = f"Topic ID: {state.get('topic_id', 'unknown')}"

    # Generate interview questions for each topic area
    questions: list[str] = []
    for topic in _INTERVIEW_TOPICS:
        context_with_topic = f"{invention_context}\nCurrent topic: {topic}"
        result = interview_module(
            conversation_history=conversation_history,
            invention_context=context_with_topic,
        )
        questions.append(result.next_question)
        # Accumulate into conversation history for subsequent questions
        conversation_history += f"\nQ ({topic}): {result.next_question}"

    # Structure the disclosure from the full transcript
    transcript = conversation_history
    structure_result = structure_module(transcript=transcript)

    # Parse the structured disclosure JSON
    disclosure_json_str = structure_result.disclosure_json
    try:
        disclosure = json.loads(disclosure_json_str)
    except (json.JSONDecodeError, TypeError):
        # If the LLM output isn't valid JSON, wrap it in a basic structure
        disclosure = {
            "technical_problem": disclosure_json_str,
            "novel_features": [],
            "implementation_details": "",
            "potential_variations": [],
        }

    duration_ms = (time.monotonic() - start) * 1000

    input_summary = (
        f"topic_id={state.get('topic_id', 'unknown')}, "
        f"messages={len(messages)}"
    )
    output_summary = (
        f"questions={len(questions)}, "
        f"disclosure_keys={list(disclosure.keys())}"
    )

    log_agent_invocation(
        logger=logger,
        name="InventionDisclosureAgent",
        input_summary=input_summary,
        output_summary=output_summary,
        duration_ms=duration_ms,
    )

    return {
        "invention_disclosure": disclosure,
        "current_step": "disclosure",
    }
