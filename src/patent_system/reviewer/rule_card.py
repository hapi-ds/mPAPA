"""RuleCard Pydantic v2 model — schema for patent office examination rule cards.

A Rule Card is a modular JSON file extracted from official patent office
examination guidelines. The LLM loads it into its system prompt to review
patent applications against specific legal frameworks.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class SourceDocument(BaseModel):
    """Reference to the official source document from which rules were extracted."""

    title: str
    edition: str
    part: str | None = None
    chapter: str | None = None
    url: str | None = None


class RuleExamples(BaseModel):
    """Compliant and non-compliant examples illustrating a rule."""

    compliant: str
    non_compliant: str


class Rule(BaseModel):
    """A single examination rule extracted from an official guideline."""

    rule_id: str
    title: str
    requirement: str
    severity: Literal["mandatory", "recommended", "optional"]
    exceptions: list[str] = Field(default_factory=list)
    examples: RuleExamples | None = None
    reference: str


class RuleCard(BaseModel):
    """A pre-compiled rule card representing a subset of patent office examination guidelines.

    Rule cards are loaded by the ReviewEngine and injected into the LLM system
    prompt so that patent reviews are grounded in specific, citable legal rules.
    """

    card_id: str
    version: str
    jurisdiction: str
    office: str
    source_document: SourceDocument
    task: str
    task_label: str
    language: str = "en"
    translated_from: str | None = None
    last_updated: date
    tags: list[str] = Field(default_factory=list)
    rules: list[Rule] = Field(default_factory=list)
    review_checklist: list[str] = Field(default_factory=list)
    system_prompt_injection: str

    @classmethod
    def from_json_file(cls, path: Path) -> RuleCard:
        """Load and validate a RuleCard from a JSON file on disk.

        Args:
            path: Path to the .json rule card file.

        Returns:
            A validated RuleCard instance.

        Raises:
            FileNotFoundError: If the path does not exist.
            pydantic.ValidationError: If the JSON does not conform to the schema.
            json.JSONDecodeError: If the file is not valid JSON.
        """
        content = path.read_text(encoding="utf-8")
        data = json.loads(content)
        return cls.model_validate(data)

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the RuleCard to a JSON string.

        Args:
            indent: Number of spaces for indentation (default 2).

        Returns:
            JSON string representation of the rule card.
        """
        return self.model_dump_json(indent=indent)
