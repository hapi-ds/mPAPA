"""Rule Card compiler — assembles extracted rules into validated RuleCard JSON."""

import json
import logging
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field

from .config import CompilerConfig

logger = logging.getLogger(__name__)


class RuleEntry(BaseModel):
    """A single normative rule within a RuleCard."""

    rule_id: str
    title: str
    requirement: str
    severity: str
    exceptions: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)
    reference: str = ""


class RuleCard(BaseModel):
    """Complete RuleCard schema for agent consumption."""

    card_id: str
    jurisdiction: str
    office: str
    source_document: str
    task: str
    task_label: str
    rules: list[RuleEntry]
    review_checklist: list[str]
    system_prompt_injection: str


CHECKLIST_PROMPT = """\
You are a patent examination assistant. Given the following extracted rules from a patent guideline, generate a concise review checklist that an agent can use to verify compliance.

Each checklist item should be a short, actionable verification statement (e.g., "Verify that independent claims contain a preamble and characterizing portion").

Jurisdiction: {jurisdiction}
Task: {task_label}
Source: {source_document}

Rules:
{rules_summary}

Output a JSON array of checklist strings (max 15 items). No markdown fencing, no explanation."""

SYSTEM_PROMPT_PROMPT = """\
You are a patent examination assistant. Given the following extracted rules from a patent guideline, generate a concise system prompt paragraph that can be injected into an agent's instructions to make it aware of these rules.

The paragraph should:
- Be 3-6 sentences
- Reference the jurisdiction and source
- Summarize the key obligations and prohibitions
- Use imperative mood

Jurisdiction: {jurisdiction}
Task: {task_label}
Source: {source_document}

Rules:
{rules_summary}

Output ONLY the system prompt paragraph as plain text. No JSON, no markdown fencing."""


class CardCompiler:
    """Compiles extracted rules into validated RuleCard JSON files.

    Args:
        config: Compiler configuration.
    """

    def __init__(self, config: CompilerConfig) -> None:
        self.config = config
        self.client = httpx.Client(timeout=120.0)

    def compile(
        self,
        rules: list[dict[str, Any]],
        card_id: str,
        jurisdiction: str,
        office: str,
        source_document: str,
        task: str,
        task_label: str,
    ) -> RuleCard:
        """Compile extracted rules into a full RuleCard.

        Args:
            rules: List of extracted rule dictionaries from RuleExtractor.
            card_id: Unique card identifier.
            jurisdiction: Patent jurisdiction (EP, US, PCT).
            office: Patent office name (EPO, USPTO, WIPO).
            source_document: Name/path of the source guideline.
            task: Task identifier (claim_structure, novelty, etc.).
            task_label: Human-readable task label.

        Returns:
            Validated RuleCard instance.
        """
        rules_summary = self._format_rules_summary(rules)

        logger.info("Generating review checklist for %s", card_id)
        checklist = self._generate_checklist(
            jurisdiction, task_label, source_document, rules_summary
        )

        logger.info("Generating system prompt injection for %s", card_id)
        system_prompt = self._generate_system_prompt(
            jurisdiction, task_label, source_document, rules_summary
        )

        # Validate rules through Pydantic
        validated_rules = [RuleEntry(**r) for r in rules]

        card = RuleCard(
            card_id=card_id,
            jurisdiction=jurisdiction,
            office=office,
            source_document=source_document,
            task=task,
            task_label=task_label,
            rules=validated_rules,
            review_checklist=checklist,
            system_prompt_injection=system_prompt,
        )

        logger.info("Compiled RuleCard %s with %d rules", card_id, len(rules))
        return card

    def write_card(self, card: RuleCard) -> Path:
        """Write a RuleCard to the output directory as JSON.

        Args:
            card: Validated RuleCard to write.

        Returns:
            Path to the written JSON file.
        """
        output_dir = self.config.output_dir / card.jurisdiction.lower()
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"{card.card_id}.json"
        output_path.write_text(
            card.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )

        logger.info("Wrote RuleCard to %s", output_path)
        return output_path

    def validate_card_file(self, path: Path) -> RuleCard:
        """Validate an existing RuleCard JSON file.

        Args:
            path: Path to the JSON file.

        Returns:
            Validated RuleCard instance.

        Raises:
            pydantic.ValidationError: If the JSON is invalid.
        """
        data = json.loads(path.read_text(encoding="utf-8"))
        return RuleCard(**data)

    def _format_rules_summary(self, rules: list[dict[str, Any]]) -> str:
        """Format rules into a concise summary for LLM prompts."""
        lines: list[str] = []
        for r in rules[:20]:  # Cap to avoid exceeding context
            severity = r.get("severity", "unknown")
            title = r.get("title", "untitled")
            requirement = r.get("requirement", "")[:200]
            lines.append(f"- [{severity}] {title}: {requirement}")
        return "\n".join(lines)

    def _generate_checklist(
        self,
        jurisdiction: str,
        task_label: str,
        source_document: str,
        rules_summary: str,
    ) -> list[str]:
        """Generate a review checklist via LLM.

        Args:
            jurisdiction: Patent jurisdiction.
            task_label: Human-readable task label.
            source_document: Source document name.
            rules_summary: Formatted rules summary.

        Returns:
            List of checklist strings.
        """
        prompt = CHECKLIST_PROMPT.format(
            jurisdiction=jurisdiction,
            task_label=task_label,
            source_document=source_document,
            rules_summary=rules_summary,
        )

        content = self._call_llm(prompt)

        try:
            checklist = json.loads(content)
            if isinstance(checklist, list):
                return [str(item) for item in checklist]
        except json.JSONDecodeError:
            logger.error("Failed to parse checklist JSON")

        return []

    def _generate_system_prompt(
        self,
        jurisdiction: str,
        task_label: str,
        source_document: str,
        rules_summary: str,
    ) -> str:
        """Generate a system prompt injection via LLM.

        Args:
            jurisdiction: Patent jurisdiction.
            task_label: Human-readable task label.
            source_document: Source document name.
            rules_summary: Formatted rules summary.

        Returns:
            System prompt paragraph.
        """
        prompt = SYSTEM_PROMPT_PROMPT.format(
            jurisdiction=jurisdiction,
            task_label=task_label,
            source_document=source_document,
            rules_summary=rules_summary,
        )

        return self._call_llm(prompt)

    def _call_llm(self, prompt: str) -> str:
        """Call the vLLM API with a prompt.

        Args:
            prompt: User prompt text.

        Returns:
            LLM response content.
        """
        payload = {
            "model": self.config.vllm_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "chat_template_kwargs": {"enable_thinking": False},
        }

        try:
            response = self.client.post(
                f"{self.config.vllm_base_url}/chat/completions",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except httpx.HTTPError as e:
            logger.error("LLM request failed: %s", e)
            return ""

    def close(self) -> None:
        """Close the HTTP client."""
        self.client.close()

    def __enter__(self) -> "CardCompiler":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
