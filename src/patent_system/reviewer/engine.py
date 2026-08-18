"""ReviewEngine: orchestrates patent review against loaded Rule Cards.

Completely independent of the LangGraph workflow — uses LM Studio directly
via OpenAI-compatible API. Runs ONE sequential LLM call per card.
"""

import json
import logging
from typing import Any

import httpx
from pydantic import BaseModel, Field

from .card_registry import CardRegistry
from .rule_card import RuleCard

logger = logging.getLogger(__name__)


class ReviewFinding(BaseModel):
    """A single finding from a Rule Card review."""

    rule_id: str
    finding: str
    severity: str = Field(description="critical | major | minor | observation")
    location: str
    suggestion: str
    compliant: bool
    reference: str


class ReviewReport(BaseModel):
    """Aggregated report from one or more Rule Card reviews."""

    card_ids_used: list[str]
    jurisdiction: str
    overall_summary: str
    findings: list[ReviewFinding]
    statistics: dict[str, Any]


class ReviewEngine:
    """Orchestrates patent review against loaded Rule Cards.

    Completely independent of the LangGraph workflow — uses LM Studio
    directly via OpenAI-compatible API (same as existing mPAPA setup).
    One LLM call per card, sequential execution.

    Args:
        lm_studio_base_url: Base URL for the OpenAI-compatible API
            (e.g. ``http://localhost:1234/v1``).
        model_name: Model identifier to use for review calls.
        card_registry: CardRegistry instance for loading Rule Cards.
    """

    def __init__(
        self,
        lm_studio_base_url: str,
        model_name: str,
        card_registry: CardRegistry,
    ) -> None:
        self.base_url = lm_studio_base_url.rstrip("/")
        self.model_name = model_name
        self.card_registry = card_registry

    async def review(
        self,
        patent_text: str,
        card_ids: list[str],
        *,
        focus_sections: list[str] | None = None,
        severity_threshold: str = "all",
    ) -> ReviewReport:
        """Run a review session against one or more Rule Cards.

        Loads each card from the registry, builds a system prompt with
        injected rules and checklist, sends to the LLM, and parses the
        structured response into findings.

        Args:
            patent_text: The patent application text to review.
            card_ids: List of card IDs to review against (one LLM call each).
            focus_sections: Optional list of sections to focus on
                (e.g. ``["claims", "description"]``).
            severity_threshold: Filter threshold — ``"all"``, ``"mandatory"``,
                or ``"should"``.

        Returns:
            ReviewReport with aggregated findings and statistics.
        """
        all_findings: list[ReviewFinding] = []
        jurisdiction = ""

        for card_id in card_ids:
            card = self.card_registry.get_card(card_id)
            if jurisdiction == "":
                jurisdiction = card.jurisdiction

            system_prompt = self._build_system_prompt(card)
            user_prompt = self._build_user_prompt(patent_text, focus_sections)

            raw_response = await self._call_llm(system_prompt, user_prompt)
            findings = self._parse_findings(raw_response, card_id)

            if severity_threshold != "all":
                findings = self._filter_by_severity(findings, severity_threshold)

            all_findings.extend(findings)

        statistics = self._compute_statistics(all_findings)
        overall_summary = self._generate_summary(all_findings, card_ids)

        return ReviewReport(
            card_ids_used=card_ids,
            jurisdiction=jurisdiction,
            overall_summary=overall_summary,
            findings=all_findings,
            statistics=statistics,
        )

    def _build_system_prompt(self, card: RuleCard) -> str:
        """Build the system prompt by injecting card rules and checklist.

        Args:
            card: The RuleCard to inject into the prompt.

        Returns:
            Formatted system prompt string.
        """
        rules_section = self._format_rules(card)
        checklist_section = "\n".join(
            f"- {item}" for item in card.review_checklist
        )

        return (
            f"You are a patent examiner for {card.office}.\n"
            f"You are reviewing a patent application against the official "
            f"examination guidelines.\n\n"
            f"=== APPLICABLE RULES ===\n"
            f"{card.system_prompt_injection}\n\n"
            f"=== RULE DETAILS ===\n"
            f"{rules_section}\n\n"
            f"=== REVIEW CHECKLIST ===\n"
            f"{checklist_section}\n\n"
            f"=== OUTPUT FORMAT ===\n"
            f"Respond with a JSON object containing:\n"
            f'- "findings": array of objects, each with:\n'
            f'  - "rule_id": string (which rule is violated/satisfied)\n'
            f'  - "finding": string (what you found)\n'
            f'  - "severity": string (critical | major | minor | observation)\n'
            f'  - "location": string (where in the patent text)\n'
            f'  - "suggestion": string (how to fix it)\n'
            f'  - "compliant": boolean (true if rule is satisfied)\n'
            f'  - "reference": string (legal reference)\n'
            f'- "summary": string (brief overall assessment)\n\n'
            f"Respond ONLY with valid JSON. No markdown, no commentary."
        )

    def _format_rules(self, card: RuleCard) -> str:
        """Format card rules into a readable text block.

        Args:
            card: The RuleCard whose rules to format.

        Returns:
            Formatted rules string.
        """
        parts: list[str] = []
        for rule in card.rules:
            block = (
                f"[{rule.rule_id}] {rule.title}\n"
                f"  Requirement: {rule.requirement}\n"
                f"  Severity: {rule.severity}\n"
            )
            if rule.exceptions:
                exceptions_str = "; ".join(rule.exceptions)
                block += f"  Exceptions: {exceptions_str}\n"
            block += f"  Reference: {rule.reference}"
            parts.append(block)
        return "\n\n".join(parts)

    def _build_user_prompt(
        self, patent_text: str, focus_sections: list[str] | None
    ) -> str:
        """Build the user prompt containing the patent text.

        Args:
            patent_text: The patent application text.
            focus_sections: Optional sections to focus the review on.

        Returns:
            Formatted user prompt string.
        """
        focus = (
            ", ".join(focus_sections) if focus_sections else "all sections"
        )
        return (
            f"Review the following patent application:\n\n"
            f"{patent_text}\n\n"
            f"Focus on: {focus}"
        )

    async def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Send a chat completion request to the LLM via OpenAI-compatible API.

        Args:
            system_prompt: The system message with injected rules.
            user_prompt: The user message with patent text.

        Returns:
            The LLM's response content as a string.

        Raises:
            httpx.HTTPStatusError: If the API returns a non-2xx status.
            RuntimeError: If the response structure is unexpected.
        """
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
            "chat_template_kwargs": {"enable_thinking": False},
        }

        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()

        data = response.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise RuntimeError(
                f"Unexpected LLM response structure: {data}"
            ) from e

    def _parse_findings(
        self, raw_response: str, card_id: str
    ) -> list[ReviewFinding]:
        """Parse the LLM's JSON response into ReviewFinding objects.

        Handles both clean JSON and JSON wrapped in markdown code fences.

        Args:
            raw_response: The raw LLM response string.
            card_id: The card ID this response belongs to (for logging).

        Returns:
            List of parsed ReviewFinding objects.
        """
        text = raw_response.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first line (```json or ```) and last line (```)
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning(
                "Failed to parse LLM response as JSON for card %s: %s",
                card_id,
                text[:200],
            )
            return []

        findings_data = data.get("findings", [])
        findings: list[ReviewFinding] = []

        for item in findings_data:
            try:
                findings.append(
                    ReviewFinding(
                        rule_id=item.get("rule_id", "unknown"),
                        finding=item.get("finding", ""),
                        severity=item.get("severity", "observation"),
                        location=item.get("location", ""),
                        suggestion=item.get("suggestion", ""),
                        compliant=item.get("compliant", False),
                        reference=item.get("reference", ""),
                    )
                )
            except Exception:
                logger.warning(
                    "Skipping malformed finding in card %s: %s",
                    card_id,
                    item,
                )

        return findings

    def _filter_by_severity(
        self, findings: list[ReviewFinding], threshold: str
    ) -> list[ReviewFinding]:
        """Filter findings by severity threshold.

        Args:
            findings: List of findings to filter.
            threshold: ``"mandatory"`` keeps critical+major;
                ``"should"`` keeps critical+major+minor.

        Returns:
            Filtered list of findings.
        """
        severity_levels = {
            "mandatory": {"critical", "major"},
            "should": {"critical", "major", "minor"},
        }
        allowed = severity_levels.get(threshold, set())
        if not allowed:
            return findings
        return [f for f in findings if f.severity in allowed]

    def _compute_statistics(
        self, findings: list[ReviewFinding]
    ) -> dict[str, Any]:
        """Compute summary statistics from all findings.

        Args:
            findings: All collected findings across cards.

        Returns:
            Dictionary with severity counts and compliance rate.
        """
        total = len(findings)
        severity_counts = {"critical": 0, "major": 0, "minor": 0, "observation": 0}
        compliant_count = 0

        for f in findings:
            severity_counts[f.severity] = severity_counts.get(f.severity, 0) + 1
            if f.compliant:
                compliant_count += 1

        return {
            "total_findings": total,
            "severity_counts": severity_counts,
            "compliant_count": compliant_count,
            "non_compliant_count": total - compliant_count,
            "compliance_rate": (
                compliant_count / total if total > 0 else 1.0
            ),
        }

    def _generate_summary(
        self, findings: list[ReviewFinding], card_ids: list[str]
    ) -> str:
        """Generate a brief overall summary of the review.

        Args:
            findings: All collected findings.
            card_ids: List of card IDs used in the review.

        Returns:
            Human-readable summary string.
        """
        total = len(findings)
        non_compliant = sum(1 for f in findings if not f.compliant)
        critical = sum(1 for f in findings if f.severity == "critical")
        major = sum(1 for f in findings if f.severity == "major")

        if non_compliant == 0:
            return (
                f"Patent review against {len(card_ids)} rule card(s) "
                f"found {total} observations with no non-compliance issues."
            )

        return (
            f"Patent review against {len(card_ids)} rule card(s) identified "
            f"{non_compliant} non-compliance issue(s) out of {total} findings "
            f"({critical} critical, {major} major)."
        )
