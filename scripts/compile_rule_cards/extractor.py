"""Rule extraction from guideline PDFs via LLM."""

import json
import logging
from pathlib import Path
from typing import Any

import httpx
import pymupdf

from .config import CompilerConfig

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """\
You are a patent law rule extraction system. Analyze the following text from an official patent guideline document and extract all normative statements.

Focus on statements containing normative language: SHALL, MUST, SHOULD, MAY NOT, MUST NOT, REQUIRED, SHALL NOT.

For each rule found, output a JSON object with these fields:
- "rule_id": A short snake_case identifier derived from the rule content (e.g. "claims_must_be_supported")
- "title": A concise title summarizing the rule (max 80 chars)
- "requirement": The full normative statement as written in the source
- "severity": One of "mandatory" (SHALL/MUST), "recommended" (SHOULD), "prohibited" (SHALL NOT/MUST NOT/MAY NOT)
- "exceptions": List of strings describing any exceptions or qualifications mentioned
- "examples": List of strings with any examples provided in the source
- "reference": The section/article/rule number from the source document

Output a JSON array of rule objects. If no normative statements are found in the chunk, output an empty array [].

Jurisdiction: {jurisdiction}
Task context: {task}
Source document: {source_name}

--- BEGIN TEXT CHUNK ---
{chunk}
--- END TEXT CHUNK ---

Output ONLY valid JSON (an array of rule objects). No markdown fencing, no explanation."""


class RuleExtractor:
    """Extracts normative rules from guideline PDFs using LLM.

    Args:
        config: Compiler configuration.
        chunk_size: Maximum characters per text chunk sent to the LLM.
        chunk_overlap: Character overlap between consecutive chunks.
    """

    def __init__(
        self,
        config: CompilerConfig,
        chunk_size: int = 4000,
        chunk_overlap: int = 500,
    ) -> None:
        self.config = config
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.client = httpx.Client(timeout=120.0)

    def extract(
        self,
        source_path: Path,
        jurisdiction: str,
        task: str,
    ) -> list[dict[str, Any]]:
        """Extract rules from a guideline PDF or HTML file.

        Args:
            source_path: Path to the PDF or HTML source file.
            jurisdiction: Patent jurisdiction (EP, US, PCT, etc.).
            task: Task context (claim_structure, novelty, inventive_step, etc.).

        Returns:
            List of extracted rule dictionaries.
        """
        text = self._extract_text(source_path)
        chunks = self._chunk_text(text)
        source_name = source_path.name

        logger.info(
            "Extracting rules from %s (%d chunks)", source_name, len(chunks)
        )

        all_rules: list[dict[str, Any]] = []
        for i, chunk in enumerate(chunks):
            logger.debug("Processing chunk %d/%d", i + 1, len(chunks))
            rules = self._process_chunk(chunk, jurisdiction, task, source_name)
            all_rules.extend(rules)

        # Deduplicate by rule_id
        seen: set[str] = set()
        deduplicated: list[dict[str, Any]] = []
        for rule in all_rules:
            rule_id = rule.get("rule_id", "")
            if rule_id and rule_id not in seen:
                seen.add(rule_id)
                deduplicated.append(rule)

        logger.info(
            "Extracted %d unique rules from %s", len(deduplicated), source_name
        )
        return deduplicated

    def _extract_text(self, source_path: Path) -> str:
        """Extract text content from PDF or HTML file.

        Args:
            source_path: Path to the source file.

        Returns:
            Extracted text content.

        Raises:
            ValueError: If the file format is unsupported.
        """
        suffix = source_path.suffix.lower()

        if suffix == ".pdf":
            doc = pymupdf.open(str(source_path))
            text_parts: list[str] = []
            for page in doc:
                text_parts.append(page.get_text())
            doc.close()
            return "\n".join(text_parts)
        elif suffix in (".html", ".htm"):
            return source_path.read_text(encoding="utf-8")
        else:
            raise ValueError(
                f"Unsupported file format: {suffix}. Use .pdf or .html/.htm"
            )

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks.

        Args:
            text: Full document text.

        Returns:
            List of text chunks.
        """
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk)
            start += self.chunk_size - self.chunk_overlap
        return chunks

    def _process_chunk(
        self,
        chunk: str,
        jurisdiction: str,
        task: str,
        source_name: str,
    ) -> list[dict[str, Any]]:
        """Send a text chunk to the LLM and parse extracted rules.

        Args:
            chunk: Text chunk to process.
            jurisdiction: Patent jurisdiction.
            task: Task context.
            source_name: Name of the source document.

        Returns:
            List of rule dictionaries extracted from this chunk.
        """
        prompt = EXTRACTION_PROMPT.format(
            jurisdiction=jurisdiction,
            task=task,
            source_name=source_name,
            chunk=chunk,
        )

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
        except httpx.HTTPError as e:
            logger.error("LLM request failed: %s", e)
            return []

        data = response.json()
        content = data["choices"][0]["message"]["content"].strip()

        try:
            rules = json.loads(content)
            if not isinstance(rules, list):
                logger.warning("LLM returned non-array JSON, wrapping")
                rules = [rules]
            return rules
        except json.JSONDecodeError as e:
            logger.error("Failed to parse LLM JSON output: %s", e)
            logger.debug("Raw output: %s", content[:500])
            return []

    def close(self) -> None:
        """Close the HTTP client."""
        self.client.close()

    def __enter__(self) -> "RuleExtractor":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
