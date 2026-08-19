"""Rule extraction from guideline PDFs via LLM — with chapter-level targeting.

Key design: PDFs often contain entire Parts (e.g. EPO Part G = all patentability chapters).
This extractor detects chapter boundaries and extracts ONLY the relevant section,
producing focused rule cards (10-30 rules) instead of monolithic ones (300+).
"""

import json
import logging
import re
from pathlib import Path
from typing import Any

import httpx
import pymupdf

from .config import CompilerConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chapter detection patterns (per jurisdiction)
# ---------------------------------------------------------------------------

# EPO: "Chapter VI – Novelty" or "Part G - Chapter VI-1" (page header)
EPO_CHAPTER_PATTERN = re.compile(
    r"^Chapter\s+(I{1,3}|IV|V|VI{0,3}|VII|VIII|IX|X)\s*[–—-]\s*(.+)$",
    re.MULTILINE,
)

# MPEP: section numbers like "2131" or "§ 2131"
MPEP_SECTION_PATTERN = re.compile(r"^§?\s*(\d{4}(?:\.\d+)?)\s+(.+)$", re.MULTILINE)

# Chapter keywords mapped to our task identifiers
CHAPTER_TASK_MAP = {
    # EPO Part G
    "patentability": ["Chapter I"],
    "subject_matter": ["Chapter II"],
    "industrial_application": ["Chapter III"],
    "state_of_art": ["Chapter IV"],
    "non_prejudicial_disclosures": ["Chapter V"],
    "novelty": ["Chapter VI"],
    "inventive_step": ["Chapter VII"],
    # EPO Part F
    "description_format": ["Chapter II"],
    "sufficiency": ["Chapter III"],
    "claim_structure": ["Chapter IV"],
    "unity": ["Chapter V"],
    "priority": ["Chapter VI"],
    # EPO Part H
    "added_matter": ["Chapter IV"],
    # EPO Part D
    "opposition": ["Chapter V"],
    # USPTO MPEP sections
    "obviousness": ["2141", "2142", "2143", "2144", "2145"],
}

# USPTO sections per task
MPEP_SECTIONS_MAP = {
    "subject_matter": (2104, 2106),
    "novelty": (2131, 2138),
    "obviousness": (2141, 2145),
    "written_description": (2161, 2164),
    "enablement": (2164, 2164),
    "definiteness": (2171, 2175),
    "claim_structure": (2171, 2175),
    "double_patenting": (804, 804),
    "restriction": (803, 808),
}


# ---------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """\
You are a patent law rule extraction system. Extract ONLY normative rules about {task_label} from the following text.

{translation_instruction}

IMPORTANT: Extract ONLY rules relevant to {task_label}. Ignore rules about other topics.
Focus on statements with normative language: SHALL, MUST, SHOULD, MAY NOT, MUST NOT, REQUIRED.

For each rule, output a JSON object:
- "rule_id": short snake_case identifier (e.g. "novelty_single_document_test")
- "title": concise title (max 80 chars)
- "requirement": the full normative statement
- "severity": "mandatory" (SHALL/MUST), "recommended" (SHOULD), or "prohibited" (SHALL NOT/MAY NOT)
- "exceptions": list of exception strings (empty list if none)
- "examples": list of example strings (empty list if none)
- "reference": section/article/rule number from source

ALL OUTPUT MUST be in English. If source is not English, translate accurately preserving legal precision.

Output ONLY a JSON array. No markdown, no explanation. Empty array [] if no relevant rules found.

Jurisdiction: {jurisdiction}
Topic: {task_label}
Source: {source_name} ({source_language})

--- TEXT ---
{chunk}
--- END ---"""

TRANSLATION_INSTRUCTIONS = {
    "en": "The source text is in English.",
    "de": "Source is GERMAN. Translate to English. (Patentanspruch→patent claim, erfinderische Tätigkeit→inventive step)",
    "ja": "Source is JAPANESE. Translate to English. (新規性→novelty, 進歩性→inventive step, 特許請求の範囲→claims)",
    "zh": "Source is CHINESE. Translate to English. (新颖性→novelty, 创造性→inventive step, 权利要求→claims)",
    "ko": "Source is KOREAN. Translate to English. (신규성→novelty, 진보성→inventive step, 특허청구범위→claims)",
    "fr": "Source is FRENCH. Translate to English. (activité inventive→inventive step, revendications→claims)",
}

TASK_LABELS = {
    "novelty": "Novelty",
    "inventive_step": "Inventive Step / Non-Obviousness",
    "claim_structure": "Claim Structure & Drafting",
    "sufficiency": "Sufficiency of Disclosure",
    "unity": "Unity of Invention",
    "subject_matter": "Patentable Subject Matter",
    "obviousness": "Obviousness (§103)",
    "written_description": "Written Description",
    "enablement": "Enablement",
    "definiteness": "Definiteness / Claim Clarity",
    "added_matter": "Added Matter / Amendments",
    "priority": "Priority",
    "industrial_application": "Industrial Application",
    "double_patenting": "Double Patenting",
    "restriction": "Restriction / Election",
    "opposition": "Opposition Grounds",
    "description_format": "Description Format & Content",
    "state_of_art": "State of the Art / Prior Art Definition",
    "non_prejudicial_disclosures": "Non-Prejudicial Disclosures / Grace Period",
}


class RuleExtractor:
    """Extracts normative rules from guideline PDFs using LLM.

    Key feature: chapter-level targeting. Instead of processing an entire PDF,
    detects chapter boundaries and extracts only the section relevant to the
    requested task. Produces focused cards (10-30 rules) suitable for small LLMs.
    """

    def __init__(
        self,
        config: CompilerConfig,
        chunk_size: int = 6000,
        chunk_overlap: int = 500,
    ) -> None:
        self.config = config
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.client = httpx.Client(timeout=180.0)

    def extract(
        self,
        source_path: Path,
        jurisdiction: str,
        task: str,
        source_language: str = "en",
    ) -> list[dict[str, Any]]:
        """Extract rules for a specific task from a guideline PDF/HTML.

        Detects chapter boundaries and processes ONLY the relevant section.

        Args:
            source_path: Path to the PDF or HTML source file.
            jurisdiction: Patent jurisdiction (EP, US, PCT, etc.).
            task: Task identifier (novelty, inventive_step, etc.).
            source_language: ISO 639-1 code (en, de, ja, zh, ko, fr).

        Returns:
            List of extracted rule dicts (focused on the task, in English).
        """
        # Extract full text with page markers
        pages = self._extract_pages(source_path)
        source_name = source_path.name

        # Target only the relevant section
        section_text = self._extract_section(pages, jurisdiction, task)
        if not section_text:
            logger.warning("Could not isolate section for %s/%s — using full text", jurisdiction, task)
            section_text = "\n".join(pages)

        section_chars = len(section_text)
        total_chars = sum(len(p) for p in pages)
        logger.info(
            "Section targeting: %s/%s → %d chars (%.0f%% of %d total)",
            jurisdiction, task, section_chars, 100 * section_chars / max(total_chars, 1), total_chars,
        )

        # Chunk and process
        chunks = self._chunk_text(section_text)
        task_label = TASK_LABELS.get(task, task.replace("_", " ").title())
        lang_label = f" [{source_language}→en]" if source_language != "en" else ""

        logger.info("Extracting %s rules from %s (%d chunks)%s", task_label, source_name, len(chunks), lang_label)

        all_rules: list[dict[str, Any]] = []
        for i, chunk in enumerate(chunks):
            logger.info("  Chunk %d/%d (%d chars)", i + 1, len(chunks), len(chunk))
            rules = self._process_chunk(chunk, jurisdiction, task, task_label, source_name, source_language)
            all_rules.extend(rules)
            logger.info("    → %d rules extracted", len(rules))

        # Deduplicate
        seen: set[str] = set()
        deduplicated: list[dict[str, Any]] = []
        for rule in all_rules:
            rule_id = rule.get("rule_id", "")
            if rule_id and rule_id not in seen:
                seen.add(rule_id)
                deduplicated.append(rule)

        logger.info("Extracted %d unique rules for %s from %s", len(deduplicated), task, source_name)
        return deduplicated

    # ------------------------------------------------------------------
    # Section targeting
    # ------------------------------------------------------------------

    def _extract_pages(self, source_path: Path) -> list[str]:
        """Extract text per page from PDF, or as single block from HTML."""
        suffix = source_path.suffix.lower()
        if suffix == ".pdf":
            doc = pymupdf.open(str(source_path))
            pages = [page.get_text() for page in doc]
            doc.close()
            return pages
        elif suffix in (".html", ".htm"):
            return [self._extract_html_content(source_path)]
        else:
            raise ValueError(f"Unsupported format: {suffix}")

    def _extract_section(self, pages: list[str], jurisdiction: str, task: str) -> str | None:
        """Find and extract only the chapter/section relevant to the task.

        For EPO: detects "Chapter X – Title" boundaries in the PDF.
        For USPTO: detects MPEP section number ranges.
        """
        jurisdiction = jurisdiction.upper()

        if jurisdiction == "EP":
            return self._extract_epo_chapter(pages, task)
        elif jurisdiction == "US":
            return self._extract_mpep_section(pages, task)
        else:
            # For PCT, JP, etc. — use full text (their PDFs are usually single-chapter)
            return None

    def _extract_epo_chapter(self, pages: list[str], task: str) -> str | None:
        """Extract a specific EPO chapter from a multi-chapter Part PDF.

        Finds the page where the chapter content starts (after TOC),
        and the page where the next chapter starts.
        """
        # Determine which chapter title to look for
        chapter_keywords = CHAPTER_TASK_MAP.get(task)
        if not chapter_keywords:
            return None

        target_chapter = chapter_keywords[0]  # e.g. "Chapter VI"
        # Extract roman numeral for matching
        target_roman = target_chapter.replace("Chapter ", "")

        # Find content pages (skip TOC pages — those are the first ~8 pages with many chapters listed)
        # Strategy: find the LAST occurrence of "Chapter X – Title" which is in the actual content
        chapter_start_page = None
        next_chapter_page = None

        # Find all pages with chapter headers (content pages, not TOC)
        chapter_pages: list[tuple[int, str]] = []
        for i, page_text in enumerate(pages):
            # Skip early TOC pages (typically pages 1-8 have condensed chapter listings)
            if i < 8:
                continue
            for match in EPO_CHAPTER_PATTERN.finditer(page_text):
                roman = match.group(1)
                title = match.group(2).strip()
                chapter_pages.append((i, roman))

        # Find our target chapter and the one after it
        for idx, (page_num, roman) in enumerate(chapter_pages):
            if roman == target_roman:
                chapter_start_page = page_num
                # Next chapter boundary
                if idx + 1 < len(chapter_pages):
                    next_chapter_page = chapter_pages[idx + 1][0]
                break

        if chapter_start_page is None:
            logger.warning("Could not find %s in PDF", target_chapter)
            return None

        # Extract pages for this chapter
        end_page = next_chapter_page if next_chapter_page else len(pages)
        section_pages = pages[chapter_start_page:end_page]

        logger.info(
            "EPO %s: pages %d-%d (%d pages)",
            target_chapter, chapter_start_page + 1, end_page, len(section_pages),
        )
        return "\n".join(section_pages)

    def _extract_mpep_section(self, pages: list[str], task: str) -> str | None:
        """Extract MPEP sections by section number range.

        MPEP has clear section numbers (2131, 2141, etc.) as page headers.
        """
        section_range = MPEP_SECTIONS_MAP.get(task)
        if not section_range:
            return None

        start_section, end_section = section_range
        start_page = None
        end_page = None

        for i, page_text in enumerate(pages):
            # Look for section numbers in page headers/content
            for match in MPEP_SECTION_PATTERN.finditer(page_text):
                section_num = int(match.group(1).split(".")[0])
                if start_page is None and section_num >= start_section:
                    start_page = i
                if section_num > end_section and end_page is None:
                    end_page = i
                    break

        if start_page is None:
            # Fallback: search for section number as plain text
            for i, page_text in enumerate(pages):
                if str(start_section) in page_text and start_page is None:
                    start_page = i
                if end_page is None and start_page is not None and str(end_section + 1) in page_text:
                    end_page = i

        if start_page is None:
            logger.warning("Could not find MPEP section %d", start_section)
            return None

        end_page = end_page or len(pages)
        section_pages = pages[start_page:end_page]

        logger.info(
            "MPEP §%d-§%d: pages %d-%d (%d pages)",
            start_section, end_section, start_page + 1, end_page, len(section_pages),
        )
        return "\n".join(section_pages)

    # ------------------------------------------------------------------
    # HTML extraction
    # ------------------------------------------------------------------

    def _extract_html_content(self, source_path: Path) -> str:
        """Extract text from HTML, stripping boilerplate."""
        import html as html_module

        html = source_path.read_text(encoding="utf-8")

        # Remove non-content blocks
        html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r"<nav[^>]*>.*?</nav>", "", html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r"<header[^>]*>.*?</header>", "", html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r"<footer[^>]*>.*?</footer>", "", html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)

        # Try main/article content
        main_match = re.search(r"<(?:main|article)[^>]*>(.*?)</(?:main|article)>", html, flags=re.DOTALL | re.IGNORECASE)
        if main_match:
            html = main_match.group(1)

        # Convert structure to text
        html = re.sub(r"<h[1-6][^>]*>(.*?)</h[1-6]>", r"\n## \1\n", html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r"<li[^>]*>", "\n- ", html, flags=re.IGNORECASE)
        html = re.sub(r"<br\s*/?>", "\n", html, flags=re.IGNORECASE)
        html = re.sub(r"<p[^>]*>", "\n", html, flags=re.IGNORECASE)
        html = re.sub(r"</p>", "\n", html, flags=re.IGNORECASE)

        text = re.sub(r"<[^>]+>", "", html)
        text = html_module.unescape(text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" +", " ", text)

        return text.strip()

    # ------------------------------------------------------------------
    # Chunking and LLM processing
    # ------------------------------------------------------------------

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks."""
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
        task_label: str,
        source_name: str,
        source_language: str = "en",
    ) -> list[dict[str, Any]]:
        """Send a chunk to the LLM for rule extraction."""
        translation_instruction = TRANSLATION_INSTRUCTIONS.get(
            source_language,
            f"Source is in '{source_language}'. Translate to English.",
        )

        prompt = EXTRACTION_PROMPT.format(
            jurisdiction=jurisdiction,
            task=task,
            task_label=task_label,
            source_name=source_name,
            source_language=source_language,
            translation_instruction=translation_instruction,
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

        # Strip markdown fencing if present
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*\n?", "", content)
            content = re.sub(r"\n?```\s*$", "", content)

        try:
            rules = json.loads(content)
            if not isinstance(rules, list):
                rules = [rules]
            return rules
        except json.JSONDecodeError as e:
            logger.error("JSON parse failed: %s (first 200 chars: %s)", e, content[:200])
            return []

    def close(self) -> None:
        """Close the HTTP client."""
        self.client.close()

    def __enter__(self) -> "RuleExtractor":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
