"""LaTeX and BibTeX export for patent documents.

Provides BibTeX generation from reference dicts and a LaTeXExporter class
for full .tex document generation with template support.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LaTeX special character escaping (for BibTeX field values)
# ---------------------------------------------------------------------------

# Sentinels to avoid double-replacement during escaping.
_BACKSLASH_SENTINEL = "\x00BACKSLASH\x00"
_TILDE_SENTINEL = "\x00TILDE\x00"
_CARET_SENTINEL = "\x00CARET\x00"


def escape_latex(text: str) -> str:
    """Escape special LaTeX characters in plain text.

    Handles all 10 special characters that have meaning in LaTeX:
    ``\\``, ``&``, ``%``, ``$``, ``#``, ``_``, ``{``, ``}``, ``~``, ``^``.

    Args:
        text: Plain text string to escape.

    Returns:
        Text with all special LaTeX characters properly escaped.
    """
    # Phase 1: Replace chars that produce multi-char sequences containing
    # other special chars ({, }) with sentinels to avoid double-escaping.
    text = text.replace("\\", _BACKSLASH_SENTINEL)
    text = text.replace("~", _TILDE_SENTINEL)
    text = text.replace("^", _CARET_SENTINEL)

    # Phase 2: Replace remaining single-char specials.
    text = text.replace("&", r"\&")
    text = text.replace("%", r"\%")
    text = text.replace("$", r"\$")
    text = text.replace("#", r"\#")
    text = text.replace("_", r"\_")
    text = text.replace("{", r"\{")
    text = text.replace("}", r"\}")

    # Phase 3: Replace sentinels with final LaTeX commands.
    text = text.replace(_BACKSLASH_SENTINEL, r"\textbackslash{}")
    text = text.replace(_TILDE_SENTINEL, r"\textasciitilde{}")
    text = text.replace(_CARET_SENTINEL, r"\textasciicircum{}")

    return text


# ---------------------------------------------------------------------------
# Citation key generation
# ---------------------------------------------------------------------------


def sanitize_citation_key(title: str, index: int) -> str:
    """Generate a unique citation key from a title and sequential index.

    The key is formed from the first 3 alphanumeric words of the title,
    joined by underscores, lowercased, with the 1-based index appended.

    Example:
        ``"Method for Wireless Communication"`` → ``method_for_wireless_1``

    Args:
        title: The reference title string.
        index: 1-based sequential number for uniqueness.

    Returns:
        A sanitized citation key string.
    """
    # Extract alphanumeric words from the title
    words = re.findall(r"[a-zA-Z0-9]+", title)
    # Take first 3 words, lowercase
    key_parts = [w.lower() for w in words[:3]]

    if not key_parts:
        key_parts = ["untitled"]

    return "_".join(key_parts) + f"_{index}"


# ---------------------------------------------------------------------------
# BibTeX entry generation
# ---------------------------------------------------------------------------


def generate_bibtex_entry(ref: dict, index: int) -> tuple[str, str]:
    """Generate a single BibTeX entry from a reference dict.

    Entry type logic:
    - ``@article`` when ``doi`` is present and ``patent_number`` is absent
    - ``@misc`` otherwise (patent_number present, or neither field present)

    Args:
        ref: A reference dict with keys like title, source, doi, patent_number,
             url, relevance_score, abstract.
        index: 1-based sequential index for citation key generation.

    Returns:
        A tuple of (citation_key, bibtex_entry_string).
    """
    title = ref.get("title", "Untitled")
    source = ref.get("source", "")
    patent_number = ref.get("patent_number", "")
    doi = ref.get("doi", "")
    url = ref.get("url", "")
    relevance_score = ref.get("relevance_score")
    abstract = ref.get("abstract", "")

    # Determine entry type
    if doi and not patent_number:
        entry_type = "article"
    else:
        entry_type = "misc"

    # Generate citation key
    citation_key = sanitize_citation_key(title, index)

    # Build fields
    fields: list[str] = []
    fields.append(f"  title = {{{escape_latex(title)}}}")
    fields.append(f"  author = {{{escape_latex(source)}}}")

    if doi:
        fields.append(f"  doi = {{{escape_latex(doi)}}}")

    if url:
        fields.append(f"  url = {{{url}}}")

    if abstract:
        fields.append(f"  abstract = {{{escape_latex(abstract)}}}")

    # Note field: include relevance_score and/or patent_number
    note_parts: list[str] = []
    if patent_number:
        note_parts.append(f"Patent Number: {patent_number}")
    if relevance_score is not None:
        note_parts.append(f"Relevance Score: {relevance_score}/100")
    if note_parts:
        note_value = "; ".join(note_parts)
        fields.append(f"  note = {{{escape_latex(note_value)}}}")

    # Build the entry string
    fields_str = ",\n".join(fields)
    entry = f"@{entry_type}{{{citation_key},\n{fields_str}\n}}"

    return citation_key, entry


def generate_bibtex(references: list[dict]) -> str:
    """Convert a list of reference dicts to a BibTeX string.

    Generates one BibTeX entry per reference with unique citation keys.
    If duplicate keys arise (e.g. from identical titles), a numeric suffix
    is appended to ensure uniqueness.

    Args:
        references: List of reference dicts with keys like title, source,
                    doi, patent_number, url, relevance_score, abstract.

    Returns:
        A complete .bib file content string with one entry per reference.
    """
    if not references:
        return ""

    entries: list[str] = []
    seen_keys: set[str] = set()

    for i, ref in enumerate(references, 1):
        citation_key, entry = generate_bibtex_entry(ref, i)

        # Ensure uniqueness — if key already seen, append extra suffix
        base_key = citation_key
        suffix = 2
        while citation_key in seen_keys:
            citation_key = f"{base_key}_{suffix}"
            suffix += 1

        # If the key was modified, rebuild the entry with the unique key
        if citation_key != base_key:
            _, entry = _build_bibtex_entry_with_key(ref, citation_key)

        seen_keys.add(citation_key)
        entries.append(entry)

    return "\n\n".join(entries) + "\n"


def _build_bibtex_entry_with_key(ref: dict, citation_key: str) -> tuple[str, str]:
    """Build a BibTeX entry string using a specific citation key.

    This is a helper for generate_bibtex() when a key collision requires
    rebuilding the entry with a different key.

    Args:
        ref: A reference dict.
        citation_key: The citation key to use.

    Returns:
        A tuple of (citation_key, bibtex_entry_string).
    """
    title = ref.get("title", "Untitled")
    source = ref.get("source", "")
    patent_number = ref.get("patent_number", "")
    doi = ref.get("doi", "")
    url = ref.get("url", "")
    relevance_score = ref.get("relevance_score")
    abstract = ref.get("abstract", "")

    # Determine entry type
    if doi and not patent_number:
        entry_type = "article"
    else:
        entry_type = "misc"

    # Build fields
    fields: list[str] = []
    fields.append(f"  title = {{{escape_latex(title)}}}")
    fields.append(f"  author = {{{escape_latex(source)}}}")

    if doi:
        fields.append(f"  doi = {{{escape_latex(doi)}}}")

    if url:
        fields.append(f"  url = {{{url}}}")

    if abstract:
        fields.append(f"  abstract = {{{escape_latex(abstract)}}}")

    # Note field: include relevance_score and/or patent_number
    note_parts: list[str] = []
    if patent_number:
        note_parts.append(f"Patent Number: {patent_number}")
    if relevance_score is not None:
        note_parts.append(f"Relevance Score: {relevance_score}/100")
    if note_parts:
        note_value = "; ".join(note_parts)
        fields.append(f"  note = {{{escape_latex(note_value)}}}")

    # Build the entry string
    fields_str = ",\n".join(fields)
    entry = f"@{entry_type}{{{citation_key},\n{fields_str}\n}}"

    return citation_key, entry


# ---------------------------------------------------------------------------
# Built-in default template (used when no template file is available)
# ---------------------------------------------------------------------------

_DEFAULT_TEMPLATE = r"""\documentclass[a4paper,11pt]{article}
% --- Fonts (LuaLaTeX/XeLaTeX for full Unicode support) ---
\usepackage{iftex}
\ifluatex
  \usepackage{fontspec}
  \defaultfontfeatures{Ligatures=TeX}
\else\ifxetex
  \usepackage{fontspec}
  \defaultfontfeatures{Ligatures=TeX}
\else
  \usepackage[utf8]{inputenc}
  \usepackage[T1]{fontenc}
  \usepackage{lmodern}
\fi\fi
\usepackage[a4paper, margin=2.5cm]{geometry}
\usepackage[colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue]{hyperref}
\title{\textbf{Patent Application}}
\author{}
\date{Generated on \today\\[0.5em]\small Generated by mPAPA}
\begin{document}
\maketitle
\tableofcontents
\newpage
\section{Claims}
%%CLAIMS%%
\section{Description}
%%DESCRIPTION%%
%%WORKFLOW_STEPS%%
%%CHAT_LOG%%
\printbibliography
\end{document}
"""


# ---------------------------------------------------------------------------
# LaTeXExporter class
# ---------------------------------------------------------------------------


class LaTeXExporter:
    """Generates .tex and .bib patent documents using optional templates.

    Uses a template file with ``%%PLACEHOLDER%%`` markers that are replaced
    with converted content. Falls back to a built-in default template when
    no template file is available.
    """

    def __init__(self, template_dir: Path, template_name: str | None = None) -> None:
        """Initialize the exporter.

        Args:
            template_dir: Path to the templates directory containing .tex template files.
            template_name: Filename of the template to use (e.g. "patent_template.tex").
                           If None or file not found, uses built-in default structure.
        """
        self._template_dir = template_dir
        self._template_name = template_name

    def export(
        self,
        claims: str,
        description: str,
        output_path: Path,
        references: list[dict] | None = None,
        chat_history: list[dict] | None = None,
        workflow_steps: dict[str, str] | None = None,
    ) -> Path:
        """Generate a .tex file (and optionally .bib) at output_path.

        Args:
            claims: Patent claims text (markdown).
            description: Patent description text (markdown).
            output_path: Destination path for the .tex file.
            references: Optional list of reference dicts.
            chat_history: Optional list of chat message dicts.
            workflow_steps: Optional mapping of step_key to content text.

        Returns:
            The output_path where the .tex file was saved.
        """
        from patent_system.export.docx_exporter import (
            STEP_DISPLAY_NAMES,
            WORKFLOW_STEP_ORDER,
        )
        from patent_system.export.markdown_latex_converter import (
            convert_markdown_to_latex,
        )

        # Load template content
        template_content = self._load_template()

        # Convert claims and description from markdown to LaTeX
        claims_latex = convert_markdown_to_latex(claims)
        description_latex = convert_markdown_to_latex(description)

        # Build workflow steps content
        workflow_latex = self._build_workflow_steps(
            workflow_steps, WORKFLOW_STEP_ORDER, STEP_DISPLAY_NAMES, convert_markdown_to_latex
        )

        # Build chat history content
        chat_latex = self._build_chat_history(chat_history)

        # Handle references / bibliography
        bib_basename = output_path.stem
        if references:
            bib_content = generate_bibtex(references)
            bib_path = output_path.with_suffix(".bib")
            bibliography_file = bib_basename
            # Generate inline bibliography for single-pass compilation
            inline_bib = self._build_inline_bibliography(references)
        else:
            bib_content = ""
            bib_path = None
            bibliography_file = ""
            inline_bib = ""

        # Replace placeholders in template
        content = template_content
        content = content.replace("%%CLAIMS%%", claims_latex)
        content = content.replace("%%DESCRIPTION%%", description_latex)
        content = content.replace("%%WORKFLOW_STEPS%%", workflow_latex)
        content = content.replace("%%REFERENCES%%", "")  # No longer used (kept for custom templates)
        content = content.replace("%%CHAT_LOG%%", chat_latex)
        content = content.replace("%%BIBLIOGRAPHY_FILE%%", bibliography_file)

        # Replace \printbibliography with inline bibliography if references exist
        # This allows single-pass compilation without biber
        if inline_bib:
            content = content.replace(r"\printbibliography", inline_bib)
        else:
            content = content.replace(r"\printbibliography", "")

        # Ensure parent directories exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write .tex file
        output_path.write_text(content, encoding="utf-8")

        # Write .bib file if references exist
        if bib_path and bib_content:
            bib_path.write_text(bib_content, encoding="utf-8")

        return output_path

    def list_available_templates(self) -> list[str]:
        """List all .tex filenames in the template directory.

        Returns:
            A list of .tex filenames found in the template directory.
            Returns an empty list if the directory does not exist.
        """
        if not self._template_dir.exists():
            return []

        return sorted(f.name for f in self._template_dir.iterdir() if f.suffix == ".tex")

    def _load_template(self) -> str:
        """Load the template content from file or return the built-in default.

        Returns:
            The template content string.
        """
        if self._template_name is None:
            return _DEFAULT_TEMPLATE

        template_path = self._template_dir / self._template_name
        if not template_path.exists():
            logger.warning(
                "Configured template '%s' not found in '%s'; using built-in default.",
                self._template_name,
                self._template_dir,
            )
            return _DEFAULT_TEMPLATE

        return template_path.read_text(encoding="utf-8")

    @staticmethod
    def _build_workflow_steps(
        workflow_steps: dict[str, str] | None,
        step_order: list[str],
        display_names: dict[str, str],
        converter: object,
    ) -> str:
        """Build LaTeX content for workflow steps in canonical order.

        Args:
            workflow_steps: Mapping of step_key to content text.
            step_order: Canonical order of step keys.
            display_names: Mapping of step_key to display name.
            converter: Function to convert markdown to LaTeX.

        Returns:
            LaTeX string with sections for each non-empty step.
        """
        if not workflow_steps:
            return ""

        parts: list[str] = []
        for step_key in step_order:
            if step_key == "patent_draft":
                continue
            content = workflow_steps.get(step_key, "")
            if content and content.strip():
                display_name = display_names.get(step_key, step_key)
                converted = converter(content)  # type: ignore[operator]
                parts.append(f"\\section{{{display_name}}}\n{converted}")

        return "\n".join(parts)

    @staticmethod
    def _build_inline_bibliography(references: list[dict]) -> str:
        """Build an inline thebibliography environment for single-pass compilation.

        This embeds references directly in the .tex file so the PDF shows
        the bibliography without needing biber/bibtex passes.

        Args:
            references: List of reference dicts.

        Returns:
            LaTeX string with a thebibliography environment.
        """
        if not references:
            return ""

        lines: list[str] = []
        lines.append(r"\begin{thebibliography}{99}")
        lines.append("")

        for i, ref in enumerate(references, 1):
            title = ref.get("title", "Untitled")
            source = ref.get("source", "")
            patent_number = ref.get("patent_number", "")
            doi = ref.get("doi", "")
            url = ref.get("url", "")
            relevance_score = ref.get("relevance_score")
            abstract = ref.get("abstract", "")

            key = sanitize_citation_key(title, i)
            lines.append(f"\\bibitem{{{key}}}")

            # Build the reference text
            entry_parts: list[str] = []
            entry_parts.append(escape_latex(title))
            if source:
                entry_parts.append(f"\\textit{{{escape_latex(source)}}}")
            if patent_number:
                entry_parts.append(f"Patent: {escape_latex(patent_number)}")
            if doi:
                entry_parts.append(f"DOI: {escape_latex(doi)}")
            if url:
                entry_parts.append(f"\\url{{{url}}}")
            if relevance_score is not None:
                entry_parts.append(f"Relevance: {relevance_score}/100")

            lines.append(". ".join(entry_parts) + ".")
            lines.append("")

        lines.append(r"\end{thebibliography}")
        return "\n".join(lines)

    @staticmethod
    def _build_chat_history(chat_history: list[dict] | None) -> str:
        """Build LaTeX content for chat history with bold role labels.

        Chat messages are converted from markdown to LaTeX to properly handle
        formatting, special characters, and structure in LLM responses.

        Args:
            chat_history: List of chat message dicts with 'role' and 'message' keys.

        Returns:
            LaTeX string with a section header and formatted chat messages.
        """
        if not chat_history:
            return ""

        from patent_system.export.markdown_latex_converter import (
            convert_markdown_to_latex,
        )

        parts: list[str] = []
        parts.append("\\section{AI Chat Log}")
        for msg in chat_history:
            role = msg.get("role", "unknown")
            message = msg.get("message", "")
            label = "You" if role == "user" else "Assistant"
            # Convert markdown content to LaTeX (handles &, #, _, etc.)
            converted_message = convert_markdown_to_latex(message)
            if not converted_message:
                converted_message = escape_latex(message)
            parts.append(f"\\textbf{{{label}:}}\n\n{converted_message}")

        return "\n\n".join(parts)
