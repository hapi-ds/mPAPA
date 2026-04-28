"""Property-based tests for LaTeX export.

Uses Hypothesis to validate correctness properties of the LaTeX export system.
"""

from __future__ import annotations

import re

from hypothesis import given, settings
from hypothesis import strategies as st

from patent_system.export.markdown_latex_converter import (
    _escape_latex_text,
    convert_markdown_to_latex,
)

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Plain text words (alphanumeric, no special chars that would be consumed by
# markdown parsing itself).
_plain_word = st.text(
    alphabet=st.characters(categories=("L", "N"), min_codepoint=65, max_codepoint=122),
    min_size=1,
    max_size=20,
).filter(lambda s: s.strip() != "")

# Generate simple markdown with headings, bold, italic, plain text, and lists.
_heading_md = st.builds(
    lambda level, text: f"{'#' * level} {text}\n\n",
    level=st.integers(min_value=1, max_value=3),
    text=_plain_word,
)

_bold_md = st.builds(lambda text: f"**{text}**", text=_plain_word)
_italic_md = st.builds(lambda text: f"*{text}*", text=_plain_word)
_plain_md = _plain_word

_list_item_md = st.builds(lambda text: f"- {text}", text=_plain_word)
_list_md = st.builds(
    lambda items: "\n".join(items) + "\n\n",
    items=st.lists(_list_item_md, min_size=1, max_size=5),
)

_paragraph_md = st.builds(
    lambda parts: " ".join(parts) + "\n\n",
    parts=st.lists(
        st.one_of(_bold_md, _italic_md, _plain_md),
        min_size=1,
        max_size=5,
    ),
)

_markdown_block = st.one_of(_heading_md, _paragraph_md, _list_md)

_markdown_document = st.builds(
    lambda blocks: "".join(blocks),
    blocks=st.lists(_markdown_block, min_size=1, max_size=5),
)

# LaTeX special characters for Property 8 testing.
_LATEX_SPECIAL_CHARS = "&%$#_{}~^\\"

_text_with_special_chars = st.text(
    alphabet=st.sampled_from(
        list("abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        + list(_LATEX_SPECIAL_CHARS)
    ),
    min_size=1,
    max_size=100,
).filter(lambda s: s.strip() != "" and any(c in s for c in _LATEX_SPECIAL_CHARS))


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _extract_plain_words(markdown_text: str) -> set[str]:
    """Extract plain text words from markdown, ignoring formatting syntax."""
    # Remove markdown formatting markers
    text = markdown_text
    # Remove heading markers
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Remove bold/italic markers
    text = re.sub(r"\*{1,3}", "", text)
    # Remove list markers
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
    # Extract words (alphanumeric sequences)
    words = set(re.findall(r"[a-zA-Z0-9]+", text))
    return words


def _extract_text_segments_from_latex(latex_output: str) -> str:
    """Extract text segments from LaTeX output (excluding commands).

    Returns the concatenated text content, stripping LaTeX commands.
    """
    # Remove LaTeX commands but keep their text arguments
    # This is a simplified extraction for testing purposes
    text = latex_output
    # Remove environment markers
    text = re.sub(r"\\begin\{[^}]+\}", "", text)
    text = re.sub(r"\\end\{[^}]+\}", "", text)
    # Remove commands like \section{}, \textbf{}, etc. but keep content
    text = re.sub(r"\\(?:section|subsection|subsubsection|textbf|textit|texttt|href)\{", "", text)
    # Remove \item
    text = re.sub(r"\\item\s*", "", text)
    # Remove \hline, \\, &
    text = re.sub(r"\\hline", "", text)
    text = re.sub(r"\\\\", "", text)
    return text


# ---------------------------------------------------------------------------
# Property 7: Markdown-to-LaTeX preserves all plain text content
# Feature: latex-export, Property 7: text preservation
# ---------------------------------------------------------------------------


class TestTextPreservation:
    """Property 7: Markdown-to-LaTeX preserves all plain text content.

    For any markdown string containing headings, bold, italic, plain text,
    and lists, converting to LaTeX SHALL produce output that contains all
    the plain text words from the markdown input.

    **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.6, 3.7**
    """

    @given(markdown=_markdown_document)
    @settings(max_examples=100)
    def test_all_plain_text_words_preserved_in_latex(
        self,
        markdown: str,
    ) -> None:
        """All plain text words from markdown appear in the LaTeX output."""
        latex_output = convert_markdown_to_latex(markdown)

        # Extract words from the original markdown (ignoring formatting syntax)
        original_words = _extract_plain_words(markdown)

        # Each word from the original should appear in the LaTeX output
        for word in original_words:
            assert word in latex_output, (
                f"Word '{word}' from markdown not found in LaTeX output.\n"
                f"Markdown: {markdown!r}\n"
                f"LaTeX: {latex_output!r}"
            )


# ---------------------------------------------------------------------------
# Property 8: No unescaped special characters in text segments
# Feature: latex-export, Property 8: no unescaped special characters
# ---------------------------------------------------------------------------


class TestNoUnescapedSpecialChars:
    """Property 8: LaTeX output contains no unescaped special characters.

    For any markdown string containing special LaTeX characters in plain text,
    converting to LaTeX SHALL produce output where all such characters in text
    segments are properly escaped.

    **Validates: Requirements 3.10, 3.13**
    """

    @given(text=_text_with_special_chars)
    @settings(max_examples=100)
    def test_escape_function_handles_all_special_chars(
        self,
        text: str,
    ) -> None:
        """The _escape_latex_text function escapes all special characters."""
        escaped = _escape_latex_text(text)

        # After escaping, no raw special characters should remain
        # that aren't part of a LaTeX command.
        # Check each special char is properly escaped:
        # - & should only appear as \&
        # - % should only appear as \%
        # - $ should only appear as \$
        # - # should only appear as \#
        # - _ should only appear as \_
        # - { should only appear as \{ or in \textbackslash{}, \textasciitilde{}, \textasciicircum{}
        # - } should only appear as \} or in \textbackslash{}, \textasciitilde{}, \textasciicircum{}
        # - ~ should only appear as \textasciitilde{}
        # - ^ should only appear as \textasciicircum{}
        # - \ should only appear as part of a command

        # Remove known escaped sequences to check for unescaped specials
        cleaned = escaped
        cleaned = cleaned.replace(r"\textbackslash{}", "")
        cleaned = cleaned.replace(r"\textasciitilde{}", "")
        cleaned = cleaned.replace(r"\textasciicircum{}", "")
        cleaned = cleaned.replace(r"\&", "")
        cleaned = cleaned.replace(r"\%", "")
        cleaned = cleaned.replace(r"\$", "")
        cleaned = cleaned.replace(r"\#", "")
        cleaned = cleaned.replace(r"\_", "")
        cleaned = cleaned.replace(r"\{", "")
        cleaned = cleaned.replace(r"\}", "")

        # After removing all escape sequences, none of the special chars
        # should remain
        for char in _LATEX_SPECIAL_CHARS:
            assert char not in cleaned, (
                f"Unescaped '{char}' found in output.\n"
                f"Input: {text!r}\n"
                f"Escaped: {escaped!r}\n"
                f"Cleaned: {cleaned!r}"
            )

    @given(text=_text_with_special_chars)
    @settings(max_examples=100)
    def test_markdown_conversion_escapes_special_chars_in_text(
        self,
        text: str,
    ) -> None:
        """Special chars passed through by markdown are escaped in LaTeX output.

        Note: Markdown itself interprets some characters as formatting
        (backslash as escape, underscores/asterisks as bold/italic, # as heading).
        This test verifies that characters which survive markdown parsing
        into text tokens are properly escaped in the LaTeX output.
        """
        # Use only characters that markdown won't interpret as formatting.
        # Filter out: \ (escape), _ (bold/italic), # (heading when at line start),
        # * (bold/italic), and sequences that form markdown syntax.
        safe_chars = "&%$~^"
        safe_text = "".join(c for c in text if c in safe_chars or c.isalnum() or c == " ")
        if not safe_text.strip() or not any(c in safe_text for c in safe_chars):
            return  # Skip if no testable special chars remain

        markdown = f"prefix {safe_text} suffix"
        latex_output = convert_markdown_to_latex(markdown)

        # Verify the escaped form of the text appears in output
        escaped_text = _escape_latex_text(safe_text)
        assert escaped_text in latex_output, (
            f"Escaped text not found in LaTeX output.\n"
            f"Input text: {safe_text!r}\n"
            f"Expected escaped: {escaped_text!r}\n"
            f"LaTeX output: {latex_output!r}"
        )


# ---------------------------------------------------------------------------
# Strategies for BibTeX property tests
# ---------------------------------------------------------------------------

from patent_system.export.latex_exporter import (
    escape_latex,
    generate_bibtex,
    generate_bibtex_entry,
    sanitize_citation_key,
)

# Strategy for generating reference titles (non-empty alphanumeric words)
_ref_title = st.text(
    alphabet=st.characters(categories=("L", "N", "Zs"), min_codepoint=32, max_codepoint=122),
    min_size=1,
    max_size=60,
).filter(lambda s: s.strip() != "" and re.search(r"[a-zA-Z0-9]", s) is not None)

# Strategy for optional string fields
_optional_str = st.one_of(st.none(), st.text(min_size=1, max_size=50).filter(lambda s: s.strip() != ""))

# Strategy for a single reference dict
_reference_dict = st.fixed_dictionaries(
    {
        "title": _ref_title,
        "source": st.text(min_size=1, max_size=30).filter(lambda s: s.strip() != ""),
    },
    optional={
        "patent_number": _optional_str,
        "doi": _optional_str,
        "url": st.one_of(st.none(), st.builds(lambda s: f"https://example.com/{s}", st.text(
            alphabet=st.characters(categories=("L", "N")), min_size=1, max_size=20
        ))),
        "relevance_score": st.one_of(st.none(), st.integers(min_value=0, max_value=100)),
        "abstract": _optional_str,
    },
).map(lambda d: {k: v for k, v in d.items() if v is not None})

# Strategy for a list of reference dicts (for testing uniqueness and round-trip)
_reference_list = st.lists(_reference_dict, min_size=1, max_size=10)

# Strategy for references with duplicate titles (for uniqueness testing)
_duplicate_title_refs = st.builds(
    lambda title, sources: [
        {"title": title, "source": src} for src in sources
    ],
    title=_ref_title,
    sources=st.lists(
        st.text(min_size=1, max_size=20).filter(lambda s: s.strip() != ""),
        min_size=2,
        max_size=5,
    ),
)


# ---------------------------------------------------------------------------
# Property 4: BibTeX entry type determined by reference fields
# Feature: latex-export, Property 4: entry type
# ---------------------------------------------------------------------------


class TestBibtexEntryType:
    """Property 4: BibTeX entry type determined by reference fields.

    For any reference dict, the generated BibTeX entry type SHALL be
    @article when a doi field is present (and no patent_number), and
    @misc otherwise.

    **Validates: Requirements 2.1, 2.2, 2.3**
    """

    @given(ref=_reference_dict)
    @settings(max_examples=100)
    def test_entry_type_matches_field_presence(self, ref: dict) -> None:
        """Entry type is @article for doi-only refs, @misc otherwise."""
        _, entry = generate_bibtex_entry(ref, 1)

        has_doi = bool(ref.get("doi"))
        has_patent = bool(ref.get("patent_number"))

        if has_doi and not has_patent:
            assert entry.startswith("@article{"), (
                f"Expected @article for ref with doi={ref.get('doi')!r} "
                f"and no patent_number, got: {entry[:30]}"
            )
        else:
            assert entry.startswith("@misc{"), (
                f"Expected @misc for ref with doi={ref.get('doi')!r}, "
                f"patent_number={ref.get('patent_number')!r}, got: {entry[:30]}"
            )


# ---------------------------------------------------------------------------
# Property 5: BibTeX citation keys are unique
# Feature: latex-export, Property 5: unique keys
# ---------------------------------------------------------------------------


class TestBibtexUniqueKeys:
    """Property 5: BibTeX citation keys are unique.

    For any list of reference dicts (including those with duplicate titles),
    all generated BibTeX citation keys SHALL be unique within the output.

    **Validates: Requirements 2.4**
    """

    @given(refs=_reference_list)
    @settings(max_examples=100)
    def test_all_keys_unique_in_generated_bibtex(self, refs: list[dict]) -> None:
        """All citation keys in generated BibTeX output are unique."""
        result = generate_bibtex(refs)
        if not result:
            return

        # Extract all citation keys from the output
        keys = re.findall(r"@(?:misc|article)\{([^,]+),", result)
        assert len(keys) == len(refs), (
            f"Expected {len(refs)} entries, found {len(keys)} keys"
        )
        assert len(keys) == len(set(keys)), (
            f"Duplicate keys found: {keys}"
        )

    @given(refs=_duplicate_title_refs)
    @settings(max_examples=100)
    def test_duplicate_titles_produce_unique_keys(self, refs: list[dict]) -> None:
        """References with identical titles still get unique keys."""
        result = generate_bibtex(refs)
        if not result:
            return

        keys = re.findall(r"@(?:misc|article)\{([^,]+),", result)
        assert len(keys) == len(refs), (
            f"Expected {len(refs)} entries, found {len(keys)} keys"
        )
        assert len(keys) == len(set(keys)), (
            f"Duplicate keys found among same-title refs: {keys}"
        )


# ---------------------------------------------------------------------------
# Property 6: BibTeX generation round-trip preserves reference metadata
# Feature: latex-export, Property 6: round-trip
# ---------------------------------------------------------------------------


class TestBibtexRoundTrip:
    """Property 6: BibTeX generation round-trip preserves reference metadata.

    For any valid list of reference dicts, generating BibTeX and then parsing
    the output back SHALL produce entries where the title, source-type mapping,
    and optional fields (url, doi, patent_number) are recoverable from the
    BibTeX text.

    **Validates: Requirements 2.5, 2.6, 2.7, 2.8, 2.9**
    """

    @given(ref=_reference_dict)
    @settings(max_examples=100)
    def test_title_recoverable_from_bibtex(self, ref: dict) -> None:
        """The title field value is present in the generated BibTeX entry."""
        _, entry = generate_bibtex_entry(ref, 1)
        title = ref.get("title", "Untitled")
        escaped_title = escape_latex(title)
        assert f"title = {{{escaped_title}}}" in entry, (
            f"Title not found in entry.\n"
            f"Expected escaped title: {escaped_title!r}\n"
            f"Entry: {entry!r}"
        )

    @given(ref=_reference_dict)
    @settings(max_examples=100)
    def test_doi_recoverable_from_bibtex(self, ref: dict) -> None:
        """When doi is present, it appears in the generated BibTeX entry."""
        _, entry = generate_bibtex_entry(ref, 1)
        doi = ref.get("doi")
        if doi:
            escaped_doi = escape_latex(doi)
            assert f"doi = {{{escaped_doi}}}" in entry, (
                f"DOI not found in entry.\n"
                f"Expected: doi = {{{escaped_doi}}}\n"
                f"Entry: {entry!r}"
            )
        else:
            assert "doi = " not in entry

    @given(ref=_reference_dict)
    @settings(max_examples=100)
    def test_url_recoverable_from_bibtex(self, ref: dict) -> None:
        """When url is present, it appears in the generated BibTeX entry."""
        _, entry = generate_bibtex_entry(ref, 1)
        url = ref.get("url")
        if url:
            assert f"url = {{{url}}}" in entry, (
                f"URL not found in entry.\n"
                f"Expected: url = {{{url}}}\n"
                f"Entry: {entry!r}"
            )
        else:
            assert "url = " not in entry

    @given(ref=_reference_dict)
    @settings(max_examples=100)
    def test_patent_number_recoverable_from_bibtex(self, ref: dict) -> None:
        """When patent_number is present, it appears in the note field (escaped)."""
        _, entry = generate_bibtex_entry(ref, 1)
        patent_number = ref.get("patent_number")
        if patent_number:
            # The patent number is embedded in the note field with LaTeX escaping
            escaped_patent = escape_latex(patent_number)
            assert f"Patent Number: {escaped_patent}" in entry, (
                f"Patent number not found in entry.\n"
                f"Expected 'Patent Number: {escaped_patent}' in note field.\n"
                f"Entry: {entry!r}"
            )
        else:
            assert "Patent Number:" not in entry

    @given(ref=_reference_dict)
    @settings(max_examples=100)
    def test_entry_type_matches_fields_roundtrip(self, ref: dict) -> None:
        """Entry type in output matches the expected type based on fields."""
        _, entry = generate_bibtex_entry(ref, 1)

        has_doi = bool(ref.get("doi"))
        has_patent = bool(ref.get("patent_number"))

        if has_doi and not has_patent:
            assert "@article{" in entry
        else:
            assert "@misc{" in entry


# ---------------------------------------------------------------------------
# Strategies for LaTeXExporter property tests (Task 3.8)
# ---------------------------------------------------------------------------

from pathlib import Path
import tempfile

from patent_system.export.latex_exporter import LaTeXExporter

# Non-empty text strategy for claims/description
_nonempty_text = st.text(
    alphabet=st.characters(categories=("L", "N", "Zs"), min_codepoint=32, max_codepoint=122),
    min_size=1,
    max_size=200,
).filter(lambda s: s.strip() != "")

# Strategy for workflow step keys (subset of canonical order, excluding patent_draft)
_VALID_STEP_KEYS = [
    "initial_idea", "claims_drafting", "prior_art_search",
    "novelty_analysis", "consistency_review", "market_potential",
    "legal_clarification", "disclosure_summary",
]

_workflow_steps_strategy = st.dictionaries(
    keys=st.sampled_from(_VALID_STEP_KEYS),
    values=_nonempty_text,
    min_size=1,
    max_size=5,
)

# Strategy for chat history messages
_chat_message = st.fixed_dictionaries({
    "role": st.sampled_from(["user", "assistant"]),
    "message": _nonempty_text,
})

_chat_history_strategy = st.lists(_chat_message, min_size=1, max_size=5)

# Strategy for file extensions (for template listing test)
_file_extension = st.sampled_from([".tex", ".docx", ".txt", ".pdf", ".bib", ".cls"])

_filename = st.builds(
    lambda name, ext: name + ext,
    name=st.text(
        alphabet=st.characters(categories=("L", "N"), min_codepoint=97, max_codepoint=122),
        min_size=1,
        max_size=10,
    ).filter(lambda s: s.strip() != ""),
    ext=_file_extension,
)


# ---------------------------------------------------------------------------
# Property 1: Export produces a .tex file
# Feature: latex-export, Property 1: tex file produced
# ---------------------------------------------------------------------------


class TestExportProducesTexFile:
    """Property 1: Export produces a .tex file.

    For any valid non-empty claims string and non-empty description string,
    calling export() SHALL produce a .tex file at the specified output_path
    that is non-empty and contains LaTeX content.

    **Validates: Requirements 1.2**
    """

    @given(claims=_nonempty_text, description=_nonempty_text)
    @settings(max_examples=100)
    def test_export_produces_nonempty_tex_file(self, claims: str, description: str) -> None:
        """Export always produces a non-empty .tex file."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            exporter = LaTeXExporter(template_dir=tmp_path)
            output = tmp_path / "output.tex"
            result = exporter.export(claims=claims, description=description, output_path=output)
            assert result == output
            assert output.exists()
            content = output.read_text()
            assert len(content) > 0


# ---------------------------------------------------------------------------
# Property 2: Export with references produces co-located .bib file
# Feature: latex-export, Property 2: bib co-located
# ---------------------------------------------------------------------------


class TestExportWithRefsBib:
    """Property 2: Export with references produces co-located .bib file.

    For any valid non-empty claims, description, and non-empty list of
    reference dicts, calling export() SHALL produce both a .tex file at
    output_path and a .bib file in the same directory with the same base name.

    **Validates: Requirements 1.3**
    """

    @given(claims=_nonempty_text, description=_nonempty_text, refs=_reference_list)
    @settings(max_examples=100)
    def test_bib_file_colocated_with_tex(
        self, claims: str, description: str, refs: list[dict]
    ) -> None:
        """A .bib file is produced alongside the .tex file when references exist."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            exporter = LaTeXExporter(template_dir=tmp_path)
            output = tmp_path / "patent.tex"
            exporter.export(
                claims=claims, description=description,
                output_path=output, references=refs,
            )
            assert output.exists()
            bib_path = tmp_path / "patent.bib"
            assert bib_path.exists()
            assert bib_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# Property 3: Export without references excludes bibliography commands
# Feature: latex-export, Property 3: no bib without refs
# ---------------------------------------------------------------------------


class TestExportNoBibWithoutRefs:
    """Property 3: Export without references excludes bibliography commands.

    For any valid non-empty claims and description, calling export() without
    references SHALL produce a .tex file that does not contain \\addbibresource.

    **Validates: Requirements 1.4**
    """

    @given(claims=_nonempty_text, description=_nonempty_text)
    @settings(max_examples=100)
    def test_no_addbibresource_without_refs(self, claims: str, description: str) -> None:
        """No \\addbibresource command when no references provided."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            exporter = LaTeXExporter(template_dir=tmp_path)
            output = tmp_path / "output.tex"
            exporter.export(claims=claims, description=description, output_path=output)
            content = output.read_text()
            assert r"\addbibresource" not in content


# ---------------------------------------------------------------------------
# Property 9: All template placeholders are replaced after export
# Feature: latex-export, Property 9: placeholders replaced
# ---------------------------------------------------------------------------


class TestPlaceholdersReplaced:
    """Property 9: All template placeholders are replaced after export.

    For any template containing all placeholder markers and any valid export
    content, the output .tex file SHALL contain none of the original
    %%...%% placeholder markers.

    **Validates: Requirements 4.2, 4.3, 4.4, 4.5, 4.6, 4.7**
    """

    @given(
        claims=_nonempty_text,
        description=_nonempty_text,
        refs=_reference_list,
        chat=_chat_history_strategy,
        steps=_workflow_steps_strategy,
    )
    @settings(max_examples=100)
    def test_no_placeholder_markers_in_output(
        self,
        claims: str,
        description: str,
        refs: list[dict],
        chat: list[dict],
        steps: dict[str, str],
    ) -> None:
        """No %%...%% markers remain in the exported .tex file."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # Use a template with all placeholders
            template_content = (
                "\\documentclass{article}\n"
                "%%REFERENCES%%\n"
                "\\begin{document}\n"
                "%%CLAIMS%%\n%%DESCRIPTION%%\n"
                "%%WORKFLOW_STEPS%%\n%%CHAT_LOG%%\n"
                "%%BIBLIOGRAPHY_FILE%%\n"
                "\\end{document}\n"
            )
            tpl_dir = tmp_path / "templates"
            tpl_dir.mkdir()
            (tpl_dir / "full.tex").write_text(template_content)

            exporter = LaTeXExporter(template_dir=tpl_dir, template_name="full.tex")
            output = tmp_path / "output.tex"
            exporter.export(
                claims=claims, description=description, output_path=output,
                references=refs, chat_history=chat, workflow_steps=steps,
            )
            content = output.read_text()
            assert "%%" not in content


# ---------------------------------------------------------------------------
# Property 10: Missing placeholders in template are tolerated
# Feature: latex-export, Property 10: missing placeholders tolerated
# ---------------------------------------------------------------------------


class TestMissingPlaceholdersTolerated:
    """Property 10: Missing placeholders in template are tolerated.

    For any template that is missing one or more %%...%% placeholders,
    calling export() SHALL succeed without error and produce a valid .tex file.

    **Validates: Requirements 4.9**
    """

    @given(
        claims=_nonempty_text,
        description=_nonempty_text,
        placeholders=st.lists(
            st.sampled_from([
                "%%CLAIMS%%", "%%DESCRIPTION%%", "%%WORKFLOW_STEPS%%",
                "%%REFERENCES%%", "%%CHAT_LOG%%", "%%BIBLIOGRAPHY_FILE%%",
            ]),
            min_size=0,
            max_size=3,
            unique=True,
        ),
    )
    @settings(max_examples=100)
    def test_export_succeeds_with_partial_template(
        self, claims: str, description: str, placeholders: list[str]
    ) -> None:
        """Export succeeds even when template has only a subset of placeholders."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # Build a template with only the selected placeholders
            template_content = "\\documentclass{article}\n\\begin{document}\n"
            template_content += "\n".join(placeholders)
            template_content += "\n\\end{document}\n"

            tpl_dir = tmp_path / "templates"
            tpl_dir.mkdir()
            (tpl_dir / "partial.tex").write_text(template_content)

            exporter = LaTeXExporter(template_dir=tpl_dir, template_name="partial.tex")
            output = tmp_path / "output.tex"
            # Should not raise
            exporter.export(
                claims=claims, description=description, output_path=output,
                references=[{"title": "Ref", "source": "Src"}],
                chat_history=[{"role": "user", "message": "Hi"}],
                workflow_steps={"initial_idea": "Idea"},
            )
            assert output.exists()
            assert output.stat().st_size > 0


# ---------------------------------------------------------------------------
# Property 11: list_available_templates returns exactly the .tex files
# Feature: latex-export, Property 11: template listing
# ---------------------------------------------------------------------------


class TestTemplateListingProperty:
    """Property 11: list_available_templates returns exactly the .tex files.

    For any directory containing a mix of .tex, .docx, .txt, and other files,
    list_available_templates() SHALL return exactly the .tex filenames and no others.

    **Validates: Requirements 1.6**
    """

    @given(filenames=st.lists(_filename, min_size=1, max_size=10, unique=True))
    @settings(max_examples=100)
    def test_returns_only_tex_files(self, filenames: list[str]) -> None:
        """Only .tex files are returned by list_available_templates()."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            for fname in filenames:
                (tmp_path / fname).write_text("content")

            exporter = LaTeXExporter(template_dir=tmp_path)
            result = exporter.list_available_templates()

            expected = sorted(f for f in filenames if f.endswith(".tex"))
            assert result == expected


# ---------------------------------------------------------------------------
# Property 12: Workflow steps appear in canonical order excluding patent_draft
# Feature: latex-export, Property 12: workflow order
# ---------------------------------------------------------------------------


class TestWorkflowOrderProperty:
    """Property 12: Workflow steps appear in canonical order excluding patent_draft.

    For any non-empty subset of workflow steps, the exported .tex file SHALL
    contain those steps as sections in the order defined by WORKFLOW_STEP_ORDER,
    and SHALL NOT include the patent_draft step.

    **Validates: Requirements 7.2**
    """

    @given(steps=_workflow_steps_strategy)
    @settings(max_examples=100)
    def test_steps_in_canonical_order(self, steps: dict[str, str]) -> None:
        """Workflow steps appear in canonical order in the output."""
        from patent_system.export.docx_exporter import (
            STEP_DISPLAY_NAMES,
            WORKFLOW_STEP_ORDER,
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            exporter = LaTeXExporter(template_dir=tmp_path)
            output = tmp_path / "output.tex"
            exporter.export(
                claims="Claims", description="Desc",
                output_path=output, workflow_steps=steps,
            )
            content = output.read_text()

            # Collect positions of step sections that appear in the output
            positions: list[tuple[int, str]] = []
            for step_key in WORKFLOW_STEP_ORDER:
                if step_key == "patent_draft":
                    continue
                if step_key in steps and steps[step_key].strip():
                    display_name = STEP_DISPLAY_NAMES[step_key]
                    section_marker = f"\\section{{{display_name}}}"
                    if section_marker in content:
                        positions.append((content.index(section_marker), step_key))

            # Verify they appear in canonical order
            step_keys_in_order = [key for _, key in sorted(positions)]
            canonical_filtered = [
                k for k in WORKFLOW_STEP_ORDER
                if k != "patent_draft" and k in steps and steps[k].strip()
            ]
            assert step_keys_in_order == canonical_filtered

    @given(
        steps=st.fixed_dictionaries({
            "patent_draft": _nonempty_text,
            "initial_idea": _nonempty_text,
        })
    )
    @settings(max_examples=100)
    def test_patent_draft_excluded(self, steps: dict[str, str]) -> None:
        """patent_draft step is never included in the output."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            exporter = LaTeXExporter(template_dir=tmp_path)
            output = tmp_path / "output.tex"
            exporter.export(
                claims="Claims", description="Desc",
                output_path=output, workflow_steps=steps,
            )
            content = output.read_text()
            assert r"\section{Patent Draft}" not in content


# ---------------------------------------------------------------------------
# Property 13: Chat history includes bold role labels
# Feature: latex-export, Property 13: chat labels
# ---------------------------------------------------------------------------


class TestChatLabelsProperty:
    """Property 13: Chat history includes bold role labels.

    For any non-empty chat history list with role and message fields,
    the exported .tex file SHALL contain \\textbf{You:} for user messages
    and \\textbf{Assistant:} for assistant messages.

    **Validates: Requirements 7.3**
    """

    @given(chat=_chat_history_strategy)
    @settings(max_examples=100)
    def test_chat_has_bold_role_labels(self, chat: list[dict]) -> None:
        """Chat messages have bold role labels in the output."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            exporter = LaTeXExporter(template_dir=tmp_path)
            output = tmp_path / "output.tex"
            exporter.export(
                claims="Claims", description="Desc",
                output_path=output, chat_history=chat,
            )
            content = output.read_text()

            for msg in chat:
                role = msg["role"]
                if role == "user":
                    assert r"\textbf{You:}" in content
                else:
                    assert r"\textbf{Assistant:}" in content
