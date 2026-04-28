"""Unit tests for BibTeX generation in latex_exporter.py.

Tests entry types, LaTeX escaping, citation key generation, and field mapping.
"""

from __future__ import annotations

from patent_system.export.latex_exporter import (
    escape_latex,
    generate_bibtex,
    generate_bibtex_entry,
    sanitize_citation_key,
)


# ---------------------------------------------------------------------------
# escape_latex() tests
# ---------------------------------------------------------------------------


class TestEscapeLatex:
    """Tests for the escape_latex() function."""

    def test_plain_text_unchanged(self) -> None:
        """Plain text without special chars passes through unchanged."""
        assert escape_latex("Hello World") == "Hello World"

    def test_ampersand_escaped(self) -> None:
        assert escape_latex("A & B") == r"A \& B"

    def test_percent_escaped(self) -> None:
        assert escape_latex("100%") == r"100\%"

    def test_dollar_escaped(self) -> None:
        assert escape_latex("$100") == r"\$100"

    def test_hash_escaped(self) -> None:
        assert escape_latex("#1") == r"\#1"

    def test_underscore_escaped(self) -> None:
        assert escape_latex("a_b") == r"a\_b"

    def test_braces_escaped(self) -> None:
        assert escape_latex("{test}") == r"\{test\}"

    def test_tilde_escaped(self) -> None:
        assert escape_latex("~") == r"\textasciitilde{}"

    def test_caret_escaped(self) -> None:
        assert escape_latex("^") == r"\textasciicircum{}"

    def test_backslash_escaped(self) -> None:
        assert escape_latex("\\") == r"\textbackslash{}"

    def test_multiple_special_chars(self) -> None:
        """Multiple special chars in one string are all escaped."""
        result = escape_latex("A & B % C $ D")
        assert r"\&" in result
        assert r"\%" in result
        assert r"\$" in result

    def test_empty_string(self) -> None:
        assert escape_latex("") == ""


# ---------------------------------------------------------------------------
# sanitize_citation_key() tests
# ---------------------------------------------------------------------------


class TestSanitizeCitationKey:
    """Tests for the sanitize_citation_key() function."""

    def test_basic_title(self) -> None:
        """First 3 words lowercased with index appended."""
        result = sanitize_citation_key("Method for Wireless Communication", 1)
        assert result == "method_for_wireless_1"

    def test_fewer_than_three_words(self) -> None:
        """Titles with fewer than 3 words use what's available."""
        result = sanitize_citation_key("Wireless", 5)
        assert result == "wireless_5"

    def test_two_words(self) -> None:
        result = sanitize_citation_key("Wireless Method", 3)
        assert result == "wireless_method_3"

    def test_special_chars_in_title(self) -> None:
        """Non-alphanumeric chars are stripped; only words extracted."""
        result = sanitize_citation_key("A Novel (Method) for: Testing!", 2)
        assert result == "a_novel_method_2"

    def test_empty_title(self) -> None:
        """Empty title falls back to 'untitled'."""
        result = sanitize_citation_key("", 1)
        assert result == "untitled_1"

    def test_only_special_chars(self) -> None:
        """Title with no alphanumeric words falls back to 'untitled'."""
        result = sanitize_citation_key("!@#$%^&*()", 3)
        assert result == "untitled_3"

    def test_numeric_words_included(self) -> None:
        """Numeric words are included in the key."""
        result = sanitize_citation_key("US12345 Patent Application", 1)
        assert result == "us12345_patent_application_1"

    def test_case_insensitive(self) -> None:
        """Output is always lowercase."""
        result = sanitize_citation_key("UPPER Case TITLE", 1)
        assert result == "upper_case_title_1"


# ---------------------------------------------------------------------------
# generate_bibtex_entry() tests — entry types
# ---------------------------------------------------------------------------


class TestGenerateBibtexEntryTypes:
    """Tests for BibTeX entry type determination."""

    def test_patent_number_produces_misc(self) -> None:
        """Reference with patent_number → @misc."""
        ref = {"title": "Test Patent", "source": "Google Patents", "patent_number": "US123"}
        key, entry = generate_bibtex_entry(ref, 1)
        assert entry.startswith("@misc{")

    def test_doi_produces_article(self) -> None:
        """Reference with doi (no patent_number) → @article."""
        ref = {"title": "Test Article", "source": "ArXiv", "doi": "10.1234/test"}
        key, entry = generate_bibtex_entry(ref, 1)
        assert entry.startswith("@article{")

    def test_both_doi_and_patent_number_produces_misc(self) -> None:
        """Reference with both doi and patent_number → @misc (patent_number takes priority)."""
        ref = {
            "title": "Test",
            "source": "Mixed",
            "doi": "10.1234/test",
            "patent_number": "US123",
        }
        key, entry = generate_bibtex_entry(ref, 1)
        assert entry.startswith("@misc{")

    def test_neither_doi_nor_patent_produces_misc(self) -> None:
        """Reference with neither doi nor patent_number → @misc."""
        ref = {"title": "Web Resource", "source": "Web"}
        key, entry = generate_bibtex_entry(ref, 1)
        assert entry.startswith("@misc{")


# ---------------------------------------------------------------------------
# generate_bibtex_entry() tests — field mapping
# ---------------------------------------------------------------------------


class TestGenerateBibtexEntryFields:
    """Tests for BibTeX field mapping."""

    def test_title_field_present(self) -> None:
        ref = {"title": "My Title", "source": "Source"}
        _, entry = generate_bibtex_entry(ref, 1)
        assert "title = {My Title}" in entry

    def test_author_field_from_source(self) -> None:
        ref = {"title": "Test", "source": "Google Patents"}
        _, entry = generate_bibtex_entry(ref, 1)
        assert "author = {Google Patents}" in entry

    def test_doi_field_included(self) -> None:
        ref = {"title": "Test", "source": "ArXiv", "doi": "10.1234/abc"}
        _, entry = generate_bibtex_entry(ref, 1)
        assert "doi = {10.1234/abc}" in entry

    def test_url_field_included(self) -> None:
        ref = {"title": "Test", "source": "Web", "url": "https://example.com"}
        _, entry = generate_bibtex_entry(ref, 1)
        assert "url = {https://example.com}" in entry

    def test_abstract_field_included(self) -> None:
        ref = {"title": "Test", "source": "Source", "abstract": "An abstract."}
        _, entry = generate_bibtex_entry(ref, 1)
        assert "abstract = {An abstract.}" in entry

    def test_relevance_score_in_note(self) -> None:
        ref = {"title": "Test", "source": "Source", "relevance_score": 85}
        _, entry = generate_bibtex_entry(ref, 1)
        assert "note = {" in entry
        assert "Relevance Score: 85/100" in entry

    def test_patent_number_in_note(self) -> None:
        ref = {"title": "Test", "source": "Source", "patent_number": "US9876543"}
        _, entry = generate_bibtex_entry(ref, 1)
        assert "note = {" in entry
        assert "Patent Number: US9876543" in entry

    def test_both_patent_and_relevance_in_note(self) -> None:
        ref = {
            "title": "Test",
            "source": "Source",
            "patent_number": "EP123",
            "relevance_score": 70,
        }
        _, entry = generate_bibtex_entry(ref, 1)
        assert "Patent Number: EP123" in entry
        assert "Relevance Score: 70/100" in entry

    def test_optional_fields_omitted_when_absent(self) -> None:
        """Only title and author are always present; others are optional."""
        ref = {"title": "Minimal", "source": "Src"}
        _, entry = generate_bibtex_entry(ref, 1)
        assert "doi" not in entry
        assert "url" not in entry
        assert "abstract" not in entry
        assert "note" not in entry

    def test_special_chars_in_title_escaped(self) -> None:
        ref = {"title": "A & B: 100% Novel", "source": "Source"}
        _, entry = generate_bibtex_entry(ref, 1)
        assert r"\&" in entry
        assert r"\%" in entry

    def test_citation_key_returned(self) -> None:
        ref = {"title": "Method for Testing", "source": "Source"}
        key, _ = generate_bibtex_entry(ref, 1)
        assert key == "method_for_testing_1"

    def test_missing_title_defaults_to_untitled(self) -> None:
        ref = {"source": "Source"}
        key, entry = generate_bibtex_entry(ref, 1)
        assert "title = {Untitled}" in entry
        assert key == "untitled_1"


# ---------------------------------------------------------------------------
# generate_bibtex() tests
# ---------------------------------------------------------------------------


class TestGenerateBibtex:
    """Tests for the generate_bibtex() function."""

    def test_empty_list_returns_empty_string(self) -> None:
        assert generate_bibtex([]) == ""

    def test_single_reference(self) -> None:
        refs = [{"title": "Test Paper", "source": "ArXiv", "doi": "10.1/x"}]
        result = generate_bibtex(refs)
        assert "@article{test_paper_1," in result
        assert "title = {Test Paper}" in result

    def test_multiple_references(self) -> None:
        refs = [
            {"title": "First Paper", "source": "ArXiv", "doi": "10.1/a"},
            {"title": "Second Paper", "source": "PubMed", "doi": "10.2/b"},
        ]
        result = generate_bibtex(refs)
        assert "@article{first_paper_1," in result
        assert "@article{second_paper_2," in result

    def test_duplicate_titles_get_unique_keys(self) -> None:
        """References with identical titles get unique citation keys."""
        refs = [
            {"title": "Same Title", "source": "Source A"},
            {"title": "Same Title", "source": "Source B"},
        ]
        result = generate_bibtex(refs)
        # Both entries should be present
        assert result.count("@misc{") == 2
        # First key is same_title_1, second gets a uniqueness suffix
        assert "same_title_1," in result
        assert "same_title_2," in result

    def test_result_ends_with_newline(self) -> None:
        refs = [{"title": "Test", "source": "Src"}]
        result = generate_bibtex(refs)
        assert result.endswith("\n")

    def test_entries_separated_by_blank_line(self) -> None:
        refs = [
            {"title": "First", "source": "A"},
            {"title": "Second", "source": "B"},
        ]
        result = generate_bibtex(refs)
        assert "\n\n" in result


# ---------------------------------------------------------------------------
# LaTeXExporter class tests (Task 3.7)
# ---------------------------------------------------------------------------

from pathlib import Path

import pytest

from patent_system.export.latex_exporter import LaTeXExporter


class TestLaTeXExporterConstructor:
    """Tests for LaTeXExporter constructor."""

    def test_accepts_template_dir_and_name(self, tmp_path: Path) -> None:
        """Constructor accepts template_dir and template_name."""
        exporter = LaTeXExporter(template_dir=tmp_path, template_name="my_template.tex")
        assert exporter._template_dir == tmp_path
        assert exporter._template_name == "my_template.tex"

    def test_template_name_defaults_to_none(self, tmp_path: Path) -> None:
        """template_name defaults to None when not provided."""
        exporter = LaTeXExporter(template_dir=tmp_path)
        assert exporter._template_name is None


class TestLaTeXExporterExport:
    """Tests for LaTeXExporter.export() method."""

    def test_generates_tex_file(self, tmp_path: Path) -> None:
        """export() generates a .tex file at the specified path."""
        exporter = LaTeXExporter(template_dir=tmp_path)
        output = tmp_path / "output.tex"
        result = exporter.export(claims="Claim 1", description="Description text", output_path=output)
        assert result == output
        assert output.exists()
        content = output.read_text()
        assert "Claim 1" in content
        assert "Description text" in content

    def test_export_with_references_generates_bib(self, tmp_path: Path) -> None:
        """export() with references generates both .tex and .bib files."""
        exporter = LaTeXExporter(template_dir=tmp_path)
        output = tmp_path / "patent.tex"
        refs = [{"title": "Test Paper", "source": "ArXiv", "doi": "10.1/x"}]
        exporter.export(
            claims="Claims", description="Desc", output_path=output, references=refs
        )
        assert output.exists()
        bib_path = tmp_path / "patent.bib"
        assert bib_path.exists()
        bib_content = bib_path.read_text()
        assert "Test Paper" in bib_content

    def test_export_with_references_inserts_bibliography(self, tmp_path: Path) -> None:
        """export() with references inserts inline bibliography in .tex file."""
        exporter = LaTeXExporter(template_dir=tmp_path)
        output = tmp_path / "patent.tex"
        refs = [{"title": "Paper", "source": "Src"}]
        exporter.export(
            claims="Claims", description="Desc", output_path=output, references=refs
        )
        content = output.read_text()
        assert r"\begin{thebibliography}" in content
        assert r"\bibitem{" in content
        assert "Paper" in content

    def test_export_without_references_no_bib_commands(self, tmp_path: Path) -> None:
        """export() without references: no bibliography in output."""
        exporter = LaTeXExporter(template_dir=tmp_path)
        output = tmp_path / "patent.tex"
        exporter.export(claims="Claims", description="Desc", output_path=output)
        content = output.read_text()
        assert r"\begin{thebibliography}" not in content
        assert r"\bibitem{" not in content
        bib_path = tmp_path / "patent.bib"
        assert not bib_path.exists()

    def test_parent_directory_creation(self, tmp_path: Path) -> None:
        """export() creates nested parent directories."""
        exporter = LaTeXExporter(template_dir=tmp_path)
        output = tmp_path / "deep" / "nested" / "dir" / "patent.tex"
        exporter.export(claims="Claims", description="Desc", output_path=output)
        assert output.exists()

    def test_placeholder_replacement(self, tmp_path: Path) -> None:
        """All placeholders are replaced in the output."""
        # Create a custom template with all placeholders
        template_content = (
            "%%CLAIMS%%\n%%DESCRIPTION%%\n%%WORKFLOW_STEPS%%\n"
            "%%REFERENCES%%\n%%CHAT_LOG%%\n%%BIBLIOGRAPHY_FILE%%"
        )
        template_file = tmp_path / "templates" / "test.tex"
        template_file.parent.mkdir(parents=True, exist_ok=True)
        template_file.write_text(template_content)

        exporter = LaTeXExporter(
            template_dir=tmp_path / "templates", template_name="test.tex"
        )
        output = tmp_path / "output.tex"
        exporter.export(
            claims="My claims",
            description="My description",
            output_path=output,
            references=[{"title": "Ref", "source": "Src"}],
            chat_history=[{"role": "user", "message": "Hello"}],
            workflow_steps={"initial_idea": "My idea"},
        )
        content = output.read_text()
        assert "%%" not in content

    def test_missing_placeholder_tolerance(self, tmp_path: Path) -> None:
        """export() succeeds with a template missing some placeholders."""
        # Template with only CLAIMS placeholder
        template_content = "\\documentclass{article}\n\\begin{document}\n%%CLAIMS%%\n\\end{document}"
        template_file = tmp_path / "templates" / "partial.tex"
        template_file.parent.mkdir(parents=True, exist_ok=True)
        template_file.write_text(template_content)

        exporter = LaTeXExporter(
            template_dir=tmp_path / "templates", template_name="partial.tex"
        )
        output = tmp_path / "output.tex"
        # Should not raise even though most placeholders are missing
        exporter.export(
            claims="Claims text",
            description="Description text",
            output_path=output,
            references=[{"title": "Ref", "source": "Src"}],
            chat_history=[{"role": "user", "message": "Hi"}],
            workflow_steps={"initial_idea": "Idea"},
        )
        assert output.exists()
        content = output.read_text()
        assert "Claims text" in content


class TestLaTeXExporterTemplateFallback:
    """Tests for template fallback behavior."""

    def test_none_template_uses_default(self, tmp_path: Path) -> None:
        """template_name=None uses built-in default template."""
        exporter = LaTeXExporter(template_dir=tmp_path, template_name=None)
        output = tmp_path / "output.tex"
        exporter.export(claims="Claims", description="Desc", output_path=output)
        content = output.read_text()
        assert r"\documentclass[a4paper,11pt]{article}" in content
        assert "Patent Application" in content

    def test_missing_template_file_uses_default(self, tmp_path: Path) -> None:
        """Non-existent template file falls back to built-in default."""
        exporter = LaTeXExporter(template_dir=tmp_path, template_name="nonexistent.tex")
        output = tmp_path / "output.tex"
        exporter.export(claims="Claims", description="Desc", output_path=output)
        content = output.read_text()
        assert r"\documentclass[a4paper,11pt]{article}" in content

    def test_existing_template_is_used(self, tmp_path: Path) -> None:
        """When template file exists, it is used instead of default."""
        template_content = "CUSTOM TEMPLATE\n%%CLAIMS%%\n%%DESCRIPTION%%"
        template_file = tmp_path / "custom.tex"
        template_file.write_text(template_content)

        exporter = LaTeXExporter(template_dir=tmp_path, template_name="custom.tex")
        output = tmp_path / "output.tex"
        exporter.export(claims="My Claims", description="My Desc", output_path=output)
        content = output.read_text()
        assert "CUSTOM TEMPLATE" in content


class TestLaTeXExporterListTemplates:
    """Tests for list_available_templates()."""

    def test_returns_tex_files_only(self, tmp_path: Path) -> None:
        """list_available_templates() returns only .tex files."""
        (tmp_path / "a.tex").write_text("tex")
        (tmp_path / "b.tex").write_text("tex")
        (tmp_path / "c.docx").write_text("docx")
        (tmp_path / "d.txt").write_text("txt")

        exporter = LaTeXExporter(template_dir=tmp_path)
        templates = exporter.list_available_templates()
        assert sorted(templates) == ["a.tex", "b.tex"]

    def test_nonexistent_dir_returns_empty(self, tmp_path: Path) -> None:
        """Non-existent template directory returns empty list."""
        exporter = LaTeXExporter(template_dir=tmp_path / "nonexistent")
        assert exporter.list_available_templates() == []


class TestLaTeXExporterWorkflowSteps:
    """Tests for workflow step ordering."""

    def test_canonical_order(self, tmp_path: Path) -> None:
        """Workflow steps appear in canonical order."""
        exporter = LaTeXExporter(template_dir=tmp_path)
        output = tmp_path / "output.tex"
        steps = {
            "novelty_analysis": "Novelty content",
            "initial_idea": "Idea content",
            "claims_drafting": "Claims drafting content",
        }
        exporter.export(
            claims="Claims", description="Desc", output_path=output, workflow_steps=steps
        )
        content = output.read_text()
        # initial_idea should come before claims_drafting, which comes before novelty_analysis
        idx_idea = content.index("Initial Idea")
        idx_claims = content.index("Claims Drafting")
        idx_novelty = content.index("Novelty Analysis")
        assert idx_idea < idx_claims < idx_novelty

    def test_patent_draft_excluded(self, tmp_path: Path) -> None:
        """patent_draft step is excluded from output."""
        exporter = LaTeXExporter(template_dir=tmp_path)
        output = tmp_path / "output.tex"
        steps = {
            "initial_idea": "Idea",
            "patent_draft": "Draft content that should not appear as workflow step",
        }
        exporter.export(
            claims="Claims", description="Desc", output_path=output, workflow_steps=steps
        )
        content = output.read_text()
        assert "Initial Idea" in content
        # patent_draft should not appear as a \section
        assert r"\section{Patent Draft}" not in content


class TestLaTeXExporterChatHistory:
    """Tests for chat history formatting."""

    def test_bold_role_labels(self, tmp_path: Path) -> None:
        """Chat history uses bold role labels."""
        exporter = LaTeXExporter(template_dir=tmp_path)
        output = tmp_path / "output.tex"
        chat = [
            {"role": "user", "message": "Hello"},
            {"role": "assistant", "message": "Hi there"},
        ]
        exporter.export(
            claims="Claims", description="Desc", output_path=output, chat_history=chat
        )
        content = output.read_text()
        assert r"\textbf{You:}" in content
        assert r"\textbf{Assistant:}" in content
        assert "Hello" in content
        assert "Hi there" in content


# ---------------------------------------------------------------------------
# Default template smoke tests (Task 4.4)
# ---------------------------------------------------------------------------


class TestDefaultTemplateSmoke:
    """Smoke tests verifying the default LaTeX template file."""

    TEMPLATE_PATH = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "patent_system"
        / "export"
        / "templates"
        / "patent_template.tex"
    )

    def test_template_file_exists(self) -> None:
        """The default template file exists at the expected path."""
        assert self.TEMPLATE_PATH.exists(), (
            f"Template not found at {self.TEMPLATE_PATH}"
        )

    def test_contains_required_packages(self) -> None:
        """Template includes all required LaTeX packages."""
        content = self.TEMPLATE_PATH.read_text(encoding="utf-8")
        required_packages = [
            "geometry",
            "microtype",
            "parskip",
            "titlesec",
            "booktabs",
            "longtable",
            "verbatim",
            "hyperref",
            "iftex",
            "fontspec",
        ]
        for pkg in required_packages:
            assert pkg in content, f"Required package '{pkg}' not found in template"

    def test_contains_all_placeholders(self) -> None:
        """Template contains all required placeholder markers."""
        content = self.TEMPLATE_PATH.read_text(encoding="utf-8")
        required_placeholders = [
            "%%CLAIMS%%",
            "%%DESCRIPTION%%",
            "%%WORKFLOW_STEPS%%",
            "%%CHAT_LOG%%",
        ]
        for placeholder in required_placeholders:
            assert placeholder in content, (
                f"Required placeholder '{placeholder}' not found in template"
            )

    def test_structural_validity_when_placeholders_replaced(self) -> None:
        """Template is structurally valid LaTeX when placeholders are empty strings."""
        content = self.TEMPLATE_PATH.read_text(encoding="utf-8")
        # Replace all placeholders with empty strings
        placeholders = [
            "%%CLAIMS%%",
            "%%DESCRIPTION%%",
            "%%WORKFLOW_STEPS%%",
            "%%REFERENCES%%",
            "%%CHAT_LOG%%",
            "%%BIBLIOGRAPHY_FILE%%",
        ]
        for ph in placeholders:
            content = content.replace(ph, "")

        # Verify balanced document structure
        assert r"\begin{document}" in content, (
            r"Missing \begin{document} after placeholder replacement"
        )
        assert r"\end{document}" in content, (
            r"Missing \end{document} after placeholder replacement"
        )
        # Verify begin comes before end
        begin_idx = content.index(r"\begin{document}")
        end_idx = content.index(r"\end{document}")
        assert begin_idx < end_idx, (
            r"\begin{document} must come before \end{document}"
        )
