"""Unit tests for the Markdown-to-LaTeX converter.

Tests each markdown element type: headings, bold, italic, code, lists,
tables, links, blockquotes, empty input, and special characters.
"""

from __future__ import annotations

from patent_system.export.markdown_latex_converter import convert_markdown_to_latex


class TestEmptyInput:
    """Tests for empty/None input handling."""

    def test_empty_string_returns_empty(self) -> None:
        assert convert_markdown_to_latex("") == ""

    def test_whitespace_only_returns_empty(self) -> None:
        assert convert_markdown_to_latex("   \n  ") == ""


class TestHeadings:
    """Tests for heading conversion to section commands."""

    def test_h1_becomes_section(self) -> None:
        result = convert_markdown_to_latex("# Title")
        assert r"\section*{Title}" in result

    def test_h2_becomes_subsection(self) -> None:
        result = convert_markdown_to_latex("## Subtitle")
        assert r"\subsection*{Subtitle}" in result

    def test_h3_becomes_subsubsection(self) -> None:
        result = convert_markdown_to_latex("### Sub-subtitle")
        assert r"\subsubsection*{Sub-subtitle}" in result

    def test_heading_with_inline_formatting(self) -> None:
        result = convert_markdown_to_latex("# **Bold** Title")
        assert r"\section*{\textbf{Bold} Title}" in result


class TestBold:
    """Tests for bold text conversion."""

    def test_bold_text(self) -> None:
        result = convert_markdown_to_latex("**bold text**")
        assert r"\textbf{bold text}" in result

    def test_bold_within_paragraph(self) -> None:
        result = convert_markdown_to_latex("This is **bold** text.")
        assert r"This is \textbf{bold} text." in result


class TestItalic:
    """Tests for italic text conversion."""

    def test_italic_text(self) -> None:
        result = convert_markdown_to_latex("*italic text*")
        assert r"\textit{italic text}" in result

    def test_italic_within_paragraph(self) -> None:
        result = convert_markdown_to_latex("This is *italic* text.")
        assert r"This is \textit{italic} text." in result


class TestBoldItalic:
    """Tests for combined bold and italic."""

    def test_bold_italic(self) -> None:
        result = convert_markdown_to_latex("***bold italic***")
        assert "textbf" in result
        assert "textit" in result
        assert "bold italic" in result


class TestInlineCode:
    """Tests for inline code conversion."""

    def test_inline_code(self) -> None:
        result = convert_markdown_to_latex("`code`")
        assert r"\texttt{code}" in result

    def test_inline_code_with_special_chars(self) -> None:
        result = convert_markdown_to_latex("`a & b`")
        assert r"\texttt{a \& b}" in result


class TestCodeBlocks:
    """Tests for fenced code block conversion."""

    def test_code_block(self) -> None:
        md = "```\nprint('hello')\n```"
        result = convert_markdown_to_latex(md)
        assert r"\begin{verbatim}" in result
        assert r"\end{verbatim}" in result
        assert "print('hello')" in result

    def test_code_block_with_language(self) -> None:
        md = "```python\nx = 1\n```"
        result = convert_markdown_to_latex(md)
        assert r"\begin{verbatim}" in result
        assert "x = 1" in result


class TestUnorderedLists:
    """Tests for unordered list conversion."""

    def test_simple_unordered_list(self) -> None:
        md = "- Item 1\n- Item 2\n- Item 3"
        result = convert_markdown_to_latex(md)
        assert r"\begin{itemize}" in result
        assert r"\end{itemize}" in result
        assert r"\item Item 1" in result
        assert r"\item Item 2" in result
        assert r"\item Item 3" in result

    def test_unordered_list_with_formatting(self) -> None:
        md = "- **Bold** item\n- *Italic* item"
        result = convert_markdown_to_latex(md)
        assert r"\textbf{Bold}" in result
        assert r"\textit{Italic}" in result


class TestOrderedLists:
    """Tests for ordered list conversion."""

    def test_simple_ordered_list(self) -> None:
        md = "1. First\n2. Second\n3. Third"
        result = convert_markdown_to_latex(md)
        assert r"\begin{enumerate}" in result
        assert r"\end{enumerate}" in result
        assert r"\item First" in result
        assert r"\item Second" in result
        assert r"\item Third" in result


class TestNestedLists:
    """Tests for nested list conversion."""

    def test_nested_unordered_list(self) -> None:
        md = "- Item 1\n  - Sub-item 1\n  - Sub-item 2\n- Item 2"
        result = convert_markdown_to_latex(md)
        # Should have nested itemize environments
        assert result.count(r"\begin{itemize}") == 2
        assert result.count(r"\end{itemize}") == 2


class TestTables:
    """Tests for table conversion."""

    def test_simple_table(self) -> None:
        md = "| Header 1 | Header 2 |\n|---|---|\n| Cell 1 | Cell 2 |"
        result = convert_markdown_to_latex(md)
        assert r"\begin{tabular}" in result
        assert r"\end{tabular}" in result
        assert r"\hline" in result
        assert r"\textbf{Header 1}" in result
        assert "Cell 1" in result
        assert "Cell 2" in result

    def test_table_with_multiple_rows(self) -> None:
        md = "| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |"
        result = convert_markdown_to_latex(md)
        assert "1" in result
        assert "4" in result


class TestLinks:
    """Tests for hyperlink conversion."""

    def test_simple_link(self) -> None:
        result = convert_markdown_to_latex("[Click here](https://example.com)")
        assert r"\href{https://example.com}{Click here}" in result

    def test_link_in_paragraph(self) -> None:
        result = convert_markdown_to_latex("Visit [our site](https://example.com) today.")
        assert r"\href{https://example.com}{our site}" in result


class TestBlockquotes:
    """Tests for blockquote conversion."""

    def test_simple_blockquote(self) -> None:
        result = convert_markdown_to_latex("> This is a quote")
        assert r"\begin{quote}" in result
        assert r"\end{quote}" in result
        assert "This is a quote" in result

    def test_multiline_blockquote(self) -> None:
        md = "> Line 1\n> Line 2"
        result = convert_markdown_to_latex(md)
        assert r"\begin{quote}" in result
        assert "Line 1" in result
        assert "Line 2" in result


class TestSpecialCharacters:
    """Tests for LaTeX special character escaping."""

    def test_ampersand_escaped(self) -> None:
        result = convert_markdown_to_latex("A & B")
        assert r"A \& B" in result

    def test_percent_escaped(self) -> None:
        result = convert_markdown_to_latex("100% done")
        assert r"100\% done" in result

    def test_dollar_escaped(self) -> None:
        result = convert_markdown_to_latex("Price: $10")
        assert r"Price: \$10" in result

    def test_hash_escaped(self) -> None:
        result = convert_markdown_to_latex("Issue #42")
        assert r"Issue \#42" in result

    def test_underscore_escaped(self) -> None:
        result = convert_markdown_to_latex("file_name")
        assert r"file\_name" in result

    def test_braces_escaped(self) -> None:
        result = convert_markdown_to_latex("use {braces}")
        assert r"use \{braces\}" in result

    def test_tilde_escaped(self) -> None:
        result = convert_markdown_to_latex("approx ~5")
        assert r"\textasciitilde{}" in result

    def test_caret_escaped(self) -> None:
        result = convert_markdown_to_latex("x^2")
        assert r"\textasciicircum{}" in result

    def test_backslash_escaped(self) -> None:
        result = convert_markdown_to_latex(r"path\to\file")
        assert r"\textbackslash{}" in result

    def test_multiple_special_chars(self) -> None:
        result = convert_markdown_to_latex("A & B % C $ D")
        assert r"\&" in result
        assert r"\%" in result
        assert r"\$" in result
