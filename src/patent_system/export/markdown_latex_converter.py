"""Markdown-to-LaTeX converter using mistune AST parsing.

Parses markdown text into an AST via mistune, then walks the tree to produce
LaTeX-formatted strings. Mirrors the existing ``markdown_converter.py`` but
targets LaTeX output instead of python-docx elements.
"""

from __future__ import annotations

import logging
from typing import Any

import mistune

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mistune markdown parser (AST mode with table plugin)
# ---------------------------------------------------------------------------
_md_parser = mistune.create_markdown(renderer="ast", plugins=["table"])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def convert_markdown_to_latex(markdown_text: str) -> str:
    """Parse markdown text and return LaTeX-formatted string.

    Uses mistune AST parsing to walk the token tree and produce
    LaTeX commands for headings, inline formatting, lists, tables,
    code blocks, blockquotes, and hyperlinks.

    Args:
        markdown_text: Markdown-formatted string to convert.

    Returns:
        LaTeX-formatted string.
    """
    if not markdown_text or not markdown_text.strip():
        return ""

    tokens: list[dict[str, Any]] = _md_parser(markdown_text)
    return _walk_tokens(tokens, list_depth=0)


# ---------------------------------------------------------------------------
# LaTeX special character escaping
# ---------------------------------------------------------------------------

# Sentinel used during escaping to avoid double-replacement.
_BACKSLASH_SENTINEL = "\x00BACKSLASH\x00"
_TILDE_SENTINEL = "\x00TILDE\x00"
_CARET_SENTINEL = "\x00CARET\x00"


def _escape_latex_text(text: str) -> str:
    """Escape special LaTeX characters in plain text segments.

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
# AST walker
# ---------------------------------------------------------------------------


def _walk_tokens(tokens: list[dict[str, Any]], list_depth: int = 0) -> str:
    """Recursively walk AST tokens and produce LaTeX string."""
    parts: list[str] = []

    for token in tokens:
        tok_type = token.get("type", "")

        if tok_type == "heading":
            parts.append(_handle_heading(token))

        elif tok_type == "paragraph":
            parts.append(_handle_paragraph(token))

        elif tok_type == "list":
            parts.append(_handle_list(token, list_depth=list_depth))

        elif tok_type == "table":
            parts.append(_handle_table(token))

        elif tok_type == "block_quote":
            parts.append(_handle_blockquote(token))

        elif tok_type == "block_code":
            parts.append(_handle_block_code(token))

        elif tok_type == "thematic_break":
            parts.append(r"\noindent\rule{\textwidth}{0.4pt}" + "\n")

        elif tok_type == "blank_line":
            continue

        else:
            # Fallback: if the token has children, recurse
            children = token.get("children")
            if children and isinstance(children, list):
                parts.append(_walk_tokens(children, list_depth=list_depth))

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Block-level handlers
# ---------------------------------------------------------------------------


def _handle_heading(token: dict[str, Any]) -> str:
    """Convert a heading token to a LaTeX section command (unnumbered).

    Uses starred variants (\\section*{}, \\subsection*{}, \\subsubsection*{})
    because headings from markdown content appear inside workflow steps and
    chat messages where LaTeX auto-numbering would clash with numbers
    already present in the text (e.g. "3. Analysis").

    Mapping:
        # → \\section*{}
        ## → \\subsection*{}
        ### → \\subsubsection*{}
    """
    level = token.get("attrs", {}).get("level", 1)
    children = token.get("children", [])
    content = _render_inline(children)

    if level == 1:
        return f"\\section*{{{content}}}"
    elif level == 2:
        return f"\\subsection*{{{content}}}"
    else:
        return f"\\subsubsection*{{{content}}}"


def _handle_paragraph(token: dict[str, Any]) -> str:
    """Convert a paragraph token to LaTeX text with inline formatting."""
    children = token.get("children", [])
    content = _render_inline(children)
    return content + "\n"


def _handle_list(token: dict[str, Any], *, list_depth: int = 0) -> str:
    """Convert a list token to an itemize or enumerate environment."""
    attrs = token.get("attrs", {})
    ordered = attrs.get("ordered", False)
    env = "enumerate" if ordered else "itemize"

    lines: list[str] = []
    lines.append(f"\\begin{{{env}}}")

    for item in token.get("children", []):
        if item.get("type") != "list_item":
            continue
        item_parts: list[str] = []
        for child in item.get("children", []):
            child_type = child.get("type", "")
            if child_type == "list":
                # Nested list
                nested = _handle_list(child, list_depth=list_depth + 1)
                item_parts.append(nested)
            elif child_type in ("block_text", "paragraph"):
                inline_content = _render_inline(child.get("children", []))
                item_parts.append(f"  \\item {inline_content}")
            else:
                # Fallback for other block types inside list items
                sub_children = child.get("children")
                if sub_children and isinstance(sub_children, list):
                    inline_content = _render_inline(sub_children)
                    item_parts.append(f"  \\item {inline_content}")

        lines.extend(item_parts)

    lines.append(f"\\end{{{env}}}")
    return "\n".join(lines)


def _handle_table(token: dict[str, Any]) -> str:
    """Convert a table token to a tabular environment."""
    # Collect header and body rows
    header_cells: list[list[dict[str, Any]]] = []
    body_rows: list[list[list[dict[str, Any]]]] = []

    for child in token.get("children", []):
        child_type = child.get("type", "")
        if child_type == "table_head":
            header_cells = [
                cell.get("children", [])
                for cell in child.get("children", [])
                if cell.get("type") == "table_cell"
            ]
        elif child_type == "table_body":
            for row in child.get("children", []):
                if row.get("type") == "table_row":
                    row_cells = [
                        cell.get("children", [])
                        for cell in row.get("children", [])
                        if cell.get("type") == "table_cell"
                    ]
                    body_rows.append(row_cells)

    num_cols = len(header_cells) if header_cells else (
        len(body_rows[0]) if body_rows else 0
    )

    if num_cols == 0:
        return ""

    col_spec = "|".join(["l"] * num_cols)
    lines: list[str] = []
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\hline")

    # Header row
    if header_cells:
        cells_text = [
            f"\\textbf{{{_render_inline(cell)}}}" for cell in header_cells
        ]
        lines.append(" & ".join(cells_text) + " \\\\")
        lines.append("\\hline")

    # Body rows
    for row_cells in body_rows:
        cells_text = [_render_inline(cell) for cell in row_cells]
        lines.append(" & ".join(cells_text) + " \\\\")

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def _handle_blockquote(token: dict[str, Any]) -> str:
    """Convert a blockquote token to a quote environment."""
    children = token.get("children", [])
    inner = _walk_tokens(children, list_depth=0)
    return f"\\begin{{quote}}\n{inner}\n\\end{{quote}}"


def _handle_block_code(token: dict[str, Any]) -> str:
    """Convert a fenced code block to a verbatim environment."""
    raw = token.get("raw", "")
    # Strip trailing newline for cleaner output
    raw = raw.rstrip("\n")
    return f"\\begin{{verbatim}}\n{raw}\n\\end{{verbatim}}"


# ---------------------------------------------------------------------------
# Inline-level handler
# ---------------------------------------------------------------------------


def _render_inline(
    children: list[dict[str, Any]],
    bold: bool = False,
    italic: bool = False,
) -> str:
    """Recursively render inline children to LaTeX string.

    Handles bold (\\textbf{}), italic (\\textit{}), code (\\texttt{}),
    links (\\href{}{}), and softbreak/linebreak.

    Args:
        children: List of inline AST tokens.
        bold: Whether inherited bold formatting is active.
        italic: Whether inherited italic formatting is active.

    Returns:
        LaTeX-formatted inline string.
    """
    parts: list[str] = []

    for child in children:
        child_type = child.get("type", "")

        if child_type == "text":
            text = _escape_latex_text(child.get("raw", ""))
            parts.append(text)

        elif child_type == "strong":
            inner = _render_inline(child.get("children", []), bold=True, italic=italic)
            parts.append(f"\\textbf{{{inner}}}")

        elif child_type == "emphasis":
            inner = _render_inline(child.get("children", []), bold=bold, italic=True)
            parts.append(f"\\textit{{{inner}}}")

        elif child_type == "codespan":
            raw = child.get("raw", "")
            parts.append(f"\\texttt{{{_escape_latex_text(raw)}}}")

        elif child_type == "link":
            url = child.get("attrs", {}).get("url", "")
            link_children = child.get("children", [])
            link_text = _render_inline(link_children)
            parts.append(f"\\href{{{url}}}{{{link_text}}}")

        elif child_type == "softbreak":
            parts.append(" ")

        elif child_type == "linebreak":
            parts.append("\\\\\n")

        else:
            # Fallback: render raw text if present, or recurse children
            raw = child.get("raw")
            if raw:
                parts.append(_escape_latex_text(raw))
            sub_children = child.get("children")
            if sub_children and isinstance(sub_children, list):
                parts.append(
                    _render_inline(sub_children, bold=bold, italic=italic)
                )

    return "".join(parts)
