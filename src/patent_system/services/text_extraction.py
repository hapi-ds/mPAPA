"""Shared text extraction utilities for PDF, DOCX, and plain text files.

Extracted from ``research_panel.py`` so that both local document uploads
and the ``FullTextDownloader`` service use the same extraction logic.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_bytes: bytes, filename: str = "document.pdf") -> str:
    """Extract text from PDF bytes using PyMuPDF with OCR fallback.

    For each page the function first attempts standard text extraction.
    If a page yields no text, it falls back to OCR via
    ``page.get_textpage_ocr``.

    Args:
        pdf_bytes: Raw PDF file content.
        filename: Original filename, used only for log messages.

    Returns:
        Extracted text with pages joined by newlines.  Returns an empty
        string when extraction fails entirely.
    """
    import fitz  # PyMuPDF

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text_parts: list[str] = []
        for page in doc:
            # Method 1: standard text extraction
            text = page.get_text("text")
            if text and text.strip():
                text_parts.append(text)
                continue
            # Method 2: OCR fallback for scanned pages
            try:
                tp = page.get_textpage_ocr(full=True)
                ocr_text = page.get_text("text", textpage=tp)
                if ocr_text and ocr_text.strip():
                    text_parts.append(ocr_text)
                    continue
            except Exception:
                pass
        doc.close()
        result = "\n".join(text_parts)
        if not result.strip():
            logger.warning(
                "PDF '%s' — no text extracted even with OCR",
                filename,
            )
        return result
    except Exception:
        logger.exception("Failed to parse PDF '%s'", filename)
        return ""


def extract_text_from_file(filename: str, content_bytes: bytes) -> str:
    """Extract text from uploaded file bytes (PDF, DOCX, or plain text).

    Dispatches to the appropriate extraction method based on the file
    extension:

    - ``.pdf`` → :func:`extract_text_from_pdf`
    - ``.docx`` → python-docx paragraph extraction
    - ``.txt`` → UTF-8 decode

    Args:
        filename: Original filename (used to determine file type).
        content_bytes: Raw file content.

    Returns:
        Extracted text content.  Returns an empty string for unsupported
        file types or when extraction fails.
    """
    lower = filename.lower()
    if lower.endswith(".txt"):
        return content_bytes.decode("utf-8", errors="replace")
    elif lower.endswith(".pdf"):
        return extract_text_from_pdf(content_bytes, filename)
    elif lower.endswith(".docx"):
        import io

        from docx import Document as DocxDoc

        doc = DocxDoc(io.BytesIO(content_bytes))
        return "\n".join(p.text for p in doc.paragraphs if p.text)
    return ""
