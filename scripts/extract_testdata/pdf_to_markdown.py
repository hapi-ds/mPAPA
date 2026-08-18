"""PDF-to-Markdown extraction pipeline for patent prosecution documents.

Uses pymupdf for text extraction from digital PDFs, and falls back to
vLLM (Qwen3.5-9B) for structuring/OCR of scanned pages.

Usage:
    # Single file
    uv run python -m scripts.extract_testdata.pdf_to_markdown \
        "testdata/Preservative Filter/US2019117738/KLueh et al_US2019117738A1_preservative removal_non final rejection.pdf"

    # Batch: all PDFs in a folder
    uv run python -m scripts.extract_testdata.pdf_to_markdown \
        --batch "testdata/Preservative Filter/US2019117738/"

    # All testdata
    uv run python -m scripts.extract_testdata.pdf_to_markdown --all
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import sys
import time
from pathlib import Path

import httpx
import pymupdf

from scripts.extract_testdata.config import ExtractionConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Think-tag stripping (Qwen3 thinking mode)
# ---------------------------------------------------------------------------

import re

_THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_UNCLOSED = re.compile(r"<think>.*", re.DOTALL)


def _strip_thinking(content: str) -> str:
    """Remove Qwen3 thinking blocks from LLM output.

    Handles:
    - Proper <think>...</think> pairs
    - Unclosed <think> (model started thinking but didn't close)
    - Multiple think blocks
    """
    if "<think>" not in content:
        return content

    # First: remove properly closed blocks
    content = _THINK_PATTERN.sub("", content)

    # Second: remove unclosed <think> (everything after it is thinking)
    if "<think>" in content:
        content = _THINK_UNCLOSED.sub("", content)

    return content.strip()


# Collapse runs of 10+ repeated characters (underscores, dashes, dots) into [BLANK]
_FILLER_PATTERN = re.compile(r"[_\-\.]{10,}")
# Collapse escaped underscores too (\_\_\_...)
_ESCAPED_FILLER = re.compile(r"(\\_){5,}")


def _clean_ocr_artifacts(content: str) -> str:
    """Post-process OCR output to remove common artifacts."""
    content = _FILLER_PATTERN.sub("[BLANK]", content)
    content = _ESCAPED_FILLER.sub("[BLANK]", content)
    return content


# ---------------------------------------------------------------------------
# Text extraction from PDF
# ---------------------------------------------------------------------------


def extract_text_from_pdf(pdf_path: Path, config: ExtractionConfig) -> list[dict]:
    """Extract text from each page of a PDF.

    Returns a list of dicts with page_num, text, and is_scanned flag.
    """
    doc = pymupdf.open(str(pdf_path))
    pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")

        is_scanned = len(text.strip()) < config.min_text_chars_per_page
        pages.append({
            "page_num": page_num + 1,
            "text": text.strip(),
            "is_scanned": is_scanned,
        })

    doc.close()
    return pages


def render_page_as_image(pdf_path: Path, page_num: int, dpi: int = 200) -> bytes:
    """Render a single PDF page as a PNG image (for OCR via LLM)."""
    doc = pymupdf.open(str(pdf_path))
    page = doc[page_num - 1]  # 0-indexed internally
    # Render at specified DPI
    zoom = dpi / 72.0
    mat = pymupdf.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img_bytes = pix.tobytes("png")
    doc.close()
    return img_bytes


# ---------------------------------------------------------------------------
# LLM-based structuring
# ---------------------------------------------------------------------------


STRUCTURE_SYSTEM_PROMPT = """\
You are a patent document specialist. Your task is to convert extracted patent office \
document text into clean, well-structured Markdown.

Rules:
- Preserve ALL content faithfully — do not summarize or omit anything
- Use proper Markdown headings (##, ###) for document sections
- Format patent claims as numbered lists
- Preserve paragraph structure
- Clean up OCR artifacts (broken words, random line breaks mid-sentence)
- Mark clearly illegible sections as [ILLEGIBLE]
- For tables, use Markdown table format
- Patent numbers, dates, and reference numbers must be preserved exactly
- If the text appears to be from a specific document type (office action, search report, etc.), \
  add a brief header identifying it

Output ONLY the Markdown content, no explanations or preamble."""

OCR_SYSTEM_PROMPT = """\
You are a patent document OCR specialist. You will receive an image of a page from a \
patent office document (office action, search report, patent application, etc.).

Your task:
- Extract ALL text content from the image
- Structure it as clean Markdown
- Use proper headings, lists, and formatting
- Preserve patent numbers, dates, claim numbers, and reference numbers exactly
- For crossed-out/strikethrough text, wrap it in ~~strikethrough~~ markdown (this is critical — it shows amendments and deletions)
- For underlined or added/inserted text, wrap it in **bold** and prefix with [ADDED:]
- Mark illegible sections as [ILLEGIBLE]
- For handwritten annotations, prefix with [HANDWRITTEN:]
- For stamps/marks, prefix with [STAMP:]
- For empty/blank form fields or signature lines, use a SINGLE [BLANK] marker — do NOT fill them with underscores or dashes
- For checkboxes, use [x] (checked) or [ ] (unchecked)
- Keep output concise — do not pad or repeat filler characters

Output ONLY the Markdown content, no explanations."""


def structure_text_with_llm(
    raw_text: str,
    doc_context: str,
    config: ExtractionConfig,
) -> str:
    """Send extracted text to LLM for structuring into clean Markdown."""
    client = httpx.Client(base_url=config.vllm_base_url, timeout=120.0)

    user_msg = f"Document context: {doc_context}\n\n---\n\nRaw extracted text:\n\n{raw_text}"

    response = client.post(
        "/chat/completions",
        json={
            "model": config.vllm_model,
            "messages": [
                {"role": "system", "content": STRUCTURE_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    response.raise_for_status()
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    client.close()

    content = _strip_thinking(content)
    return _clean_ocr_artifacts(content)


def ocr_page_with_llm(
    image_bytes: bytes,
    doc_context: str,
    config: ExtractionConfig,
) -> str:
    """Send a page image to the LLM for OCR extraction."""
    client = httpx.Client(base_url=config.vllm_base_url, timeout=180.0)

    img_b64 = base64.b64encode(image_bytes).decode("utf-8")

    response = client.post(
        "/chat/completions",
        json={
            "model": config.vllm_model,
            "messages": [
                {"role": "system", "content": OCR_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Document context: {doc_context}\n\nExtract all text from this page:",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_b64}",
                            },
                        },
                    ],
                },
            ],
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    response.raise_for_status()
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    client.close()

    content = _strip_thinking(content)
    return _clean_ocr_artifacts(content)


# ---------------------------------------------------------------------------
# Main extraction pipeline
# ---------------------------------------------------------------------------


def derive_doc_context(pdf_path: Path) -> str:
    """Derive document context from filename for LLM guidance."""
    name = pdf_path.stem
    parts = []

    # Detect document type from filename patterns
    type_markers = {
        "non final rejection": "USPTO Non-Final Office Action",
        "non-final rejection": "USPTO Non-Final Office Action",
        "final rejection": "USPTO Final Office Action",
        "restriction election": "USPTO Restriction/Election Requirement",
        "reply to restriction": "Response to Restriction/Election",
        "notice of allowance": "USPTO Notice of Allowance",
        "search report": "Patent Search Report",
        "search results": "EPO Extended Search Report",
        "communication patentability": "EPO Communication on Patentability (Art. 94(3))",
        "reply after": "Applicant Response/Reply",
        "reply to": "Applicant Response/Reply",
        "replay to": "Applicant Response/Reply",  # typo in filename
        "response to communication": "Response to Office Communication",
        "amended claims": "Amended Patent Claims",
        "claims after": "Amended Patent Claims",
        "patent pending": "Published Patent Application",
        "accepted": "Granted Patent",
        "listed documents": "IDS / Cited Documents List",
        "drawings amendment": "Amended Drawings",
    }

    name_lower = name.lower()
    for marker, doc_type in type_markers.items():
        if marker in name_lower:
            parts.append(f"Document type: {doc_type}")
            break

    # Detect jurisdiction
    if "US20" in name or "US2019" in name:
        parts.append("Jurisdiction: USPTO (United States)")
    elif "EP" in name:
        parts.append("Jurisdiction: EPO (European Patent Office)")
    elif "WO20" in name:
        parts.append("Jurisdiction: PCT/WIPO (International)")

    # Add the full filename
    parts.append(f"Filename: {name}")

    return "; ".join(parts) if parts else f"Patent document: {name}"


def extract_single_pdf(pdf_path: Path, config: ExtractionConfig) -> str:
    """Extract a single PDF to structured Markdown.

    Returns the Markdown content as a string.
    """
    logger.info(f"Extracting: {pdf_path.name} ({pdf_path.stat().st_size / 1024:.0f} KB)")

    doc_context = derive_doc_context(pdf_path)
    pages = extract_text_from_pdf(pdf_path, config)

    total_pages = len(pages)
    scanned_pages = sum(1 for p in pages if p["is_scanned"])
    digital_pages = total_pages - scanned_pages

    logger.info(f"  Pages: {total_pages} total, {digital_pages} digital, {scanned_pages} scanned")

    # Process pages in chunks
    markdown_parts: list[str] = []

    # Group consecutive pages by type for efficient processing
    i = 0
    while i < len(pages):
        page = pages[i]

        if page["is_scanned"]:
            # OCR path: render page as image, send to LLM
            logger.info(f"  OCR page {page['page_num']}...")
            img_bytes = render_page_as_image(pdf_path, page["page_num"], config.dpi_for_ocr)
            ocr_text = ocr_page_with_llm(img_bytes, doc_context, config)
            markdown_parts.append(f"<!-- Page {page['page_num']} (OCR) -->\n{ocr_text}")
            i += 1
        else:
            # Digital path: collect consecutive digital pages and structure together
            chunk_texts = []
            chunk_start = page["page_num"]
            while i < len(pages) and not pages[i]["is_scanned"]:
                chunk_texts.append(f"--- Page {pages[i]['page_num']} ---\n{pages[i]['text']}")
                i += 1
                if len(chunk_texts) >= config.max_pages_per_chunk:
                    break

            combined = "\n\n".join(chunk_texts)
            chunk_end = chunk_start + len(chunk_texts) - 1

            if len(combined.strip()) < 100:
                # Very little text — might be a mostly-image page, skip LLM
                markdown_parts.append(
                    f"<!-- Pages {chunk_start}-{chunk_end}: minimal text content -->\n{combined}"
                )
            else:
                logger.info(f"  Structuring pages {chunk_start}-{chunk_end}...")
                structured = structure_text_with_llm(combined, doc_context, config)
                markdown_parts.append(
                    f"<!-- Pages {chunk_start}-{chunk_end} -->\n{structured}"
                )

    # Assemble final document
    header = f"# {pdf_path.stem}\n\n"
    header += f"**Source**: `{pdf_path.name}`\n"
    header += f"**Pages**: {total_pages} ({digital_pages} digital, {scanned_pages} OCR)\n"
    header += f"**Context**: {doc_context}\n\n---\n\n"

    return header + "\n\n".join(markdown_parts)


def determine_output_path(pdf_path: Path, config: ExtractionConfig) -> Path:
    """Determine the output .md path preserving testdata subfolder structure."""
    try:
        # Get relative path from testdata dir
        rel = pdf_path.relative_to(config.testdata_dir)
    except ValueError:
        # Not under testdata — use just the filename
        rel = Path(pdf_path.stem)

    # Replace .pdf extension with .md
    output_path = config.output_dir / rel.with_suffix(".md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def find_all_pdfs(directory: Path) -> list[Path]:
    """Find all PDF files in a directory recursively."""
    return sorted(directory.rglob("*.pdf"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for PDF extraction."""
    parser = argparse.ArgumentParser(
        description="Extract patent PDFs from testdata/ to structured Markdown",
    )
    parser.add_argument(
        "path",
        nargs="?",
        help="Path to a single PDF file or directory (with --batch)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process all PDFs in the given directory",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all PDFs in the entire testdata/ directory",
    )
    parser.add_argument(
        "--priority",
        type=int,
        choices=[1, 2, 3, 4],
        help="Process only priority N documents (1=examiner actions, 2=replies, 3=amendments, 4=patents)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of files to process (useful with --priority or --all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be extracted without running LLM",
    )
    parser.add_argument(
        "--vllm-url",
        default="http://localhost:8000/v1",
        help="vLLM API base URL (default: http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3.5-9B",
        help="Model name for vLLM (default: Qwen/Qwen3.5-9B)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip PDFs that already have a .md output file",
    )

    args = parser.parse_args()

    config = ExtractionConfig(
        vllm_base_url=args.vllm_url,
        vllm_model=args.model,
    )

    # Determine files to process
    pdfs: list[Path] = []

    if args.priority:
        from scripts.extract_testdata.priority import get_priority_files
        groups = get_priority_files()
        group_keys = [
            "priority_1_examiner",
            "priority_2_replies",
            "priority_3_amendments",
            "priority_4_patents",
        ]
        # Include all groups up to and including the requested priority
        for key in group_keys[: args.priority]:
            pdfs.extend(groups[key])
        logger.info(f"Priority mode: levels 1-{args.priority} ({len(pdfs)} files)")
    elif args.all:
        pdfs = find_all_pdfs(config.testdata_dir)
    elif args.path:
        path = Path(args.path)
        if path.is_file() and path.suffix.lower() == ".pdf":
            pdfs = [path]
        elif path.is_dir() or args.batch:
            target = path if path.is_dir() else Path(args.path)
            pdfs = find_all_pdfs(target)
        else:
            logger.error(f"Not a PDF file or directory: {path}")
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(0)

    if not pdfs:
        logger.warning("No PDF files found.")
        sys.exit(0)

    logger.info(f"Found {len(pdfs)} PDF files to process")

    # Filter existing if requested
    if args.skip_existing:
        original_count = len(pdfs)
        pdfs = [
            p for p in pdfs
            if not determine_output_path(p, config).exists()
        ]
        skipped = original_count - len(pdfs)
        if skipped:
            logger.info(f"Skipping {skipped} already-converted files")

    # Apply limit
    if args.limit and len(pdfs) > args.limit:
        logger.info(f"Limiting to first {args.limit} files (of {len(pdfs)})")
        pdfs = pdfs[: args.limit]

    if args.dry_run:
        logger.info("DRY RUN — would process:")
        for pdf in pdfs:
            output = determine_output_path(pdf, config)
            size_mb = pdf.stat().st_size / (1024 * 1024)
            logger.info(f"  {pdf.name} ({size_mb:.1f} MB) → {output}")
        sys.exit(0)

    # Process files
    total = len(pdfs)
    success = 0
    failed = 0

    for i, pdf in enumerate(pdfs, 1):
        output_path = determine_output_path(pdf, config)
        logger.info(f"\n[{i}/{total}] Processing: {pdf.name}")

        try:
            start = time.time()
            markdown = extract_single_pdf(pdf, config)
            elapsed = time.time() - start

            output_path.write_text(markdown, encoding="utf-8")
            logger.info(f"  ✓ Written: {output_path} ({elapsed:.1f}s)")
            success += 1

        except httpx.ConnectError:
            logger.error(
                "  ✗ Cannot connect to vLLM at %s — is it running?",
                config.vllm_base_url,
            )
            logger.error("    Start with: ./scripts/start-vllm-9b.sh")
            sys.exit(1)

        except Exception as exc:
            logger.error(f"  ✗ Failed: {exc}")
            failed += 1

    logger.info(f"\nDone: {success} succeeded, {failed} failed out of {total}")


if __name__ == "__main__":
    main()
