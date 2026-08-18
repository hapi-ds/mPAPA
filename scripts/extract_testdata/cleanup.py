"""Post-processing cleanup for already-extracted markdown files.

Removes:
- <think>...</think> blocks (Qwen3 thinking mode artifacts)
- Metadata headers (added by the extraction script, not needed for reviewer input)

Usage:
    # Clean all converted files
    uv run python -m scripts.extract_testdata.cleanup

    # Dry run (show what would change)
    uv run python -m scripts.extract_testdata.cleanup --dry-run

    # Clean a specific file
    uv run python -m scripts.extract_testdata.cleanup path/to/file.md
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

from scripts.extract_testdata.config import ExtractionConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Patterns to remove
_THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_UNCLOSED = re.compile(r"<think>.*", re.DOTALL)
_PAGE_COMMENT = re.compile(r"<!-- Page \d+.*?-->\n?")
_HEADER_BLOCK = re.compile(
    r"^# .+?\n\n\*\*Source\*\*:.+?\n\*\*Pages\*\*:.+?\n\*\*Context\*\*:.+?\n\n---\n\n",
    re.DOTALL,
)
_FILLER_PATTERN = re.compile(r"[_\-\.]{10,}")
_ESCAPED_FILLER = re.compile(r"(\\_){5,}")


def clean_markdown(content: str, *, strip_header: bool = True) -> str:
    """Clean a markdown file of extraction artifacts.

    Args:
        content: Raw markdown content.
        strip_header: Remove the metadata header block added by extraction.

    Returns:
        Cleaned markdown content.
    """
    original_len = len(content)

    # 1. Remove think blocks (closed)
    content = _THINK_PATTERN.sub("", content)

    # 2. Remove unclosed think blocks
    if "<think>" in content:
        content = _THINK_UNCLOSED.sub("", content)

    # 3. Remove page comments (<!-- Page N ... -->)
    content = _PAGE_COMMENT.sub("", content)

    # 4. Remove metadata header block
    if strip_header:
        content = _HEADER_BLOCK.sub("", content)

    # 5. Collapse repeated filler characters (underscores, dashes) into [BLANK]
    content = _FILLER_PATTERN.sub("[BLANK]", content)
    content = _ESCAPED_FILLER.sub("[BLANK]", content)

    # 6. Collapse excessive blank lines (3+ → 2)
    content = re.sub(r"\n{3,}", "\n\n", content)

    # 7. Strip leading/trailing whitespace
    content = content.strip() + "\n"

    return content


def clean_file(path: Path, *, dry_run: bool = False, strip_header: bool = True) -> bool:
    """Clean a single markdown file. Returns True if file was modified."""
    original = path.read_text(encoding="utf-8")
    cleaned = clean_markdown(original, strip_header=strip_header)

    if cleaned == original:
        return False

    if dry_run:
        removed_chars = len(original) - len(cleaned)
        logger.info(f"  Would clean: {path.name} (remove {removed_chars} chars)")
    else:
        path.write_text(cleaned, encoding="utf-8")
        removed_chars = len(original) - len(cleaned)
        logger.info(f"  Cleaned: {path.name} (removed {removed_chars} chars)")

    return True


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Clean extraction artifacts from converted markdown files",
    )
    parser.add_argument(
        "path",
        nargs="?",
        help="Path to a specific .md file to clean",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without modifying files",
    )
    parser.add_argument(
        "--keep-header",
        action="store_true",
        help="Keep the metadata header (Source/Pages/Context block)",
    )

    args = parser.parse_args()
    config = ExtractionConfig()

    if args.path:
        path = Path(args.path)
        if not path.exists():
            logger.error(f"File not found: {path}")
            sys.exit(1)
        files = [path]
    else:
        # Find all .md files in the converted directory
        if not config.output_dir.exists():
            logger.error(f"No converted directory found: {config.output_dir}")
            sys.exit(1)
        files = sorted(config.output_dir.rglob("*.md"))

    if not files:
        logger.info("No markdown files found to clean.")
        sys.exit(0)

    logger.info(f"{'DRY RUN: ' if args.dry_run else ''}Processing {len(files)} files...")

    modified = 0
    for f in files:
        if clean_file(f, dry_run=args.dry_run, strip_header=not args.keep_header):
            modified += 1

    logger.info(f"{'Would modify' if args.dry_run else 'Modified'}: {modified}/{len(files)} files")


if __name__ == "__main__":
    main()
