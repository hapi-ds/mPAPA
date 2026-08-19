"""CLI entry point for the Rule Card batch compiler.

Usage:
    uv run python scripts/compile_cards.py \\
        --jurisdiction EP \\
        --task inventive_step \\
        --source scripts/compile_rule_cards/sources/EPO_Guidelines_Part_G_Chapter_VII.pdf \\
        --card-id epo_inventive_step

    uv run python scripts/compile_cards.py --validate-all
    uv run python scripts/compile_cards.py --all --jurisdiction EP
"""

import argparse
import logging
import sys
from pathlib import Path

from compile_rule_cards.compiler import CardCompiler
from compile_rule_cards.config import CompilerConfig
from compile_rule_cards.extractor import RuleExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Map jurisdiction codes to office names
OFFICE_MAP: dict[str, str] = {
    "EP": "EPO",
    "US": "USPTO",
    "PCT": "WIPO",
    "DE": "DPMA",
    "CN": "CNIPA",
    "JP": "JPO",
    "KR": "KIPO",
}

# Map task identifiers to human-readable labels
TASK_LABELS: dict[str, str] = {
    "claim_structure": "Claim Structure & Drafting",
    "novelty": "Novelty Assessment",
    "inventive_step": "Inventive Step / Non-Obviousness",
    "sufficiency": "Sufficiency of Disclosure",
    "clarity": "Claim Clarity",
    "unity": "Unity of Invention",
    "priority": "Priority & Filing",
    "amendment": "Amendment Practice",
    "description": "Description Requirements",
}


def compile_single(
    config: CompilerConfig,
    source: Path,
    jurisdiction: str,
    task: str,
    card_id: str,
    source_language: str = "en",
) -> None:
    """Compile a single source file into a RuleCard.

    Non-English sources are automatically translated to English during extraction.

    Args:
        config: Compiler configuration.
        source: Path to the source PDF/HTML file.
        jurisdiction: Jurisdiction code (EP, US, PCT, etc.).
        task: Task identifier.
        card_id: Output card ID.
        source_language: ISO 639-1 code of source language (en, de, ja, zh, ko).
    """
    if not source.exists():
        logger.error("Source file not found: %s", source)
        sys.exit(1)

    office = OFFICE_MAP.get(jurisdiction.upper(), jurisdiction.upper())
    task_label = TASK_LABELS.get(task, task.replace("_", " ").title())

    with RuleExtractor(config) as extractor:
        rules = extractor.extract(source, jurisdiction, task, source_language=source_language)

    if not rules:
        logger.warning("No rules extracted from %s", source)
        sys.exit(1)

    with CardCompiler(config) as compiler:
        card = compiler.compile(
            rules=rules,
            card_id=card_id,
            jurisdiction=jurisdiction.upper(),
            office=office,
            source_document=source.name,
            task=task,
            task_label=task_label,
        )
        output_path = compiler.write_card(card)

    logger.info("✓ Compiled %s → %s (%d rules)", source.name, output_path, len(rules))


def compile_all(config: CompilerConfig, jurisdiction: str | None = None) -> None:
    """Compile all source files in the sources directory.

    Args:
        config: Compiler configuration.
        jurisdiction: Optional filter by jurisdiction (derived from filename prefix).
    """
    sources_dir = config.sources_dir
    if not sources_dir.exists():
        logger.error("Sources directory not found: %s", sources_dir)
        sys.exit(1)

    source_files = list(sources_dir.glob("*.pdf")) + list(sources_dir.glob("*.html"))

    if not source_files:
        logger.warning("No source files found in %s", sources_dir)
        sys.exit(1)

    if jurisdiction:
        source_files = [
            f for f in source_files if f.stem.upper().startswith(jurisdiction.upper())
        ]

    logger.info("Compiling %d source files", len(source_files))

    for source_file in sorted(source_files):
        # Derive metadata from filename convention: {Office}_{Topic}_{Details}.pdf
        stem_parts = source_file.stem.split("_", 2)
        derived_jurisdiction = stem_parts[0].upper() if stem_parts else "UNKNOWN"
        derived_task = stem_parts[1].lower() if len(stem_parts) > 1 else "general"
        derived_card_id = source_file.stem.lower()

        logger.info("Processing: %s", source_file.name)
        try:
            compile_single(
                config, source_file, derived_jurisdiction, derived_task, derived_card_id
            )
        except Exception as e:
            logger.error("Failed to compile %s: %s", source_file.name, e)
            continue


def validate_all(config: CompilerConfig) -> None:
    """Validate all existing RuleCard JSON files against the Pydantic model.

    Args:
        config: Compiler configuration.
    """
    output_dir = config.output_dir
    if not output_dir.exists():
        logger.error("Output directory not found: %s", output_dir)
        sys.exit(1)

    card_files = list(output_dir.rglob("*.json"))
    if not card_files:
        logger.warning("No card files found in %s", output_dir)
        return

    compiler = CardCompiler(config)
    errors = 0

    for card_file in sorted(card_files):
        try:
            compiler.validate_card_file(card_file)
            logger.info("✓ Valid: %s", card_file)
        except Exception as e:
            logger.error("✗ Invalid: %s — %s", card_file, e)
            errors += 1

    compiler.close()

    if errors:
        logger.error("%d/%d cards failed validation", errors, len(card_files))
        sys.exit(1)
    else:
        logger.info("All %d cards valid", len(card_files))


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Rule Card Batch Compiler — extract rules from patent guidelines via LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  uv run python scripts/compile_cards.py \\
      --jurisdiction EP --task inventive_step \\
      --source scripts/compile_rule_cards/sources/EPO_Guidelines_Part_G_Chapter_VII.pdf \\
      --card-id epo_inventive_step

  uv run python scripts/compile_cards.py --validate-all
  uv run python scripts/compile_cards.py --all --jurisdiction EP
""",
    )

    parser.add_argument(
        "--jurisdiction",
        type=str,
        help="Patent jurisdiction code (EP, US, PCT, DE, CN, JP, KR)",
    )
    parser.add_argument(
        "--task",
        type=str,
        help="Task identifier (claim_structure, novelty, inventive_step, etc.)",
    )
    parser.add_argument(
        "--source",
        type=Path,
        help="Path to source PDF or HTML guideline file",
    )
    parser.add_argument(
        "--card-id",
        type=str,
        help="Output card identifier",
    )
    parser.add_argument(
        "--validate-all",
        action="store_true",
        help="Validate all existing RuleCard JSON files",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Compile all sources in the sources directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory (default: rule-cards/)",
    )
    parser.add_argument(
        "--vllm-url",
        type=str,
        default=None,
        help="Override vLLM base URL",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override model name",
    )
    parser.add_argument(
        "--language", "-L",
        type=str,
        default="en",
        help="Source language ISO code (en, de, ja, zh, ko, fr). Non-English sources are translated to English during extraction. Default: en",
    )

    args = parser.parse_args()

    config = CompilerConfig()
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.vllm_url:
        config.vllm_base_url = args.vllm_url
    if args.model:
        config.vllm_model = args.model

    if args.validate_all:
        validate_all(config)
        return

    if args.all:
        compile_all(config, jurisdiction=args.jurisdiction)
        return

    # Single file compilation
    if not args.source:
        parser.error("--source is required for single-file compilation")
    if not args.jurisdiction:
        parser.error("--jurisdiction is required for single-file compilation")
    if not args.task:
        parser.error("--task is required for single-file compilation")
    if not args.card_id:
        parser.error("--card-id is required for single-file compilation")

    compile_single(config, args.source, args.jurisdiction, args.task, args.card_id, source_language=args.language)


if __name__ == "__main__":
    main()
