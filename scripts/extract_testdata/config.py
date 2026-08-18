"""Configuration for testdata extraction pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExtractionConfig:
    """Configuration for the PDF-to-Markdown extraction pipeline."""

    # vLLM connection
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_model: str = "Qwen/Qwen3.5-9B"
    max_tokens: int = 8192
    temperature: float = 0.1

    # Paths
    testdata_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent / "testdata")
    output_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent / "testdata" / "converted")

    # Processing
    max_pages_per_chunk: int = 3  # Pages sent to LLM at once for structuring
    dpi_for_ocr: int = 300  # Resolution for page rendering when OCR needed
    min_text_chars_per_page: int = 50  # Below this = likely scanned, needs OCR

    def __post_init__(self) -> None:
        """Ensure output directory exists."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
