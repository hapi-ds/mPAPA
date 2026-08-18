"""Configuration for the Rule Card batch compiler."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CompilerConfig:
    """Configuration for the Rule Card compilation pipeline.

    Attributes:
        vllm_base_url: Base URL for the vLLM OpenAI-compatible API.
        vllm_model: Model identifier served by vLLM.
        max_tokens: Maximum tokens for LLM generation.
        temperature: Sampling temperature (low for deterministic extraction).
        output_dir: Directory where compiled RuleCard JSON files are written.
        sources_dir: Directory containing source guideline PDFs/HTML files.
    """

    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_model: str = "Qwen/Qwen3.6-27B"
    max_tokens: int = 8192
    temperature: float = 0.1
    output_dir: Path = field(default_factory=lambda: Path("rule-cards/"))
    sources_dir: Path = field(
        default_factory=lambda: Path("scripts/compile_rule_cards/sources/")
    )
