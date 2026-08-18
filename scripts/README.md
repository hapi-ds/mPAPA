# Scripts

Developer/maintainer tools. Not needed by mPAPA users.

## Prerequisites

- NVIDIA GPU with sufficient VRAM (RTX PRO 6000 or similar)
- CUDA 13.3 toolkit
- vLLM installed in a separate venv (see start scripts)
- Models downloaded: `Qwen/Qwen3.5-9B` (vision), `Qwen/Qwen3.6-27B`

## Testdata Extraction

Converts scanned patent PDFs in `testdata/` to structured Markdown using OCR via Qwen3.5-9B vision model.

```bash
# 1. Start vLLM with 9B multimodal model
./scripts/start-vllm-9b.sh

# 2. Extract priority documents (examiner actions first — smallest, most useful)
uv run python -m scripts.extract_testdata.pdf_to_markdown --priority 1 --limit 10

# 3. Extract all priority 1 + 2 (examiner actions + replies)
uv run python -m scripts.extract_testdata.pdf_to_markdown --priority 2

# 4. Extract everything
uv run python -m scripts.extract_testdata.pdf_to_markdown --all

# Dry run (see what would be processed)
uv run python -m scripts.extract_testdata.pdf_to_markdown --all --dry-run

# Skip already converted
uv run python -m scripts.extract_testdata.pdf_to_markdown --all --skip-existing
```

Output goes to `testdata/converted/` (gitignored).

## Rule Card Compilation (future)

```bash
# Start vLLM with 27B model
./scripts/start-vllm-27b.sh

# Compile a specific card
uv run python scripts/compile_cards.py --jurisdiction EP --task claim_structure
```

## Model Cache

Both start scripts point to the shared model cache at:
`/home/hako/02_work/02_tyro/REGEN_3/models/`

No need to re-download models.
