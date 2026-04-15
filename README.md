# mPAPA — my Personal Artificial Patent Attorney

[![Tests](https://github.com/OWNER/REPO/actions/workflows/build.yml/badge.svg)](https://github.com/OWNER/REPO/actions/workflows/build.yml)
[![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/OWNER/YOUR_GIST_ID/raw/coverage-badge.json)](https://github.com/OWNER/REPO/actions/workflows/build.yml)
[![Version](https://img.shields.io/github/v/release/OWNER/REPO?label=version)](https://github.com/OWNER/REPO/releases/latest)

A locally-hosted, AI-powered patent analysis and drafting system. mPAPA orchestrates a pipeline of specialized agents to help patent professionals research prior art, analyze novelty, and draft patent applications — all while keeping sensitive invention data on your machine.

## How It Works

mPAPA runs a multi-agent workflow powered by LangGraph. Each step builds on the previous one, and you can pause, review, and resume at any point.

```
┌─────────────────┐
│  1. Disclosure   │  Interactive interview extracts structured invention details
└────────┬────────┘
         ▼
┌─────────────────┐
│ 2. Prior Art     │  Searches DEPATISnet, Google Patents, Google Scholar, ArXiv, PubMed
│    Search        │
└────────┬────────┘
         ▼
┌─────────────────┐
│ 3. Novelty       │  RAG-powered comparison of your invention against prior art
│    Analysis      │
└────────┬────────┘
         ▼
┌─────────────────┐
│ 4. Claims        │  Generates claims in German/European patent style
│    Drafting      │
└────────┬────────┘
         ▼
┌─────────────────┐     ┌──────────────┐
│ 5. Consistency   │────▶│ Revision     │  Up to 3 automatic revision loops,
│    Review        │◀────│ Loop         │  then human review if needed
└────────┬────────┘     └──────────────┘
         ▼
┌─────────────────┐
│ 6. Description   │  Full patent specification: Technical Field, Background Art,
│    Drafting      │  Summary, Detailed Description, Drawings, Applicability
└────────┬────────┘
         ▼
┌─────────────────┐
│ 7. DOCX Export   │  Export with optional custom templates
└─────────────────┘
```

All LLM inference runs locally via LM Studio. Only public patent/scientific database queries go over the network.

## Quick Start

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager
- [LM Studio](https://lmstudio.ai/) running locally with a loaded model

### Setup

```bash
# Clone the repo
git clone https://github.com/your-user/mPAPA.git
cd mPAPA

# Install dependencies
uv sync

# Copy the example env file and adjust as needed
cp .env.example .env

# Start LM Studio and load a model (e.g. Llama 3, Mistral, etc.)

# Run the app
uv run python -m patent_system.main
```

Open your browser at `http://localhost:8080`.

### Usage

1. Create a new topic in the left sidebar
2. Click "Generate Patent Draft" to start the agent workflow
3. The system walks through disclosure, search, analysis, drafting, and review
4. Edit claims and description in the expandable editors
5. Export to DOCX when ready

The AI Chat tab lets you ask questions about your collected patents and papers using RAG-powered retrieval.

## Configuration

All settings are managed via environment variables with the `PATENT_` prefix, or a `.env` file in the project root. See `.env.example` for all available options.

## Running Tests

```bash
uv run pytest -q
```

## Project Structure

```
src/patent_system/
├── main.py              # App entry point
├── config.py            # Pydantic Settings
├── logging_config.py    # Structured JSON logging
├── exceptions.py        # Custom exception hierarchy
├── db/                  # SQLite schema, models, repositories
├── agents/              # LangGraph workflow and agent nodes
├── dspy_modules/        # DSPy signatures and module wrappers
├── rag/                 # Embedding service, RAG engine, citation graph
├── parsers/             # Source-specific parsers (DEPATISnet, ArXiv, etc.)
├── export/              # DOCX exporter with template support
├── monitoring/          # Background prior art monitoring scheduler
└── gui/                 # NiceGUI web interface panels
```

## License

See [LICENSE](LICENSE).
