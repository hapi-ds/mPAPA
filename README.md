# mPAPA — my Personal Artificial Patent Attorney

<p align="center">
  <a href="https://github.com/hapi-ds/mPAPA/releases">
    <img src="https://img.shields.io/github/v/release/hapi-ds/mPAPA?style=flat-square&color=blue" alt="Latest Release">
  </a>
  <img src="https://img.shields.io/badge/python-3.13-blue?style=flat-square&logo=python&logoColor=white" alt="Python Version">
  <a href="https://github.com/hapi-ds/mPAPA/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/hapi-ds/mPAPA?style=flat-square" alt="License">
  </a>
</p>

<p align="center">
  <a href="https://github.com/hapi-ds/mPAPA/actions">
    <img src="https://github.com/hapi-ds/mPAPA/actions/workflows/build.yml/badge.svg" alt="CI Pipeline Status">
  </a>
  <a href="https://github.com/hapi-ds/mPAPA/actions">
    <img src="https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/hapi-ds/b90471b021c2812e72f3ff038d22508d/raw/coverage-badge.json" alt="Coverage">
  </a>
</p>



**Your invention. Your machine. Your patent. Zero data leaks.**

---

Stop uploading your billion-dollar idea to ChatGPT's servers. Stop waiting weeks for a first draft.

**mPAPA is a fully local AI patent drafting system.** It runs on YOUR machine*. Your invention data never touches the internet. Ever.

*Any recent PC will do. Just 6 GB+ RAM for LLM. GPU optional. Works great with models like Gemma-4-E2B.

---

### What it does in 30 minutes flat:

🔍 **Searches 5 patent/paper databases simultaneously** — EPO, Google Patents, Google Scholar, ArXiv, PubMed. 100+ references collected and analyzed automatically.

💬 **AI Chat over your entire research** — Ask questions, explore prior art, refine your claims. RAG-powered, context-aware, backed by every reference you've collected.

⚡ **9-step AI workflow generates your complete patent draft:**
1. Invention disclosure review
2. Claims drafting (European patent format)
3. Prior art landscape analysis
4. Novelty assessment
5. Consistency review
6. Market potential analysis
7. Legal & IP clarification
8. Disclosure summary
9. Full patent specification

**Every step: AI generates → you review → you edit → you continue.** Full control. Full transparency.

📄 **Export to DOCX** — Professional formatting, proper styles, header/footer, ready for your attorney.

Stored locally in a database for each topic, so you can pick up right where you left off at any time.

---

### Why mPAPA destroys the alternatives:

| | mPAPA | ChatGPT / Claude | Patent Attorney |
|---|---|---|---|
| **Data privacy** | 100% local | Your IP on their servers | NDA required |
| **Cost** | Free / Open Source | $20-200/month | $5,000-50,000 |
| **Speed** | 30 minutes | Hours of prompting | 2-8 weeks |
| **Prior art search** | 5 sources, automated | Manual, one at a time | Manual, expensive |
| **Structured workflow** | 9-step guided process | Unstructured chat | Black box |
| **Editable at every step** | Yes | Start over | Revision rounds |
| **Works offline** | Yes | No | No |

---

### The tech that makes it possible:

- **LM Studio** — Run any open-source LLM locally. Llama, Qwen, Mistral. Your choice.
- **LangGraph** — Multi-step AI workflows with human-in-the-loop review at every stage.
- **DSPy** — Structured, optimizable prompts. Not fragile prompt strings.
- **LlamaIndex** — RAG over your entire reference collection. Local embeddings.
- **NiceGUI** — Clean web interface. No Electron bloat. No cloud dependency.
- **SQLite** — Everything persisted. Close the app, reopen tomorrow, pick up where you left off.

---

### Who is this for?

- **Independent inventors** who can't afford $15K for a patent application
- **Startup founders** who need to file fast before competitors
- **R&D engineers** who want a solid first draft before engaging counsel
- **Patent professionals** who want AI assistance without data privacy risks
- **Academic researchers** exploring patentability of their work

---

## Installation

### Option A: Windows Executable (easiest)

No Python required. Just download and run.

1. Go to the [Releases page](https://github.com/OWNER/REPO/releases/latest)
2. Download `main.exe` from the latest release
3. Place it in a folder of your choice (e.g. `C:\mPAPA\`)
4. Copy `.env.example` next to the exe and rename it to `.env` — adjust settings if needed
5. Double-click `main.exe` or run it from a terminal
6. Open `http://localhost:8080` in your browser

On first launch, mPAPA automatically creates `data/` and `logs/` folders next to the executable.

### Option B: From Source (for developers)

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
# Clone the repo
git clone https://github.com/OWNER/REPO.git
cd mPAPA

# Install dependencies
uv sync

# Copy the example env file and adjust as needed
cp .env.example .env

# Run the app
uv run python -m patent_system.main
```

Open your browser at `http://localhost:8080`.

---

## Setting Up LM Studio

mPAPA needs a local LLM running via [LM Studio](https://lmstudio.ai/). Here's how to get it going:

### 1. Install LM Studio

- Download from [lmstudio.ai](https://lmstudio.ai/) (Windows, macOS, Linux)
- Run the installer — no special configuration needed

### 2. Download a model

Open LM Studio and search for a model in the Discover tab. Recommended options for small machines:

| Model | Size | Notes |
|---|---|---|
| **Gemma 3 4B** | ~3 GB | Fast, good quality, runs on most machines |
| **Qwen 2.5 7B** | ~5 GB | Strong reasoning, good for patent language |
| **Llama 3.1 8B** | ~5 GB | Well-rounded, widely tested |
| **Mistral 7B** | ~4 GB | Good balance of speed and quality |

Pick one that fits your RAM. 8 GB system RAM is enough for 4B models, 16 GB for 7-8B models. A GPU speeds things up but isn't required.

### 3. Download an embedding model

In LM Studio's Discover tab, also search for and download an embedding model:

- **nomic-embed-text-v1.5** (recommended, ~270 MB)

### 4. Start the local server

- Go to the **Developer** tab in LM Studio
- Load your chosen chat model
- Load the embedding model
- Click **Start Server**
- The server runs at `http://localhost:1234/v1` by default — this matches mPAPA's default config

That's it. Leave LM Studio running and start mPAPA.

> **Tip:** If you change the LM Studio port or URL, update `PATENT_LM_STUDIO_BASE_URL` in your `.env` file.

---

## Usage

1. Create a new topic in the left sidebar
2. Insert your patent idea and some search terms
3. Add your local reference resources/papers/patents
4. Start search - reference patents and other literature will be downloaded and a RAG database is build
5. Chat with AI - ask for better key words, first patent idea improvements and refine/repeat from step 2 - al based on found references and your input
6. Click "Generate Patent Draft" to start the agent workflow
7. The system walks through disclosure, search, analysis, drafting, and review
8. Edit claims and description in the expandable editors
9. Change and re-run as you like
10. Export to DOCX when ready

You can make a break whenever you want - all results are saved to a local db and are recalled when you select a topic.

Tip: You can easily change the appearance of the export by customising the template in ./src/export/templates.

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

---

*mPAPA — because your invention is too valuable to upload to someone else's server.*

*Built by [koehler](https://koehler.eu.com). Open source. Local first. Always.*
