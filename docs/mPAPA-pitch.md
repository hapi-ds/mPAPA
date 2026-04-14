# mPAPA — my Personal Artificial Patent Agent

**Your invention. Your machine. Your patent. Zero data leaks.**

---

Stop paying patent attorneys $500/hour to do what AI can do in minutes. Stop uploading your billion-dollar idea to ChatGPT's servers. Stop waiting weeks for a first draft.

**mPAPA is the first fully local AI patent drafting system.** It runs on YOUR machine*. Your invention data never touches the internet. Ever.

*Very good results can already be achieved with LLMs as small as 4.5 GB (GEMMA-4-e2b). When combined with an embedded model, even “standard” graphics cards or newer laptops are sufficient (VRAM >8GB).

---

### What it does in 30 minutes flat:

🔍 **Searches 5 patent/paper databases simultaneously** — EPO, Google Patents, Google Scholar, ArXiv, PubMed. 120+ references collected and analyzed automatically.

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
| **Cost** | i don't know yet | $20-200/month | $5,000-50,000 |
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

### One command. That's it.

```bash
uv run python -m patent_system.main
```

Open `http://localhost:8080`. Describe your invention. Let the AI work. Export your patent draft.

**Your idea stays on your machine. Where it belongs.**

---

*mPAPA — because your invention is too valuable to upload to someone else's server.*

*Built by [Antlia](https://antlia.io). Open source. Local first. Always.*
