# Architecture Guide: Building Local AI-Powered Document Generation Systems

A practical guide based on mPAPA — a local-first, AI-powered patent analysis and drafting system. Use this as a blueprint for similar projects in pharma, medical devices, regulatory compliance, or any domain requiring structured AI-assisted document generation.

---

## 1. The Application Pattern

The core pattern is a **three-phase iterative workflow** with a final document generation step:

```
┌─────────────────────────────────────────────────────────────┐
│  Tab 1: RESEARCH          Tab 2: AI CHAT       Tab 3: DRAFT │
│                                                              │
│  Idea + Search Terms  ──→  RAG Chat         ──→  Multi-Step  │
│  Source Selection           (refine idea,        Workflow     │
│  Reference Search           explore refs)        (AI agents)  │
│  Local Doc Upload                                Document     │
│       ↑                        │                 Export       │
│       └────── refine ──────────┘                             │
└─────────────────────────────────────────────────────────────┘
```

### Phase 1: Research (Input + Reference Collection)
- User formulates their idea/description and search terms
- System searches multiple external sources (switchable on/off)
- User uploads local documents (PDFs, DOCX, TXT)
- All references are stored locally with embeddings for RAG
- User can refine search terms and re-search iteratively

### Phase 2: AI Chat (Exploration + Refinement)
- RAG-powered chat over all collected references + idea + search terms
- User explores the reference landscape, asks questions
- Insights feed back into refining the idea and search terms (back to Tab 1)
- Chat history is persisted per topic

### Phase 3: Document Generation (Multi-Step Workflow)
- Step-by-step AI workflow using all prior context (idea, references, chat insights)
- Each step: AI generates → user reviews/edits → continues or re-runs
- Final export to DOCX with proper formatting, header/footer, styles

**The key insight:** Tabs 1 and 2 are iterative — the user goes back and forth refining. Tab 3 consumes everything from Tabs 1 and 2 as input to the AI agents.

---

## 2. Tech Stack

| Layer | Technology | Why |
|---|---|---|
| **Web UI** | NiceGUI | Python-native, reactive, no JS build step |
| **Agent Orchestration** | LangGraph | State machines with `interrupt_after` for human review |
| **LLM Prompting** | DSPy | Structured signatures, chain-of-thought, domain-specific |
| **LLM Backend** | LM Studio | Local inference, OpenAI-compatible API, multiple models |
| **Embeddings** | LM Studio `/v1/embeddings` | Same local server, no separate embedding service |
| **RAG** | LlamaIndex | Vector index per topic, backed by LM Studio embeddings |
| **Data Validation** | Pydantic + pydantic-settings | Settings from `.env`, domain models |
| **Document Export** | python-docx | DOCX with styles, markdown conversion, templates |
| **Database** | SQLite | Zero-config, file-based, foreign keys, stores everything |
| **Package Manager** | uv | Fast, reliable, replaces pip/poetry/conda |

### Key Principle: Everything Local
- All LLM inference runs on your machine via LM Studio
- All embeddings generated locally via LM Studio's `/v1/embeddings` endpoint
- All data stored in a local SQLite file
- Only public database queries (ArXiv, PubMed, patent offices) go over the network
- Sensitive data never leaves the local machine

### LM Studio Setup
LM Studio exposes an OpenAI-compatible API at `http://localhost:1234/v1`. You need:
- A **chat/completion model** for all AI agents (e.g., Qwen, Llama, Mistral)
- An **embedding model** for RAG (e.g., `text-embedding-nomic-embed-text-v1.5`)

Both are configured via environment variables — no code changes needed to switch models.

---

## 3. Project Structure

```
src/your_system/
├── main.py                  # Entry point: settings, DB, workflow, NiceGUI launch
├── config.py                # AppSettings (pydantic-settings, env prefix)
├── logging_config.py        # Structured JSON logging with event helpers
├── exceptions.py            # Custom exception hierarchy (AgentError base)
│
├── agents/                  # LangGraph workflow + agent nodes
│   ├── state.py             # WorkflowState TypedDict (flows through all nodes)
│   ├── graph.py             # StateGraph: nodes, edges, interrupt_after
│   ├── step_one.py          # Agent node: step_one_node(state) -> dict
│   ├── step_two.py          # Each step = one file, one node function
│   └── ...
│
├── db/                      # SQLite persistence
│   ├── schema.py            # All CREATE TABLE statements + init_schema()
│   ├── models.py            # Pydantic domain models
│   └── repository.py        # Repository classes (one per table, CRUD)
│
├── dspy_modules/            # DSPy prompt definitions
│   ├── signatures.py        # DSPy Signature classes (the "prompts")
│   └── modules.py           # DSPy Module wrappers (ChainOfThought)
│
├── rag/                     # RAG subsystem
│   ├── embeddings.py        # LM Studio embedding service (OpenAI-compatible)
│   └── engine.py            # RAG engine: per-topic vector index + query
│
├── parsers/                 # External source parsers (if applicable)
│   ├── base.py              # BaseSourceParser ABC
│   └── source_x.py          # One parser per external source
│
├── export/                  # Document generation
│   ├── docx_exporter.py     # DOCX export with markdown-to-docx conversion
│   └── templates/           # .docx template files (styles, header, footer)
│
└── gui/                     # NiceGUI web interface
    ├── layout.py            # Main page: header, drawer, tab navigation
    ├── research_panel.py    # Tab 1: idea, search terms, sources, results
    ├── chat_panel.py        # Tab 2: RAG-powered AI chat
    └── workflow_panel.py    # Tab 3: multi-step AI workflow + export

tests/
├── conftest.py              # Shared fixtures (in-memory DB, mock LLM)
├── unit/                    # One test file per source module
├── integration/             # Cross-module flows, agent tests
└── property/                # Hypothesis property-based tests
```

---

## 4. Tab 1: Research Panel

### What It Does
1. User enters a primary description and additional search terms
2. User selects which external sources to query (checkboxes, on/off)
3. System searches each source **one term at a time** (avoids URL/query limits)
4. Results are stored in SQLite with embeddings for RAG
5. User can upload local documents (PDF, DOCX, TXT)
6. User can refine terms and re-search iteratively

### Search Term Validation
Validate terms before searching and warn about problematic formatting:
- Quotes + AND operators → "use simple phrases instead"
- Wildcards → "not supported by all sources"
- Very long terms → "consider splitting"

### Per-Term Query Pattern
**Critical:** Never join all search terms into one giant query. Each term = one API call.

```python
for i, term in enumerate(search_terms):
    if i > 0:
        time.sleep(rate_limit_delay)
    results = query_source(term)
    all_results.extend(deduplicate(results))
```

This pattern is used for ALL sources (ArXiv, PubMed, Google Scholar, Google Patents, EPO OPS). It avoids URL length limits, API errors, and gives better results.

### Embedding + Storage
Every reference (patent or paper) gets:
1. Stored in SQLite (title, abstract, source, full text if available)
2. An embedding vector generated via LM Studio's `/v1/embeddings`
3. The embedding stored as packed float32 bytes in the same SQLite row

```python
class EmbeddingService:
    def __init__(self, model_name, api_base="http://localhost:1234/v1"):
        self._client = openai.OpenAI(base_url=api_base, api_key="not-needed")

    def generate_embedding(self, text: str) -> bytes:
        resp = self._client.embeddings.create(input=[text], model=self._model_name)
        vector = resp.data[0].embedding
        return struct.pack(f"{len(vector)}f", *vector)
```

---

## 5. Tab 2: AI Chat (RAG-Powered)

### What It Does
1. Builds a per-topic vector index from all references + local documents
2. User asks questions → system retrieves relevant chunks → LLM answers with context
3. Chat history is persisted per topic
4. Insights from chat feed back into refining idea/search terms (Tab 1)

### RAG Architecture
```
All references + local docs
        ↓
    Chunking + Embedding (LM Studio)
        ↓
    Per-topic VectorStoreIndex (LlamaIndex)
        ↓
    User query → similarity search → top-K chunks → LLM context → answer
```

### Implementation
```python
class RAGEngine:
    def __init__(self, embed_model: LMStudioEmbedding):
        Settings.embed_model = embed_model

    def index_documents(self, topic_id, docs):
        # Build/update VectorStoreIndex for this topic
        ...

    def query(self, topic_id, query_text, top_k=5) -> list[dict]:
        # Retrieve relevant chunks
        ...
```

The embedding model is the same LM Studio instance used for LLM inference — just a different model loaded (e.g., `text-embedding-nomic-embed-text-v1.5`).

---

## 6. Tab 3: Multi-Step Workflow

### 6.1 Define Your Steps

```python
WORKFLOW_STEPS = [
    "initial_input",       # Read-only: shows idea + search terms from Tab 1
    "step_two",            # AI generates based on input + references
    "step_three",          # AI analyzes previous output + references
    ...
    "final_output",        # AI generates the final document
]
```

**For a DQ/IQ/OQ/PQ system:**
```python
WORKFLOW_STEPS = [
    "system_description",      # Read-only: from Tab 1
    "risk_assessment",         # AI: identify risks from system + references
    "dq_protocol",             # AI: Design Qualification protocol
    "iq_protocol",             # AI: Installation Qualification
    "oq_protocol",             # AI: Operational Qualification
    "pq_protocol",             # AI: Performance Qualification
    "traceability_matrix",     # AI: requirements-to-test traceability
    "summary_report",          # AI: qualification summary
]
```

### 6.2 State TypedDict

All data flows through a single TypedDict:

```python
class WorkflowState(TypedDict):
    project_id: int
    input_data: str                # From Tab 1 (idea + search terms)
    references: list[dict]         # From Tab 1 (pre-loaded from local DB)
    step_two_output: str           # Generated by step 2
    step_three_output: str         # Generated by step 3
    ...
    current_step: str
```

**Key:** Pre-load references from the local DB into the state before starting the graph. The workflow nodes should NEVER make external network requests — they only use what's already collected.

### 6.3 DSPy Signatures

Each AI step gets a DSPy Signature. The docstring and field descriptions ARE your prompt:

```python
class DraftIQProtocol(dspy.Signature):
    """Draft an Installation Qualification protocol.

    Verify that the system is installed correctly per manufacturer specs.
    Include: scope, equipment list, installation checks, acceptance criteria,
    deviations handling, and sign-off sections.
    Follow GAMP 5, FDA 21 CFR Part 11, EU Annex 11.
    """
    system_description: str = dspy.InputField(desc="System description and intended use")
    risk_assessment: str = dspy.InputField(desc="Risk assessment findings")
    dq_protocol: str = dspy.InputField(desc="Approved DQ protocol for reference")
    reference_summary: str = dspy.InputField(
        desc="Summary of all reference documents including regulations and SOPs"
    )
    iq_protocol: str = dspy.OutputField(
        desc="Complete IQ protocol with scope, checks, criteria, and sign-off sections"
    )
```

**Lesson:** Be specific in descriptions. Mention standards, regulations, expected sections. The LLM follows these closely.

### 6.4 Agent Nodes

Each node follows the same pattern:

```python
def step_node(state: WorkflowState) -> dict[str, Any]:
    start = time.monotonic()

    # 1. Extract inputs from state
    input_a = state.get("field_a", "")
    input_b = _prepare_text(state.get("field_b"))  # Handle str/dict/None

    # 2. Call DSPy module
    module = MyModule()
    try:
        prediction = module(input_a=input_a, input_b=input_b)
    except (ConnectionError, OSError, ...) as exc:
        raise LLMConnectionError(f"LLM unreachable: {exc}") from exc

    # 3. Log and return
    log_agent_invocation(logger, name="StepAgent", ...)
    return {"step_output": prediction.output_field, "current_step": "step_name"}
```

**Always handle string/dict/None for state values** — earlier nodes may serialize dicts to strings.

### 6.5 Graph Wiring

```python
def build_workflow(checkpointer):
    graph = StateGraph(WorkflowState)

    steps = [
        ("initial_input", passthrough_node),      # No LLM, no interrupt
        ("step_two", step_two_node),              # LLM + interrupt_after
        ("step_three", step_three_node),           # LLM + interrupt_after
        ...
        ("final_output", final_node),              # LLM, no interrupt (last)
    ]

    interrupt_after = []
    for i, (key, fn) in enumerate(steps):
        graph.add_node(key, fn)
        if 0 < i < len(steps) - 1:  # Skip first (passthrough) and last
            interrupt_after.append(key)

    graph.set_entry_point(steps[0][0])
    for i in range(len(steps) - 1):
        graph.add_edge(steps[i][0], steps[i + 1][0])
    graph.add_edge(steps[-1][0], END)

    return graph.compile(checkpointer=checkpointer, interrupt_after=interrupt_after)
```

**Critical:** Use `interrupt_after` at compile time. Do NOT call `interrupt()` inside nodes — it prevents the return value from being applied to state.

### 6.6 UI Flow

```
Step 1 (read-only): Shows idea + search terms from Tab 1
  → User clicks "Continue to Next Step"
  → Graph starts: passthrough → step 2 (LLM) → interrupt_after
  → Step 2 textarea fills with AI output
  → User reviews, edits if needed → clicks "Continue"
  → Graph resumes: step 3 (LLM) → interrupt_after
  → ...repeat until final step completes
  → Export button becomes available
```

### 6.7 Re-run Pattern

After the initial run, each step gets a "Rerun this Step" button:

```python
async def _rerun_single_step(step_key):
    old_content = panel_state["step_contents"][step_key]
    textarea.value = ""  # Clear immediately — user sees it's working

    state = _build_state_from_current_panel_contents()
    result = await asyncio.to_thread(node_fn, state)  # Call node directly
    new_content = extract_content(result)

    textarea.value = new_content
    # Show Keep/Revert confirmation dialog
```

**Key lesson:** Don't restart the LangGraph for re-runs. Call the node function directly — simpler and more reliable.

### 6.8 State Injection for Edits

When the user edits a step's content and presses Continue, inject edits into the checkpoint before resuming:

```python
if is_resuming:
    state_updates = {}
    for step_key, state_field in STEP_TO_STATE_FIELD.items():
        edited = panel_state["step_contents"].get(step_key)
        if edited:
            state_updates[state_field] = edited
    if state_updates:
        await asyncio.to_thread(workflow.update_state, config, state_updates)
    stream_input = None  # Resume
```

---

## 7. Persistence

### 7.1 Database Schema Pattern

```sql
-- Projects/topics
CREATE TABLE projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Workflow step content + status
CREATE TABLE workflow_steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL,
    step_key TEXT NOT NULL,
    content TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'pending',  -- 'pending' or 'completed'
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (project_id) REFERENCES projects(id),
    UNIQUE(project_id, step_key)
);

-- References (from external search)
CREATE TABLE references (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL,
    title TEXT, abstract TEXT, source TEXT,
    embedding BLOB,  -- packed float32 vector
    ...
);

-- Chat history
CREATE TABLE chat_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL,
    role TEXT, message TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Local uploaded documents
CREATE TABLE local_documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL,
    filename TEXT, content TEXT,
    embedding BLOB
);
```

### 7.2 Repository Pattern

One class per table, all follow the same pattern:

```python
class WorkflowStepRepository:
    def __init__(self, conn: sqlite3.Connection): ...
    def upsert(self, project_id, step_key, content, status): ...
    def get_by_project(self, project_id) -> list[dict]: ...
    def get_step(self, project_id, step_key) -> dict | None: ...
    def reset_from_step(self, project_id, step_key): ...
```

### 7.3 Save Strategy

| Event | What to save | Status |
|---|---|---|
| AI generates content (streaming) | Content to memory + DB | `"pending"` |
| User clicks "Continue" | Read textarea → DB | `"completed"` |
| User clicks "Save Edits" | Read textarea → DB | `"completed"` |
| User clicks "Rerun" → "Keep" | New content → DB | `"completed"` |
| Graph completes (final step) | All steps with content → DB | `"completed"` |

**Never overwrite `"completed"` with `"pending"`** during streaming.

---

## 8. Document Export

### 8.1 Markdown-to-DOCX

LLMs output markdown. Convert to proper DOCX:

```python
def _add_markdown_content(doc, text):
    for line in text.split("\n"):
        if re.match(r'^#{2,6}\s+', line):       # ## Heading
            _safe_add_heading(doc, heading_text, level)
        elif re.match(r'^[-*]\s+', line):        # - bullet
            _safe_add_list(doc, item_text, 'List Bullet')
        elif re.match(r'^\d+\.\s+', line):       # 1. numbered
            _safe_add_list(doc, item_text, 'List Number')
        else:                                     # paragraph
            p = doc.add_paragraph()
            _add_inline_formatting(p, line)       # **bold**, *italic*, `code`
```

### 8.2 Template with Styles

Create templates programmatically to guarantee all styles exist:

```python
doc = Document()
doc.styles['Heading 1'].font.size = Pt(16)
doc.styles['Heading 1'].font.bold = True
# Add header: centered title + bottom border
# Add footer: copyright | confidential | Page X of Y
doc.save('templates/template.docx')
```

### 8.3 Safe Style Usage

Always handle missing styles (custom templates may not have them):

```python
def _safe_add_heading(doc, text, level=1):
    try:
        doc.add_heading(text, level=level)
    except KeyError:
        p = doc.add_paragraph()
        run = p.add_run(text)
        run.bold = True
        run.font.size = Pt(18 - level * 2)
```

---

## 9. Configuration

All settings via environment variables with a prefix:

```python
class AppSettings(BaseSettings):
    lm_studio_base_url: str = "http://localhost:1234/v1"
    lm_studio_api_key: str = "not-needed"
    embedding_model_name: str = "text-embedding-nomic-embed-text-v1.5"
    database_path: Path = Path("data/system.db")
    docx_template_dir: Path = Path("src/system/export/templates")
    docx_template_name: str | None = None
    search_max_results_per_source: int = 10
    search_request_delay_seconds: float = 3.0
    ...
    model_config = {"env_file": ".env", "env_prefix": "YOUR_PREFIX_"}
```

---

## 10. Lessons Learned

### What Worked
1. **Three-tab iterative flow** — research → chat → generate, with back-and-forth
2. **LM Studio for everything** — LLM + embeddings from one local server
3. **DSPy signatures as domain prompts** — descriptive docstrings guide the LLM
4. **`interrupt_after` at compile time** — reliable pause/resume
5. **Per-term queries** — one search term = one API call, avoids all URL/query limits
6. **Direct node calls for re-runs** — bypass the graph, call the function
7. **SQLite for everything** — workflow state, checkpoints, documents, embeddings, chat
8. **Pre-load references into graph state** — workflow nodes never hit the network
9. **Markdown-to-DOCX conversion** — LLM outputs markdown, export converts to styles

### What to Avoid
1. **`interrupt()` inside nodes** — prevents return value from being applied to state
2. **Joining search terms into one query** — causes URL length / API errors
3. **`autogrow` on NiceGUI textareas** — unreliable; use fixed height + scroll
4. **Auto-marking steps complete during streaming** — user should confirm
5. **Assuming state values are always dicts** — they may be strings after serialization
6. **Recursive async calls** — causes stack overflow in NiceGUI's event loop
7. **Restarting the graph for re-runs** — graph always starts from entry point
8. **External network calls in workflow nodes** — pre-load everything into state

---

## 11. Getting Started

```bash
# 1. Create project
uv init my-system
uv add nicegui langgraph langgraph-checkpoint-sqlite dspy-ai \
      llama-index llama-index-embeddings-huggingface \
      python-docx pydantic-settings openai

# 2. Set up LM Studio
# - Load a chat model (e.g., Qwen 2.5, Llama 3.1, Mistral)
# - Load an embedding model (e.g., nomic-embed-text-v1.5)
# - Start the server on port 1234

# 3. Create .env
echo 'YOUR_PREFIX_LM_STUDIO_BASE_URL=http://localhost:1234/v1' > .env
echo 'YOUR_PREFIX_EMBEDDING_MODEL_NAME=text-embedding-nomic-embed-text-v1.5' >> .env

# 4. Build iteratively
# - Start with Tab 1 (input + search)
# - Add Tab 2 (RAG chat)
# - Add Tab 3 (workflow + export)
```

---

## 12. Example: Qualification Documentation Generator

### Tabs
1. **System Input** — describe the system, upload existing docs (URS, FRS, SOPs), add search terms for regulatory references
2. **Regulatory Chat** — RAG chat over uploaded docs + regulatory references, explore requirements
3. **Qualification Workflow** — 8-step AI generation of DQ/IQ/OQ/PQ documents

### Workflow Steps
| Step | Agent | Input | Output |
|---|---|---|---|
| 1. System Description | Passthrough | From Tab 1 | Read-only display |
| 2. Risk Assessment | LLM | System desc + references | Risk matrix, critical parameters |
| 3. DQ Protocol | LLM | System + risks + refs | Design Qualification protocol |
| 4. IQ Protocol | LLM | System + risks + DQ + refs | Installation Qualification |
| 5. OQ Protocol | LLM | System + risks + IQ + refs | Operational Qualification |
| 6. PQ Protocol | LLM | System + risks + OQ + refs | Performance Qualification |
| 7. Traceability | LLM | All protocols + requirements | Requirements traceability matrix |
| 8. Summary | LLM | All of the above | Qualification summary report |

### DSPy Signature Example
```python
class DraftOQProtocol(dspy.Signature):
    """Draft an Operational Qualification protocol for a computerized system.

    Verify that the system operates correctly throughout all anticipated
    operating ranges. Include: scope, test cases for each critical function,
    boundary conditions, error handling, acceptance criteria per GAMP 5
    risk-based approach. Reference FDA 21 CFR Part 11 for electronic records
    and EU Annex 11 for computerized systems.
    """
    system_description: str = dspy.InputField(desc="System description and intended use")
    risk_assessment: str = dspy.InputField(desc="Risk assessment with critical parameters")
    iq_protocol: str = dspy.InputField(desc="Approved IQ protocol for reference")
    reference_summary: str = dspy.InputField(
        desc="Summary of regulatory references, SOPs, and uploaded documents"
    )
    oq_protocol: str = dspy.OutputField(
        desc="Complete OQ protocol: scope, test cases, boundary tests, "
        "error handling, acceptance criteria, deviations procedure, sign-off"
    )
```

### Example: RAG Knowledge Base for Pharma Documents
Same architecture, but Tab 3 becomes a **query interface** instead of a generation workflow:
- Tab 1: Upload SOPs, batch records, validation docs, regulatory guidelines
- Tab 2: RAG chat — "What does SOP-123 say about temperature monitoring?"
- Tab 3: Report generator — AI summarizes findings, generates gap analyses, audit prep docs
