# Patent Reviewer — Development Status

## ✅ Phase 1: Core Module (Complete)

- [x] **Spec & requirements** — `.kiro/specs/patent-reviewer/`
- [x] **Rule Card model** — `src/patent_system/reviewer/rule_card.py` (Pydantic v2, JSON serialization)
- [x] **Card Registry** — `src/patent_system/reviewer/card_registry.py` (local cache, GitHub download, offline)
- [x] **Review Engine** — `src/patent_system/reviewer/engine.py` (prompt builder, LLM call, response parser)
- [x] **Report renderer** — `src/patent_system/reviewer/report.py` (Markdown + DOCX export)
- [x] **GUI Panel** — `src/patent_system/gui/reviewer_panel.py` (NiceGUI tab: paste/import/use-draft, card selection, async review, severity cards, export)
- [x] **DB persistence** — `review_sessions` + `review_findings` tables, repositories, save_report()
- [x] **"Review Draft" button** — in Draft Panel, switches to Patent Review tab
- [x] **JSON Schema** — `rule-cards/schema/rule-card-v1.schema.json`
- [x] **3 hand-authored Rule Cards** — EPO claim structure (12 rules), EPO novelty (9 rules), USPTO §101 (7 rules)
- [x] **Testdata extraction scripts** — `scripts/extract_testdata/` (OCR via Qwen3.5-9B multimodal)
- [x] **vLLM start scripts** — `scripts/start-vllm-9b.sh`, `scripts/start-vllm-27b.sh` (300W GPU limit)
- [x] **Rule Card batch compiler** — `scripts/compile_rule_cards/` (CLI pipeline using Qwen3.6-27B)
- [x] **Prosecution test suite** — 9 integration tests, all passing (3.5 min with live LLM)
- [x] **34 testdata files converted** — from scanned PDFs via OCR (EP, PCT, US prosecution histories)

## 🔲 Phase 2: Full EP + US Coverage (Next)

### Rule Card Production Pipeline

The full pipeline is now: **Download → (Translate) → Compile → Review → Commit**

```bash
# 1. Download all guideline PDFs (original language)
uv run python scripts/compile_rule_cards/download_sources.py

# 2. Start vLLM with 27B model (for compilation + translation)
./scripts/start-vllm-27b.sh

# 3. Compile a card (translates non-English sources automatically)
uv run python scripts/compile_cards.py --jurisdiction EP --task inventive_step \
    --source scripts/compile_rule_cards/sources/EP/epo_inventive_step.pdf
```

**Key design decision**: Sources are downloaded in their ORIGINAL language
(Japanese for JPO, Chinese for CNIPA, Korean for KIPO, German for DPMA).
Translation to English is handled by the rule card compiler via the local LLM.
This means no waiting for official English translations — we always have the latest rules.

**Configure sources**: Edit `scripts/compile_rule_cards/sources.toml`
- Comment out jurisdictions or chapters you don't need
- Add new sources as they become available
- Run with `--dry-run` to check for updates without downloading

**EPO cards needed:**
- [ ] `epo_inventive_step` — Art. 56, Problem-Solution Approach (Part G-VII)
- [ ] `epo_sufficiency` — Art. 83, Sufficiency of Disclosure (Part F-III)
- [ ] `epo_unity` — Art. 82, Unity of Invention (Part F-V)
- [ ] `epo_added_matter` — Art. 123(2), Amendments (Part H-IV)
- [ ] `epo_industrial_application` — Art. 57 (Part G-III)
- [ ] `epo_priority` — Art. 87-89, Priority (Part A-III)

**USPTO cards needed:**
- [ ] `uspto_novelty` — 35 U.S.C. §102 (MPEP 2131-2138)
- [ ] `uspto_obviousness` — 35 U.S.C. §103, KSR/Graham (MPEP 2141-2145)
- [ ] `uspto_written_description` — 35 U.S.C. §112(a) (MPEP 2161-2164)
- [ ] `uspto_enablement` — 35 U.S.C. §112(a) (MPEP 2164)
- [ ] `uspto_definiteness` — 35 U.S.C. §112(b) (MPEP 2171-2175)
- [ ] `uspto_double_patenting` — MPEP 804
- [ ] `uspto_restriction` — 35 U.S.C. §121 (MPEP 803-808)

### Test Suite Expansion

- [ ] Add tests for inventive step assessment (once cards exist)
- [ ] Add tests for §103 obviousness (once cards exist)
- [ ] Add cross-jurisdiction comparison tests
- [ ] Benchmark reviewer accuracy against real examiner findings in prosecution history

## 🔲 Phase 3: Additional Jurisdictions

- [ ] **PCT** — WIPO ISPE Guidelines (prior art search, written opinion)
- [ ] **CNIPA** — Chinese Patent Examination Guidelines (translate from Chinese)
- [ ] **JPO** — Japanese Examination Guidelines (official English translation available)
- [ ] **KIPO** — Korean Examination Guidelines
- [ ] **IPO India** — Indian Patent Office Manual
- [ ] **DPMA** — German Prüfungsrichtlinien (translate from German)

## 🔲 Phase 4: Advanced Features

- [ ] Multi-jurisdiction comparative review (run EP + US simultaneously, highlight differences)
- [ ] Amendment suggestion engine (propose claim rewording to address findings)
- [ ] Prosecution strategy recommendations
- [ ] Rule Card update notifications (detect when new guideline editions are published)
- [ ] Community-contributed card validation pipeline

## Architecture Notes

```
src/patent_system/reviewer/     # Core module (independent of main workflow)
├── __init__.py
├── rule_card.py                # Pydantic models: RuleCard, Rule, SourceDocument
├── card_registry.py            # Local cache + GitHub download, offline-capable
├── engine.py                   # ReviewEngine: prompt builder → LLM → response parser
├── report.py                   # Markdown + DOCX rendering
├── persistence.py              # save_report() helper
└── repository.py               # ReviewSessionRepository, ReviewFindingRepository

rule-cards/                     # Pre-compiled JSON rule cards (committed to repo)
├── schema/rule-card-v1.schema.json
├── index.json                  # Card manifest
├── EP/                         # European Patent Office cards
└── US/                         # USPTO cards

scripts/
├── compile_rule_cards/         # Batch compiler (uses Qwen3.6-27B via vLLM)
├── extract_testdata/           # PDF→Markdown OCR (uses Qwen3.5-9B via vLLM)
├── start-vllm-9b.sh           # OCR extraction model
├── start-vllm-27b.sh          # Card compilation model
└── compile_cards.py            # CLI entry point
```

## How to Produce New Rule Cards

1. Download the source guideline PDF (e.g., from epo.org or USPTO MPEP)
2. Place it in `scripts/compile_rule_cards/sources/`
3. Start vLLM: `./scripts/start-vllm-27b.sh`
4. Run: `uv run python scripts/compile_cards.py --jurisdiction EP --task <task_name> --source <path_to_pdf>`
5. Review the generated JSON, validate against schema
6. Move to `rule-cards/<jurisdiction>/` and update `rule-cards/index.json`
7. Commit

## Test Commands

```bash
# Unit tests (no LLM needed)
uv run pytest tests/unit -q

# Integration tests with live LLM (needs vLLM on :8000 or LM Studio on :1234)
uv run pytest tests/integration/test_reviewer_prosecution.py -v

# All tests
uv run pytest -q
```
