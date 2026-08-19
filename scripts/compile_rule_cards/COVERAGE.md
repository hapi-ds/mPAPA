# Rule Card Coverage — Guideline Chapters & Rationale

This document explains which patent examination guideline chapters are covered by the Rule Card system, why they were selected, and what was excluded.

## Design Principle

The Patent Reviewer checks a **patent text** (claims + description) against examination rules. We only include chapters that contain **substantive rules** a patent application can violate. Purely procedural chapters (how to file, deadlines, fee payment) are excluded because they don't produce reviewable findings against a patent draft.

---

## EPO Guidelines for Examination (Parts A–H)

### ✅ Included (Substantive — reviewable against patent text)

| Part | Chapter | Card ID | What it checks |
|------|---------|---------|----------------|
| **F-II** | Content of Application | `epo_description_format` | Description structure (technical field, background, detailed description, drawings, abstract) |
| **F-III** | Sufficiency of Disclosure | `epo_sufficiency` | Can a skilled person reproduce the invention from the description? (Art. 83) |
| **F-IV** | Claims | `epo_claim_structure` | Two-part form, clarity, conciseness, support, dependent claims (Art. 84, Rule 43) |
| **F-V** | Unity of Invention | `epo_unity` | Do all claims relate to one inventive concept? (Art. 82) |
| **F-VI** | Priority | `epo_priority` | Is priority validly claimed? Same invention test (Art. 87-89) |
| **G-II** | Inventions | `epo_patentable_inventions` | Is it patentable subject matter? Exclusions: discoveries, math, software as such (Art. 52) |
| **G-III** | Industrial Application | `epo_industrial_application` | Can it be made or used in industry? (Art. 57) |
| **G-IV** | State of the Art | `epo_state_of_art` | What counts as prior art? Effective dates, Art. 54(2) definition |
| **G-V** | Non-Prejudicial Disclosures | `epo_non_prejudicial_disclosures` | Grace period: when own disclosure doesn't destroy novelty (Art. 55) |
| **G-VI** | Novelty | `epo_novelty` | Is the claim new vs. each prior art document individually? (Art. 54) |
| **G-VII** | Inventive Step | `epo_inventive_step` | Problem-solution approach, obviousness, could/would test (Art. 56) |
| **H-IV** | Amendments | `epo_added_matter` | Does amended text add subject matter beyond original filing? (Art. 123(2)(3)) |
| **D-V** | Opposition (Substantive) | `epo_opposition` | Grounds for opposition: novelty, inventive step, sufficiency, added matter |

### ❌ Excluded (Procedural — no patent text to check against)

| Part | Title | Why excluded |
|------|-------|-------------|
| **General** | Preliminary remarks | Administrative overview of the EPO |
| **A** | Formalities Examination | Filing requirements: signatures, fees, translations, formal deficiencies, drawings format. Nothing about patent content quality. |
| **B** | Search | How examiners conduct prior art searches. Internal process — an applicant can't violate "search rules." |
| **C** | Procedural Aspects of Substantive Examination | Communication formats, time limits, stages of examination, division's internal workflow. Process, not substance. |
| **D-I to D-IV, D-VI to D-X** | Opposition Procedure | Who can oppose, time limits, procedure steps, costs. Only D-V (substantive grounds) is included. |
| **E** | General Procedural Matters | Oral proceedings, evidence, interruptions, time limits, loss of rights, appeals. All procedure. |
| **F-I** | Introduction to Part F | Overview only, no normative content. |
| **G-I** | Introduction to Patentability | Brief overview, no rules. |
| **H-I to H-III, H-V, H-VI** | Amendment Procedure & Examples | Right to amend (procedural), admissibility (procedural), correction of errors. Only H-IV (substantive Art. 123 test) is included. |

---

## USPTO — MPEP Chapters

### ✅ Included

| MPEP Sections | Card ID | What it checks |
|---------------|---------|----------------|
| 2104-2106 | `uspto_subject_matter` | §101 Alice/Mayo: abstract ideas, laws of nature, natural phenomena |
| 2131-2138 | `uspto_novelty` | §102 AIA: anticipation, all-elements test, inherency, grace period |
| 2141-2145 | `uspto_obviousness` | §103 KSR/Graham: obviousness, motivation to combine, teaching away |
| 2161-2164 | `uspto_written_description` | §112(a) written description: possession of the invention |
| 2164 | `uspto_enablement` | §112(a) enablement: can a skilled person make/use without undue experimentation? |
| 2171-2175 | `uspto_definiteness` | §112(b): are claims definite? Means-plus-function, relative terms |
| 804 | `uspto_double_patenting` | Same invention / obviousness-type double patenting, terminal disclaimers |
| 803-808 | `uspto_restriction` | §121: restriction requirement, election, rejoinder |

### ❌ Excluded

| MPEP Chapter | Why |
|---|---|
| 100-199 | Secrecy, national security orders |
| 200-299 | Types of applications, filing dates — procedural |
| 300-399 | Ownership, assignment — procedural |
| 400-499 | Representative matters — procedural |
| 500-599 | Receipt of applications — procedural |
| 600-699 | Parts of application (formal requirements only) |
| 700-799 | Examination procedures — internal examiner workflow |
| 900-999 | Prior art search tools — internal |
| 1000-1099 | IDS, duty of disclosure — partially relevant but procedural |
| 1200-1299 | Appeals — procedural |
| 1300-1399 | Allowance, issue — procedural |
| 1400-1499 | Correction of patents — procedural |
| 1800-1899 | PCT — covered separately under PCT jurisdiction |
| 2200-2299 | Citation of prior art — procedural |
| 2300-2399 | Interference — rare, legacy |
| 2700-2799 | Patent terms — procedural |

---

## PCT — WIPO ISPE Guidelines

### ✅ Included

| Chapter | Card ID | What it checks |
|---------|---------|----------------|
| 15 (Prior Art) | `pct_novelty` | International search: what constitutes relevant prior art |
| 3 (Written Opinion) | `pct_inventive_step` | Written opinion on patentability |
| 1 (Unity) | `pct_unity` | Unity of invention for international applications |

---

## JPO — 審査基準 (Japanese, translated during compilation)

### ✅ Included

| Chapter | Card ID | What it checks |
|---------|---------|----------------|
| Part III Ch.2 | `jpo_novelty` | 新規性 — Novelty |
| Part III Ch.2 | `jpo_inventive_step` | 進歩性 — Inventive step |
| Part III Ch.1 | `jpo_claim_structure` | 発明の認定 — Claim interpretation |
| Part I | `jpo_sufficiency` | 明細書の記載要件 — Description requirements |
| Part III Ch.3 | `jpo_industrial_application` | 産業上利用可能性 — Industrial applicability |

---

## Jurisdictions pending (manual source acquisition)

| Jurisdiction | Status | Notes |
|---|---|---|
| **CNIPA (China)** | Disabled | No stable public URL for examination guidelines. Place Chinese PDF manually. |
| **KIPO (Korea)** | Disabled | No freely available English or Korean PDF download link. |
| **IPO India** | Disabled | 2019 manual available — dated but usable. |
| **DPMA (Germany)** | Disabled | German guidelines available. Largely overlaps with EPO for substantive law. |

---

## How to update when new editions are published

1. Check the office website for new PDF URLs (see `source_registry.py` header for links)
2. Update the URL and edition string in `scripts/compile_rule_cards/source_registry.py`
3. Run `uv run python scripts/compile_rule_cards/download_sources.py` — it detects changes automatically
4. Re-compile affected cards with `uv run python scripts/compile_cards.py --jurisdiction EP`
