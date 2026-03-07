# Logical Fallacy Detector — Project Context 

## 1) Project Overview
This project builds a **Logical Fallacy Detector** that identifies logical fallacies in argumentative text (and optionally in **two-speaker discussion transcripts** produced from audio). The core deliverable is a preprocessing + modeling pipeline that:

- Collects and cleans fallacy-labeled datasets
- Normalizes everything into a single canonical schema
- Produces train/dev/test splits (leakage-safe)
- Trains a classifier to predict:
  - **Coarse label:** `fallacy` vs `no_fallacy`
  - **Fine label:** fallacy type (e.g., ad hominem, straw man, etc.)
- (Optional) Supports an **audio→transcript** path (diarization + ASR), then runs the same fallacy detector on the transcript.

Primary focus for Week 6: **Data collection + cleaning + preprocessing** and writing it up in the Overleaf paper template.

---

## 2) Scope and Assumptions
### In scope
- Text fallacy detection on short-to-medium argument spans (comments, debate turns, paragraphs).
- Unified dataset creation from multiple sources.
- Reproducible preprocessing pipeline with logs and reports.

### Optional extensions (only if time permits)
- Two-speaker transcript processing (diarization + ASR).
- Context-aware classification using previous turns as context.

### Out of scope (for now)
- Full argument-mining (claim/premise extraction) beyond lightweight features.
- Real-time deployment requirements (can be prototyped later).

---

## 3) Target Labels (Taxonomy)
The model will output a fine-grained fallacy label from a fixed taxonomy.

**Recommended target taxonomy (7 + none):**
- `ad_hominem`
- `straw_man`
- `appeal_to_authority`
- `false_dilemma`
- `false_cause`
- `hasty_generalization`
- `slippery_slope`
- `none` (no fallacy)

Optional:
- `other` (if datasets contain fallacies outside the target taxonomy)

Additionally:
- `label_coarse` is derived:
  - `fallacy` if `label_fine != none`
  - `no_fallacy` if `label_fine == none`

---

## 4) Datasets (Planned)
### Core datasets (recommended)
1) **LOGIC / Logical Fallacy Detection dataset**
   - Provides fallacy-labeled text examples.

2) **CoCoLoFa**
   - Fallacy-labeled content closer to online discussion / comments.
   - Often includes structured splits (train/dev/test), and sometimes metadata like `article_id`.

### Optional add-ons (only if needed)
- **Argotario** (multilingual fallacies; useful if expanding beyond English)
- **IAC (Internet Argument Corpus)** (not fallacy-labeled by default; useful for extra “no-fallacy” argumentative text if needed)

Dataset handling principles:
- All sources must be converted into one canonical record format.
- All dataset-specific labels must be mapped into the target taxonomy via a documented mapping.

---

## 5) Canonical Data Schema (Data Contract)
All samples are normalized into a common format for training and evaluation.

Each record (JSONL) includes:

- `id`: unique stable ID (string)
- `source`: dataset name (e.g., `logic`, `cocolofa`)
- `text_raw`: original text
- `text_clean`: cleaned/normalized text used for training
- `label_fine`: one of the taxonomy labels
- `label_coarse`: `fallacy` or `no_fallacy`
- `meta`: object/dict with extra info (dataset-specific), e.g.:
  - `article_id`, `comment_id`, `topic`, `language`, `url`, etc.

### Optional (dialogue / transcript fields)
- `speaker`: `A`/`B`/`unknown`
- `turn_id`: integer ordering for dialogue turns
- `start_time`, `end_time`: timestamps (seconds)
- `context_text`: previous 1–2 turns or parent comment context

---

## 6) Preprocessing Pipeline (What the code will do)
### Stage A — Import / Standardize
- Read raw dataset files (CSV/JSON/JSONL)
- Convert each example into the canonical schema
- Save to `data/interim/normalized/*.jsonl`

### Stage B — Text Normalization (minimal, reversible)
Goal: remove formatting noise without destroying rhetorical cues.

**Operations**
- Unicode normalization (NFKC)
- Whitespace normalization (collapse repeated whitespace)
- Strip HTML tags (especially for comments)
- Replace URLs with `<URL>`
- Replace user mentions with `<USER>` (if applicable)
- Keep punctuation (important for fallacy patterns)
- Keep casing unless using a lowercased model checkpoint

Outputs:
- `text_raw` unchanged
- `text_clean` normalized

### Stage C — Label Harmonization
- Map dataset label names → target taxonomy via `label_mapping.md`
- Handle:
  - unknown labels: map to `other` or drop (choose one and document it)
  - multi-label samples: keep primary label OR duplicate sample per label (document choice)

### Stage D — Quality Filters
Drop samples that are:
- empty after cleaning
- too short (e.g., `< 5 tokens`)
- optionally too long (truncate or drop; document)

Optional:
- language filter (English-only)

### Stage E — Deduplication + Leakage Control
- Exact duplicate removal using hash of `text_clean`
- Optional near-duplicate removal (TF-IDF/embeddings similarity)

Leakage-safe split rules:
- If metadata provides `article_id` (or thread id), do a **grouped split** so related items don’t span train/test.
- If creating TTS/ASR variants, keep all variants of the same base sample in the same split.

### Stage F — Train/Dev/Test Split and Export
- If dataset already provides splits (e.g., CoCoLoFa), preserve them and clean within split.
- Otherwise:
  - stratify by `label_fine`
  - group split by `article_id` when available
  - export:
    - `data/processed/train.jsonl`
    - `data/processed/dev.jsonl`
    - `data/processed/test.jsonl`

### Stage G — Data Health Report
Auto-generate a markdown report for the write-up:
- counts before/after per dataset
- label distribution
- drop reasons and counts
- dedup counts
- text length stats

Save to:
- `reports/data_health.md`

---

## 7) Repository / Folder Structure
Suggested structure:

project/
- data/
  - raw/
    - logic/
    - cocolofa/
    - (optional) argotario/
    - (optional) iac/
  - interim/
    - normalized/
    - label_mapped/
    - filtered/
    - deduped/
  - processed/
    - train.jsonl
    - dev.jsonl
    - test.jsonl
- src/
  - preprocessing/
    - 01_import.py
    - 02_normalize_text.py
    - 03_label_mapping.py
    - 04_quality_filters.py
    - 05_deduplicate.py
    - 06_split.py
    - 07_export.py
  - utils/
    - io.py
    - text.py
- reports/
  - data_health.md
  - label_mapping.md
- README.md

---

## 8) Week 6 Deliverable Requirements (Assignment)
Objective: **Collect and clean the data needed for the project** and **preprocess it**.

What must be true by submission:
- Raw datasets exist in `data/raw/`
- Cleaned + standardized dataset exists in `data/processed/` as train/dev/test
- The Overleaf paper includes:
  - what datasets were used and why
  - preprocessing steps
  - before/after counts
  - final split sizes and label distribution

---

## 9) Modeling Plan (Brief Context)
(Not required for Week 6, but guides preprocessing choices.)

- Baseline model: transformer text classifier (e.g., BERT/RoBERTa)
- Inputs:
  - `text_clean` (optionally with `context_text`)
- Outputs:
  - fine label prediction
  - derived coarse label

Evaluation:
- Macro F1 for fine labels (handles imbalance)
- Accuracy / F1 for coarse fallacy detection
- Confusion matrix to see common confusions (e.g., straw man vs false dilemma)

---

## 10) Design Decisions (to document)
These should be decided and written down in `reports/label_mapping.md` and `reports/data_health.md`:
- Target taxonomy and mapping decisions
- Multi-label handling (primary vs expanded)
- Drop/truncate rules for long texts
- Split method (stratified + grouped where possible)
- Deduplication approach and thresholds (if using near-dup)

---

## 11) Definition of Done (for preprocessing)
Preprocessing is considered complete when:
- All datasets are converted into the canonical schema
- Labels are mapped into the target taxonomy consistently
- Junk/empty/invalid records are removed
- Duplicates are removed (at least exact duplicates)
- Leakage-safe train/dev/test split exists
- A data health report exists to support the Week 6 write-up