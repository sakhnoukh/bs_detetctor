# BS-Detector — Logical Fallacy Detection

An NLP pipeline for detecting logical fallacies in argumentative text.
Built as a university project (Year 3, Semester 2).

## Project Overview

The system classifies text spans into fine-grained fallacy types:

| Label | Description |
|-------|-------------|
| `ad_hominem` | Attacking the person rather than the argument |
| `appeal_to_authority` | Using authority as evidence without justification |
| `false_cause` | Assuming causation from correlation |
| `false_dilemma` | Presenting only two options when more exist |
| `hasty_generalization` | Drawing broad conclusions from limited evidence |
| `slippery_slope` | Assuming one event will lead to extreme consequences |
| `straw_man` | Misrepresenting someone's argument |
| `none` | No fallacy detected |
| `other` | Fallacy outside the core taxonomy |

A binary coarse label (`fallacy` / `no_fallacy`) is derived automatically.

## Repository Structure

```
bs_detetctor/
├── data/
│   ├── raw/                  # Original datasets (not committed)
│   │   ├── cocolofa/
│   │   └── logic/
│   ├── interim/              # Intermediate pipeline outputs
│   │   ├── normalized/
│   │   ├── label_mapped/
│   │   ├── filtered/
│   │   └── deduped/
│   ├── processed/            # Final train/dev/test JSONL splits
│   └── features/             # Extracted feature matrices
├── src/
│   ├── preprocessing/        # 7-stage data pipeline (01–07)
│   ├── features/             # Feature extraction modules
│   │   ├── build_features.py # CLI orchestrator
│   │   ├── tfidf.py          # TF-IDF (word + char n-grams)
│   │   ├── embeddings.py     # Transformer sentence embeddings
│   │   ├── rhetorical.py     # Hand-crafted argumentation features
│   │   └── ner.py            # Optional spaCy NER counts
│   └── utils/                # Shared I/O and text helpers
├── reports/                  # Auto-generated markdown reports
├── project.md                # Full project specification
└── DATA.md                   # Dataset documentation
```

## Datasets

- **CoCoLoFa** (EMNLP 2024) — fallacy-labelled news comments
- **LOGIC / LogicClimate** — fallacy-labelled argumentative text

Raw data is not committed. Place files in `data/raw/` as described in `DATA.md`.

## Setup

```bash
# Create virtual environment (Python 3.12 recommended)
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Optional: install spaCy model for NER features
python -m spacy download en_core_web_sm
```

## Running the Pipelines

### 1. Preprocessing (stages 01–07)

Each stage reads from the previous stage's output. Run from the project root:

```bash
python src/preprocessing/01_import.py
python src/preprocessing/02_normalize_text.py
python src/preprocessing/03_label_mapping.py
python src/preprocessing/04_quality_filters.py
python src/preprocessing/05_deduplicate.py
python src/preprocessing/06_split.py
python src/preprocessing/07_report.py
```

Output: `data/processed/{train,dev,test}.jsonl`

### 2. Feature extraction

```bash
python -m src.features.build_features \
    --input_dir data/processed \
    --output_dir data/features \
    --use_context 0
```

Key options:

| Flag | Default | Description |
|------|---------|-------------|
| `--use_context` | `0` | Prepend `context_text` to embeddings |
| `--embed_model` | `all-MiniLM-L6-v2` | Sentence-transformer model |
| `--tfidf_max_features` | `50000` | Max vocab size per TF-IDF vectoriser |
| `--tfidf_min_df` | `2` | Minimum document frequency |
| `--embed_batch_size` | `64` | Batch size for encoding |

Output: `data/features/` containing `.npz`, `.npy`, `.txt`, and `feature_manifest.json`.

## Data Schema

Every record in the processed JSONL files follows:

```json
{
  "id": "cocolofa_train_5584",
  "source": "cocolofa",
  "text_raw": "original text …",
  "text_clean": "normalised text …",
  "label_fine": "none",
  "label_coarse": "no_fallacy",
  "meta": { "article_id": 262, "..." : "..." }
}
```

## Reports

- `reports/data_health.md` — pipeline statistics, label distributions, text length stats
- `reports/feature_extraction.md` — feature descriptions and rationale
- `reports/label_mapping.md` — taxonomy mapping documentation
