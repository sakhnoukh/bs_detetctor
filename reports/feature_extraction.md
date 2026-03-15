# Feature Extraction — Summary & Rationale

This document describes the feature extraction pipeline for the
BS-Detector logical-fallacy detection project.  All features are computed
from the cleaned text (`text_clean`) and, optionally, from surrounding
context (`context_text`).

## 1  TF-IDF Features (baseline)

| Parameter | Default |
|-----------|---------|
| Word n-grams | (1, 2) |
| Char n-grams | (3, 5) |
| `max_features` | 50 000 per vectoriser |
| `min_df` | 2 |
| Sub-linear TF | yes |

**Rationale.**  Bag-of-words and character n-grams provide a strong,
interpretable baseline for text classification.  Word bigrams capture
common fallacy-specific collocations (e.g., *"everyone knows"*,
*"you should"*), while character n-grams add robustness to
morphological variation and misspellings.  The two vectorisers are
fitted on the training split only and their outputs are horizontally
stacked into a single sparse matrix per split.

## 2  Transformer Embeddings

| Parameter | Default |
|-----------|---------|
| Model | `sentence-transformers/all-MiniLM-L6-v2` |
| Dimensionality | 384 |
| Backend | sentence-transformers (fallback: HuggingFace transformers + mean-pooling) |

**Rationale.**  Pre-trained sentence embeddings capture deep semantic
information that surface-level features miss, such as implicit reasoning
patterns, irony, and subtle shifts in argumentation.  MiniLM-L6-v2 is
chosen for its balance between quality and compute cost (6-layer
distilled BERT, ≈ 22 M parameters).  When `context_text` is available
and `--use_context 1` is set, context is prepended with a `[SEP]`
token to let the model attend over the broader argumentative
environment.

## 3  Rhetorical / Argumentation Features

A hand-crafted set of lightweight numeric features designed to capture
surface-level rhetorical cues frequently associated with logical
fallacies:

| Feature group | Signals captured |
|---------------|------------------|
| Length (chars, words, avg word len) | Verbosity, complexity |
| Punctuation counts (`!`, `?`, `,`, `.`) | Emotional emphasis, questioning |
| Negation tokens (*not, no, never, n't*) | Denial patterns |
| Discourse markers (*because, therefore, however, but, so, if, then*) | Argumentation structure |
| Certainty words (*always, never, obviously, clearly*) | Overconfidence, hasty generalisation |
| Hedge words (*maybe, probably, might, could, perhaps*) | Evasiveness, weakening |
| Second-person pronouns (*you, your, yourself*) | Ad-hominem / audience addressing |
| ALL-CAPS token ratio | Shouting, emphasis |

**Rationale.**  These features are motivated by argumentation theory and
prior work on propaganda / persuasion detection (Da San Martino et al.,
2019; Habernal et al., 2018).  They are fast to compute, fully
interpretable, and complement both TF-IDF and embedding representations.

## 4  NER Features (optional)

If spaCy (`en_core_web_sm`) is available, per-sample entity counts are
extracted for: `PERSON`, `ORG`, `GPE`, `NORP`.

**Rationale.**  Appeal-to-authority and ad-hominem fallacies frequently
reference named entities.  Entity density may also correlate with the
argumentative complexity of a passage.

## 5  Labels

A deterministic label encoder maps `label_fine` values to integer IDs
(sorted alphabetically).  The mapping is persisted in
`feature_manifest.json` for downstream reproducibility.

## 6  Outputs

All artefacts are stored in `data/features/`.  Sparse TF-IDF matrices
are saved as `.npz`; dense arrays (embeddings, extra features, labels)
as `.npy`; sample IDs as plain-text `.txt` files.  A JSON manifest
records all parameters, model names, package versions, and a UTC
timestamp.

## 7  Reproducibility

- TF-IDF is deterministic given the same input.
- Transformer seeds are set explicitly (default 42).
- Feature order and manifest enable exact re-creation of any run.
