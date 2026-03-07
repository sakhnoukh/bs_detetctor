0) Choose the datasets (recommended for your project)
Core (do these first)

LOGIC / LogicClimate (logical fallacy detection dataset released with “Logical Fallacy Detection”)

CoCoLoFa (EMNLP 2024; has ready-made train/dev/test JSON files)

Optional (only if you need extra data or multilingual)

Argotario (fallacies collected via serious game; multilingual)

IAC (Internet Argument Corpus) (for debate-style text; not fallacy labels by default, but useful for “non-fallacy” / argument structure tasks)

Why this combo works: LOGIC gives clean fallacy examples; CoCoLoFa gives “news comment / online discussion” style samples close to your real use case.

1) Set up your repo structure (so preprocessing is reproducible)

In Windsurf, create this skeleton:

project/
  data/
    raw/
      logic/
      cocolofa/
      argotario/        (optional)
      iac/              (optional)
    interim/
      normalized/
      label_mapped/
      deduped/
    processed/
      train.jsonl
      dev.jsonl
      test.jsonl
  src/
    preprocessing/
      01_download_or_import.py
      02_normalize_text.py
      03_label_mapping.py
      04_quality_filters.py
      05_deduplicate.py
      06_split.py
      07_export.py
    utils/
      text.py
      io.py
  reports/
    data_health.md
    label_mapping.md

The rule: each script reads from previous stage folder and writes to the next.

2) Define a single “data contract” (the schema all datasets become)

Make a unified record format so you can merge datasets safely:

Canonical schema

id (unique, stable)

source (logic / cocolofa / …)

text_raw

text_clean

label_fine (your fallacy type, or none)

label_coarse (fallacy or no_fallacy)

meta (JSON object: dataset-specific fields like article_id, comment_id, etc.)

This one decision removes 80% of future pain.

3) Download / import the datasets
CoCoLoFa (easy start)

Clone or download the repo and place train.json, dev.json, test.json in data/raw/cocolofa/.

LOGIC

Pull from the official repo associated with the paper and store in data/raw/logic/.

Optional datasets

Argotario repo/data if you want multilingual later.

IAC for additional debate-style text if you want more “clean non-fallacy” argumentative content.

4) Normalize text (minimal + reversible)

Create 02_normalize_text.py that:

Unicode normalize (NFKC)

Normalize whitespace:

collapse multiple spaces

standardize newlines

Remove HTML tags (mainly useful for comment datasets)

Replace:

URLs → <URL>

user mentions → <USER> (if present)

Keep punctuation (important for fallacies)

Output both:

text_raw untouched

text_clean normalized

Important: Don’t aggressively remove stopwords, punctuation, or “emotional” words; fallacies are rhetorical.

5) Harmonize labels across datasets (map to your taxonomy)

You need one label set to train cleanly.

A) Define your target label set

For example (matches common assignments):

ad_hominem

straw_man

appeal_to_authority

false_dilemma

false_cause

hasty_generalization

slippery_slope

none

other (optional bucket if datasets contain extra types)

B) Build a mapping file (and document it)

Create reports/label_mapping.md and a mapping dict in 03_label_mapping.py.

Rules:

If a dataset label is exact match → map directly.

If a dataset has fallacies you’re not modeling → map to other or drop (choose one and be consistent).

If a sample is multi-label:

Option 1 (simpler): keep only the primary label

Option 2 (better): duplicate the sample per label (but watch leakage)

CoCoLoFa is already structured for fallacy classification; LOGIC also provides fallacy types (you’ll still normalize naming).

6) Quality filters (remove junk without biasing the task)

Create 04_quality_filters.py:

Drop samples that are:

empty after cleaning

too short (e.g., < 5 tokens) → usually noise

extremely long (optional) → e.g., truncate to 512–1024 tokens depending on your model plan

Optional:

language detection and keep English-only (if your datasets include multilingual)

Log what you dropped:

count dropped per reason

examples of dropped rows (a few)

This will help your Week 6 write-up.

7) Deduplicate + prevent leakage (super important when combining datasets)

Create 05_deduplicate.py:

A) Exact duplicates

hash text_clean (e.g., sha1)

keep first occurrence, drop the rest

B) Near-duplicates (optional but nice)

Use a cheap similarity method:

TF-IDF cosine similarity OR

sentence embeddings + cosine threshold

Drop near-dups across splits (or group them)

Leakage rule

If CoCoLoFa has article_id, keep all comments from the same article grouped when splitting (so the model doesn’t “see the same topic framing” in train and test). That’s a strong, defensible choice.

8) Split strategy (train/dev/test)

If you use CoCoLoFa’s provided splits, keep them intact (clean within split).

If you merge datasets:

Stratify by label_fine

Group split by article_id (for CoCoLoFa) and by source if needed

Typical split: 80/10/10

Save 3 JSONL files with the canonical schema.

9) Create a “data health report” (this is what your Week 6 write-up needs)

Generate reports/data_health.md automatically:

Include:

Total samples before/after cleaning (per dataset)

Label distribution after mapping

Avg / median text length

% removed by each filter

Deduplication counts

This one report basically becomes your “Preprocessing” section in Overleaf.

10) (Optional) Add dialogue-ready preprocessing (if you have transcripts/audio)

If your project includes two-speaker discussions:

Store per-turn records with:

speaker, turn_id, start_time, end_time

Normalize each turn’s text the same way

Create an additional field:

context_text = previous 1–2 turns (this boosts fallacy detection a lot)