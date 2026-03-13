"""Lightweight rhetorical / argumentation features.

All features are computed from *text_clean* (the normalised text).  Returns a
fixed-length float32 numpy array per sample.
"""

from __future__ import annotations

import re
from typing import List

import numpy as np

# ── word lists ────────────────────────────────────────────────────────────────
NEGATION_TOKENS = {"not", "no", "never", "n't"}
DISCOURSE_MARKERS = {"because", "therefore", "however", "but", "so", "if", "then"}
CERTAINTY_WORDS = {"always", "never", "obviously", "clearly"}
HEDGE_WORDS = {"maybe", "probably", "might", "could", "perhaps"}
SECOND_PERSON = {"you", "your", "yourself"}

# Feature names (order matters – matches column indices)
FEATURE_NAMES: List[str] = [
    "num_chars",
    "num_words",
    "avg_word_len",
    "count_excl",
    "count_quest",
    "count_comma",
    "count_period",
    "negation_count",
    "discourse_count",
    "certainty_count",
    "hedge_count",
    "second_person_count",
    "allcaps_ratio",
]

_WORD_RE = re.compile(r"\S+")


def _features_single(text: str) -> List[float]:
    """Compute rhetorical feature vector for a single text string."""
    chars = len(text)
    tokens = _WORD_RE.findall(text)
    n_words = len(tokens) if tokens else 1  # avoid div-by-zero
    avg_word_len = np.mean([len(t) for t in tokens]).item() if tokens else 0.0

    lower_tokens = [t.lower() for t in tokens]

    count_excl = text.count("!")
    count_quest = text.count("?")
    count_comma = text.count(",")
    count_period = text.count(".")

    negation = sum(1 for t in lower_tokens if t in NEGATION_TOKENS or t.endswith("n't"))
    discourse = sum(1 for t in lower_tokens if t in DISCOURSE_MARKERS)
    certainty = sum(1 for t in lower_tokens if t in CERTAINTY_WORDS)
    hedges = sum(1 for t in lower_tokens if t in HEDGE_WORDS)
    second_p = sum(1 for t in lower_tokens if t in SECOND_PERSON)

    allcaps_count = sum(1 for t in tokens if t.isupper() and len(t) > 1)
    allcaps_ratio = allcaps_count / n_words

    return [
        float(chars),
        float(n_words),
        avg_word_len,
        float(count_excl),
        float(count_quest),
        float(count_comma),
        float(count_period),
        float(negation),
        float(discourse),
        float(certainty),
        float(hedges),
        float(second_p),
        allcaps_ratio,
    ]


def extract_rhetorical(texts: List[str]) -> np.ndarray:
    """Return (N, D) float32 array of rhetorical features for *texts*."""
    feats = [_features_single(t) for t in texts]
    return np.array(feats, dtype=np.float32)
