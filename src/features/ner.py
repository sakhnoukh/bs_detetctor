"""Optional NER-based features using spaCy.

Extracts per-sample entity counts for common entity types.  Gracefully skips
with a warning if spaCy or the model is not installed.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

ENTITY_TYPES: List[str] = ["PERSON", "ORG", "GPE", "NORP"]


def load_spacy_model(model_name: str = "en_core_web_sm"):
    """Try to load a spaCy model; return None on failure.

    Call once and pass the result into :func:`extract_ner` to avoid
    reloading the model for every split.
    """
    try:
        import spacy
        return spacy.load(model_name, disable=["parser", "lemmatizer"])
    except ImportError:
        logger.warning("spaCy is not installed – NER features will be skipped.")
        return None
    except Exception as exc:
        logger.warning("spaCy failed to load (%s) – NER features will be skipped.", exc)
        return None


def extract_ner(
    texts: List[str],
    *,
    nlp: Optional[object] = None,
    model_name: str = "en_core_web_sm",
    batch_size: int = 256,
) -> Tuple[Optional[np.ndarray], bool]:
    """Return (features_array, enabled) or (None, False) if unavailable.

    Pass a pre-loaded spaCy model via *nlp* to avoid reloading per split.
    *features_array* has shape (N, len(ENTITY_TYPES)), dtype float32.
    """
    if nlp is None:
        nlp = load_spacy_model(model_name)
    if nlp is None:
        return None, False

    counts: List[List[int]] = []
    for doc in nlp.pipe(texts, batch_size=batch_size):
        row = {etype: 0 for etype in ENTITY_TYPES}
        for ent in doc.ents:
            if ent.label_ in row:
                row[ent.label_] += 1
        counts.append([row[e] for e in ENTITY_TYPES])

    return np.array(counts, dtype=np.float32), True
