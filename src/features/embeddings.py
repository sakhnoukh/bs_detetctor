"""Transformer-based sentence embeddings.

Uses sentence-transformers if available; falls back to raw HuggingFace
transformers with mean-pooling otherwise.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _set_seeds(seed: int = 42) -> None:
    import random, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _prepare_texts(
    texts: List[str],
    contexts: Optional[List[str]],
    use_context: bool,
) -> List[str]:
    """Optionally prepend context_text with [SEP] separator."""
    if not use_context or contexts is None:
        return texts
    prepared: List[str] = []
    for ctx, txt in zip(contexts, texts):
        if ctx and ctx.strip():
            prepared.append(f"{ctx.strip()} [SEP] {txt}")
        else:
            prepared.append(txt)
    return prepared


def load_model(model_name: str = DEFAULT_MODEL) -> Any:
    """Load and return a sentence-transformer (or HF fallback) model once.

    Returns a tuple of (backend, model) where backend is
    ``"sentence-transformers"`` or ``"hf-transformers"``.
    """
    try:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading sentence-transformers model: %s", model_name)
        return ("sentence-transformers", SentenceTransformer(model_name))
    except ImportError:
        import torch
        from transformers import AutoModel, AutoTokenizer
        logger.warning(
            "sentence-transformers not installed; loading HuggingFace transformers model."
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        return ("hf-transformers", (tokenizer, model, device))


def _encode_sentence_transformers(
    model: Any,
    texts: List[str],
    batch_size: int,
) -> np.ndarray:
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return embeddings.astype(np.float32)


def _encode_hf_transformers(
    model_bundle: Any,
    texts: List[str],
    batch_size: int,
) -> np.ndarray:
    import torch

    tokenizer, model, device = model_bundle

    all_embeddings: list[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        encoded = tokenizer(
            batch, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            output = model(**encoded)
        # Mean-pool over token dimension, respecting attention mask
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        summed = (output.last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        mean_pooled = (summed / counts).cpu().numpy().astype(np.float32)
        all_embeddings.append(mean_pooled)
    return np.vstack(all_embeddings)


def encode_texts(
    texts: List[str],
    contexts: Optional[List[str]] = None,
    *,
    use_context: bool = False,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 64,
    seed: int = 42,
    _loaded_model: Optional[Any] = None,
) -> np.ndarray:
    """Encode *texts* into dense embeddings (float32 ndarray).

    Pass a pre-loaded model via *_loaded_model* (output of :func:`load_model`)
    to avoid reloading on every call.
    """
    _set_seeds(seed)
    prepared = _prepare_texts(texts, contexts, use_context)

    backend, model = _loaded_model or load_model(model_name)
    if backend == "sentence-transformers":
        return _encode_sentence_transformers(model, prepared, batch_size)
    return _encode_hf_transformers(model, prepared, batch_size)
