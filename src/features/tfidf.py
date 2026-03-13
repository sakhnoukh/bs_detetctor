"""TF-IDF feature extraction (word + char n-grams).

Fits on training data only; transforms dev/test with the same vectoriser.
Saves sparse matrices as .npz via scipy.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf(
    train_texts: List[str],
    dev_texts: List[str],
    test_texts: List[str],
    *,
    word_ngram_range: Tuple[int, int] = (1, 2),
    char_ngram_range: Tuple[int, int] = (3, 5),
    max_features: int = 50_000,
    min_df: int = 2,
) -> Tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix, dict]:
    """Return (train, dev, test) sparse TF-IDF matrices and params dict.

    Two vectorisers are fitted on *train_texts* (word-level and char-level)
    and their outputs are horizontally stacked for each split.
    """
    word_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=word_ngram_range,
        max_features=max_features,
        min_df=min_df,
        sublinear_tf=True,
        dtype=np.float32,
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=char_ngram_range,
        max_features=max_features,
        min_df=min_df,
        sublinear_tf=True,
        dtype=np.float32,
    )

    # Fit on train only
    w_train = word_vec.fit_transform(train_texts)
    c_train = char_vec.fit_transform(train_texts)

    w_dev = word_vec.transform(dev_texts)
    c_dev = char_vec.transform(dev_texts)

    w_test = word_vec.transform(test_texts)
    c_test = char_vec.transform(test_texts)

    X_train = sp.hstack([w_train, c_train], format="csr")
    X_dev = sp.hstack([w_dev, c_dev], format="csr")
    X_test = sp.hstack([w_test, c_test], format="csr")

    params = {
        "word_ngram_range": list(word_ngram_range),
        "char_ngram_range": list(char_ngram_range),
        "max_features": max_features,
        "min_df": min_df,
        "word_vocab_size": len(word_vec.vocabulary_),
        "char_vocab_size": len(char_vec.vocabulary_),
        "total_dim": X_train.shape[1],
    }
    return X_train, X_dev, X_test, params


def save_sparse(matrix: sp.csr_matrix, path: Path) -> None:
    """Save a sparse matrix as .npz."""
    sp.save_npz(str(path), matrix)
