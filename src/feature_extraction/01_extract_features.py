import sys
import re
import json
import joblib
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from utils.io import read_jsonl

PROCESSED = ROOT / "data" / "processed"
FEATURES = ROOT / "data" / "features"
REPORTS = ROOT / "reports"


_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)
_UPPER_RE = re.compile(r"[A-Z]")


@dataclass(frozen=True)
class FeatureConfig:
    word_ngram_range: tuple[int, int] = (1, 2)
    char_ngram_range: tuple[int, int] = (3, 5)
    max_word_features: int = 50000
    max_char_features: int = 50000
    min_df: int = 2
    lowercase: bool = False


def _engineered_features(texts: list[str]) -> sparse.csr_matrix:
    rows: list[list[float]] = []

    for t in texts:
        tokens = _WORD_RE.findall(t)
        n_tokens = len(tokens)
        n_chars = len(t)

        n_q = t.count("?")
        n_excl = t.count("!")
        n_comma = t.count(",")
        n_dot = t.count(".")

        has_url = 1.0 if "<URL>" in t else 0.0
        has_user = 1.0 if "<USER>" in t else 0.0

        n_upper = len(_UPPER_RE.findall(t))
        upper_ratio = (n_upper / n_chars) if n_chars else 0.0

        token_len_mean = (sum(len(x) for x in tokens) / n_tokens) if n_tokens else 0.0

        rows.append([
            float(n_tokens),
            float(n_chars),
            float(n_q),
            float(n_excl),
            float(n_comma),
            float(n_dot),
            float(has_url),
            float(has_user),
            float(upper_ratio),
            float(token_len_mean),
        ])

    arr = np.asarray(rows, dtype=np.float32)
    return sparse.csr_matrix(arr)


def _load_split(split: str) -> tuple[list[str], list[str]]:
    path = PROCESSED / f"{split}.jsonl"
    records = read_jsonl(path)
    texts = [str(r.get("text_clean", "")) for r in records]
    labels = [str(r.get("label_fine", "other")) for r in records]
    return texts, labels


def _label_vocab(train_labels: list[str]) -> tuple[list[str], dict[str, int]]:
    classes = sorted(set(train_labels))
    if "other" not in classes:
        classes.append("other")
    to_id = {c: i for i, c in enumerate(classes)}
    return classes, to_id


def main():
    FEATURES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    cfg = FeatureConfig()

    x_train_text, y_train = _load_split("train")
    x_dev_text, y_dev = _load_split("dev")
    x_test_text, y_test = _load_split("test")

    classes, label_to_id = _label_vocab(y_train)

    other_id = label_to_id["other"]
    y_train_ids = np.asarray([label_to_id.get(y, other_id) for y in y_train], dtype=np.int64)
    y_dev_ids = np.asarray([label_to_id.get(y, other_id) for y in y_dev], dtype=np.int64)
    y_test_ids = np.asarray([label_to_id.get(y, other_id) for y in y_test], dtype=np.int64)

    word_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=cfg.word_ngram_range,
        min_df=cfg.min_df,
        max_features=cfg.max_word_features,
        lowercase=cfg.lowercase,
    )

    char_vec = TfidfVectorizer(
        analyzer="char",
        ngram_range=cfg.char_ngram_range,
        min_df=cfg.min_df,
        max_features=cfg.max_char_features,
        lowercase=cfg.lowercase,
    )

    xw_train = word_vec.fit_transform(x_train_text)
    xw_dev = word_vec.transform(x_dev_text)
    xw_test = word_vec.transform(x_test_text)

    xc_train = char_vec.fit_transform(x_train_text)
    xc_dev = char_vec.transform(x_dev_text)
    xc_test = char_vec.transform(x_test_text)

    xe_train = _engineered_features(x_train_text)
    xe_dev = _engineered_features(x_dev_text)
    xe_test = _engineered_features(x_test_text)

    x_train = sparse.hstack([xw_train, xc_train, xe_train], format="csr")
    x_dev = sparse.hstack([xw_dev, xc_dev, xe_dev], format="csr")
    x_test = sparse.hstack([xw_test, xc_test, xe_test], format="csr")

    sparse.save_npz(FEATURES / "x_train.npz", x_train)
    sparse.save_npz(FEATURES / "x_dev.npz", x_dev)
    sparse.save_npz(FEATURES / "x_test.npz", x_test)

    np.save(FEATURES / "y_train.npy", y_train_ids)
    np.save(FEATURES / "y_dev.npy", y_dev_ids)
    np.save(FEATURES / "y_test.npy", y_test_ids)

    (FEATURES / "classes.json").write_text(json.dumps(classes, indent=2), encoding="utf-8")
    (FEATURES / "feature_config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    joblib.dump(word_vec, FEATURES / "tfidf_word.joblib")
    joblib.dump(char_vec, FEATURES / "tfidf_char.joblib")

    engineered_names = [
        "n_tokens",
        "n_chars",
        "n_question",
        "n_exclamation",
        "n_comma",
        "n_dot",
        "has_url",
        "has_user",
        "upper_ratio",
        "token_len_mean",
    ]
    (FEATURES / "engineered_feature_names.json").write_text(
        json.dumps(engineered_names, indent=2),
        encoding="utf-8",
    )

    report_lines = [
        "# Feature Extraction Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## 1. Feature sets",
        "",
        "- TF–IDF word n-grams (1–2)",
        "- TF–IDF character n-grams (3–5)",
        "- Engineered surface features (10 dims): " + ", ".join(f"`{n}`" for n in engineered_names),
        "",
        "## 2. Dataset sizes",
        "",
        f"- train: {len(y_train):,}",
        f"- dev: {len(y_dev):,}",
        f"- test: {len(y_test):,}",
        "",
        "## 3. Vocabulary sizes",
        "",
        f"- word TF–IDF vocab: {len(word_vec.vocabulary_):,}",
        f"- char TF–IDF vocab: {len(char_vec.vocabulary_):,}",
        "",
        "## 4. Final feature dimensionality",
        "",
        f"- X_train shape: {x_train.shape}",
        f"- X_dev shape: {x_dev.shape}",
        f"- X_test shape: {x_test.shape}",
        "",
        "## 5. Output artifacts",
        "",
        "Saved under `data/features/`:",
        "- `x_{train,dev,test}.npz` (sparse CSR)",
        "- `y_{train,dev,test}.npy` (int labels)",
        "- `classes.json` (label id mapping)",
        "- `tfidf_word.joblib`, `tfidf_char.joblib`",
        "- `engineered_feature_names.json`, `feature_config.json`",
    ]

    (REPORTS / "feature_extraction.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("\nweek 7 feature extraction complete")
    print("  features written to:", FEATURES)
    print("  report written to:", REPORTS / "feature_extraction.md")


if __name__ == "__main__":
    main()
