"""CLI orchestrator for feature extraction.

Usage
-----
    python -m src.features.build_features \
        --input_dir data/processed \
        --output_dir data/features \
        --use_context 1
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── local modules ─────────────────────────────────────────────────────────────
from src.features.tfidf import build_tfidf, save_sparse
from src.features.embeddings import encode_texts, load_model, DEFAULT_MODEL
from src.features.rhetorical import extract_rhetorical, FEATURE_NAMES as RHETORICAL_NAMES
from src.features.ner import extract_ner, load_spacy_model, ENTITY_TYPES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SPLITS = ("train", "dev", "test")


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of dicts."""
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}  invalid JSON – {exc}") from exc
    return rows


def _extract_fields(
    rows: List[Dict[str, Any]], split_name: str
) -> Tuple[List[str], List[str], List[str], List[str], List[Optional[str]]]:
    """Extract (ids, texts, labels_fine, labels_coarse, contexts) with validation."""
    ids: List[str] = []
    texts: List[str] = []
    labels_fine: List[str] = []
    labels_coarse: List[str] = []
    contexts: List[Optional[str]] = []

    for i, row in enumerate(rows):
        for col in ("id", "text_clean", "label_fine", "label_coarse"):
            if col not in row:
                raise KeyError(
                    f"Row {i} in {split_name} is missing required column '{col}'."
                )
        ids.append(str(row["id"]))
        texts.append(row["text_clean"])
        labels_fine.append(row["label_fine"])
        labels_coarse.append(row["label_coarse"])
        contexts.append(row.get("context_text"))

    return ids, texts, labels_fine, labels_coarse, contexts


def _build_label_encoder(
    train_labels: List[str],
    dev_labels: List[str],
    test_labels: List[str],
) -> Tuple[Dict[str, int], np.ndarray, np.ndarray, np.ndarray]:
    """Create a deterministic label→int mapping and encode all splits."""
    all_labels = sorted(set(train_labels) | set(dev_labels) | set(test_labels))
    label2id: Dict[str, int] = {lbl: idx for idx, lbl in enumerate(all_labels)}
    y_train = np.array([label2id[l] for l in train_labels], dtype=np.int64)
    y_dev = np.array([label2id[l] for l in dev_labels], dtype=np.int64)
    y_test = np.array([label2id[l] for l in test_labels], dtype=np.int64)
    return label2id, y_train, y_dev, y_test


def _pkg_versions() -> Dict[str, str]:
    """Collect versions of key packages."""
    versions: Dict[str, str] = {}
    try:
        import sklearn; versions["scikit-learn"] = sklearn.__version__
    except Exception:
        pass
    try:
        import sentence_transformers; versions["sentence-transformers"] = sentence_transformers.__version__
    except Exception:
        pass
    try:
        import transformers; versions["transformers"] = transformers.__version__
    except Exception:
        pass
    try:
        import torch; versions["torch"] = torch.__version__
    except Exception:
        pass
    try:
        import spacy; versions["spacy"] = spacy.__version__
    except Exception:
        pass
    try:
        import scipy; versions["scipy"] = scipy.__version__
    except Exception:
        pass
    versions["numpy"] = np.__version__
    return versions


# ── smoke tests ───────────────────────────────────────────────────────────────

def _smoke_test(
    n_train: int,
    n_dev: int,
    n_test: int,
    tfidf_shapes: Tuple,
    embed_shapes: Tuple,
    extra_shapes: Tuple,
    y_shapes: Tuple,
) -> None:
    """Quick assertions to verify alignment."""
    for name, expected, actual in [
        ("tfidf_train", n_train, tfidf_shapes[0]),
        ("tfidf_dev", n_dev, tfidf_shapes[1]),
        ("tfidf_test", n_test, tfidf_shapes[2]),
        ("embed_train", n_train, embed_shapes[0]),
        ("embed_dev", n_dev, embed_shapes[1]),
        ("embed_test", n_test, embed_shapes[2]),
        ("extra_train", n_train, extra_shapes[0]),
        ("extra_dev", n_dev, extra_shapes[1]),
        ("extra_test", n_test, extra_shapes[2]),
        ("y_train", n_train, y_shapes[0]),
        ("y_dev", n_dev, y_shapes[1]),
        ("y_test", n_test, y_shapes[2]),
    ]:
        assert expected == actual, (
            f"Shape mismatch for {name}: expected {expected} rows, got {actual}"
        )
    logger.info("Smoke tests passed ✓")


# ── main ──────────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Feature extraction pipeline for BS-Detector."
    )
    parser.add_argument(
        "--input_dir", type=str, default="data/processed",
        help="Directory containing train/dev/test JSONL files.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/features",
        help="Directory to write feature artifacts.",
    )
    parser.add_argument(
        "--use_context", type=int, default=0, choices=[0, 1],
        help="Whether to prepend context_text to text_clean for embeddings.",
    )
    parser.add_argument(
        "--embed_model", type=str, default=DEFAULT_MODEL,
        help="HuggingFace model name for sentence embeddings.",
    )
    parser.add_argument(
        "--tfidf_max_features", type=int, default=50_000,
        help="Max vocabulary size per TF-IDF vectoriser.",
    )
    parser.add_argument(
        "--tfidf_min_df", type=int, default=2,
        help="Minimum document frequency for TF-IDF.",
    )
    parser.add_argument(
        "--embed_batch_size", type=int, default=64,
        help="Batch size for transformer encoding.",
    )
    args = parser.parse_args(argv)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    use_context = bool(args.use_context)

    t0 = time.time()

    # ── 1. Load data ──────────────────────────────────────────────────────
    logger.info("Loading JSONL splits from %s …", input_dir)
    data: Dict[str, List[Dict[str, Any]]] = {}
    fields: Dict[str, Tuple] = {}
    for split in SPLITS:
        data[split] = _load_jsonl(input_dir / f"{split}.jsonl")
        fields[split] = _extract_fields(data[split], split)
        logger.info("  %s: %d samples", split, len(data[split]))

    ids = {s: fields[s][0] for s in SPLITS}
    texts = {s: fields[s][1] for s in SPLITS}
    labels_fine = {s: fields[s][2] for s in SPLITS}
    contexts = {s: fields[s][4] for s in SPLITS}

    # ── 2. Labels ─────────────────────────────────────────────────────────
    logger.info("Encoding labels …")
    label2id, y_train, y_dev, y_test = _build_label_encoder(
        labels_fine["train"], labels_fine["dev"], labels_fine["test"]
    )
    y = {"train": y_train, "dev": y_dev, "test": y_test}
    logger.info("  %d unique fine labels: %s", len(label2id), list(label2id.keys()))

    # ── 3. TF-IDF ─────────────────────────────────────────────────────────
    logger.info("Building TF-IDF features …")
    tfidf_train, tfidf_dev, tfidf_test, tfidf_params = build_tfidf(
        texts["train"], texts["dev"], texts["test"],
        max_features=args.tfidf_max_features,
        min_df=args.tfidf_min_df,
    )
    tfidf = {"train": tfidf_train, "dev": tfidf_dev, "test": tfidf_test}
    logger.info("  TF-IDF dim: %d", tfidf_train.shape[1])

    # ── 4. Embeddings ─────────────────────────────────────────────────────
    logger.info("Encoding transformer embeddings (%s) …", args.embed_model)
    embed_model = load_model(args.embed_model)
    embed: Dict[str, np.ndarray] = {}
    for split in SPLITS:
        logger.info("  encoding %s …", split)
        embed[split] = encode_texts(
            texts[split],
            contexts[split],
            use_context=use_context,
            model_name=args.embed_model,
            batch_size=args.embed_batch_size,
            _loaded_model=embed_model,
        )
    logger.info("  Embedding dim: %d", embed["train"].shape[1])

    # ── 5. Rhetorical features ────────────────────────────────────────────
    logger.info("Extracting rhetorical features …")
    rhetorical: Dict[str, np.ndarray] = {}
    for split in SPLITS:
        rhetorical[split] = extract_rhetorical(texts[split])
    extra_names = list(RHETORICAL_NAMES)

    # ── 6. NER features (optional) ────────────────────────────────────────
    logger.info("Attempting NER features …")
    ner_enabled = False
    spacy_nlp = load_spacy_model()
    if spacy_nlp is not None:
        for split in SPLITS:
            ner_feats, enabled = extract_ner(texts[split], nlp=spacy_nlp)
            if enabled and ner_feats is not None:
                ner_enabled = True
                rhetorical[split] = np.hstack([rhetorical[split], ner_feats])
        extra_names.extend([f"ner_{e.lower()}" for e in ENTITY_TYPES])
        logger.info("  NER enabled – added %d entity features.", len(ENTITY_TYPES))
    else:
        logger.info("  NER skipped (dependency not available).")

    # ── 7. Save artefacts ─────────────────────────────────────────────────
    logger.info("Saving artefacts to %s …", output_dir)
    for split in SPLITS:
        save_sparse(tfidf[split], output_dir / f"tfidf_{split}.npz")
        np.save(output_dir / f"embed_{split}.npy", embed[split])
        np.save(output_dir / f"extra_{split}.npy", rhetorical[split])
        np.save(output_dir / f"labels_{split}.npy", y[split])
        (output_dir / f"ids_{split}.txt").write_text(
            "\n".join(ids[split]) + "\n", encoding="utf-8"
        )

    # ── 8. Smoke test ─────────────────────────────────────────────────────
    _smoke_test(
        len(data["train"]), len(data["dev"]), len(data["test"]),
        tfidf_shapes=(tfidf["train"].shape[0], tfidf["dev"].shape[0], tfidf["test"].shape[0]),
        embed_shapes=(embed["train"].shape[0], embed["dev"].shape[0], embed["test"].shape[0]),
        extra_shapes=(rhetorical["train"].shape[0], rhetorical["dev"].shape[0], rhetorical["test"].shape[0]),
        y_shapes=(y["train"].shape[0], y["dev"].shape[0], y["test"].shape[0]),
    )

    # ── 9. Manifest ───────────────────────────────────────────────────────
    elapsed = round(time.time() - t0, 1)
    manifest: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed,
        "splits": {s: len(data[s]) for s in SPLITS},
        "use_context": use_context,
        "tfidf": tfidf_params,
        "embedding": {
            "model": args.embed_model,
            "dim": int(embed["train"].shape[1]),
            "batch_size": args.embed_batch_size,
        },
        "extra_features": extra_names,
        "ner_enabled": ner_enabled,
        "label_mapping": label2id,
        "package_versions": _pkg_versions(),
    }
    manifest_path = output_dir / "feature_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    logger.info("Manifest written to %s", manifest_path)
    logger.info("Feature extraction complete in %.1fs.", elapsed)


if __name__ == "__main__":
    main()
