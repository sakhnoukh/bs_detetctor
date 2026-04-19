#!/usr/bin/env python3
"""Evaluate an improved (or any) fine-tuned transformer model.

Usage:
  python src/evaluation/02_evaluate_improved.py                                     # default: weighted_ce
  python src/evaluation/02_evaluate_improved.py --run improved_focal_roberta-base_seed42
  python src/evaluation/02_evaluate_improved.py --run finetune_roberta-base_seed42   # baseline
"""

import sys
import json
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from utils.io import read_jsonl

PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"


def _load_split(split: str, text_field: str, label_field: str) -> tuple[list[str], list[str]]:
    path = PROCESSED / f"{split}.jsonl"
    records = read_jsonl(path)
    texts = [str(r.get(text_field, "")) for r in records]
    labels = [str(r.get(label_field, "other")) for r in records]
    return texts, labels


def _write_confusion_csv(labels: list[str], matrix: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("," + ",".join(labels) + "\n")
        for i, row in enumerate(matrix):
            f.write(labels[i] + "," + ",".join(str(int(x)) for x in row) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned transformer")
    parser.add_argument("--run", type=str,
                        default="improved_weighted_ce_roberta-base_seed42",
                        help="Run name (model directory under models/)")
    parser.add_argument("--split", type=str, default="test",
                        choices=["test", "dev"], help="Split to evaluate on")
    cli_args = parser.parse_args()

    from datasets import Dataset
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        classification_report,
        confusion_matrix,
    )

    run_name = cli_args.run
    eval_split = cli_args.split
    text_field = "text_clean"
    label_field = "label_fine"
    max_length = 256

    model_dir = MODELS / run_name
    label_space_path = model_dir / "label_space.json"

    if not model_dir.exists() or not label_space_path.exists():
        print(f"Cannot evaluate: missing model artifacts at {model_dir}")
        print("Run training first:")
        print("  python src/modeling/02_finetune_improved.py")
        sys.exit(1)

    label_space = json.loads(label_space_path.read_text(encoding="utf-8"))
    labels: list[str] = list(label_space["labels"])
    label2id: dict[str, int] = {k: int(v) for k, v in label_space["label2id"].items()}
    other_id = int(label2id.get("other", len(labels) - 1))

    texts, raw_labels = _load_split(eval_split, text_field, label_field)
    y_true = np.asarray([label2id.get(l, other_id) for l in raw_labels], dtype=np.int64)

    ds = Dataset.from_dict({"text": texts, "label": y_true.tolist()})

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    ds = ds.map(tokenize, batched=True, remove_columns=["text"])
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))

    args = TrainingArguments(
        output_dir=str(model_dir / "eval_tmp"),
        per_device_eval_batch_size=16,
        report_to=[],
    )

    trainer = Trainer(model=model, args=args, processing_class=tokenizer)
    pred = trainer.predict(ds)
    logits = pred.predictions
    y_pred = np.argmax(logits, axis=-1)

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))

    REPORTS.mkdir(parents=True, exist_ok=True)

    # Use run_name in filenames to avoid overwriting baseline results
    safe_name = run_name.replace("/", "-")
    cm_path = REPORTS / f"confusion_matrix_{safe_name}.csv"
    _write_confusion_csv(labels, cm, cm_path)

    cls_report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(labels))),
        target_names=labels,
        digits=4,
        zero_division=0,
    )

    # Per-class F1 scores for comparison
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0, labels=list(range(len(labels))))
    per_class_dict = {labels[i]: float(per_class_f1[i]) for i in range(len(labels))}

    summary = {
        "run_name": run_name,
        "model_dir": str(model_dir),
        "eval_split": eval_split,
        "n_samples": int(len(y_true)),
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "per_class_f1": per_class_dict,
        "labels": labels,
        "confusion_matrix_csv": str(cm_path),
    }

    eval_json_path = REPORTS / f"evaluation_{safe_name}.json"
    (eval_json_path).write_text(json.dumps({"summary": summary}, indent=2), encoding="utf-8")

    md_lines = [
        f"# Evaluation Report: {run_name}",
        "",
        f"Model dir: `{model_dir}`",
        f"Eval split: **{eval_split}**",
        "",
        "## Metrics",
        "",
        f"- accuracy: {acc:.4f}",
        f"- macro-F1: {f1_macro:.4f}",
        f"- weighted-F1: {f1_weighted:.4f}",
        "",
        "## Per-class F1",
        "",
        "| Class | F1-Score |",
        "|-------|----------|",
    ]
    for lbl, f1_val in sorted(per_class_dict.items(), key=lambda x: x[1]):
        md_lines.append(f"| `{lbl}` | {f1_val:.4f} |")

    md_lines.extend([
        "",
        "## Confusion matrix",
        "",
        f"Saved to: `{cm_path}`",
        "",
        "## Full classification report",
        "",
        "```",
        cls_report.strip(),
        "```",
        "",
    ])

    eval_md_path = REPORTS / f"evaluation_{safe_name}.md"
    (eval_md_path).write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"\nevaluation complete for {run_name}")
    print(f"  accuracy:    {acc:.4f}")
    print(f"  macro-F1:    {f1_macro:.4f}")
    print(f"  weighted-F1: {f1_weighted:.4f}")
    print(f"  report: {eval_md_path}")
    print(f"  confusion matrix: {cm_path}")


if __name__ == "__main__":
    main()
