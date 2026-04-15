import sys
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from utils.io import read_jsonl

PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"


@dataclass(frozen=True)
class EvalConfig:
    run_name: str = "finetune_roberta-base_seed42"
    text_field: str = "text_clean"
    label_field: str = "label_fine"
    max_length: int = 256


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
    """Evaluate a fine-tuned transformer model.

    This script depends on artifacts produced by Week 8:
      - models/<run_name>/ (HF model + tokenizer)
      - models/<run_name>/label_space.json

    Usage:
      python src/evaluation/01_evaluate_transformer.py

    If Week 8 hasn't been run yet, the script will exit with instructions.
    """

    from datasets import Dataset
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        classification_report,
        confusion_matrix,
    )

    cfg = EvalConfig()
    model_dir = MODELS / cfg.run_name

    label_space_path = model_dir / "label_space.json"
    if not model_dir.exists() or not label_space_path.exists():
        print("Week 9 evaluation cannot run yet: missing Week 8 artifacts.")
        print(f"Expected: {model_dir}")
        print("Run Week 8 first:")
        print("  python src/modeling/01_finetune_transformer.py")
        sys.exit(1)

    label_space = json.loads(label_space_path.read_text(encoding="utf-8"))
    labels: list[str] = list(label_space["labels"])
    label2id: dict[str, int] = {k: int(v) for k, v in label_space["label2id"].items()}
    other_id = int(label2id.get("other", len(labels) - 1))

    test_texts, test_labels = _load_split("test", cfg.text_field, cfg.label_field)
    y_true = np.asarray([label2id.get(l, other_id) for l in test_labels], dtype=np.int64)

    ds = Dataset.from_dict({"text": test_texts, "label": y_true.tolist()})

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=cfg.max_length)

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

    cm_path = REPORTS / "confusion_matrix_week9.csv"
    _write_confusion_csv(labels, cm, cm_path)

    cls_report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(labels))),
        target_names=labels,
        digits=4,
        zero_division=0,
    )

    summary = {
        "run_name": cfg.run_name,
        "model_dir": str(model_dir),
        "n_test": int(len(y_true)),
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "labels": labels,
        "confusion_matrix_csv": str(cm_path),
    }

    (REPORTS / "evaluation_week9.json").write_text(json.dumps({"config": asdict(cfg), "summary": summary}, indent=2), encoding="utf-8")

    md_lines = [
        "# Week 9 Evaluation Report",
        "",
        f"Model run: `{cfg.run_name}`",
        f"Model dir: `{model_dir}`",
        "",
        "## Metrics (test set)",
        "",
        f"- accuracy: {acc:.4f}",
        f"- macro-F1: {f1_macro:.4f}",
        f"- weighted-F1: {f1_weighted:.4f}",
        "",
        "## Confusion matrix",
        "",
        f"Saved to: `{cm_path}`",
        "",
        "## Per-class report",
        "",
        "```",
        cls_report.strip(),
        "```",
        "",
        "## Notes",
        "",
        "- The label space includes `other` as a real class.",
        "- If you want additional analysis (top confusions, error examples), extend this report after inspection.",
    ]

    (REPORTS / "evaluation_week9.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print("\nweek 9 evaluation complete")
    print("  report written to:", REPORTS / "evaluation_week9.md")
    print("  confusion matrix written to:", cm_path)


if __name__ == "__main__":
    main()
