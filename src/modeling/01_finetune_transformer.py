import sys
import json
import random
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
class TrainConfig:
    model_name: str = "roberta-base"
    text_field: str = "text_clean"
    label_field: str = "label_fine"
    max_length: int = 256

    num_train_epochs: int = 3
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 16
    warmup_ratio: float = 0.06

    seed: int = 42


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _load_split(split: str, text_field: str, label_field: str) -> tuple[list[str], list[str]]:
    path = PROCESSED / f"{split}.jsonl"
    records = read_jsonl(path)
    texts = [str(r.get(text_field, "")) for r in records]
    labels = [str(r.get(label_field, "other")) for r in records]
    return texts, labels


def _build_label_space(train_labels: list[str]) -> tuple[list[str], dict[str, int]]:
    labels = sorted(set(train_labels))
    if "other" not in labels:
        labels.append("other")
    label2id = {l: i for i, l in enumerate(labels)}
    return labels, label2id


def main():
    """Fine-tune a transformer classifier on `data/processed/*.jsonl`.

    Usage (defaults):
      python src/modeling/01_finetune_transformer.py

    This script intentionally avoids project-specific dependencies beyond HuggingFace Transformers.
    """

    # Local imports so repo can still be imported without ML deps.
    from datasets import Dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )
    from sklearn.metrics import f1_score, accuracy_score

    cfg = TrainConfig()
    _set_seed(cfg.seed)

    MODELS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    train_texts, train_labels = _load_split("train", cfg.text_field, cfg.label_field)
    dev_texts, dev_labels = _load_split("dev", cfg.text_field, cfg.label_field)
    test_texts, test_labels = _load_split("test", cfg.text_field, cfg.label_field)

    label_list, label2id = _build_label_space(train_labels)
    id2label = {v: k for k, v in label2id.items()}

    def encode_labels(labels: list[str]) -> list[int]:
        other_id = label2id["other"]
        return [label2id.get(l, other_id) for l in labels]

    train_ds = Dataset.from_dict({"text": train_texts, "label": encode_labels(train_labels)})
    dev_ds = Dataset.from_dict({"text": dev_texts, "label": encode_labels(dev_labels)})
    test_ds = Dataset.from_dict({"text": test_texts, "label": encode_labels(test_labels)})

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=cfg.max_length,
        )

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    dev_ds = dev_ds.map(tokenize, batched=True, remove_columns=["text"])
    test_ds = test_ds.map(tokenize, batched=True, remove_columns=["text"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
            "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
        }

    run_name = f"finetune_{cfg.model_name.replace('/', '-')}_seed{cfg.seed}"
    output_dir = MODELS / run_name

    args = TrainingArguments(
        output_dir=str(output_dir),
        run_name=run_name,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        warmup_ratio=cfg.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=2,
        seed=cfg.seed,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    train_result = trainer.train()

    dev_metrics = trainer.evaluate(eval_dataset=dev_ds)
    test_metrics = trainer.evaluate(eval_dataset=test_ds)

    trainer.save_model()
    tokenizer.save_pretrained(str(output_dir))

    (output_dir / "label_space.json").write_text(
        json.dumps({"labels": label_list, "label2id": label2id}, indent=2),
        encoding="utf-8",
    )
    (output_dir / "train_config.json").write_text(
        json.dumps(asdict(cfg), indent=2),
        encoding="utf-8",
    )

    report = {
        "run_name": run_name,
        "output_dir": str(output_dir),
        "n_labels": len(label_list),
        "labels": label_list,
        "train_runtime_seconds": float(train_result.metrics.get("train_runtime", 0.0)),
        "dev_metrics": dev_metrics,
        "test_metrics": test_metrics,
    }

    (REPORTS / "modeling_week8.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_lines = [
        "# Week 8 Modeling Report",
        "",
        f"Run: `{run_name}`",
        f"Model: `{cfg.model_name}`",
        f"Labels: {len(label_list)} classes (including `other`)",
        "",
        "## Dev metrics",
        "",
        f"- accuracy: {dev_metrics.get('eval_accuracy', 'N/A')}",
        f"- macro-F1: {dev_metrics.get('eval_f1_macro', 'N/A')}",
        f"- weighted-F1: {dev_metrics.get('eval_f1_weighted', 'N/A')}",
        "",
        "## Test metrics",
        "",
        f"- accuracy: {test_metrics.get('eval_accuracy', 'N/A')}",
        f"- macro-F1: {test_metrics.get('eval_f1_macro', 'N/A')}",
        f"- weighted-F1: {test_metrics.get('eval_f1_weighted', 'N/A')}",
        "",
        "## Artifacts",
        "",
        f"- model + tokenizer saved to: `{output_dir}`",
        "- label space: `label_space.json`",
        "- training config: `train_config.json`",
        "",
    ]
    (REPORTS / "modeling_week8.md").write_text("\n".join(md_lines), encoding="utf-8")

    print("\nweek 8 modeling complete")
    print("  model saved to:", output_dir)
    print("  report written to:", REPORTS / "modeling_week8.md")


if __name__ == "__main__":
    main()
