#!/usr/bin/env python3
"""Improved transformer fine-tuning with class-weighted loss, focal loss,
and label smoothing to address class imbalance.

Compared to 01_finetune_transformer.py (baseline):
  - Class-weighted cross-entropy loss (inverse frequency)
  - Optional focal loss (gamma=2) for hard-example mining
  - Label smoothing (0.1)
  - Longer training (5 epochs) with early stopping patience

Usage:
  python src/modeling/02_finetune_improved.py                     # weighted CE (default)
  python src/modeling/02_finetune_improved.py --loss focal         # focal loss
  python src/modeling/02_finetune_improved.py --loss weighted_ce   # explicit weighted CE
"""

import sys
import json
import argparse
import random
from collections import Counter
from dataclasses import dataclass, asdict, field
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

    num_train_epochs: int = 5
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 16
    warmup_ratio: float = 0.06
    label_smoothing: float = 0.1
    early_stopping_patience: int = 2

    loss_type: str = "weighted_ce"   # "weighted_ce" or "focal"
    focal_gamma: float = 2.0
    focal_alpha: float = 1.0

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


def _compute_class_weights(encoded_labels: list[int], num_classes: int) -> np.ndarray:
    """Compute inverse-frequency class weights, normalized to sum to num_classes."""
    counts = Counter(encoded_labels)
    total = len(encoded_labels)
    weights = np.zeros(num_classes, dtype=np.float32)
    for cls_id in range(num_classes):
        count = counts.get(cls_id, 1)
        weights[cls_id] = total / (num_classes * count)
    return weights


def main():
    parser = argparse.ArgumentParser(description="Improved transformer fine-tuning")
    parser.add_argument("--loss", type=str, default="weighted_ce",
                        choices=["weighted_ce", "focal"],
                        help="Loss function type")
    cli_args = parser.parse_args()

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from datasets import Dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        EarlyStoppingCallback,
    )
    from sklearn.metrics import f1_score, accuracy_score

    cfg = TrainConfig(loss_type=cli_args.loss)
    _set_seed(cfg.seed)

    MODELS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────
    train_texts, train_labels = _load_split("train", cfg.text_field, cfg.label_field)
    dev_texts, dev_labels = _load_split("dev", cfg.text_field, cfg.label_field)
    test_texts, test_labels = _load_split("test", cfg.text_field, cfg.label_field)

    label_list, label2id = _build_label_space(train_labels)
    id2label = {v: k for k, v in label2id.items()}

    def encode_labels(labels: list[str]) -> list[int]:
        other_id = label2id["other"]
        return [label2id.get(l, other_id) for l in labels]

    encoded_train = encode_labels(train_labels)
    train_ds = Dataset.from_dict({"text": train_texts, "label": encoded_train})
    dev_ds = Dataset.from_dict({"text": dev_texts, "label": encode_labels(dev_labels)})
    test_ds = Dataset.from_dict({"text": test_texts, "label": encode_labels(test_labels)})

    # ── Class weights ─────────────────────────────────────────────────
    class_weights = _compute_class_weights(encoded_train, len(label_list))
    print(f"\nClass weights ({cfg.loss_type}):")
    for i, lbl in enumerate(label_list):
        count = Counter(encoded_train).get(i, 0)
        print(f"  {lbl:25s}  n={count:5d}  weight={class_weights[i]:.3f}")

    # ── Tokenizer & datasets ─────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=cfg.max_length)

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    dev_ds = dev_ds.map(tokenize, batched=True, remove_columns=["text"])
    test_ds = test_ds.map(tokenize, batched=True, remove_columns=["text"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ── Model ─────────────────────────────────────────────────────────
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )

    # ── Custom Trainer with weighted / focal loss ─────────────────────
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32)

    class WeightedTrainer(Trainer):
        """Trainer subclass that supports class-weighted CE and focal loss."""

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits

            device = logits.device
            w = weight_tensor.to(device)

            if cfg.loss_type == "focal":
                ce = F.cross_entropy(logits, labels, weight=w, reduction="none")
                pt = torch.exp(-ce)
                focal_term = (1 - pt) ** cfg.focal_gamma
                loss = (cfg.focal_alpha * focal_term * ce).mean()
            else:
                loss = F.cross_entropy(logits, labels, weight=w,
                                       label_smoothing=cfg.label_smoothing)

            return (loss, outputs) if return_outputs else loss

    # ── Metrics ───────────────────────────────────────────────────────
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
            "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
        }

    # ── Training arguments ────────────────────────────────────────────
    run_name = f"improved_{cfg.loss_type}_{cfg.model_name.replace('/', '-')}_seed{cfg.seed}"
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

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)],
    )

    # ── Train ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Training: {run_name}")
    print(f"  Loss: {cfg.loss_type}")
    print(f"  Epochs: {cfg.num_train_epochs} (early stopping patience={cfg.early_stopping_patience})")
    print(f"  Label smoothing: {cfg.label_smoothing}")
    print(f"{'='*60}\n")

    train_result = trainer.train()

    dev_metrics = trainer.evaluate(eval_dataset=dev_ds)
    test_metrics = trainer.evaluate(eval_dataset=test_ds)

    # ── Save artifacts ────────────────────────────────────────────────
    trainer.save_model()
    tokenizer.save_pretrained(str(output_dir))

    (output_dir / "label_space.json").write_text(
        json.dumps({"labels": label_list, "label2id": label2id}, indent=2),
        encoding="utf-8",
    )

    config_dict = asdict(cfg)
    config_dict["class_weights"] = {label_list[i]: float(class_weights[i])
                                     for i in range(len(label_list))}
    (output_dir / "train_config.json").write_text(
        json.dumps(config_dict, indent=2),
        encoding="utf-8",
    )

    # ── Report ────────────────────────────────────────────────────────
    report = {
        "run_name": run_name,
        "output_dir": str(output_dir),
        "loss_type": cfg.loss_type,
        "n_labels": len(label_list),
        "labels": label_list,
        "class_weights": {label_list[i]: float(class_weights[i])
                          for i in range(len(label_list))},
        "train_runtime_seconds": float(train_result.metrics.get("train_runtime", 0.0)),
        "dev_metrics": dev_metrics,
        "test_metrics": test_metrics,
    }

    report_name = f"modeling_improved_{cfg.loss_type}"
    (REPORTS / f"{report_name}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_lines = [
        f"# Improved Modeling Report ({cfg.loss_type})",
        "",
        f"Run: `{run_name}`",
        f"Model: `{cfg.model_name}`",
        f"Loss: **{cfg.loss_type}** with class weights",
        f"Labels: {len(label_list)} classes (including `other`)",
        f"Epochs: {cfg.num_train_epochs} (early stopping patience={cfg.early_stopping_patience})",
        f"Label smoothing: {cfg.label_smoothing}",
        "",
        "## Class weights",
        "",
        "| Class | Train count | Weight |",
        "|-------|-------------|--------|",
    ]
    for i, lbl in enumerate(label_list):
        count = Counter(encoded_train).get(i, 0)
        md_lines.append(f"| `{lbl}` | {count} | {class_weights[i]:.3f} |")

    md_lines.extend([
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
    ])
    (REPORTS / f"{report_name}.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(f"\nimproved modeling complete ({cfg.loss_type})")
    print("  model saved to:", output_dir)
    print("  report written to:", REPORTS / f"{report_name}.md")


if __name__ == "__main__":
    main()
