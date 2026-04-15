#!/usr/bin/env python3
"""Generate visualizations for BS-Detector results."""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Paths
REPORTS = Path(__file__).resolve().parents[2] / "reports" / "results"
OUTPUT = Path(__file__).resolve().parents[2] / "reports" / "figures"
OUTPUT.mkdir(parents=True, exist_ok=True)

# Per-class data from evaluation report
class_data = {
    "ad_hominem": {"precision": 0.6552, "recall": 0.7037, "f1": 0.6786, "support": 54},
    "appeal_to_authority": {"precision": 0.7191, "recall": 0.7033, "f1": 0.7111, "support": 91},
    "false_cause": {"precision": 0.4815, "recall": 0.4333, "f1": 0.4561, "support": 30},
    "false_dilemma": {"precision": 0.8065, "recall": 0.7353, "f1": 0.7692, "support": 68},
    "hasty_generalization": {"precision": 0.6818, "recall": 0.5208, "f1": 0.5906, "support": 144},
    "none": {"precision": 0.7870, "recall": 0.8070, "f1": 0.7969, "support": 316},
    "other": {"precision": 0.7619, "recall": 0.8348, "f1": 0.7967, "support": 460},
    "slippery_slope": {"precision": 0.8481, "recall": 0.8590, "f1": 0.8535, "support": 78},
    "straw_man": {"precision": 0.6923, "recall": 0.3600, "f1": 0.4737, "support": 25},
}

df = pd.DataFrame(class_data).T
df = df.sort_values("f1", ascending=True)

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 10

# 1. F1-Score by Class
fig, ax = plt.subplots(figsize=(10, 6))
colors = ["#e74c3c" if f1 < 0.5 else "#f39c12" if f1 < 0.65 else "#27ae60" for f1 in df["f1"]]
bars = ax.barh(df.index, df["f1"], color=colors, edgecolor="black", linewidth=0.5)
ax.set_xlim(0, 1)
ax.set_xlabel("F1-Score", fontsize=12)
ax.set_title("Per-Class F1-Score on Test Set (RoBERTa-base)", fontsize=14, fontweight="bold")
ax.axvline(x=0.681, color="gray", linestyle="--", alpha=0.7, label="Macro-F1: 0.681")

for bar, f1, support in zip(bars, df["f1"], df["support"]):
    width = bar.get_width()
    ax.text(width + 0.02, bar.get_y() + bar.get_height()/2, 
            f"{f1:.3f} (n={int(support)})", va="center", fontsize=9)

ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(OUTPUT / "01_f1_by_class.png", dpi=300, bbox_inches="tight")
print(f"Saved: {OUTPUT / '01_f1_by_class.png'}")
plt.close()

# 2. Precision vs Recall
fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(df["recall"], df["precision"], 
                     s=df["support"]*3, 
                     c=df["f1"], 
                     cmap="RdYlGn", 
                     vmin=0.4, vmax=0.9,
                     edgecolors="black", linewidth=1)

for idx, row in df.iterrows():
    ax.annotate(idx.replace("_", " "), (row["recall"], row["precision"]), 
                xytext=(5, 5), textcoords="offset points", fontsize=9)

ax.set_xlabel("Recall", fontsize=12)
ax.set_ylabel("Precision", fontsize=12)
ax.set_title("Precision vs Recall by Class\n(Bubble size = support, Color = F1)", fontsize=14, fontweight="bold")
plt.colorbar(scatter, label="F1-Score")
plt.tight_layout()
plt.savefig(OUTPUT / "02_precision_recall.png", dpi=300, bbox_inches="tight")
print(f"Saved: {OUTPUT / '02_precision_recall.png'}")
plt.close()

# 3. Confusion Matrix Heatmap
labels = ["ad_hominem", "appeal_to_authority", "false_cause", "false_dilemma", 
          "hasty_generalization", "none", "other", "slippery_slope", "straw_man"]

# Load confusion matrix
cm_path = REPORTS / "confusion_matrix_week9.csv"
cm_data = pd.read_csv(cm_path, index_col=0)

fig, ax = plt.subplots(figsize=(12, 10))
# Normalize by row (true labels) for better visualization
cm_normalized = cm_data.div(cm_data.sum(axis=1), axis=0)

sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="YlOrRd", 
            xticklabels=labels, yticklabels=labels, ax=ax, 
            cbar_kws={"label": "Proportion of True Label"})
ax.set_xlabel("Predicted Label", fontsize=12)
ax.set_ylabel("True Label", fontsize=12)
ax.set_title("Normalized Confusion Matrix (Test Set)", fontsize=14, fontweight="bold")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT / "03_confusion_matrix.png", dpi=300, bbox_inches="tight")
print(f"Saved: {OUTPUT / '03_confusion_matrix.png'}")
plt.close()

# 4. Support vs Performance
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(df["support"], df["f1"], s=200, c=df["f1"], cmap="RdYlGn", 
           edgecolors="black", linewidth=1, vmin=0.4, vmax=0.9)

for idx, row in df.iterrows():
    ax.annotate(idx.replace("_", " "), (row["support"], row["f1"]), 
                xytext=(5, 5), textcoords="offset points", fontsize=9)

ax.set_xlabel("Number of Test Samples (Support)", fontsize=12)
ax.set_ylabel("F1-Score", fontsize=12)
ax.set_title("Class Imbalance vs Performance", fontsize=14, fontweight="bold")
ax.axhline(y=0.681, color="gray", linestyle="--", alpha=0.7, label="Macro-F1")
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT / "04_support_vs_f1.png", dpi=300, bbox_inches="tight")
print(f"Saved: {OUTPUT / '04_support_vs_f1.png'}")
plt.close()

print(f"\nAll figures saved to: {OUTPUT}")
