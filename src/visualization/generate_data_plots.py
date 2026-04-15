#!/usr/bin/env python3
"""Generate visualizations for data distribution."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Paths
REPORTS = Path(__file__).resolve().parents[2] / "reports"
OUTPUT = REPORTS / "figures"
OUTPUT.mkdir(parents=True, exist_ok=True)

# Set style
sns.set_style("whitegrid")
plt.rcParams["font.size"] = 10

# Data from data_health.md report
label_dist = {
    "ad_hominem": 377,
    "appeal_to_authority": 743,
    "false_cause": 274,
    "false_dilemma": 738,
    "hasty_generalization": 1098,
    "none": 3128,
    "other": 3985,
    "slippery_slope": 649,
    "straw_man": 184,
}

source_dist = {
    "cocolofa": 7702,
    "logic_climate": 1071,
    "logic_edu": 2403,
}

# 1. Label Distribution (Pie Chart)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# All classes
colors = plt.cm.Set3(range(len(label_dist)))
wedges, texts, autotexts = ax1.pie(
    label_dist.values(), 
    labels=[l.replace("_", " ") for l in label_dist.keys()],
    autopct="%1.1f%%",
    colors=colors,
    startangle=90,
    explode=[0.05 if v < 300 else 0 for v in label_dist.values()]  # Explode rare classes
)
ax1.set_title("Label Distribution (All Splits Combined)\nn=11,176", fontsize=14, fontweight="bold")

# 2. Dataset Source Distribution
bars = ax2.bar(source_dist.keys(), source_dist.values(), 
               color=["#3498db", "#e74c3c", "#2ecc71"],
               edgecolor="black", linewidth=1)
ax2.set_ylabel("Number of Samples", fontsize=12)
ax2.set_title("Data by Source", fontsize=14, fontweight="bold")

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    pct = height / sum(source_dist.values()) * 100
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f"{int(height)}\n({pct:.1f}%)",
             ha="center", va="bottom", fontsize=10)

plt.tight_layout()
plt.savefig(OUTPUT / "05_data_distribution.png", dpi=300, bbox_inches="tight")
print(f"Saved: {OUTPUT / '05_data_distribution.png'}")
plt.close()

# 3. Split Distribution
fig, ax = plt.subplots(figsize=(10, 6))
splits = {"Train": 7870, "Dev": 2040, "Test": 1266}
colors_split = ["#27ae60", "#f39c12", "#e74c3c"]

bars = ax.bar(splits.keys(), splits.values(), color=colors_split, edgecolor="black", linewidth=1)
ax.set_ylabel("Number of Samples", fontsize=12)
ax.set_title("Train / Dev / Test Split", fontsize=14, fontweight="bold")

for bar, (name, val) in zip(bars, splits.items()):
    height = bar.get_height()
    pct = val / sum(splits.values()) * 100
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f"{int(val)}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig(OUTPUT / "06_split_distribution.png", dpi=300, bbox_inches="tight")
print(f"Saved: {OUTPUT / '06_split_distribution.png'}")
plt.close()

# 4. Pipeline Flow Diagram
fig, ax = plt.subplots(figsize=(14, 6))

stages = ["Import\n11,506", "Quality\nFilter\n11,458", "Deduplicate\n11,176", "Final\n11,176"]
x_pos = range(len(stages))
values = [11506, 11458, 11176, 11176]
colors_flow = ["#3498db", "#9b59b6", "#e67e22", "#27ae60"]

bars = ax.bar(x_pos, values, color=colors_flow, edgecolor="black", linewidth=1, width=0.6)

# Add arrows and drop counts
for i in range(len(stages)-1):
    dropped = values[i] - values[i+1]
    if dropped > 0:
        ax.annotate(f"-{dropped}", 
                    xy=(i+0.5, (values[i] + values[i+1])/2),
                    fontsize=11, color="red", fontweight="bold",
                    ha="center")

ax.set_xticks(x_pos)
ax.set_xticklabels(stages)
ax.set_ylabel("Number of Records", fontsize=12)
ax.set_title("Data Pipeline Flow", fontsize=14, fontweight="bold")
ax.set_ylim(0, 13000)

# Add percentage kept
for bar, val in zip(bars, values):
    pct = val / 11506 * 100
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 200,
            f"{pct:.1f}% retained",
            ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT / "07_pipeline_flow.png", dpi=300, bbox_inches="tight")
print(f"Saved: {OUTPUT / '07_pipeline_flow.png'}")
plt.close()

# 5. Rare Class Highlight
fig, ax = plt.subplots(figsize=(12, 6))

rare_classes = {k: v for k, v in label_dist.items() if v < 400}
rare_sorted = dict(sorted(rare_classes.items(), key=lambda x: x[1]))

colors_rare = ["#e74c3c" if v < 200 else "#f39c12" for v in rare_sorted.values()]
bars = ax.barh([k.replace("_", " ") for k in rare_sorted.keys()], 
               rare_sorted.values(), 
               color=colors_rare,
               edgecolor="black", linewidth=1)

ax.set_xlabel("Number of Samples", fontsize=12)
ax.set_title("Underrepresented Classes (< 400 samples)", fontsize=14, fontweight="bold")
ax.axvline(x=100, color="red", linestyle="--", alpha=0.7, label="Critical threshold")

for bar, val in zip(bars, rare_sorted.values()):
    width = bar.get_width()
    ax.text(width + 5, bar.get_y() + bar.get_height()/2,
            f"{val} ({val/11176*100:.1f}%)",
            va="center", fontsize=10)

ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT / "08_rare_classes.png", dpi=300, bbox_inches="tight")
print(f"Saved: {OUTPUT / '08_rare_classes.png'}")
plt.close()

print(f"\nAll data visualizations saved to: {OUTPUT}")
