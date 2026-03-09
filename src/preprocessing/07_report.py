import sys
import json
from pathlib import Path
from collections import Counter
from datetime import date

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from utils.io import read_jsonl

PROCESSED = ROOT / "data" / "processed"
INTERIM_NORMALIZED = ROOT / "data" / "interim" / "normalized"
INTERIM_FILTERED = ROOT / "data" / "interim" / "filtered"
INTERIM_DEDUPED = ROOT / "data" / "interim" / "deduped"
REPORTS = ROOT / "reports"
QUALITY_SUMMARY_PATH = REPORTS / "quality_filter_stats.json"
DEDUP_SUMMARY_PATH = REPORTS / "dedup_stats.json"


def count_jsonl(folder: Path, pattern: str = "*.jsonl") -> int:
    total = 0
    for p in folder.glob(pattern):
        with open(p, encoding="utf-8") as f:
            total += sum(1 for line in f if line.strip())
    return total


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def text_length_stats(records: list[dict]) -> dict:
    lengths = [len(r["text_clean"].split()) for r in records]
    if not lengths:
        return {}
    lengths.sort()
    n = len(lengths)
    return {
        "min": lengths[0],
        "max": lengths[-1],
        "mean": round(sum(lengths) / n, 1),
        "median": lengths[n // 2],
    }


def label_table(records: list[dict]) -> str:
    dist = Counter(r["label_fine"] for r in records)
    total = sum(dist.values())
    rows = ["| Label | Count | % |", "|-------|-------|---|"]
    for label in sorted(dist):
        cnt = dist[label]
        pct = 100 * cnt / total if total else 0
        rows.append(f"| `{label}` | {cnt:,} | {pct:.1f}% |")
    rows.append(f"| **Total** | **{total:,}** | 100% |")
    return "\n".join(rows)


def before_after_source_table(before: dict, after: dict) -> str:
    keys = sorted(set(before) | set(after))
    rows = [
        "| Source | Before quality filter | After quality filter | Removed |",
        "|--------|------------------------|----------------------|---------|",
    ]
    total_before = 0
    total_after = 0
    for src in keys:
        b = int(before.get(src, 0))
        a = int(after.get(src, 0))
        rows.append(f"| `{src}` | {b:,} | {a:,} | {b - a:,} |")
        total_before += b
        total_after += a
    rows.append(f"| **Total** | **{total_before:,}** | **{total_after:,}** | **{total_before - total_after:,}** |")
    return "\n".join(rows)


def drop_reason_table(drop_reasons: dict) -> str:
    rows = ["| Reason | Count |", "|--------|-------|"]
    total = 0
    for reason, count in sorted(drop_reasons.items()):
        rows.append(f"| `{reason}` | {int(count):,} |")
        total += int(count)
    rows.append(f"| **Total** | **{total:,}** |")
    return "\n".join(rows)


def source_table(records: list[dict]) -> str:
    dist = Counter(r["source"] for r in records)
    total = sum(dist.values())
    rows = ["| Source | Count | % |", "|--------|-------|---|"]
    for src in sorted(dist):
        cnt = dist[src]
        pct = 100 * cnt / total if total else 0
        rows.append(f"| `{src}` | {cnt:,} | {pct:.1f}% |")
    rows.append(f"| **Total** | **{total:,}** | 100% |")
    return "\n".join(rows)


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)

    quality_summary = load_json(QUALITY_SUMMARY_PATH)
    dedup_summary = load_json(DEDUP_SUMMARY_PATH)

    all_records: list[dict] = []
    split_records: dict[str, list[dict]] = {}
    for split in ("train", "dev", "test"):
        path = PROCESSED / f"{split}.jsonl"
        if path.exists():
            recs = read_jsonl(path)
            split_records[split] = recs
            all_records.extend(recs)
        else:
            split_records[split] = []

    n_normalized = count_jsonl(INTERIM_NORMALIZED)
    n_filtered = count_jsonl(INTERIM_FILTERED)
    n_deduped = count_jsonl(INTERIM_DEDUPED)
    n_final = len(all_records)
    removed_total = n_normalized - n_final
    removed_pct = (100 * removed_total / n_normalized) if n_normalized else 0.0

    stats = text_length_stats(all_records)
    coarse_dist = Counter(r["label_coarse"] for r in all_records)

    lines = [
        f"# Data Health Report",
        f"",
        f"Generated: {date.today()}",
        f"",
        f"---",
        f"",
        f"## 1. Pipeline Stage Counts",
        f"",
        f"| Stage | Records |",
        f"|-------|---------|",
        f"| After import (Stage 1) | {n_normalized:,} |",
        f"| After quality filter (Stage 4) | {n_filtered:,} |",
        f"| After deduplication (Stage 5) | {n_deduped:,} |",
        f"| **Final processed total** | **{n_final:,}** |",
        f"",
        f"Records removed: {removed_total:,} ({removed_pct:.1f}% of raw)",
        f"",
        f"---",
        f"",
        f"## 2. Final Split Sizes",
        f"",
        f"| Split | Records |",
        f"|-------|---------|",
    ]
    for split in ("train", "dev", "test"):
        lines.append(f"| {split} | {len(split_records[split]):,} |")
    lines += [
        f"| **Total** | **{n_final:,}** |",
        f"",
        f"---",
        f"",
        f"## 3. Label Distribution (all splits combined)",
        f"",
        label_table(all_records),
        f"",
        f"### Coarse label split",
        f"",
        f"| Coarse Label | Count |",
        f"|--------------|-------|",
        f"| `fallacy` | {coarse_dist.get('fallacy', 0):,} |",
        f"| `no_fallacy` | {coarse_dist.get('no_fallacy', 0):,} |",
        f"",
        f"---",
        f"",
        f"## 4. Data Sources",
        f"",
        source_table(all_records),
        f"",
        f"---",
        f"",
        f"## 5. Before/After Counts Per Source (Stage 4 quality filter)",
        f"",
    ]

    if quality_summary:
        lines += [
            before_after_source_table(
                quality_summary.get("source_counts_before", {}),
                quality_summary.get("source_counts_after", {}),
            ),
            f"",
            f"### Stage 4 Drop Reason Breakdown",
            f"",
            drop_reason_table(quality_summary.get("drop_reasons", {})),
        ]
    else:
        lines += [
            "_No quality filter summary found. Run `04_quality_filters.py` to include this section._",
        ]

    lines += [
        f"",
        f"---",
        f"",
        f"## 6. Deduplication Summary (Stage 5)",
        f"",
    ]

    if dedup_summary:
        lines += [
            f"- Retention policy: {dedup_summary.get('retention_policy', 'N/A')}",
            f"- Input records: {dedup_summary.get('total_input', 0):,}",
            f"- Kept records: {dedup_summary.get('total_kept', 0):,}",
            f"- Dropped (same file): {dedup_summary.get('drop_same_file', 0):,}",
            f"- Dropped (cross file): {dedup_summary.get('drop_cross_file', 0):,}",
            f"- Dropped (cross split): {dedup_summary.get('drop_cross_split', 0):,}",
        ]
    else:
        lines += [
            "_No dedup summary found. Run `05_deduplicate.py` to include this section._",
        ]

    lines += [
        f"",
        f"---",
        f"",
        f"## 7. Text Length Statistics (tokens, whitespace-split)",
        f"",
        f"| Stat | Value |",
        f"|------|-------|",
        f"| Min | {stats.get('min', 'N/A')} |",
        f"| Max | {stats.get('max', 'N/A')} |",
        f"| Mean | {stats.get('mean', 'N/A')} |",
        f"| Median | {stats.get('median', 'N/A')} |",
        f"",
        f"---",
        f"",
        f"## 8. Label Distribution Per Split",
        f"",
    ]

    for split in ("train", "dev", "test"):
        recs = split_records[split]
        lines += [
            f"### {split.capitalize()} ({len(recs):,} records)",
            f"",
            label_table(recs),
            f"",
        ]

    report = "\n".join(lines)
    out_path = REPORTS / "data_health.md"
    out_path.write_text(report, encoding="utf-8")
    print(f"report written to: {out_path}")


if __name__ == "__main__":
    main()
