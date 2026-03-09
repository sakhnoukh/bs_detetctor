import sys
import hashlib
import json
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from utils.io import read_jsonl, write_jsonl

FILTERED = ROOT / "data" / "interim" / "filtered"
DEDUPED = ROOT / "data" / "interim" / "deduped"
REPORTS = ROOT / "reports"
SUMMARY_PATH = REPORTS / "dedup_stats.json"


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def split_key_from_filename(filename: str) -> str:
    name = filename.lower()
    if "train" in name:
        return "train"
    if "dev" in name:
        return "dev"
    if "test" in name:
        return "test"
    return "unknown"


def split_priority(filename: str) -> int:
    split = split_key_from_filename(filename)
    if split == "train":
        return 0
    if split == "dev":
        return 1
    if split == "test":
        return 2
    return 3


def main():
    DEDUPED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    all_records: list[tuple[str, dict]] = []
    for path in sorted(FILTERED.glob("*.jsonl")):
        for r in read_jsonl(path):
            all_records.append((path.name, r))

    # Explicit retention policy: when duplicates collide, prefer train over dev over test.
    # Keep original in-file order (stable sort on file-level keys only).
    all_records.sort(key=lambda x: (split_priority(x[0]), x[0]))

    total_in = len(all_records)

    seen: dict[str, tuple[str, dict]] = {}
    cross_file_dups = 0
    cross_split_dups = 0
    same_file_dups = 0
    deduped_by_file: dict[str, list[dict]] = defaultdict(list)
    source_before: Counter = Counter()
    source_after: Counter = Counter()

    for fname, r in all_records:
        source_before[r["source"]] += 1
        h = sha1(r["text_clean"])
        if h in seen:
            orig_fname, _ = seen[h]
            if orig_fname != fname:
                cross_file_dups += 1
                if split_key_from_filename(orig_fname) != split_key_from_filename(fname):
                    cross_split_dups += 1
            else:
                same_file_dups += 1
        else:
            seen[h] = (fname, r)
            deduped_by_file[fname].append(r)
            source_after[r["source"]] += 1

    total_out = 0
    for fname in sorted(deduped_by_file):
        records = deduped_by_file[fname]
        write_jsonl(records, DEDUPED / fname)
        total_out += len(records)

    dropped = total_in - total_out
    print(f"\nstage 5 complete.")
    print(f"  input:                   {total_in:,}")
    print(f"  kept:                    {total_out:,}")
    print(f"  dropped (same file):     {same_file_dups:,}")
    print(f"  dropped (cross-file):    {cross_file_dups:,}")
    print(f"  dropped (cross-split):   {cross_split_dups:,}")
    print(f"  total dropped:           {dropped:,}")

    summary = {
        "total_input": total_in,
        "total_kept": total_out,
        "total_dropped": dropped,
        "drop_same_file": same_file_dups,
        "drop_cross_file": cross_file_dups,
        "drop_cross_split": cross_split_dups,
        "retention_policy": "prefer train > dev > test when duplicate text appears across files",
        "source_counts_before": dict(source_before),
        "source_counts_after": dict(source_after),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\ndedup summary written to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
