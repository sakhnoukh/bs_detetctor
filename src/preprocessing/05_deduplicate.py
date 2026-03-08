"""
Stage 5 — Deduplication

Reads from data/interim/filtered/,
removes exact duplicates by SHA-1 hash of text_clean,
writes to data/interim/deduped/.

Strategy:
  - Hash text_clean for every record.
  - On collision, keep the first occurrence and drop the rest.
  - Report cross-dataset duplicates separately for transparency.
"""
import sys
import hashlib
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from utils.io import read_jsonl, write_jsonl

FILTERED = ROOT / "data" / "interim" / "filtered"
DEDUPED = ROOT / "data" / "interim" / "deduped"


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def main():
    DEDUPED.mkdir(parents=True, exist_ok=True)

    # Load all records from all files, tagging which file they came from
    all_records: list[tuple[str, dict]] = []
    for path in sorted(FILTERED.glob("*.jsonl")):
        for r in read_jsonl(path):
            all_records.append((path.name, r))

    total_in = len(all_records)

    # Deduplicate: hash → first-seen (filename, record)
    seen: dict[str, tuple[str, dict]] = {}
    cross_dataset_dups = 0
    same_file_dups = 0

    deduped_by_file: dict[str, list[dict]] = defaultdict(list)

    for fname, r in all_records:
        h = sha1(r["text_clean"])
        if h in seen:
            orig_fname, _ = seen[h]
            if orig_fname != fname:
                cross_dataset_dups += 1
            else:
                same_file_dups += 1
        else:
            seen[h] = (fname, r)
            deduped_by_file[fname].append(r)

    # Write per-file deduplicated output
    total_out = 0
    for fname, records in deduped_by_file.items():
        write_jsonl(records, DEDUPED / fname)
        total_out += len(records)

    dropped = total_in - total_out
    print(f"\nStage 5 complete.")
    print(f"  Input:                 {total_in:,}")
    print(f"  Kept:                  {total_out:,}")
    print(f"  Dropped (same file):   {same_file_dups:,}")
    print(f"  Dropped (cross-dataset): {cross_dataset_dups:,}")
    print(f"  Total dropped:         {dropped:,}")


if __name__ == "__main__":
    main()
