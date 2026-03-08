"""
Stage 6 — Train / Dev / Test Split and Export

Split strategy:
  - CoCoLoFa: use its pre-existing splits. All cocolofa_train* → train, etc.
  - LOGIC: already provided as train/dev/test splits — preserve them too.
  - Merge all train files → train.jsonl, dev files → dev.jsonl, etc.

Leakage control:
  - CoCoLoFa comments from the same article stay in the same split
    (preserved by using the existing splits as-is).
  - LOGIC edu and climate are kept in their original splits.

Outputs:
  data/processed/train.jsonl
  data/processed/dev.jsonl
  data/processed/test.jsonl
"""
import sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from utils.io import read_jsonl, write_jsonl

DEDUPED = ROOT / "data" / "interim" / "deduped"
PROCESSED = ROOT / "data" / "processed"


def split_key(filename: str) -> str:
    """Infer train/dev/test from filename."""
    name = filename.lower()
    if "train" in name:
        return "train"
    if "dev" in name:
        return "dev"
    if "test" in name:
        return "test"
    raise ValueError(f"Cannot determine split from filename: {filename}")


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)

    buckets: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}

    for path in sorted(DEDUPED.glob("*.jsonl")):
        try:
            split = split_key(path.name)
        except ValueError as e:
            print(f"  Skipping: {e}")
            continue
        records = read_jsonl(path)
        buckets[split].extend(records)
        print(f"  {path.name} → {split} ({len(records):,} records)")

    for split, records in buckets.items():
        out_path = PROCESSED / f"{split}.jsonl"
        write_jsonl(records, out_path)

    # Summary
    print("\nFinal split sizes:")
    total = 0
    for split, records in buckets.items():
        label_dist = Counter(r["label_fine"] for r in records)
        print(f"  {split:6s}: {len(records):6,} records")
        for label, cnt in sorted(label_dist.items()):
            print(f"           {label:25s} {cnt:5d}")
        total += len(records)
    print(f"  {'TOTAL':6s}: {total:6,} records")
    print(f"\nStage 6 complete. Processed data in: {PROCESSED}")


if __name__ == "__main__":
    main()
