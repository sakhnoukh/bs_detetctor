import sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from utils.io import read_jsonl, write_jsonl

DEDUPED = ROOT / "data" / "interim" / "deduped"
PROCESSED = ROOT / "data" / "processed"


def split_key(filename: str) -> str:
    name = filename.lower()
    if "train" in name:
        return "train"
    if "dev" in name:
        return "dev"
    if "test" in name:
        return "test"
    raise ValueError(f"cannot determine split from filename: {filename}")


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)

    buckets: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}

    for path in sorted(DEDUPED.glob("*.jsonl")):
        try:
            split = split_key(path.name)
        except ValueError as e:
            print(f"  skipping: {e}")
            continue
        records = read_jsonl(path)
        buckets[split].extend(records)
        print(f"  {path.name} → {split} ({len(records):,} records)")

    for split, records in buckets.items():
        write_jsonl(records, PROCESSED / f"{split}.jsonl")

    print("\nfinal split sizes:")
    total = 0
    for split, records in buckets.items():
        label_dist = Counter(r["label_fine"] for r in records)
        print(f"  {split:6s}: {len(records):6,} records")
        for label, cnt in sorted(label_dist.items()):
            print(f"           {label:25s} {cnt:5d}")
        total += len(records)
    print(f"  {'total':6s}: {total:6,} records")
    print(f"\nstage 6 complete. processed data in: {PROCESSED}")


if __name__ == "__main__":
    main()
