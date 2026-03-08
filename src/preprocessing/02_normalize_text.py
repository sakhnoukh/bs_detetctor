"""
Stage 2 — Text Normalization

Reads normalized JSONL from data/interim/normalized/,
fills in text_clean using minimal, reversible normalization,
writes to data/interim/normalized/ (in-place update of each file).

Normalization rules:
  - Unicode NFKC
  - Strip HTML tags
  - Replace URLs → <URL>
  - Replace @mentions → <USER>
  - Collapse repeated whitespace
  - Keep punctuation and casing (important for fallacy rhetoric)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from utils.io import read_jsonl, write_jsonl
from utils.text import normalize

NORMALIZED = ROOT / "data" / "interim" / "normalized"


def main():
    jsonl_files = sorted(NORMALIZED.glob("*.jsonl"))
    if not jsonl_files:
        print("No files found in", NORMALIZED, "— run 01_import.py first.")
        return

    total = 0
    for path in jsonl_files:
        records = read_jsonl(path)
        for r in records:
            r["text_clean"] = normalize(r["text_raw"])
        write_jsonl(records, path)
        total += len(records)

    print(f"\nStage 2 complete. Normalized text_clean for {total:,} records.")


if __name__ == "__main__":
    main()
