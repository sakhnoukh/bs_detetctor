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
        print("no files found in", NORMALIZED, "— run 01_import.py first.")
        return

    total = 0
    for path in jsonl_files:
        records = read_jsonl(path)
        for r in records:
            r["text_clean"] = normalize(r["text_raw"])
        write_jsonl(records, path)
        total += len(records)

    print(f"\nstage 2 complete. normalized text_clean for {total:,} records.")


if __name__ == "__main__":
    main()
