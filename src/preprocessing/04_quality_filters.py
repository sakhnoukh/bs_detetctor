import sys
import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from utils.io import read_jsonl, write_jsonl

MAPPED = ROOT / "data" / "interim" / "label_mapped"
FILTERED = ROOT / "data" / "interim" / "filtered"
REPORTS = ROOT / "reports"
SUMMARY_PATH = REPORTS / "quality_filter_stats.json"

MIN_TOKENS = 5
MAX_TOKENS = 512


def token_count(text: str) -> int:
    return len(text.split())


def truncate(text: str, max_tokens: int) -> str:
    tokens = text.split()
    return " ".join(tokens[:max_tokens])


def main():
    FILTERED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    drop_reasons: Counter = Counter()
    source_before: Counter = Counter()
    source_after: Counter = Counter()
    per_file: dict[str, dict[str, int]] = {}
    total_in = 0
    total_out = 0

    for path in sorted(MAPPED.glob("*.jsonl")):
        records = read_jsonl(path)
        for r in records:
            source_before[r["source"]] += 1
        total_in += len(records)
        kept = []
        file_drop_reasons: Counter = Counter()
        for r in records:
            text = r["text_clean"]

            if not text:
                drop_reasons["empty"] += 1
                file_drop_reasons["empty"] += 1
                continue

            n = token_count(text)

            if n < MIN_TOKENS:
                drop_reasons["too_short"] += 1
                file_drop_reasons["too_short"] += 1
                continue

            if n > MAX_TOKENS:
                r["text_clean"] = truncate(text, MAX_TOKENS)
                drop_reasons["truncated"] += 1
                file_drop_reasons["truncated"] += 1

            kept.append(r)
            source_after[r["source"]] += 1

        write_jsonl(kept, FILTERED / path.name)
        total_out += len(kept)
        per_file[path.name] = {
            "input": len(records),
            "kept": len(kept),
            "dropped": len(records) - len(kept),
            "empty": file_drop_reasons.get("empty", 0),
            "too_short": file_drop_reasons.get("too_short", 0),
            "truncated": file_drop_reasons.get("truncated", 0),
        }

    dropped = total_in - total_out
    print(f"\nstage 4 complete.")
    print(f"  input:   {total_in:,}")
    print(f"  kept:    {total_out:,}")
    print(f"  dropped: {dropped:,}")
    print("\ndrop/action breakdown:")
    for reason, cnt in sorted(drop_reasons.items()):
        print(f"  {reason:15s}  {cnt:5d}")

    summary = {
        "total_input": total_in,
        "total_kept": total_out,
        "total_dropped": dropped,
        "drop_reasons": dict(drop_reasons),
        "source_counts_before": dict(source_before),
        "source_counts_after": dict(source_after),
        "per_file": per_file,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nquality filter summary written to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
