import sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from utils.io import read_jsonl, write_jsonl

MAPPED = ROOT / "data" / "interim" / "label_mapped"
FILTERED = ROOT / "data" / "interim" / "filtered"

MIN_TOKENS = 5
MAX_TOKENS = 512


def token_count(text: str) -> int:
    return len(text.split())


def truncate(text: str, max_tokens: int) -> str:
    tokens = text.split()
    return " ".join(tokens[:max_tokens])


def main():
    FILTERED.mkdir(parents=True, exist_ok=True)

    drop_reasons: Counter = Counter()
    total_in = 0
    total_out = 0

    for path in sorted(MAPPED.glob("*.jsonl")):
        records = read_jsonl(path)
        total_in += len(records)
        kept = []
        for r in records:
            text = r["text_clean"]

            if not text:
                drop_reasons["empty"] += 1
                continue

            n = token_count(text)

            if n < MIN_TOKENS:
                drop_reasons["too_short"] += 1
                continue

            if n > MAX_TOKENS:
                r["text_clean"] = truncate(text, MAX_TOKENS)
                drop_reasons["truncated"] += 1

            kept.append(r)

        write_jsonl(kept, FILTERED / path.name)
        total_out += len(kept)

    dropped = total_in - total_out
    print(f"\nstage 4 complete.")
    print(f"  input:   {total_in:,}")
    print(f"  kept:    {total_out:,}")
    print(f"  dropped: {dropped:,}")
    print("\ndrop/action breakdown:")
    for reason, cnt in sorted(drop_reasons.items()):
        print(f"  {reason:15s}  {cnt:5d}")


if __name__ == "__main__":
    main()
