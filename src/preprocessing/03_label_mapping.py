import sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from utils.io import read_jsonl, write_jsonl

NORMALIZED = ROOT / "data" / "interim" / "normalized"
MAPPED = ROOT / "data" / "interim" / "label_mapped"

LOGIC_MAP: dict[str, str] = {
    "ad_hominem":             "ad_hominem",
    "false_causality":        "false_cause",
    "false_dilemma":          "false_dilemma",
    "faulty_generalization":  "hasty_generalization",
    "appeal_to_emotion":      "other",
    "ad_populum":             "other",
    "circular_reasoning":     "other",
    "fallacy_of_relevance":   "other",
    "fallacy_of_extension":   "straw_man",  # misrepresenting scope of an argument
    "equivocation":           "other",
    "fallacy_of_logic":       "other",
    "fallacy_of_credibility": "appeal_to_authority",
    "intentional":            "other",
}

COCOLOFA_MAP: dict[str, str] = {
    "none":                     "none",
    "appeal_to_authority":      "appeal_to_authority",
    "appeal_to_majority":       "other",
    "appeal_to_nature":         "other",
    "appeal_to_tradition":      "other",
    "appeal_to_worse_problems": "other",
    "false_dilemma":            "false_dilemma",
    "hasty_generalization":     "hasty_generalization",
    "slippery_slope":           "slippery_slope",
}

SOURCE_MAPS: dict[str, dict[str, str]] = {
    "logic_edu":     LOGIC_MAP,
    "logic_climate": LOGIC_MAP,
    "cocolofa":      COCOLOFA_MAP,
}

VALID_LABELS = {
    "ad_hominem", "appeal_to_authority", "false_dilemma", "false_cause",
    "hasty_generalization", "slippery_slope", "straw_man", "none", "other",
}


def map_label(raw_label: str, source: str) -> str:
    mapping = SOURCE_MAPS.get(source, {})
    mapped = mapping.get(raw_label)
    if mapped is None:
        if raw_label in VALID_LABELS:
            return raw_label
        return "other"
    return mapped


def coarse(label_fine: str) -> str:
    return "no_fallacy" if label_fine == "none" else "fallacy"


def main():
    MAPPED.mkdir(parents=True, exist_ok=True)

    unmapped_counter: Counter = Counter()
    total = 0

    for path in sorted(NORMALIZED.glob("*.jsonl")):
        records = read_jsonl(path)
        for r in records:
            raw = r["label_fine"]
            source = r["source"]
            mapped = map_label(raw, source)
            if mapped == "other":
                unmapped_counter[(source, raw)] += 1
            r["label_fine"] = mapped
            r["label_coarse"] = coarse(mapped)
        out_path = MAPPED / path.name
        write_jsonl(records, out_path)
        total += len(records)

    print(f"\nstage 3 complete. mapped labels for {total:,} records.")

    if unmapped_counter:
        print("\nlabels mapped to 'other' (source, raw_label → count):")
        for (src, lbl), cnt in sorted(unmapped_counter.items()):
            print(f"  {src:20s}  {lbl:30s}  {cnt:5d}")


if __name__ == "__main__":
    main()
