import sys
from pathlib import Path

import pandas as pd
import json

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from utils.io import write_jsonl

RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "interim" / "normalized"


def _logic_label(raw: str) -> str:
    return raw.strip().lower().replace(" ", "_")


def import_logic(domain: str, split: str) -> list[dict]:
    path = RAW / "logic" / f"{domain}_{split}.csv"
    df = pd.read_csv(path)

    # edu uses "updated_label"; climate uses "logical_fallacies"
    label_col = "updated_label" if "updated_label" in df.columns else "logical_fallacies"

    records = []
    for i, row in df.iterrows():
        text = str(row["source_article"]).strip()
        label = _logic_label(str(row[label_col]))
        records.append({
            "id": f"logic_{domain}_{split}_{i}",
            "source": f"logic_{domain}",
            "text_raw": text,
            "text_clean": "",
            "label_fine": label,
            "label_coarse": "",
            "meta": {
                "domain": domain,
                "original_split": split,
            },
        })
    return records


def _cocolofa_label(raw: str) -> str:
    return raw.strip().lower().replace(" ", "_")


def import_cocolofa(split: str) -> list[dict]:
    path = RAW / "cocolofa" / f"{split}.json"
    articles = json.loads(path.read_text(encoding="utf-8"))

    records = []
    for article in articles:
        article_id = article["id"]
        for comment in article.get("comments", []):
            cid = comment["id"]
            text = str(comment["comment"]).strip()
            label = _cocolofa_label(comment["fallacy"])
            records.append({
                "id": f"cocolofa_{split}_{cid}",
                "source": "cocolofa",
                "text_raw": text,
                "text_clean": "",
                "label_fine": label,
                "label_coarse": "",
                "meta": {
                    "article_id": article_id,
                    "comment_id": cid,
                    "respond_to": comment.get("respond_to", ""),
                    "worker_id": comment.get("worker_id"),
                    "original_split": split,
                },
            })
    return records


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    for split in ("train", "dev", "test"):
        recs = import_logic("edu", split)
        write_jsonl(recs, OUT / f"logic_edu_{split}.jsonl")

    for split in ("train", "dev", "test"):
        recs = import_logic("climate", split)
        write_jsonl(recs, OUT / f"logic_climate_{split}.jsonl")

    for split in ("train", "dev", "test"):
        recs = import_cocolofa(split)
        write_jsonl(recs, OUT / f"cocolofa_{split}.jsonl")

    print("\nstage 1 complete. files in:", OUT)


if __name__ == "__main__":
    main()
