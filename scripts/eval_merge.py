import argparse
import csv
import json
from pathlib import Path


def load_summary(path):
    if not Path(path).exists():
        return []
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, default="xverse,mosaic,psr")
    parser.add_argument("--eval_dir", type=str, default="eval_outputs")
    parser.add_argument("--out", type=str, default="eval_outputs/compare_summary.csv")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    rows = []
    for model in models:
        summary_path = Path(args.eval_dir) / f"{model}_summary.json"
        for item in load_summary(summary_path):
            row = {
                "model": model,
                "subject_count": item.get("subject_count"),
                "clip": item.get("clip"),
                "nido": item.get("nido"),
                "count": item.get("count"),
            }
            scr = item.get("scr", {}) or {}
            for k, v in scr.items():
                row[f"scr@{k}"] = v
            rows.append(row)

    if not rows:
        return

    fieldnames = ["model", "subject_count", "clip", "nido"]
    scr_keys = sorted({k for r in rows for k in r.keys() if k.startswith("scr@")})
    fieldnames += scr_keys + ["count"]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


if __name__ == "__main__":
    main()
