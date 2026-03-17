import argparse
import json
import math
from pathlib import Path
import time

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def load_meta_records(results_root, model_name):
    model_root = Path(results_root) / model_name
    if not model_root.exists():
        return []
    records = []
    for meta_path in model_root.rglob("meta.json"):
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict):
            data = [data]
        for rec in data:
            rec["_meta_path"] = str(meta_path)
            records.append(rec)
    return records


def load_image(path):
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return Image.open(p).convert("RGB")


def normalize(x):
    return x / x.norm(dim=-1, keepdim=True)


def encode_text(processor, model, device, text):
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        feats = model.get_text_features(**inputs)
    return normalize(feats)


def encode_image(processor, model, device, image):
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        feats = model.get_image_features(**inputs)
    return normalize(feats)


def cosine(a, b):
    return float((a * b).sum(dim=-1).mean().item())


def evaluate_record(processor, model, device, rec, thresholds):
    prompt = rec.get("prompt")
    output_path = rec.get("output_path")
    subjects = rec.get("subjects") or []
    if not prompt or not output_path or not subjects:
        return None
    gen_img = load_image(output_path)
    if gen_img is None:
        return None
    gen_feat = encode_image(processor, model, device, gen_img)
    text_feat = encode_text(processor, model, device, prompt)
    clip_score = cosine(gen_feat, text_feat)

    subject_feats = []
    for s in subjects:
        img = load_image(s.get("image"))
        if img is None:
            continue
        subject_feats.append(encode_image(processor, model, device, img))
    if not subject_feats:
        nido = None
        scr = {t: None for t in thresholds}
    else:
        sims = [cosine(gen_feat, sf) for sf in subject_feats]
        nido = float(sum(sims) / len(sims))
        scr = {}
        for t in thresholds:
            collapsed = sum(1 for s in sims if s < t)
            scr[t] = collapsed / len(sims)

    result = {
        "prompt_id": rec.get("prompt_id") or rec.get("id") or rec.get("index"),
        "seed": rec.get("seed"),
        "subject_count": len(subjects),
        "clip": clip_score,
        "nido": nido,
        "scr": scr,
        "prompt": prompt,
        "output_path": output_path,
    }
    return result


def mean(values):
    vals = [v for v in values if v is not None and not math.isnan(v)]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def aggregate(results, thresholds):
    by_level = {}
    for r in results:
        lvl = r["subject_count"]
        by_level.setdefault(lvl, []).append(r)
    summary = []
    for lvl in sorted(by_level.keys()):
        items = by_level[lvl]
        clip_mean = mean([x["clip"] for x in items])
        nido_mean = mean([x["nido"] for x in items])
        scr_mean = {}
        for t in thresholds:
            scr_mean[str(t)] = mean([x["scr"].get(t) for x in items if x["scr"].get(t) is not None])
        summary.append(
            {
                "subject_count": lvl,
                "clip": clip_mean,
                "nido": nido_mean,
                "scr": scr_mean,
                "count": len(items),
            }
        )
    return summary


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path, rows, thresholds):
    header = ["subject_count", "clip", "nido"] + [f"scr@{t}" for t in thresholds] + ["count"]
    lines = [",".join(header)]
    for r in rows:
        line = [
            str(r["subject_count"]),
            "" if r["clip"] is None else f"{r['clip']:.6f}",
            "" if r["nido"] is None else f"{r['nido']:.6f}",
        ]
        for t in thresholds:
            val = r["scr"].get(str(t))
            line.append("" if val is None else f"{val:.6f}")
        line.append(str(r["count"]))
        lines.append(",".join(line))
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, default="xverse,mosaic,psr")
    parser.add_argument("--results_root", type=str, default="results")
    parser.add_argument("--out_dir", type=str, default="eval_outputs")
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--thresholds", type=str, default="0.2,0.25,0.3")
    parser.add_argument("--max_records", type=int, default=0)
    args = parser.parse_args()

    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    processor = CLIPProcessor.from_pretrained(args.clip_model)
    model = CLIPModel.from_pretrained(args.clip_model).to(device)
    model.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    for model_name in models:
        records = load_meta_records(args.results_root, model_name)
        results = []
        for rec in records[: args.max_records or None]:
            r = evaluate_record(processor, model, device, rec, thresholds)
            if r is not None:
                results.append(r)
        summary = aggregate(results, thresholds)
        payload = {
            "model": model_name,
            "time": int(time.time()),
            "results": results,
            "summary": summary,
            "thresholds": thresholds,
        }
        write_json(out_dir / f"{model_name}_results.json", payload)
        write_json(out_dir / f"{model_name}_summary.json", summary)
        write_csv(out_dir / f"{model_name}_summary.csv", summary, thresholds)


if __name__ == "__main__":
    main()
