import argparse
import json
import math
import os
from pathlib import Path
import time

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import torchvision.transforms as T


def load_meta_records(results_root, model_name):
    # Fix for path joining issue
    base_root = Path(results_root)
    # If the user passed just "." or "" it might resolve differently
    model_root = base_root / model_name
    
    print(f"DEBUG: Looking for meta.json files in {model_root.resolve()}")
    if not model_root.exists():
        print(f"DEBUG: Directory {model_root.resolve()} does not exist!")
        return []
        
    records = []
    found_count = 0
    for meta_path in model_root.rglob("meta.json"):
        found_count += 1
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"DEBUG: Error reading {meta_path}: {e}")
            continue
            
        if isinstance(data, dict):
            data = [data]
            
        for rec in data:
            rec["_meta_path"] = str(meta_path)
            records.append(rec)
            
    print(f"DEBUG: Found {found_count} meta.json files containing {len(records)} records")
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
        if not isinstance(feats, torch.Tensor):
            if hasattr(feats, "text_embeds"):
                feats = feats.text_embeds
            else:
                feats = feats.pooler_output
    return normalize(feats)


def encode_image(processor, model, device, image):
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        feats = model.get_image_features(**inputs)
        if not isinstance(feats, torch.Tensor):
            if hasattr(feats, "image_embeds"):
                feats = feats.image_embeds
            else:
                feats = feats.pooler_output
    return normalize(feats)


def encode_dino_image(model, device, image):
    # DINO requires standard ImageNet normalization
    transform = T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    img_t = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = model(img_t)
    return normalize(feats)


def cosine(a, b):
    return float((a * b).sum(dim=-1).mean().item())


def get_scene_type(prompt_id_str):
    """
    Inference scene type from prompt_id.
    1-5, 16-20, 31-35, 46-50, 61-65: neutral (No interaction)
    6-10, 21-25, 36-40, 51-55, 66-70: occlusion
    11-15, 26-30, 41-45, 56-60, 71-75: interaction
    """
    try:
        pid = int(prompt_id_str)
        mod = (pid - 1) % 15
        if mod < 5:
            return "neutral"
        elif mod < 10:
            return "occlusion"
        else:
            return "interaction"
    except (ValueError, TypeError):
        return "unknown"


def evaluate_record(processor, model, dino_model, device, rec, thresholds):
    prompt = rec.get("prompt")
    output_path = rec.get("output_path")
    prompt_id = rec.get("prompt_id") or rec.get("id") or rec.get("index")
    subjects = rec.get("subjects") or []
    if not prompt or not output_path or not subjects:
        return None
    gen_img = load_image(output_path)
    if gen_img is None:
        return None
    gen_feat = encode_image(processor, model, device, gen_img)
    text_feat = encode_text(processor, model, device, prompt)
    clip_score = cosine(gen_feat, text_feat)
    
    if dino_model is not None:
        gen_dino_feat = encode_dino_image(dino_model, device, gen_img)
    else:
        gen_dino_feat = None

    subject_feats = []
    subject_dino_feats = []
    for s in subjects:
        img = load_image(s.get("image"))
        if img is None:
            continue
        subject_feats.append(encode_image(processor, model, device, img))
        if dino_model is not None:
            subject_dino_feats.append(encode_dino_image(dino_model, device, img))
            
    if not subject_feats:
        nido = None
        nido_dino = None
        scr = {t: None for t in thresholds}
    else:
        sims = [cosine(gen_feat, sf) for sf in subject_feats]
        nido = float(sum(sims) / len(sims))
        
        if dino_model is not None and subject_dino_feats:
            dino_sims = [cosine(gen_dino_feat, sf) for sf in subject_dino_feats]
            nido_dino = float(sum(dino_sims) / len(dino_sims))
        else:
            nido_dino = None
            
        scr = {}
        for t in thresholds:
            collapsed = sum(1 for s in sims if s < t)
            scr[t] = collapsed / len(sims)

    scene_type = get_scene_type(prompt_id)

    result = {
        "prompt_id": prompt_id,
        "seed": rec.get("seed"),
        "subject_count": len(subjects),
        "scene_type": scene_type,
        "clip": clip_score,
        "nido": nido,
        "nido_dino": nido_dino,
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
    # Base aggregation by subject count
    by_level = {}
    # Aggregation by subject count + scene type
    by_level_scene = {}
    
    for r in results:
        lvl = r["subject_count"]
        scene = r["scene_type"]
        by_level.setdefault(lvl, []).append(r)
        
        scene_key = f"{lvl}_{scene}"
        by_level_scene.setdefault(scene_key, []).append(r)
        
    summary = []
    
    # Process overall subject count
    for lvl in sorted(by_level.keys()):
        items = by_level[lvl]
        clip_mean = mean([x["clip"] for x in items])
        nido_mean = mean([x["nido"] for x in items])
        nido_dino_mean = mean([x.get("nido_dino") for x in items if x.get("nido_dino") is not None])
        scr_mean = {}
        for t in thresholds:
            scr_mean[str(t)] = mean([x["scr"].get(t) for x in items if x["scr"].get(t) is not None])
        summary.append(
            {
                "subject_count": lvl,
                "scene_type": "all",
                "clip": clip_mean,
                "nido": nido_mean,
                "nido_dino": nido_dino_mean,
                "scr": scr_mean,
                "count": len(items),
            }
        )
        
    # Process subject count + scene type breakdown
    for scene_key in sorted(by_level_scene.keys()):
        items = by_level_scene[scene_key]
        lvl = items[0]["subject_count"]
        scene = items[0]["scene_type"]
        
        clip_mean = mean([x["clip"] for x in items])
        nido_mean = mean([x["nido"] for x in items])
        nido_dino_mean = mean([x.get("nido_dino") for x in items if x.get("nido_dino") is not None])
        scr_mean = {}
        for t in thresholds:
            scr_mean[str(t)] = mean([x["scr"].get(t) for x in items if x["scr"].get(t) is not None])
        summary.append(
            {
                "subject_count": lvl,
                "scene_type": scene,
                "clip": clip_mean,
                "nido": nido_mean,
                "nido_dino": nido_dino_mean,
                "scr": scr_mean,
                "count": len(items),
            }
        )
        
    return summary


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path, rows, thresholds):
    header = ["subject_count", "scene_type", "clip", "nido", "nido_dino"] + [f"scr@{t}" for t in thresholds] + ["count"]
    lines = [",".join(header)]
    for r in rows:
        line = [
            str(r["subject_count"]),
            r["scene_type"],
            "" if r["clip"] is None else f"{r['clip']:.6f}",
            "" if r["nido"] is None else f"{r['nido']:.6f}",
            "" if r.get("nido_dino") is None else f"{r['nido_dino']:.6f}",
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
    parser.add_argument("--results_root", type=str, default="./results")
    parser.add_argument("--out_dir", type=str, default="./eval_outputs")
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
    
    # Load DINO model
    try:
        print("Loading DINO model...")
        dino_model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=True).to(device)
        dino_model.eval()
    except Exception as e:
        print(f"Warning: Failed to load DINO model. DINO evaluation will be skipped. Error: {e}")
        dino_model = None

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    for model_name in models:
        print(f"Evaluating model: {model_name}")
        records = load_meta_records(args.results_root, model_name)
        results = []
        for rec in records[: args.max_records or None]:
            r = evaluate_record(processor, model, dino_model, device, rec, thresholds)
            if r is not None:
                results.append(r)
        if not results:
            print(f"  No records found for {model_name}")
            continue
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
