import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
from PIL import Image
from diffusers import FluxPipeline


def load_jobs(job_path):
    text = Path(job_path).read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        return json.loads(text)
    jobs = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        jobs.append(json.loads(line))
    return jobs


def ensure_repo(repo_dir, repo_url):
    repo_path = Path(repo_dir)
    if repo_path.exists():
        return
    subprocess.run(["git", "clone", "--depth", "1", repo_url, str(repo_path)], check=True)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def resolve_image_path(path):
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str((PROJECT_ROOT / p).resolve())

def normalize_subjects(images, names, captions):
    images = list(images or [])
    if not images:
        return []
    if names and len(names) != len(images):
        raise ValueError("subject names length does not match images length")
    if captions and len(captions) != len(images):
        raise ValueError("subject captions length does not match images length")
    subjects = []
    for i, image_path in enumerate(images):
        name = names[i] if names else Path(image_path).stem.replace("_", " ")
        caption = captions[i] if captions else name
        subjects.append({"image": image_path, "name": name, "caption": caption})
    return subjects


def build_output_path(out_root, prompt_id, seed):
    prompt_dir = Path(out_root) / str(prompt_id)
    prompt_dir.mkdir(parents=True, exist_ok=True)
    return prompt_dir / f"{seed}.png"


def update_meta(out_root, prompt_id, record):
    prompt_dir = Path(out_root) / str(prompt_id)
    meta_path = prompt_dir / "meta.json"
    if meta_path.exists():
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            data = [data]
    else:
        data = []
    data.append(record)
    meta_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def init_model(base_model, lora_repo, weight_name, adapter_name, device, dtype):
    pipe = FluxPipeline.from_pretrained(base_model, torch_dtype=dtype).to(device)
    pipe.load_lora_weights(lora_repo, weight_name=weight_name, adapter_name=adapter_name)
    pipe.set_adapters([adapter_name], [1])
    return pipe


def run_job(args, job, pipe, Condition, generate, process_image):
    prompt_id = job.get("prompt_id") or job.get("id") or job.get("index") or "unknown"
    prompt = job.get("prompt")
    if not prompt:
        raise ValueError(f"missing prompt for job {prompt_id}")
    subjects = job.get("subjects") or []
    if not subjects:
        raise ValueError(f"missing subjects for job {prompt_id}")
    normalized_subjects = []
    for s in subjects:
        img_path = resolve_image_path(s["image"])
        normalized = dict(s)
        normalized["image"] = img_path
        normalized_subjects.append(normalized)
    seed = int(job.get("seed", args.seed))
    output_path = build_output_path(args.out_root, prompt_id, seed)
    if output_path.exists():
        return {"prompt_id": prompt_id, "seed": seed, "skipped": True, "output_path": str(output_path)}

    ref_imgs = []
    for s in normalized_subjects:
        img_path = s["image"]
        if os.path.exists(img_path):
            pil_img = process_image(img_path, target_size=args.ref_size, pad_color=(255, 255, 255), scale=args.ref_scale)
        else:
            pil_img = Image.new("RGB", (args.ref_size, args.ref_size), (0, 0, 0))
        ref_imgs.append(pil_img)

    position_deltas = []
    for i in range(len(ref_imgs)):
        position_deltas.append([0, -(args.ref_size * (i + 1)) // 16])

    conditions = [Condition(appearance, "subject", position_deltas[i]) for i, appearance in enumerate(ref_imgs)]

    with torch.no_grad():
        result = generate(
            pipe,
            prompt=prompt,
            conditions=conditions,
            num_inference_steps=args.num_inference_steps,
            num_images_per_prompt=1,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
            generator=torch.Generator("cuda").manual_seed(seed),
        )[0]
    if len(result) == 0:
        raise ValueError(f"empty result for {prompt_id}")
    result_img = result[0]
    result_img.save(str(output_path))
    return {"prompt_id": prompt_id, "seed": seed, "skipped": False, "output_path": str(output_path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=str)
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--prompt_id", type=str, default="single")
    parser.add_argument("--subjects", nargs="*", default=[])
    parser.add_argument("--subject_names", nargs="*")
    parser.add_argument("--subject_captions", nargs="*")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--ref_size", type=int, default=512)
    parser.add_argument("--ref_scale", type=float, default=0.9)
    parser.add_argument("--base_model", type=str, default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--lora_repo", type=str, default="ByteDance-FanQie/MOSAIC")
    parser.add_argument("--weight_name", type=str, default="subject_512.safetensors")
    parser.add_argument("--adapter_name", type=str, default="subject")
    parser.add_argument("--out_root", type=str, default="results/mosaic")
    parser.add_argument("--mosaic_dir", type=str, default="MOSAIC-main")
    parser.add_argument("--repo_url", type=str, default="https://github.com/bytedance-fanqie-ai/MOSAIC")
    parser.add_argument("--continue_on_error", action="store_true")
    args = parser.parse_args()

    ensure_repo(args.mosaic_dir, args.repo_url)
    if args.mosaic_dir not in sys.path:
        sys.path.insert(0, args.mosaic_dir)
    from src.flux_omini import Condition, generate
    from utils import process_image

    jobs = []
    if args.jobs:
        jobs = load_jobs(args.jobs)
    elif args.prompt and args.subjects:
        subjects = normalize_subjects(args.subjects, args.subject_names, args.subject_captions)
        jobs = [{"prompt_id": args.prompt_id, "prompt": args.prompt, "subjects": subjects, "seed": args.seed}]
    else:
        raise ValueError("provide --jobs or --prompt with --subjects")

    Path(args.out_root).mkdir(parents=True, exist_ok=True)
    device = "cuda"
    dtype = torch.bfloat16
    pipe = init_model(args.base_model, args.lora_repo, args.weight_name, args.adapter_name, device, dtype)

    for job in jobs:
        try:
            result = run_job(args, job, pipe, Condition, generate, process_image)
            record = {
                "model": "mosaic",
                "prompt_id": result["prompt_id"],
                "seed": result["seed"],
                "prompt": job.get("prompt"),
                "subjects": job.get("subjects"),
                "output_path": result["output_path"],
                "skipped": result["skipped"],
                "time": int(time.time()),
            }
            update_meta(args.out_root, result["prompt_id"], record)
        except Exception as e:
            record = {
                "model": "mosaic",
                "prompt_id": job.get("prompt_id"),
                "seed": job.get("seed", args.seed),
                "prompt": job.get("prompt"),
                "subjects": job.get("subjects"),
                "error": str(e),
                "time": int(time.time()),
            }
            update_meta(args.out_root, job.get("prompt_id", "unknown"), record)
            if not args.continue_on_error:
                raise

    del pipe
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
