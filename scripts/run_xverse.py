import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def parse_bool_list(values):
    if values is None:
        return None
    parsed = []
    for v in values:
        if isinstance(v, bool):
            parsed.append(v)
            continue
        s = str(v).strip().lower()
        if s in {"1", "true", "t", "yes", "y"}:
            parsed.append(True)
        elif s in {"0", "false", "f", "no", "n"}:
            parsed.append(False)
        else:
            raise ValueError(f"invalid boolean value: {v}")
    return parsed


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

def normalize_subjects(images, names, captions, idips):
    images = list(images or [])
    if not images:
        return []
    if names and len(names) != len(images):
        raise ValueError("subject names length does not match images length")
    if captions and len(captions) != len(images):
        raise ValueError("subject captions length does not match images length")
    if idips and len(idips) != len(images):
        raise ValueError("subject idips length does not match images length")
    subjects = []
    for i, image_path in enumerate(images):
        name = names[i] if names else Path(image_path).stem.replace("_", " ")
        caption = captions[i] if captions else name
        idip = idips[i] if idips else True
        subjects.append({"image": image_path, "name": name, "caption": caption, "idip": idip})
    return subjects


def build_output_path(out_root, prompt_id, seed):
    out_root = Path(out_root)
    if not out_root.is_absolute():
        out_root = PROJECT_ROOT / out_root
    prompt_dir = out_root / str(prompt_id)
    prompt_dir.mkdir(parents=True, exist_ok=True)
    return prompt_dir / f"{seed}.png"


def update_meta(out_root, prompt_id, record):
    out_root = Path(out_root)
    if not out_root.is_absolute():
        out_root = PROJECT_ROOT / out_root
    prompt_dir = out_root / str(prompt_id)
    prompt_dir.mkdir(parents=True, exist_ok=True)
    meta_path = prompt_dir / "meta.json"
    if meta_path.exists():
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            data = [data]
    else:
        data = []
        
    for i, existing in enumerate(data):
        if existing.get("seed") == record.get("seed") and existing.get("model") == record.get("model"):
            data[i] = record
            meta_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            return
            
    data.append(record)
    meta_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def run_job(args, job):
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
    images = [s["image"] for s in normalized_subjects]
    captions = [s.get("caption") or s.get("name") or Path(s["image"]).stem for s in normalized_subjects]
    idips = [bool(s.get("idip", True)) for s in normalized_subjects]
    seed = int(job.get("seed", args.seed))
    output_path = build_output_path(args.out_root, prompt_id, seed)
    if output_path.exists():
        return {"prompt_id": prompt_id, "seed": seed, "skipped": True, "output_path": str(output_path)}

    cmd = [
        sys.executable,
        "inference_single_sample.py",
        "--prompt",
        prompt,
        "--seed",
        str(seed),
        "--cond_size",
        str(args.cond_size),
        "--target_height",
        str(args.height),
        "--target_width",
        str(args.width),
        "--weight_id",
        str(args.weight_id),
        "--weight_ip",
        str(args.weight_ip),
        "--latent_lora_scale",
        str(args.latent_lora_scale),
        "--vae_lora_scale",
        str(args.vae_lora_scale),
        "--vae_skip_iter_s1",
        str(args.vae_skip_iter_s1),
        "--vae_skip_iter_s2",
        str(args.vae_skip_iter_s2),
        "--num_inference_steps",
        str(args.num_inference_steps),
        "--save_path",
        str(output_path),
        "--num_images",
        "1",
        "--images",
        *images,
        "--captions",
        *captions,
        "--idips",
        *["true" if v else "false" for v in idips],
    ]
    if args.use_low_vram:
        cmd += ["--use_low_vram", "True"]
    if args.use_lower_vram:
        cmd += ["--use_lower_vram", "True"]
    if args.dit_quant:
        cmd += ["--dit_quant", args.dit_quant]

    env = os.environ.copy()
    for kv in args.env:
        if "=" not in kv:
            raise ValueError(f"invalid env format: {kv}")
        k, v = kv.split("=", 1)
        env[k] = v

    subprocess.run(cmd, cwd=args.xverse_dir, env=env, check=True)
    return {"prompt_id": prompt_id, "seed": seed, "skipped": False, "output_path": str(output_path)}

def run_jobs_batch(args):
    jobs_path = Path(args.jobs)
    if not jobs_path.is_absolute():
        jobs_path = PROJECT_ROOT / jobs_path
    cmd = [
        sys.executable,
        "inference_single_sample.py",
        "--jobs",
        str(jobs_path),
        "--out_root",
        args.out_root,
        "--cond_size",
        str(args.cond_size),
        "--target_height",
        str(args.height),
        "--target_width",
        str(args.width),
        "--weight_id",
        str(args.weight_id),
        "--weight_ip",
        str(args.weight_ip),
        "--latent_lora_scale",
        str(args.latent_lora_scale),
        "--vae_lora_scale",
        str(args.vae_lora_scale),
        "--vae_skip_iter_s1",
        str(args.vae_skip_iter_s1),
        "--vae_skip_iter_s2",
        str(args.vae_skip_iter_s2),
        "--num_inference_steps",
        str(args.num_inference_steps),
        "--num_images",
        "1",
    ]
    if args.use_low_vram:
        cmd += ["--use_low_vram", "True"]
    if args.use_lower_vram:
        cmd += ["--use_lower_vram", "True"]
    if args.dit_quant:
        cmd += ["--dit_quant", args.dit_quant]
    if args.continue_on_error:
        cmd += ["--continue_on_error"]

    env = os.environ.copy()
    for kv in args.env:
        if "=" not in kv:
            raise ValueError(f"invalid env format: {kv}")
        k, v = kv.split("=", 1)
        env[k] = v

    subprocess.run(cmd, cwd=args.xverse_dir, env=env, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=str)
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--prompt_id", type=str, default="single")
    parser.add_argument("--subjects", nargs="*", default=[])
    parser.add_argument("--subject_names", nargs="*")
    parser.add_argument("--subject_captions", nargs="*")
    parser.add_argument("--subject_idips", nargs="*")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--cond_size", type=int, default=256)
    parser.add_argument("--weight_id", type=float, default=3)
    parser.add_argument("--weight_ip", type=float, default=5)
    parser.add_argument("--latent_lora_scale", type=float, default=0.85)
    parser.add_argument("--vae_lora_scale", type=float, default=1.3)
    parser.add_argument("--vae_skip_iter_s1", type=float, default=0.05)
    parser.add_argument("--vae_skip_iter_s2", type=float, default=0.8)
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--dit_quant", type=str, default="int8-quanto")
    parser.add_argument("--use_low_vram", action="store_true")
    parser.add_argument("--use_lower_vram", action="store_true")
    parser.add_argument("--out_root", type=str, default="results/xverse")
    parser.add_argument("--xverse_dir", type=str, default="XVerse-main")
    parser.add_argument("--repo_url", type=str, default="https://github.com/bytedance/XVerse")
    parser.add_argument("--env", nargs="*", default=[])
    parser.add_argument("--continue_on_error", action="store_true")
    args = parser.parse_args()

    args.subject_idips = parse_bool_list(args.subject_idips)
    ensure_repo(args.xverse_dir, args.repo_url)

    jobs = []
    if args.jobs:
        run_jobs_batch(args)
        jobs = load_jobs(args.jobs)
        for job in jobs:
            prompt_id = job.get("prompt_id") or job.get("id") or job.get("index") or "unknown"
            seed = int(job.get("seed", args.seed))
            output_path = build_output_path(args.out_root, prompt_id, seed)
            if output_path.exists():
                record = {
                    "model": "xverse",
                    "prompt_id": prompt_id,
                    "seed": seed,
                    "prompt": job.get("prompt"),
                    "subjects": job.get("subjects"),
                    "output_path": str(output_path),
                    "skipped": False,
                    "time": int(time.time()),
                }
                update_meta(args.out_root, prompt_id, record)
        return
    elif args.prompt and args.subjects:
        subjects = normalize_subjects(args.subjects, args.subject_names, args.subject_captions, args.subject_idips)
        jobs = [{"prompt_id": args.prompt_id, "prompt": args.prompt, "subjects": subjects, "seed": args.seed}]
    else:
        raise ValueError("provide --jobs or --prompt with --subjects")

    Path(args.out_root).mkdir(parents=True, exist_ok=True)
    for job in jobs:
        try:
            result = run_job(args, job)
            record = {
                "model": "xverse",
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
                "model": "xverse",
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


if __name__ == "__main__":
    main()
