import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


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
    
    # avoid duplicates
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
    seed = int(job.get("seed", args.seed))
    output_path = build_output_path(args.out_root, prompt_id, seed)
    if output_path.exists():
        return {"prompt_id": prompt_id, "seed": seed, "skipped": True, "output_path": str(output_path)}

    images = [s["image"] for s in normalized_subjects]
    names = [s.get("name") or Path(s["image"]).stem for s in normalized_subjects]
    captions = [s.get("caption") or s.get("name") or Path(s["image"]).stem for s in normalized_subjects]

    cmd = [
        sys.executable,
        str(args.runner_path),
        "--prompt",
        prompt,
        "--seed",
        str(seed),
        "--output",
        str(output_path),
        "--subjects",
        *images,
    ]
    if names:
        cmd.extend(["--subject_names", *names])
    if captions:
        cmd.extend(["--subject_captions", *captions])
    env = os.environ.copy()
    subprocess.run(cmd, cwd=args.runner_cwd or None, env=env, check=True)
    return {"prompt_id": prompt_id, "seed": seed, "skipped": False, "output_path": str(output_path)}

def run_jobs_batch(args):
    jobs_path = Path(args.jobs)
    if not jobs_path.is_absolute():
        jobs_path = PROJECT_ROOT / jobs_path
    cmd = [
        sys.executable,
        str(args.runner_path),
        "--jobs",
        str(jobs_path),
        "--out_root",
        args.out_root,
    ]
    if args.continue_on_error:
        cmd += ["--continue_on_error"]
    env = os.environ.copy()
    subprocess.run(cmd, cwd=args.runner_cwd or None, env=env, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=str)
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--prompt_id", type=str, default="single")
    parser.add_argument("--subjects", nargs="*", default=[])
    parser.add_argument("--subject_names", nargs="*")
    parser.add_argument("--subject_captions", nargs="*")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_root", type=str, default="results/psr")
    parser.add_argument("--runner", type=str, required=True)
    parser.add_argument("--runner_cwd", type=str)
    parser.add_argument("--psr_dir", type=str, default="PSR-main")
    parser.add_argument("--repo_url", type=str, default="https://github.com/wang-shulei/PSR")
    parser.add_argument("--continue_on_error", action="store_true")
    args = parser.parse_args()

    ensure_repo(args.psr_dir, args.repo_url)
    runner_path = Path(args.runner)
    if not runner_path.is_absolute():
        psr_dir = Path(args.psr_dir)
        if runner_path.parts and runner_path.parts[0] == psr_dir.name:
            runner_path = (PROJECT_ROOT / runner_path).resolve()
        else:
            runner_path = (psr_dir / runner_path).resolve()
    else:
        runner_path = runner_path.resolve()
    args.runner_path = runner_path
    if args.runner_cwd is None:
        args.runner_cwd = str(Path(args.psr_dir))

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
                    "model": "psr",
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
        subjects = normalize_subjects(args.subjects, args.subject_names, args.subject_captions)
        jobs = [{"prompt_id": args.prompt_id, "prompt": args.prompt, "subjects": subjects, "seed": args.seed}]
    else:
        raise ValueError("provide --jobs or --prompt with --subjects")

    Path(args.out_root).mkdir(parents=True, exist_ok=True)
    for job in jobs:
        try:
            result = run_job(args, job)
            record = {
                "model": "psr",
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
                "model": "psr",
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
