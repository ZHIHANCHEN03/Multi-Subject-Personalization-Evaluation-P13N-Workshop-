import argparse
import json
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

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

def resolve_image_path(path):
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str((PROJECT_ROOT / p).resolve())

def build_subject_captions(subjects, subject_names, subject_captions):
    if subject_captions:
        return list(subject_captions)
    if subject_names:
        return list(subject_names)
    return [Path(p).stem.replace("_", " ") for p in subjects]


def inject_subjects(prompt, captions):
    for i, cap in enumerate(captions, start=1):
        token = f"ENT{i}"
        if token in prompt:
            prompt = prompt.replace(token, cap)
    if any(f"ENT{i}" in prompt for i in range(1, len(captions) + 1)):
        return prompt
    if captions:
        return f"{prompt}. Subjects: {', '.join(captions)}."
    return prompt


def load_pipeline(model_id, device, dtype):
    try:
        from diffusers import FluxPipeline
        return FluxPipeline.from_pretrained(model_id, torch_dtype=dtype).to(device)
    except Exception:
        pass
    try:
        from diffusers import StableDiffusionXLPipeline
        return StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=dtype).to(device)
    except Exception:
        from diffusers import StableDiffusionPipeline
        return StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype).to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=str)
    parser.add_argument("--out_root", type=str, default="results/psr")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str)
    parser.add_argument("--subjects", nargs="*", default=[])
    parser.add_argument("--subject_names", nargs="*")
    parser.add_argument("--subject_captions", nargs="*")
    parser.add_argument("--model_id", type=str, default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    pipe = load_pipeline(args.model_id, device, dtype)

    if args.jobs:
        jobs = load_jobs(args.jobs)
        out_root = Path(args.out_root)
        if not out_root.is_absolute():
            out_root = PROJECT_ROOT / out_root
        out_root.mkdir(parents=True, exist_ok=True)
        for job in jobs:
            try:
                prompt_id = job.get("prompt_id") or job.get("id") or job.get("index") or "unknown"
                prompt = job.get("prompt")
                subjects = job.get("subjects") or []
                if not prompt or not subjects:
                    raise ValueError(f"missing prompt or subjects for job {prompt_id}")
                seed = int(job.get("seed", args.seed))
                output_path = out_root / str(prompt_id) / f"{seed}.png"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                if output_path.exists():
                    continue
                images = [resolve_image_path(s["image"]) for s in subjects]
                names = [s.get("name") or Path(s["image"]).stem for s in subjects]
                captions = [s.get("caption") or s.get("name") or Path(s["image"]).stem for s in subjects]
                captions = build_subject_captions(images, names, captions)
                final_prompt = inject_subjects(prompt, captions)
                generator = torch.Generator(device).manual_seed(seed)
                image = pipe(
                    prompt=final_prompt,
                    height=args.height,
                    width=args.width,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance,
                    generator=generator,
                ).images[0]
                image.save(output_path)
            except Exception:
                if not args.continue_on_error:
                    raise
        return

    if not args.prompt:
        raise ValueError("prompt is required when --jobs is not provided")
    if not args.output:
        raise ValueError("output is required when --jobs is not provided")
    captions = build_subject_captions(args.subjects, args.subject_names, args.subject_captions)
    prompt = inject_subjects(args.prompt, captions)
    generator = torch.Generator(device).manual_seed(args.seed)
    image = pipe(
        prompt=prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        generator=generator,
    ).images[0]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)


if __name__ == "__main__":
    main()
