import argparse
from pathlib import Path
import torch


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
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, required=True)
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

    captions = build_subject_captions(args.subjects, args.subject_names, args.subject_captions)
    prompt = inject_subjects(args.prompt, captions)

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    pipe = load_pipeline(args.model_id, device, dtype)
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
