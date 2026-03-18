import argparse
import json
import re
from pathlib import Path


def normalize_name(name):
    name = name.lower().strip()
    name = name.replace("(", "").replace(")", "")
    name = name.replace("-", " ").replace("_", " ")
    for prefix in ["a ", "an ", "the "]:
        if name.startswith(prefix):
            name = name[len(prefix):]
    name = re.sub(r"[^a-z0-9]+", "", name)
    return name


def build_image_map(images_dir):
    images_dir = Path(images_dir)
    image_map = {}
    for p in images_dir.iterdir():
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        key = normalize_name(p.stem)
        image_map[key] = p
    return image_map


def resolve_subject_image(subject, image_map, alias_map):
    if subject in alias_map:
        return alias_map[subject]
    key = normalize_name(subject)
    return image_map.get(key)


def parse_prompts(prompts_path):
    lines = [l.rstrip("\n") for l in Path(prompts_path).read_text(encoding="utf-8").splitlines()]
    items = []
    i = 0
    current_type = None
    while i < len(lines):
        line = lines[i].strip()
        if line.lower() == "no interaction":
            current_type = "no_interaction"
            i += 1
            continue
        if line.lower() == "occlusion":
            current_type = "occlusion"
            i += 1
            continue
        if line.lower() == "interaction":
            current_type = "interaction"
            i += 1
            continue
        m = re.match(r"#(\d+)\s+\[(.+?)\]", line)
        if m:
            prompt_id = m.group(1).zfill(2)
            subjects_raw = m.group(2)
            subjects = [s.strip() for s in subjects_raw.split("·")]
            j = i + 1
            prompt = None
            while j < len(lines):
                t = lines[j].strip()
                if t:
                    prompt = t
                    break
                j += 1
            if prompt:
                items.append((prompt_id, prompt, subjects, current_type))
            i = j + 1
            continue
        i += 1
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=str, required=True)
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--out", type=str, default="jobs.jsonl")
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_path = Path(args.out)
    if out_path.exists() and not args.overwrite:
        return

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    image_map = build_image_map(args.images_dir)
    alias_map = {
        "motor(cycle)": image_map.get(normalize_name("motorcycle")),
        "motorcycle": image_map.get(normalize_name("motorcycle")),
        "stuffed bear": image_map.get(normalize_name("stuff bear")),
        "t-shirt": image_map.get(normalize_name("tshirt")),
        "black woman": image_map.get(normalize_name("black women")),
        "black man": image_map.get(normalize_name("blackman")),
        "middle eastern man": image_map.get(normalize_name("middle eastern man")),
        "western woman": image_map.get(normalize_name("western woman")),
        "asian woman": image_map.get(normalize_name("asian woman")),
        "old man": image_map.get(normalize_name("old man")),
        "vintage camera": image_map.get(normalize_name("vintage camera")),
        "puzzle cube": image_map.get(normalize_name("puzzle cube")),
        "berry bowl": image_map.get(normalize_name("berry bowl")),
        "leather handbag": image_map.get(normalize_name("leather handbag")),
        "siamese cat": image_map.get(normalize_name("siamese cat")),
    }

    items = parse_prompts(args.prompts)
    lines = []
    for prompt_id, prompt, subjects, scene_type in items:
        subject_objs = []
        for s in subjects:
            img_path = resolve_subject_image(s, image_map, alias_map)
            if img_path is None:
                raise ValueError(f"cannot find image for subject: {s}")
            subject_objs.append(
                {
                    "image": str(img_path),
                    "name": s,
                    "caption": s,
                    "idip": True,
                }
            )
        if scene_type:
            labeled_prompt_id = f"{prompt_id}_{scene_type}"
        else:
            labeled_prompt_id = prompt_id
        for seed in seeds:
            job = {
                "prompt_id": labeled_prompt_id,
                "prompt": prompt,
                "seed": seed,
                "scene_type": scene_type,
                "subjects": subject_objs,
            }
            lines.append(json.dumps(job, ensure_ascii=False))

    out_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
