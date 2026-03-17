# Multi-Subject-Personalization-Evaluation-P13N-Workshop-
Study whether subject-driven diffusion models can compose many identities under interaction and occlusion.

## Structure
- XVerse-main, MOSAIC-main, PSR-main: model repositories
- val_dataset: subject images and prompts
- scripts/: generation and evaluation scripts
- results/{model}/{prompt_id}/{seed}.png + meta.json: generation outputs
- eval_outputs: evaluation outputs

## Quick Start

Generate:

```bash
bash start.sh gen --jobs jobs.jsonl --psr-runner psr_infer.py
```

Evaluate:

```bash
bash start.sh eval --models xverse,mosaic,psr
```

Generate then evaluate:

```bash
bash start.sh all --jobs jobs.jsonl --psr-runner psr_infer.py
```

Run with default benchmark from val_dataset/prompts_50.txt:

```bash
bash start.sh all
```

Cross-model summary:

```
eval_outputs/compare_summary.csv
```

## Input Format

jobs.jsonl uses one JSON object per line:

```json
{"prompt_id":"001","prompt":"ENT1 hugging ENT2","seed":42,"subjects":[{"image":"val_dataset/dog.jpg","name":"dog","caption":"a dog","idip":true},{"image":"val_dataset/panda.jpg","name":"panda","caption":"a panda","idip":true}]}
```

Fields:
- prompt_id: string or number
- prompt: text prompt
- seed: integer
- subjects: list of subject objects with image, name, caption, idip

## PSR Runner

The default PSR runner is provided at PSR-main/psr_infer.py and is invoked by:

```bash
--psr-runner psr_infer.py
```

It accepts the same input format as other models and uses a configurable model_id for generation.

## Requirements and Setup

General:
- GPU recommended
- Python and pip available in environment
- Current environment: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

XVerse:
- Checkpoints need to be downloaded:

```bash
cd XVerse-main/checkpoints
bash ./download_ckpts.sh
cd ..
```

- Download face recognition model and place into checkpoints:
  - model_ir_se50.pth from InsightFace_Pytorch

- Export runtime environment variables:

```bash
export FLORENCE2_MODEL_PATH="./XVerse-main/checkpoints/Florence-2-large"
export SAM2_MODEL_PATH="./XVerse-main/checkpoints/sam2.1_hiera_large.pt"
export FACE_ID_MODEL_PATH="./XVerse-main/checkpoints/model_ir_se50.pth"
export CLIP_MODEL_PATH="./XVerse-main/checkpoints/clip-vit-large-patch14"
export FLUX_MODEL_PATH="./XVerse-main/checkpoints/FLUX.1-dev"
export DPG_VQA_MODEL_PATH="./XVerse-main/checkpoints/mplug_visual-question-answering_coco_large_en"
export DINO_MODEL_PATH="./XVerse-main/checkpoints/dino-vits16"
```

MOSAIC:
- Weights are loaded from HuggingFace at runtime by default
- Requires internet access on first run

PSR:
- PSR runner provided at PSR-main/psr_infer.py

HuggingFace access:
- If model weights are public, no token is required
- If a repo is gated or requires license acceptance, you must login

```bash
huggingface-cli login
```

Or set a token:

```bash
export HF_TOKEN=your_token
```

If any of the above requires manual changes, keep them in this README and do not hard-code them in scripts.

## Dependency Isolation

XVerse, MOSAIC, and PSR have different dependency constraints. The start.sh script installs each model into a separate venv by default and runs models sequentially to avoid GPU memory pressure.

Use the shared environment if you want:

```bash
bash start.sh --no-venv --models xverse,mosaic
```

## Generation Interface

Each run script supports single prompt or batch jobs.

Single prompt:

```bash
python run_xverse.py --prompt "A dog and a panda" --subjects val_dataset/dog.jpg val_dataset/panda.jpg
python run_mosaic.py --prompt "A dog and a panda" --subjects val_dataset/dog.jpg val_dataset/panda.jpg
python run_psr.py --prompt "A dog and a panda" --subjects val_dataset/dog.jpg val_dataset/panda.jpg --runner your_psr_runner.py
```

Batch jobs:

```bash
python run_xverse.py --jobs jobs.jsonl
python run_mosaic.py --jobs jobs.jsonl
python run_psr.py --jobs jobs.jsonl --runner your_psr_runner.py
```

Output format:

```
results/{model}/{prompt_id}/{seed}.png
results/{model}/{prompt_id}/meta.json
```

meta.json contains prompt, subjects, seed, output_path, and errors if any.

## Evaluation

Run evaluation for one or multiple models:

```bash
python eval.py --models xverse
python eval.py --models xverse,mosaic,psr
```

Outputs:

```
eval_outputs/{model}_results.json
eval_outputs/{model}_summary.json
eval_outputs/{model}_summary.csv
```
