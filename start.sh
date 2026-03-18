#!/usr/bin/env bash
set -euo pipefail

MODELS="xverse,mosaic,psr"
INSTALL=1
USE_VENV=1
JOBS=""
PSR_RUNNER="psr_infer.py"
PSR_RUNNER_CWD=""

MODE="${1:-}"
if [[ -n "$MODE" ]]; then
  shift 1
fi

usage() {
  cat <<'EOF'
Usage:
  bash start.sh gen  --jobs jobs.jsonl
  bash start.sh eval --models xverse,mosaic,psr
  bash start.sh all  --jobs jobs.jsonl
  bash start.sh all

Options:
  --models         Comma-separated model list (default: xverse,mosaic,psr)
  --jobs           Job file (jsonl or json array)
  --psr-runner     PSR inference entry script
  --psr-runner-cwd Runner working directory
  --no-install     Skip installing requirements
  --no-venv        Use current environment
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models)
      MODELS="$2"
      shift 2
      ;;
    --no-install)
      INSTALL=0
      shift 1
      ;;
    --no-venv)
      USE_VENV=0
      shift 1
      ;;
    --jobs)
      JOBS="$2"
      shift 2
      ;;
    --psr-runner)
      PSR_RUNNER="$2"
      shift 2
      ;;
    --psr-runner-cwd)
      PSR_RUNNER_CWD="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      shift 1
      ;;
  esac
done

model_dir() {
  case "$1" in
    xverse) echo "XVerse-main" ;;
    mosaic) echo "MOSAIC-main" ;;
    psr) echo "PSR-main" ;;
    *) echo "" ;;
  esac
}

model_python() {
  local model="$1"
  if [[ "$USE_VENV" -eq 1 ]]; then
    local vdir=".venv_shared"
    if [[ ! -d "$vdir" ]]; then
      python -m venv "$vdir"
    fi
    echo "$PWD/$vdir/bin/python"
  else
    echo "python"
  fi
}

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

ensure_xverse_checkpoints() {
  local repo_root
  repo_root="$(pwd -P)"
  local ckpt_dir="$repo_root/XVerse-main/checkpoints"
  local missing=0
  if [[ ! -f "$ckpt_dir/sam2.1_hiera_large.pt" ]]; then missing=1; fi
  if [[ ! -d "$ckpt_dir/FLUX.1-dev" ]]; then missing=1; fi
  if [[ ! -d "$ckpt_dir/Florence-2-large" ]]; then missing=1; fi
  if [[ ! -d "$ckpt_dir/clip-vit-large-patch14" ]]; then missing=1; fi
  if [[ ! -d "$ckpt_dir/dino-vits16" ]]; then missing=1; fi
  if [[ ! -d "$ckpt_dir/mplug_visual-question-answering_coco_large_en" ]]; then missing=1; fi
  if [[ ! -d "$ckpt_dir/XVerse" ]]; then missing=1; fi
  if [[ "$missing" -eq 1 ]]; then
    log "downloading XVerse checkpoints..."
    if ! command -v huggingface-cli &> /dev/null; then
      log "installing huggingface-cli..."
      local py
      py=$(model_python "xverse")
      "$py" -m pip install -U "huggingface_hub[cli]<1.0" || { echo "failed to install huggingface-cli"; exit 1; }
    fi
    if ! command -v huggingface-cli &> /dev/null; then
      export HUGGINGFACE_CLI="$py -m huggingface_hub.commands.huggingface_cli"
    fi
    export HUGGINGFACE_PY="$py"
    (cd "$ckpt_dir" && bash ./download_ckpts.sh)
    unset HUGGINGFACE_PY
  fi
  if [[ "${SKIP_FACE_MODEL:-1}" -ne 1 ]]; then
    if [[ ! -f "$ckpt_dir/model_ir_se50.pth" ]]; then
      log "downloading model_ir_se50.pth..."
      local face_url="https://huggingface.co/lithiumice/insightface/resolve/main/InsightFace_Pytorch%2Bmodel_ir_se50.pth"
      local face_url_alt="https://www.dropbox.com/s/kzo52d9neybjxsb/model_ir_se50.pth?dl=1"
      if command -v wget &> /dev/null; then
        if [[ -n "${HF_TOKEN:-}" ]]; then
          wget --header="Authorization: Bearer $HF_TOKEN" -O "$ckpt_dir/model_ir_se50.pth" "$face_url" || wget -O "$ckpt_dir/model_ir_se50.pth" "$face_url_alt" || { echo "failed to download face model"; exit 1; }
        else
          wget -O "$ckpt_dir/model_ir_se50.pth" "$face_url" || wget -O "$ckpt_dir/model_ir_se50.pth" "$face_url_alt" || { echo "failed to download face model"; exit 1; }
        fi
      elif command -v curl &> /dev/null; then
        if [[ -n "${HF_TOKEN:-}" ]]; then
          curl -L -H "Authorization: Bearer $HF_TOKEN" "$face_url" -o "$ckpt_dir/model_ir_se50.pth" || curl -L "$face_url_alt" -o "$ckpt_dir/model_ir_se50.pth" || { echo "failed to download face model"; exit 1; }
        else
          curl -L "$face_url" -o "$ckpt_dir/model_ir_se50.pth" || curl -L "$face_url_alt" -o "$ckpt_dir/model_ir_se50.pth" || { echo "failed to download face model"; exit 1; }
        fi
      else
        echo "Please install wget or curl to download model_ir_se50.pth"
        exit 1
      fi
    fi
  else
    log "skipping model_ir_se50.pth download"
  fi
  if [[ -z "${FLORENCE2_MODEL_PATH:-}" ]]; then export FLORENCE2_MODEL_PATH="$ckpt_dir/Florence-2-large"; fi
  if [[ -z "${SAM2_MODEL_PATH:-}" ]]; then export SAM2_MODEL_PATH="$ckpt_dir/sam2.1_hiera_large.pt"; fi
  if [[ -z "${FACE_ID_MODEL_PATH:-}" && -f "$ckpt_dir/model_ir_se50.pth" ]]; then export FACE_ID_MODEL_PATH="$ckpt_dir/model_ir_se50.pth"; fi
  if [[ -z "${CLIP_MODEL_PATH:-}" ]]; then export CLIP_MODEL_PATH="$ckpt_dir/clip-vit-large-patch14"; fi
  if [[ -z "${FLUX_MODEL_PATH:-}" ]]; then export FLUX_MODEL_PATH="$ckpt_dir/FLUX.1-dev"; fi
  if [[ -z "${DPG_VQA_MODEL_PATH:-}" ]]; then export DPG_VQA_MODEL_PATH="$ckpt_dir/mplug_visual-question-answering_coco_large_en"; fi
  if [[ -z "${DINO_MODEL_PATH:-}" ]]; then export DINO_MODEL_PATH="$ckpt_dir/dino-vits16"; fi
}

install_requirements() {
  local model="$1"
  local dir="$2"
  if [[ -f "$dir/requirements.txt" ]]; then
    local py
    py=$(model_python "$model")
    if [[ "$USE_VENV" -eq 1 ]]; then
      local vdir req_hash_file current_hash
      vdir="$(dirname "$(dirname "$py")")"
      req_hash_file="$vdir/.requirements.sha256"
      REQ_FILE="$dir/requirements.txt" current_hash="$("$py" - <<'PY'
import hashlib
import os

path = os.environ["REQ_FILE"]
with open(path, "rb") as f:
    data = f.read()
print(hashlib.sha256(data).hexdigest())
PY
)"
      if [[ -f "$req_hash_file" ]]; then
        local saved_hash
        saved_hash="$(cat "$req_hash_file")"
        if [[ "$saved_hash" == "$current_hash" ]]; then
          log "requirements unchanged for $model, skipping install"
          return
        fi
      fi
    fi
    log "installing requirements for $model from $dir/requirements.txt"
    "$py" -m pip install -r "$dir/requirements.txt"
    if [[ "${req_hash_file:-}" != "" ]]; then
      echo "$current_hash" > "$req_hash_file"
    fi
  fi
}

run_model() {
  local model="$1"
  local args=()
  local py
  py=$(model_python "$model" "$(model_dir "$model")")
  log "running model: $model"
  args+=(--jobs "$JOBS")
  if [[ "$model" == "xverse" ]]; then
    "$py" scripts/run_xverse.py "${args[@]}"
  elif [[ "$model" == "mosaic" ]]; then
    "$py" scripts/run_mosaic.py "${args[@]}"
  elif [[ "$model" == "psr" ]]; then
    if [[ -z "$PSR_RUNNER" ]]; then
      echo "psr needs --psr-runner"
      exit 1
    fi
    local psr_args=(--runner "$PSR_RUNNER")
    if [[ -n "$PSR_RUNNER_CWD" ]]; then
      psr_args+=(--runner_cwd "$PSR_RUNNER_CWD")
    fi
    "$py" scripts/run_psr.py "${args[@]}" "${psr_args[@]}"
  fi
  log "finished model: $model"
}

IFS=',' read -r -a model_arr <<< "$MODELS"

if [[ "$INSTALL" -eq 1 && "$MODE" != "eval" ]]; then
  for m in "${model_arr[@]}"; do
    if [[ "$m" == "xverse" ]]; then
      install_requirements "xverse" "XVerse-main"
    elif [[ "$m" == "mosaic" ]]; then
      install_requirements "mosaic" "MOSAIC-main"
    elif [[ "$m" == "psr" ]]; then
      install_requirements "psr" "PSR-main"
    fi
  done
fi

mkdir -p results eval_outputs

if [[ "$MODE" == "gen" ]]; then
  log "mode=gen"
  if [[ " $MODELS " == *"xverse"* ]]; then
    ensure_xverse_checkpoints
  fi
  if [[ -z "$JOBS" ]]; then
    if [[ -f "jobs.jsonl" ]]; then
      JOBS="jobs.jsonl"
    else
      log "generating jobs.jsonl from val_dataset/prompts_50.txt"
      python scripts/generate_jobs.py --prompts val_dataset/prompts_50.txt --images_dir val_dataset --out jobs.jsonl
      JOBS="jobs.jsonl"
    fi
  fi
  log "using jobs file: $JOBS"
  for m in "${model_arr[@]}"; do
    run_model "$m"
  done
elif [[ "$MODE" == "eval" ]]; then
  log "mode=eval"
  eval_py=$(model_python "${model_arr[0]}")
  log "running eval.py"
  "$eval_py" scripts/eval.py --models "$MODELS"
elif [[ "$MODE" == "all" ]]; then
  log "mode=all"
  if [[ " $MODELS " == *"xverse"* ]]; then
    ensure_xverse_checkpoints
  fi
  if [[ -z "$JOBS" ]]; then
    if [[ -f "jobs.jsonl" ]]; then
      JOBS="jobs.jsonl"
    else
      log "generating jobs.jsonl from val_dataset/prompts_50.txt"
      python scripts/generate_jobs.py --prompts val_dataset/prompts_50.txt --images_dir val_dataset --out jobs.jsonl
      JOBS="jobs.jsonl"
    fi
  fi
  log "using jobs file: $JOBS"
  for m in "${model_arr[@]}"; do
    run_model "$m"
  done
  eval_py=$(model_python "${model_arr[0]}")
  log "running eval.py"
  "$eval_py" scripts/eval.py --models "$MODELS"
  log "running eval_merge.py"
  "$eval_py" scripts/eval_merge.py --models "$MODELS"
  log "all done"
else
  usage
  exit 1
fi
