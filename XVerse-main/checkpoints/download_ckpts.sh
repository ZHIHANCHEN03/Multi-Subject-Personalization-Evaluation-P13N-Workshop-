set -euo pipefail

if [[ -n "${HUGGINGFACE_PY:-}" ]]; then
    PY="$HUGGINGFACE_PY"
elif command -v python3 &> /dev/null; then
    PY="python3"
else
    PY="python"
fi

if command -v wget &> /dev/null; then
    CMD="wget"
elif command -v curl &> /dev/null; then
    CMD="curl -L -O"
else
    echo "Please install wget or curl to download the checkpoints."
    exit 1
fi

hf_download() {
    local repo="$1"
    local out_dir="$2"
    if command -v huggingface-cli &> /dev/null; then
        if huggingface-cli download "$repo" --local-dir "$out_dir"; then
            return 0
        fi
    fi
    if [[ -n "${HUGGINGFACE_CLI:-}" ]]; then
        if $HUGGINGFACE_CLI download "$repo" --local-dir "$out_dir"; then
            return 0
        fi
    fi
    if $PY - <<'PY'
import importlib.util
import sys
ok = importlib.util.find_spec("huggingface_hub.commands.huggingface_cli") is not None
sys.exit(0 if ok else 1)
PY
    then
        if $PY -m huggingface_hub.commands.huggingface_cli download "$repo" --local-dir "$out_dir"; then
            return 0
        fi
    fi
    if $PY - <<'PY'
import importlib.util
import sys
ok = importlib.util.find_spec("huggingface_hub.cli") is not None
sys.exit(0 if ok else 1)
PY
    then
        if $PY -m huggingface_hub.cli download "$repo" --local-dir "$out_dir"; then
            return 0
        fi
    fi
    echo "Failed to download $repo. Ensure huggingface-cli or huggingface_hub CLI is available."
    exit 1
}

ensure_hf_cli() {
    if command -v huggingface-cli &> /dev/null; then
        return
    fi
    if $PY - <<'PY'
import importlib.util
import sys
ok = importlib.util.find_spec("huggingface_hub") is not None
sys.exit(0 if ok else 1)
PY
    then
        return
    fi
    echo "huggingface_hub not found, installing..."
    $PY -m pip install -U "huggingface_hub[cli]<1.0"
}

# Define the URLs for SAM 2.1 checkpoints
SAM2p1_BASE_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824"
sam2p1_hiera_l_url="${SAM2p1_BASE_URL}/sam2.1_hiera_large.pt"

echo "Downloading sam2.1_hiera_large.pt checkpoint..."
$CMD $sam2p1_hiera_l_url || { echo "Failed to download checkpoint from $sam2p1_hiera_l_url"; exit 1; }

ensure_hf_cli

echo "Downloading FLUX.1-dev checkpoint..."
hf_download black-forest-labs/FLUX.1-dev ./FLUX.1-dev

echo "Downloading Florence-2-large checkpoint..."
hf_download microsoft/Florence-2-large ./Florence-2-large

echo "Downloading clip-vit-large-patch14 checkpoint..."
hf_download openai/clip-vit-large-patch14 ./clip-vit-large-patch14

echo "Downloading DINO checkpoint..."
hf_download facebook/dino-vits16 ./dino-vits16

echo "Downloading DPG VQA checkpoint..."
hf_download xingjianleng/mplug_visual-question-answering_coco_large_en ./mplug_visual-question-answering_coco_large_en

echo "Downloading XVerse checkpoint..."
hf_download ByteDance/XVerse ./XVerse

echo "All checkpoints are downloaded successfully."
