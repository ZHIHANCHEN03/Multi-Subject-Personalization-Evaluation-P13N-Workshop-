set -euo pipefail

if command -v wget &> /dev/null; then
    CMD="wget"
elif command -v curl &> /dev/null; then
    CMD="curl -L -O"
else
    echo "Please install wget or curl to download the checkpoints."
    exit 1
fi

if command -v huggingface-cli &> /dev/null; then
    HF_CMD="huggingface-cli"
else
    HF_CMD="${HUGGINGFACE_CLI:-python -m huggingface_hub.commands.huggingface_cli}"
fi

# Define the URLs for SAM 2.1 checkpoints
SAM2p1_BASE_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824"
sam2p1_hiera_l_url="${SAM2p1_BASE_URL}/sam2.1_hiera_large.pt"

echo "Downloading sam2.1_hiera_large.pt checkpoint..."
$CMD $sam2p1_hiera_l_url || { echo "Failed to download checkpoint from $sam2p1_hiera_l_url"; exit 1; }

echo "Downloading FLUX.1-dev checkpoint..."
$HF_CMD download black-forest-labs/FLUX.1-dev --local-dir ./FLUX.1-dev

echo "Downloading Florence-2-large checkpoint..."
$HF_CMD download microsoft/Florence-2-large --local-dir ./Florence-2-large

echo "Downloading clip-vit-large-patch14 checkpoint..."
$HF_CMD download openai/clip-vit-large-patch14 --local-dir ./clip-vit-large-patch14

echo "Downloading DINO checkpoint..."
$HF_CMD download facebook/dino-vits16 --local-dir ./dino-vits16

echo "Downloading DPG VQA checkpoint..."
$HF_CMD download xingjianleng/mplug_visual-question-answering_coco_large_en --local-dir ./mplug_visual-question-answering_coco_large_en

echo "Downloading XVerse checkpoint..."
$HF_CMD download ByteDance/XVerse --local-dir ./XVerse

echo "Downloading FLUX.1-schnell checkpoint..."
$HF_CMD download black-forest-labs/FLUX.1-schnell --local-dir ./FLUX.1-schnell

echo "Downloading diffusers/FLUX.1-dev-bnb-4bit checkpoint..."
$HF_CMD download diffusers/FLUX.1-dev-bnb-4bit --local-dir ./FLUX.1-dev-bnb-4bit

echo "Downloading city96/FLUX.1-dev-gguf checkpoint..."
$HF_CMD download city96/FLUX.1-dev-gguf --local-dir ./FLUX.1-dev-gguf

echo "All checkpoints are downloaded successfully."
