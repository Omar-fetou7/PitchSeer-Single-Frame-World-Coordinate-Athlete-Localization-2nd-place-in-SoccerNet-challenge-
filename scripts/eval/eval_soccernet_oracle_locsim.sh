#!/bin/bash

set -euo pipefail

# === EDIT THESE (or override via env vars) ===
DATA_ROOT="${DATA_ROOT:-/path/to/data}"                       # contains images/{train,val,test}/ and annotations/{train,val,test}.json
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/path/to/checkpoints}"      # trained ViTPose checkpoints
SPLIT="${SPLIT:-val}"                                          # val | test
CHECKPOINT_NAME="${CHECKPOINT_NAME:-vitpose_large_3dloss_lambda0.5_epoch3.pth}"
# =============================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/soccernet/ViTPose_large_3dloss_soccernet_4k_256x192.py}"
ANN_PATH="${DATA_ROOT}/annotations/${SPLIT}.json"
IMG_PREFIX="${DATA_ROOT}/images/${SPLIT}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/evaluation/oracle_detection}"

CHECKPOINTS=(
  "${CHECKPOINT_DIR}/${CHECKPOINT_NAME}"
)
cd "${REPO_ROOT}"
mkdir -p slurm_logs "${OUTPUT_DIR}"

export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0

echo "=== ViTPose SoccerNet Oracle LocSim Evaluation ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

# conda run -n vit --no-capture-output python tools/dataset/prepare_soccernet_4k.py 

conda run -n vit --no-capture-output python -u tools/analysis/eval_soccernet_oracle_locsim.py \
  "${CONFIG_PATH}" \
  --checkpoint "${CHECKPOINTS[@]}" \
  --ann-file "${ANN_PATH}" \
  --img-prefix "${IMG_PREFIX}" \
  --output-dir "${OUTPUT_DIR}" \
  --device cuda:0 \
  --samples-per-gpu 256 \
  --workers-per-gpu 8 \
  --overwrite
