#!/bin/bash

set -euo pipefail

# === EDIT THESE (or override via env vars) ===
DATA_ROOT="${DATA_ROOT:-/path/to/data}"                        # contains images/{train,val,test}/ and annotations/{train,val,test}.json
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/path/to/checkpoints}"       # trained ViTPose checkpoints
DETECTOR_DIR="${DETECTOR_DIR:-/path/to/detector_weights}"      # YOLO/RT-DETR weights
SPLIT="${SPLIT:-val}"                                          # val | test
CHECKPOINT_NAME="${CHECKPOINT_NAME:-vitpose_large_3dloss_lambda0.5_epoch3.pth}"
DETECTOR_NAME="${DETECTOR_NAME:-yolo_best.pt}"
# =============================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/soccernet/ViTPose_large_3dloss_soccernet_4k_256x192.py}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${CHECKPOINT_DIR}/${CHECKPOINT_NAME}}"
DETECTOR_WEIGHTS="${DETECTOR_WEIGHTS:-${DETECTOR_DIR}/${DETECTOR_NAME}}"
ANN_PATH="${DATA_ROOT}/annotations/${SPLIT}.json"
IMG_PREFIX="${DATA_ROOT}/images/${SPLIT}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/evaluation/yolo_detection}"
METRICS_NAME="${METRICS_NAME:-}"  # leave empty to use default: <checkpoint-stem>_yolo_best

cd "${REPO_ROOT}"

export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0

# echo "=== ViTPose SoccerNet YOLO LocSim Evaluation ==="
# echo "Date: $(date)"
# echo "Host: $(hostname)"
# echo "GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
# echo ""

conda run -n vit --no-capture-output python -u tools/analysis/eval_soccernet_yolo_locsim.py \
  "${CONFIG_PATH}" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --detector-weights "${DETECTOR_WEIGHTS}" \
  --ann-file "${ANN_PATH}" \
  --img-prefix "${IMG_PREFIX}" \
  --output-dir "${OUTPUT_DIR}" \
  --device cuda:0 \
  --samples-per-gpu 256 \
  --workers-per-gpu 4 \
  --detector-batch-size 256 \
  --overwrite \
  ${METRICS_NAME:+--metrics-name "${METRICS_NAME}"}
