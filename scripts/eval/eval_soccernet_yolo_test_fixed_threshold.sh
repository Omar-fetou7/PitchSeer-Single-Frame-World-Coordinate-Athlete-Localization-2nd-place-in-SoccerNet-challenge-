#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
CONFIG_PATH="${REPO_ROOT}/configs/eval/soccernet_yolo_test_fixed_threshold.yaml"

cd "${REPO_ROOT}"

#export CUDA_VISIBLE_DEVICES=2
export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

echo "=== ViTPose SoccerNet YOLO Test Fixed-Threshold Evaluation ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

conda run -n vit --no-capture-output python -u tools/analysis/eval_soccernet_from_config.py \
  --config "${CONFIG_PATH}"
