#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
CONFIG_PATH="configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/soccernet/ViTPose_small_simple_soccernet_trainval_cosine_256x192.py"
WORK_DIR="work_dirs/soccernet_trainval_cosine"

cd "${REPO_ROOT}"
mkdir -p slurm_logs "${WORK_DIR}"

export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=5

echo "=== ViTPose SoccerNet Train+Val Cosine Training ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

conda run -n vit --no-capture-output python -u tools/train.py "${CONFIG_PATH}" --work-dir "${WORK_DIR}" --seed 0
