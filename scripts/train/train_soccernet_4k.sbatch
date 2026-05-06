#!/bin/bash

set -euo pipefail

# === EDIT THIS ===
DATA_ROOT="/path/to/data"   # contains raw 4K annotations/{train,val,test}.json (input to prepare step) and images/
# =================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
CONFIG_PATH="configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/soccernet/ViTPose_small_simple_soccernet_4k_256x192.py"
WORK_DIR="work_dirs/soccernet_4k_nomosaic"

cd "${REPO_ROOT}"
mkdir -p slurm_logs "${WORK_DIR}"

export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

echo "=== ViTPose SoccerNet 4K Training (No Mosaic) ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

conda run -n vitpose --no-capture-output python tools/dataset/prepare_soccernet_4k.py --ann-dir "${DATA_ROOT}/annotations"
conda run -n vitpose --no-capture-output python -u tools/train.py "${CONFIG_PATH}" --work-dir "${WORK_DIR}" --seed 0
