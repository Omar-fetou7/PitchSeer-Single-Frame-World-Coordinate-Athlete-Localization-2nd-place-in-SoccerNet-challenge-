#!/bin/bash
#SBATCH --job-name=vitpose-rtdetr-locsim
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH -p cscc-gpu-p
#SBATCH --account=cscc-users
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --cpus-per-task=16
#SBATCH --output=/path/to/repo/slurm_logs/vitpose_rtdetr_locsim_%j.log

set -euo pipefail

# === EDIT THIS ===
DATA_ROOT="/path/to/data"   # contains raw 4K annotations/{train,val,test}.json (input to prepare step)
# =================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
CONFIG_PATH="${REPO_ROOT}/configs/eval/soccernet_rtdetr_locsim.yaml"

cd "${REPO_ROOT}"
mkdir -p slurm_logs

export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "=== ViTPose SoccerNet RT-DETR LocSim Evaluation ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

conda run -n vitpose --no-capture-output python tools/dataset/prepare_soccernet_4k.py --ann-dir "${DATA_ROOT}/annotations"

conda run -n vitpose --no-capture-output python -u tools/analysis/eval_soccernet_from_config.py \
  --config "${CONFIG_PATH}"
