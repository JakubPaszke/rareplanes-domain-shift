#!/usr/bin/env bash
#SBATCH --job-name=final-smoke
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=final-smoke-%j.out
#SBATCH --error=final-smoke-%j.err

set -euo pipefail

module load miniconda
conda activate rareplanes

cd /work/${USER}/rareplanes-domain-shift

DATA_DIR="${DATA_DIR:-/work/${USER}/rareplanes-data/data}"
DEVICE="${DEVICE:-0}"
BATCH="${BATCH:-32}"
WORKERS="${WORKERS:-4}"
EPOCHS="${EPOCHS:-3}"

echo "DATA_DIR=${DATA_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi

python --version
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"

python src/train_final_model.py \
  --smoke \
  --data-dir "${DATA_DIR}" \
  --epochs "${EPOCHS}" \
  --batch "${BATCH}" \
  --workers "${WORKERS}" \
  --device "${DEVICE}" \
  --skip-benchmark \
  "$@"
