#!/usr/bin/env bash
#SBATCH --job-name=train-final
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=24:00:00
#SBATCH --output=train-final-%j.out
#SBATCH --error=train-final-%j.err

set -euo pipefail

module load miniconda
conda activate rareplanes

cd /work/${USER}/rareplanes-domain-shift

DATA_DIR="${DATA_DIR:-/work/${USER}/rareplanes-data/data}"
DEVICE="${DEVICE:-0}"
BATCH="${BATCH:-64}"
WORKERS="${WORKERS:-4}"
EPOCHS="${EPOCHS:-60}"
FULL_DOWNLOAD_WORKERS="${FULL_DOWNLOAD_WORKERS:-32}"

echo "HOST=$(hostname)"
echo "DATA_DIR=${DATA_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi

python --version
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"

python src/train_final_model.py \
  --data-dir "${DATA_DIR}" \
  --epochs "${EPOCHS}" \
  --batch "${BATCH}" \
  --workers "${WORKERS}" \
  --device "${DEVICE}" \
  --full-download-workers "${FULL_DOWNLOAD_WORKERS}" \
  "$@"
