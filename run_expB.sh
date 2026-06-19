#!/usr/bin/env bash
#SBATCH --job-name=expB-onfly
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=expB-%j.out
#SBATCH --error=expB-%j.err

module load miniconda
conda activate rareplanes

cd /work/$USER/rareplanes-domain-shift || exit 1

echo "HOST=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

python --version
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"

python src/run_expB_onfly_cluster.py \
  --variants B1 B2 B3 \
  --src-dataset data/yolo/synthetic_10k \
  --dataset-tag 10k_onfly \
  --epochs 20 \
  --batch 64 \
  --workers 4 \
  --device 0
