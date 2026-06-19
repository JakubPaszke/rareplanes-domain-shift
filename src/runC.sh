#!/usr/bin/env bash
#SBATCH --job-name=expC-final
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=expC-%j.out
#SBATCH --error=expC-%j.err

module load miniconda
conda activate rareplanes

cd /work/$USER/rareplanes-domain-shift || exit 1

echo "HOST=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

python --version
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"

python expC.py \
  --data-dir /work/$USER/rareplanes-data/data \
  --batch 64 \
  --workers 4 \
  --device 0
