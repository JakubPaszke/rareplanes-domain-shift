#!/usr/bin/env bash
# Baseline synthetic->real na klastrze.
#
# Trening: pelny synthetic YOLO dataset.
# Ewaluacja raportowa: realny holdout COCO przez src/eval_per_size.py.
#
# Domyslnie nazwa runa zaczyna sie od baseline_synth i zawiera SLURM_JOB_ID,
# zeby nie kolidowac ze starszymi wynikami typu syn45k_to_real_baseline.
#
# Przyklady:
#   sbatch run_baseline_synth_cluster.sh
#   EPOCHS=100 sbatch run_baseline_synth_cluster.sh
#   RUN_NAME=baseline_synth_manual_$(date +%Y%m%d_%H%M%S) sbatch run_baseline_synth_cluster.sh
#   PREPARE_YOLO=0 sbatch run_baseline_synth_cluster.sh

#SBATCH --job-name=baseline-synth
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=baseline-synth-%j.out
#SBATCH --error=baseline-synth-%j.err

set -euo pipefail

module load miniconda
conda activate rareplanes

REPO_DIR="${REPO_DIR:-/work/${USER}/rareplanes-domain-shift}"
DATA_DIR="${DATA_DIR:-/work/${USER}/rareplanes-data/data}"

cd "${REPO_DIR}"

DEVICE="${DEVICE:-0}"
MODEL="${MODEL:-yolov10n.pt}"
EPOCHS="${EPOCHS:-60}"
BATCH="${BATCH:-64}"
IMGSZ="${IMGSZ:-512}"
SEED="${SEED:-42}"
WORKERS="${WORKERS:-4}"
PATIENCE="${PATIENCE:-20}"
CACHE="${CACHE:-}"
PREPARE_YOLO="${PREPARE_YOLO:-1}"
MIN_SYN_TRAIN="${MIN_SYN_TRAIN:-30000}"
MIN_REAL_TEST="${MIN_REAL_TEST:-2500}"

RUN_ID="${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-baseline_synth_45k_yolov10n_img${IMGSZ}_e${EPOCHS}_s${SEED}_${RUN_ID}}"

SYN_DATA="${SYN_DATA:-data/yolo/synthetic_aircraft/data.yaml}"
REAL_DATA="${REAL_DATA:-data/yolo/real_aircraft/data.yaml}"
REAL_IMG_DIR="${REAL_IMG_DIR:-data/real/PS-RGB_tiled/PS-RGB_tiled}"
REAL_COCO_GT="${REAL_COCO_GT:-data/real/annotations/instances_test_aircraft.json}"
SYN_TRAIN_IMG_DIR="${SYN_TRAIN_IMG_DIR:-data/yolo/synthetic_aircraft/images/train}"

echo "HOST=$(hostname)"
echo "REPO_DIR=${REPO_DIR}"
echo "DATA_DIR=${DATA_DIR}"
echo "RUN_NAME=${RUN_NAME}"
echo "MODEL=${MODEL}"
echo "EPOCHS=${EPOCHS} BATCH=${BATCH} IMGSZ=${IMGSZ} SEED=${SEED}"
echo "DEVICE=${DEVICE} WORKERS=${WORKERS} PATIENCE=${PATIENCE}"
echo "MIN_SYN_TRAIN=${MIN_SYN_TRAIN} MIN_REAL_TEST=${MIN_REAL_TEST}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi

python --version
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"

if [ ! -e data ]; then
  if [ ! -d "${DATA_DIR}" ]; then
    echo "[error] Nie widze katalogu DATA_DIR=${DATA_DIR}" >&2
    echo "        Ustaw DATA_DIR albo najpierw przygotuj/pobierz dane." >&2
    exit 1
  fi
  ln -s "${DATA_DIR}" data
  echo "[link] data -> ${DATA_DIR}"
elif [ -L data ]; then
  echo "[ok] data symlink -> $(readlink data)"
else
  echo "[ok] data istnieje jako katalog w repo"
fi

if [ "${PREPARE_YOLO}" = "1" ]; then
  echo "[prepare] COCO -> YOLO synthetic"
  python src/coco_to_yolo.py --domain synthetic --classes aircraft --val-frac 0.15 --seed "${SEED}"

  echo "[prepare] COCO -> YOLO real"
  python src/coco_to_yolo.py --domain real --classes aircraft --val-frac 0.15 --seed "${SEED}"
else
  echo "[prepare] pomijam COCO -> YOLO (PREPARE_YOLO=0)"
fi

for required in "${SYN_DATA}" "${REAL_DATA}" "${REAL_IMG_DIR}" "${REAL_COCO_GT}"; do
  if [ ! -e "${required}" ]; then
    echo "[error] Brakuje wymaganego artefaktu: ${required}" >&2
    exit 1
  fi
done

SYN_TRAIN_COUNT="$(find "${SYN_TRAIN_IMG_DIR}" -maxdepth 1 -name '*.png' | wc -l)"
REAL_TEST_COUNT="$(find "${REAL_IMG_DIR}" -maxdepth 1 -name '*.png' | wc -l)"
echo "[check] synthetic train images=${SYN_TRAIN_COUNT}"
echo "[check] real test images=${REAL_TEST_COUNT}"

if [ "${SYN_TRAIN_COUNT}" -lt "${MIN_SYN_TRAIN}" ]; then
  echo "[error] Za malo obrazow synthetic train: ${SYN_TRAIN_COUNT} < ${MIN_SYN_TRAIN}" >&2
  echo "        Sprawdz data/synthetic/images/train albo ustaw MIN_SYN_TRAIN, jesli celowo robisz mniejszy baseline." >&2
  exit 1
fi

if [ "${REAL_TEST_COUNT}" -lt "${MIN_REAL_TEST}" ]; then
  echo "[error] Za malo obrazow real test: ${REAL_TEST_COUNT} < ${MIN_REAL_TEST}" >&2
  echo "        Sprawdz data/real/PS-RGB_tiled/PS-RGB_tiled albo ustaw MIN_REAL_TEST." >&2
  exit 1
fi

echo "[train] synthetic -> ${RUN_NAME}"
TRAIN_CMD=(
  python src/train_yolo.py
  --data "${SYN_DATA}"
  --name "${RUN_NAME}"
  --model "${MODEL}"
  --epochs "${EPOCHS}"
  --batch "${BATCH}"
  --imgsz "${IMGSZ}"
  --seed "${SEED}"
  --device "${DEVICE}"
  --workers "${WORKERS}"
  --patience "${PATIENCE}"
  --val-data "${REAL_DATA}"
)

if [ -n "${CACHE}" ]; then
  TRAIN_CMD+=(--cache "${CACHE}")
fi

printf '[cmd]'
printf ' %q' "${TRAIN_CMD[@]}"
printf '\n'
"${TRAIN_CMD[@]}"

WEIGHTS="runs/${RUN_NAME}/weights/best.pt"
if [ ! -s "${WEIGHTS}" ]; then
  echo "[error] Nie widze wag po treningu: ${WEIGHTS}" >&2
  exit 1
fi

echo "[eval] COCO real holdout -> results/per_size/${RUN_NAME}.json"
python src/eval_per_size.py \
  --weights "${WEIGHTS}" \
  --img-dir "${REAL_IMG_DIR}" \
  --coco-gt "${REAL_COCO_GT}" \
  --name "${RUN_NAME}" \
  --imgsz "${IMGSZ}" \
  --device "${DEVICE}"

echo "[done] baseline synthetic->real"
echo "  Ultralytics metrics: results/baselines/${RUN_NAME}.json"
echo "  COCO real holdout:   results/per_size/${RUN_NAME}.json"
echo "  Weights:             ${WEIGHTS}"
