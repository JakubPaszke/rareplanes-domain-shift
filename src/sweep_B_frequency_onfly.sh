#!/usr/bin/env bash
set -euo pipefail

# Eksperyment B on-the-fly: degradacja obrazow podczas treningu, bez zapisu PNG.
# Zaklada, ze istnieja:
#   data/yolo/synthetic_10k/data.yaml albo data/yolo/synthetic_1k/data.yaml
#   data/real/PS-RGB_tiled/PS-RGB_tiled/
#   data/real/annotations/instances_test_aircraft.json

SRC_DATASET="${SRC_DATASET:-data/yolo/synthetic_10k}"
DATASET_TAG="${DATASET_TAG:-10k_onfly}"
EPOCHS="${EPOCHS:-60}"
BATCH="${BATCH:-64}"
WORKERS="${WORKERS:-2}"
DEVICE="${DEVICE:-0}"
VARIANTS="${VARIANTS:-B1 B2 B3}"
FREQ_PROB="${FREQ_PROB:-1.0}"
MODEL="${MODEL:-yolov10n.pt}"

echo "[sweep B onfly] SRC_DATASET=$SRC_DATASET DATASET_TAG=$DATASET_TAG"
echo "[sweep B onfly] EPOCHS=$EPOCHS BATCH=$BATCH WORKERS=$WORKERS DEVICE=$DEVICE"
echo "[sweep B onfly] VARIANTS=$VARIANTS FREQ_PROB=$FREQ_PROB MODEL=$MODEL"

has_variant() {
  case " ${VARIANTS} " in
    *" $1 "*) return 0 ;;
    *) return 1 ;;
  esac
}

run_variant() {
  local variant="$1"
  local run="$2"
  shift 2

  echo "[$variant] trening on-the-fly"
  python3 -u src/train_yolo_freq_onfly.py \
    --data "$SRC_DATASET/data.yaml" \
    --degrade-root "$SRC_DATASET/images/train" \
    --name "$run" \
    --model "$MODEL" \
    --epochs "$EPOCHS" \
    --batch "$BATCH" \
    --imgsz 512 \
    --seed 42 \
    --device "$DEVICE" \
    --workers "$WORKERS" \
    --freq-prob "$FREQ_PROB" \
    "$@"

  echo "[$variant] ewaluacja real holdout"
  python3 src/eval_per_size.py \
    --weights "runs/${run}/weights/best.pt" \
    --img-dir data/real/PS-RGB_tiled/PS-RGB_tiled \
    --coco-gt data/real/annotations/instances_test_aircraft.json \
    --device "$DEVICE" \
    --name "$run"
}

if has_variant B1; then
  run_variant B1 "expB1_blur_noise_onfly_${DATASET_TAG}_ml" \
    --blur-radius 0.4 \
    --noise-sigma 5
fi

if has_variant B2; then
  run_variant B2 "expB2_noise_onfly_${DATASET_TAG}_ml" \
    --noise-sigma 8
fi

if has_variant B3; then
  run_variant B3 "expB3_blur_noise_jpeg_onfly_${DATASET_TAG}_ml" \
    --blur-radius 0.6 \
    --noise-sigma 6 \
    --jpeg-quality-min 75
fi
