#!/usr/bin/env bash
set -euo pipefail

# Eksperyment C: synthetic subset + mala porcja real train/val.
# Zaklada, ze istnieja:
#   data/yolo/synthetic_10k/data.yaml albo data/yolo/synthetic_1k/data.yaml
#   data/yolo/real_aircraft/data.yaml
#   data/real/PS-RGB_tiled/PS-RGB_tiled/
#   data/real/annotations/instances_test_aircraft.json

SRC_DATASET="${SRC_DATASET:-data/yolo/synthetic_10k}"
DATASET_TAG="${DATASET_TAG:-10k}"
EPOCHS="${EPOCHS:-60}"
BATCH="${BATCH:-64}"
WORKERS="${WORKERS:-8}"
DEVICE="${DEVICE:-0}"
PCTS="${PCTS:-1 5 10 25}"
FRACS="${FRACS:-0.01 0.05 0.10 0.25}"
IMG_SIZE="${IMG_SIZE:-512}"
MODEL="${MODEL:-yolov10n.pt}"

read -r -a pcts <<< "$PCTS"
read -r -a fracs <<< "$FRACS"

if [ "${#pcts[@]}" -ne "${#fracs[@]}" ]; then
  echo "PCTS i FRACS musza miec tyle samo elementow" >&2
  echo "PCTS=$PCTS" >&2
  echo "FRACS=$FRACS" >&2
  exit 2
fi

echo "[sweep C] SRC_DATASET=$SRC_DATASET DATASET_TAG=$DATASET_TAG"
echo "[sweep C] EPOCHS=$EPOCHS BATCH=$BATCH WORKERS=$WORKERS DEVICE=$DEVICE IMG_SIZE=$IMG_SIZE"
echo "[sweep C] PCTS=$PCTS FRACS=$FRACS MODEL=$MODEL"

for i in "${!pcts[@]}"; do
  pct="${pcts[$i]}"
  frac="${fracs[$i]}"
  dataset="mixed_syn${DATASET_TAG}_real${pct}pct"
  run="expC_${pct}pct_real_${DATASET_TAG}_ml"

  echo "[C ${pct}%] dataset=${dataset} real_frac=${frac}"

  python3 src/make_mixed_dataset.py \
    --syn-src "$SRC_DATASET" \
    --real-src data/yolo/real_aircraft \
    --name "${dataset}" \
    --real-frac "${frac}" \
    --seed 42 \
    --overwrite

  python3 src/train_yolo.py \
    --data "data/yolo/${dataset}/data.yaml" \
    --name "${run}" \
    --model "$MODEL" \
    --epochs "$EPOCHS" \
    --batch "$BATCH" \
    --imgsz "$IMG_SIZE" \
    --seed 42 \
    --device "$DEVICE" \
    --workers "$WORKERS" \
    --val-data "data/yolo/${dataset}/data.yaml"

  python3 src/eval_per_size.py \
    --weights "runs/${run}/weights/best.pt" \
    --img-dir data/real/PS-RGB_tiled/PS-RGB_tiled \
    --coco-gt data/real/annotations/instances_test_aircraft.json \
    --device "$DEVICE" \
    --imgsz "$IMG_SIZE" \
    --name "${run}"
done
