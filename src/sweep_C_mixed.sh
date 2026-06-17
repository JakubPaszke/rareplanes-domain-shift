#!/usr/bin/env bash
set -euo pipefail

# Eksperyment C: synthetic 10k + mala porcja real train/val.
# Zaklada, ze istnieja:
#   data/yolo/synthetic_10k/data.yaml
#   data/yolo/real_aircraft/data.yaml
#   data/real/PS-RGB_tiled/PS-RGB_tiled/
#   data/real/annotations/instances_test_aircraft.json

EPOCHS="${EPOCHS:-60}"
BATCH="${BATCH:-64}"
WORKERS="${WORKERS:-8}"
DEVICE="${DEVICE:-0}"

pcts=(1 5 10 25)
fracs=(0.01 0.05 0.10 0.25)

for i in "${!pcts[@]}"; do
  pct="${pcts[$i]}"
  frac="${fracs[$i]}"
  dataset="mixed_syn10k_real${pct}pct"
  run="expC_${pct}pct_real_10k_ml"

  python3 src/make_mixed_dataset.py \
    --syn-src data/yolo/synthetic_10k \
    --real-src data/yolo/real_aircraft \
    --name "${dataset}" \
    --real-frac "${frac}" \
    --seed 42 \
    --overwrite

  python3 src/train_yolo.py \
    --data "data/yolo/${dataset}/data.yaml" \
    --name "${run}" \
    --epochs "$EPOCHS" \
    --batch "$BATCH" \
    --imgsz 512 \
    --seed 42 \
    --device "$DEVICE" \
    --workers "$WORKERS" \
    --val-data "data/yolo/${dataset}/data.yaml"

  python3 src/eval_per_size.py \
    --weights "runs/${run}/weights/best.pt" \
    --img-dir data/real/PS-RGB_tiled/PS-RGB_tiled \
    --coco-gt data/real/annotations/instances_test_aircraft.json \
    --device "$DEVICE" \
    --name "${run}"
done
