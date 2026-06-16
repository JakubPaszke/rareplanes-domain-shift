#!/usr/bin/env bash
set -euo pipefail

# Eksperyment B: degradacja czestotliwosciowa synthetic 10k -> ewaluacja na real.
# Zaklada, ze istnieja:
#   data/yolo/synthetic_10k/data.yaml
#   data/real/PS-RGB_tiled/PS-RGB_tiled/
#   data/real/annotations/instances_test_aircraft.json

EPOCHS="${EPOCHS:-60}"
BATCH="${BATCH:-64}"
WORKERS="${WORKERS:-8}"

python3 src/make_frequency_degraded_dataset.py \
  --src data/yolo/synthetic_10k \
  --name synthetic_10k_b1_blur_noise \
  --blur-radius 0.4 \
  --noise-sigma 5 \
  --seed 42 \
  --overwrite

python3 src/train_yolo.py \
  --data data/yolo/synthetic_10k_b1_blur_noise/data.yaml \
  --name expB1_blur_noise_10k_ml \
  --epochs "$EPOCHS" \
  --batch "$BATCH" \
  --imgsz 512 \
  --seed 42 \
  --workers "$WORKERS" \
  --val-data data/yolo/synthetic_10k_b1_blur_noise/data.yaml

python3 src/eval_per_size.py \
  --weights runs/expB1_blur_noise_10k_ml/weights/best.pt \
  --img-dir data/real/PS-RGB_tiled/PS-RGB_tiled \
  --coco-gt data/real/annotations/instances_test_aircraft.json \
  --name expB1_blur_noise_10k_ml

python3 src/make_frequency_degraded_dataset.py \
  --src data/yolo/synthetic_10k \
  --name synthetic_10k_b2_noise \
  --noise-sigma 8 \
  --seed 42 \
  --overwrite

python3 src/train_yolo.py \
  --data data/yolo/synthetic_10k_b2_noise/data.yaml \
  --name expB2_noise_10k_ml \
  --epochs "$EPOCHS" \
  --batch "$BATCH" \
  --imgsz 512 \
  --seed 42 \
  --workers "$WORKERS" \
  --val-data data/yolo/synthetic_10k_b2_noise/data.yaml

python3 src/eval_per_size.py \
  --weights runs/expB2_noise_10k_ml/weights/best.pt \
  --img-dir data/real/PS-RGB_tiled/PS-RGB_tiled \
  --coco-gt data/real/annotations/instances_test_aircraft.json \
  --name expB2_noise_10k_ml

python3 src/make_frequency_degraded_dataset.py \
  --src data/yolo/synthetic_10k \
  --name synthetic_10k_b3_blur_noise_jpeg \
  --blur-radius 0.6 \
  --noise-sigma 6 \
  --jpeg-quality-min 75 \
  --seed 42 \
  --overwrite

python3 src/train_yolo.py \
  --data data/yolo/synthetic_10k_b3_blur_noise_jpeg/data.yaml \
  --name expB3_blur_noise_jpeg_10k_ml \
  --epochs "$EPOCHS" \
  --batch "$BATCH" \
  --imgsz 512 \
  --seed 42 \
  --workers "$WORKERS" \
  --val-data data/yolo/synthetic_10k_b3_blur_noise_jpeg/data.yaml

python3 src/eval_per_size.py \
  --weights runs/expB3_blur_noise_jpeg_10k_ml/weights/best.pt \
  --img-dir data/real/PS-RGB_tiled/PS-RGB_tiled \
  --coco-gt data/real/annotations/instances_test_aircraft.json \
  --name expB3_blur_noise_jpeg_10k_ml
