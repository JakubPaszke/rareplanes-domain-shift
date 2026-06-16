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
DEVICE="${DEVICE:-0}"
VARIANTS="${VARIANTS:-B1 B2 B3}"
CLEANUP_DATASETS="${CLEANUP_DATASETS:-0}"

echo "[sweep B] EPOCHS=$EPOCHS BATCH=$BATCH WORKERS=$WORKERS DEVICE=$DEVICE"
echo "[sweep B] VARIANTS=$VARIANTS CLEANUP_DATASETS=$CLEANUP_DATASETS"

has_variant() {
  case " ${VARIANTS} " in
    *" $1 "*) return 0 ;;
    *) return 1 ;;
  esac
}

cleanup_dataset() {
  if [ "$CLEANUP_DATASETS" = "1" ]; then
    python3 -c "import shutil; shutil.rmtree('$1', ignore_errors=True)"
  fi
}

if has_variant B1; then
echo "[B1] tworze dataset: blur + noise"
python3 -u src/make_frequency_degraded_dataset.py \
  --src data/yolo/synthetic_10k \
  --name synthetic_10k_b1_blur_noise \
  --blur-radius 0.4 \
  --noise-sigma 5 \
  --seed 42 \
  --overwrite

echo "[B1] trening"
python3 src/train_yolo.py \
  --data data/yolo/synthetic_10k_b1_blur_noise/data.yaml \
  --name expB1_blur_noise_10k_ml \
  --epochs "$EPOCHS" \
  --batch "$BATCH" \
  --imgsz 512 \
  --seed 42 \
  --device "$DEVICE" \
  --workers "$WORKERS" \
  --val-data data/yolo/synthetic_10k_b1_blur_noise/data.yaml

echo "[B1] ewaluacja real holdout"
python3 src/eval_per_size.py \
  --weights runs/expB1_blur_noise_10k_ml/weights/best.pt \
  --img-dir data/real/PS-RGB_tiled/PS-RGB_tiled \
  --coco-gt data/real/annotations/instances_test_aircraft.json \
  --device "$DEVICE" \
  --name expB1_blur_noise_10k_ml
cleanup_dataset "data/yolo/synthetic_10k_b1_blur_noise"
fi

if has_variant B2; then
echo "[B2] tworze dataset: noise"
python3 -u src/make_frequency_degraded_dataset.py \
  --src data/yolo/synthetic_10k \
  --name synthetic_10k_b2_noise \
  --noise-sigma 8 \
  --seed 42 \
  --overwrite

echo "[B2] trening"
python3 src/train_yolo.py \
  --data data/yolo/synthetic_10k_b2_noise/data.yaml \
  --name expB2_noise_10k_ml \
  --epochs "$EPOCHS" \
  --batch "$BATCH" \
  --imgsz 512 \
  --seed 42 \
  --device "$DEVICE" \
  --workers "$WORKERS" \
  --val-data data/yolo/synthetic_10k_b2_noise/data.yaml

echo "[B2] ewaluacja real holdout"
python3 src/eval_per_size.py \
  --weights runs/expB2_noise_10k_ml/weights/best.pt \
  --img-dir data/real/PS-RGB_tiled/PS-RGB_tiled \
  --coco-gt data/real/annotations/instances_test_aircraft.json \
  --device "$DEVICE" \
  --name expB2_noise_10k_ml
cleanup_dataset "data/yolo/synthetic_10k_b2_noise"
fi

if has_variant B3; then
echo "[B3] tworze dataset: blur + noise + JPEG"
python3 -u src/make_frequency_degraded_dataset.py \
  --src data/yolo/synthetic_10k \
  --name synthetic_10k_b3_blur_noise_jpeg \
  --blur-radius 0.6 \
  --noise-sigma 6 \
  --jpeg-quality-min 75 \
  --seed 42 \
  --overwrite

echo "[B3] trening"
python3 src/train_yolo.py \
  --data data/yolo/synthetic_10k_b3_blur_noise_jpeg/data.yaml \
  --name expB3_blur_noise_jpeg_10k_ml \
  --epochs "$EPOCHS" \
  --batch "$BATCH" \
  --imgsz 512 \
  --seed 42 \
  --device "$DEVICE" \
  --workers "$WORKERS" \
  --val-data data/yolo/synthetic_10k_b3_blur_noise_jpeg/data.yaml

echo "[B3] ewaluacja real holdout"
python3 src/eval_per_size.py \
  --weights runs/expB3_blur_noise_jpeg_10k_ml/weights/best.pt \
  --img-dir data/real/PS-RGB_tiled/PS-RGB_tiled \
  --coco-gt data/real/annotations/instances_test_aircraft.json \
  --device "$DEVICE" \
  --name expB3_blur_noise_jpeg_10k_ml
cleanup_dataset "data/yolo/synthetic_10k_b3_blur_noise_jpeg"
fi
