#!/usr/bin/env bash
# Eksperyment A - sweep augmentacji fotometrycznych (HSV jitter) na podzbiorze 10k.
# 3 sily: slaby / sredni / mocny. Kazdy: trening 60 epok + ewaluacja na realnym tescie.
# Motywacja: synthetic jasniejszy (V 133 vs 72) i mniej nasycony (S 0.12 vs 0.30) niz real.
#
# Uzycie: bash src/sweep_A_photometric.sh
# Wymaga: data/yolo/synthetic_10k (z make_subset.py). Cache labeli przyspiesza restart.
set -e
cd "$(dirname "$0")/.."

run() {  # nazwa hsv_s hsv_v
  local name=$1 s=$2 v=$3
  echo ">>> $name (hsv_s=$s hsv_v=$v) $(date)"
  python3 src/train_yolo.py --data data/yolo/synthetic_10k/data.yaml --name "$name" \
    --epochs 60 --batch 64 --imgsz 512 --seed 42 --patience 20 \
    --hsv_h 0.015 --hsv_s "$s" --hsv_v "$v" --val-data data/yolo/synthetic_10k/data.yaml
  python3 src/eval_per_size.py --weights "runs/$name/weights/best.pt" \
    --img-dir data/real/PS-RGB_tiled/PS-RGB_tiled \
    --coco-gt data/real/annotations/instances_test_aircraft.json --name "$name"
}

echo "===== SWEEP A: augmentacje fotometryczne na 10k ====="
run expA1_weak_10k   0.4 0.3
run expA2_med_10k    0.7 0.5
run expA3_strong_10k 0.9 0.7
echo "===== SWEEP A DONE $(date) ====="
