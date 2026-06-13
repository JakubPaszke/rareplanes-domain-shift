#!/usr/bin/env bash
# Wznowienie sweepu A po przerwaniu: A1 z checkpointu (epoka 24/60), potem A2/A3 od zera.
set -e
cd "$(dirname "$0")/.."

echo ">>> A1 RESUME z last.pt $(date)"
python3 - <<'PY'
from ultralytics import YOLO
m = YOLO("runs/expA1_weak_10k/weights/last.pt")
m.train(resume=True)
PY
python3 src/eval_per_size.py --weights runs/expA1_weak_10k/weights/best.pt \
  --img-dir data/real/PS-RGB_tiled/PS-RGB_tiled \
  --coco-gt data/real/annotations/instances_test_aircraft.json --name expA1_weak_10k

run() {
  local name=$1 s=$2 v=$3
  echo ">>> $name (hsv_s=$s hsv_v=$v) $(date)"
  python3 src/train_yolo.py --data data/yolo/synthetic_10k/data.yaml --name "$name" \
    --epochs 60 --batch 64 --imgsz 512 --seed 42 --patience 20 \
    --hsv_h 0.015 --hsv_s "$s" --hsv_v "$v" --val-data data/yolo/synthetic_10k/data.yaml
  python3 src/eval_per_size.py --weights "runs/$name/weights/best.pt" \
    --img-dir data/real/PS-RGB_tiled/PS-RGB_tiled \
    --coco-gt data/real/annotations/instances_test_aircraft.json --name "$name"
}
run expA2_med_10k    0.7 0.5
run expA3_strong_10k 0.9 0.7
echo "===== SWEEP A DONE (resumed) $(date) ====="
