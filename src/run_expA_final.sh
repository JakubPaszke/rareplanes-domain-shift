#!/usr/bin/env bash
# Finalny eksperyment A: slaby HSV jitter (s=0.4, v=0.3) na pelnych 45k.
# Prawdziwy plik skryptu => set -e dziala (eval NIE poleci po crashu treningu).
set -euo pipefail
cd "$(dirname "$0")/.."

echo "Start FINAL A $(date)"
python3 src/train_yolo.py --data data/yolo/synthetic_aircraft/data.yaml --name expA_final_45k \
  --epochs 100 --batch 64 --imgsz 512 --seed 42 --patience 25 --workers 4 \
  --hsv_h 0.015 --hsv_s 0.4 --hsv_v 0.3 \
  --val-data data/yolo/synthetic_aircraft/data.yaml
echo "trening OK, eval na realnym tescie..."
python3 src/eval_per_size.py --weights runs/expA_final_45k/weights/best.pt \
  --img-dir data/real/PS-RGB_tiled/PS-RGB_tiled \
  --coco-gt data/real/annotations/instances_test_aircraft.json --name expA_final_45k
echo "DONE FINAL A $(date)"
