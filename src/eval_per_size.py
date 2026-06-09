"""
Ewaluacja detektora YOLO w stylu COCO z rozbiciem AP na rozmiar obiektu.

Daje AP osobno dla small/medium/large (wymog PDF), czego domyslny walidator
ultralytics nie raportuje. Dziala na dowolnym checkpointcie i dowolnym zbiorze
testowym COCO (kluczowe dla baseline'u synthetic->real: model syntetyczny,
adnotacje realne).

Uzycie:
  python3 src/eval_per_size.py \
    --weights runs/real_baseline_yolov10n/weights/best.pt \
    --img-dir data/real/PS-RGB_tiled/PS-RGB_tiled \
    --coco-gt data/real/annotations/instances_test_aircraft.json \
    --name real_baseline
"""
import argparse
import json
import contextlib
import io
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--img-dir", required=True, help="katalog z obrazami testowymi")
    ap.add_argument("--coco-gt", required=True, help="adnotacje COCO ground-truth (test)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--conf", type=float, default=0.001, help="niski prog -> pelna krzywa PR")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    from ultralytics import YOLO
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt = json.load(open(args.coco_gt))
    # mapowanie file_name -> image_id; oraz category mapping (YOLO cls 0..N -> coco cat id)
    fname_to_id = {im["file_name"]: im["id"] for im in gt["images"]}
    cat_ids = sorted(c["id"] for c in gt["categories"])
    yolo_to_cat = {i: cid for i, cid in enumerate(cat_ids)}

    img_dir = Path(args.img_dir)
    model = YOLO(args.weights)

    # predykcja na wszystkich obrazach testowych obecnych w GT
    detections = []
    img_files = [img_dir / fn for fn in fname_to_id if (img_dir / fn).exists()]
    print(f"[eval] obrazow do predykcji: {len(img_files)} / {len(fname_to_id)} w GT")

    # batche, zeby nie zalac pamieci
    B = 64
    for i in range(0, len(img_files), B):
        batch = img_files[i:i + B]
        results = model.predict(batch, imgsz=args.imgsz, conf=args.conf,
                                device=args.device, verbose=False)
        for f, r in zip(batch, results):
            iid = fname_to_id[f.name]
            boxes = r.boxes
            if boxes is None:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), sc, c in zip(xyxy, confs, clss):
                detections.append({
                    "image_id": iid,
                    "category_id": yolo_to_cat.get(int(c), cat_ids[0]),
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(sc),
                })

    print(f"[eval] detekcji: {len(detections)}")
    coco_gt = COCO(args.coco_gt)
    coco_dt = coco_gt.loadRes(detections) if detections else coco_gt.loadRes([])

    ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
    with contextlib.redirect_stdout(io.StringIO()):
        ev.evaluate(); ev.accumulate(); ev.summarize()
    s = ev.stats  # standardowe 12 metryk COCO

    metrics = {
        "AP@[.5:.95]": float(s[0]),
        "AP@.5": float(s[1]),
        "AP@.75": float(s[2]),
        "AP_small": float(s[3]),
        "AP_medium": float(s[4]),
        "AP_large": float(s[5]),
        "AR@1": float(s[6]),
        "AR@10": float(s[7]),
        "AR@100": float(s[8]),
        "AR_small": float(s[9]),
        "AR_medium": float(s[10]),
        "AR_large": float(s[11]),
    }
    out = {"name": args.name, "weights": args.weights, "coco_gt": args.coco_gt,
           "n_images": len(img_files), "n_detections": len(detections),
           "metrics": metrics}

    res_dir = ROOT / "results" / "per_size"
    res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / f"{args.name}.json").write_text(json.dumps(out, indent=2))
    print("\n=== AP per rozmiar (COCO) ===")
    for k, v in metrics.items():
        print(f"  {k:14s}: {v:.4f}")
    print(f"[zapisano] {res_dir / (args.name + '.json')}")


if __name__ == "__main__":
    main()
