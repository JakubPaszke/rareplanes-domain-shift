"""
Trening YOLO z degradacja czestotliwosciowa synthetic on-the-fly.

Zamiast tworzyc osobny katalog PNG dla B1/B2/B3, ten skrypt tymczasowo podmienia
`cv2.imread` w procesie treningu. Gdy Ultralytics wczytuje obraz ze wskazanego
katalogu train, obraz dostaje blur/noise/JPEG w locie. Labelki i struktura YOLO
pozostaja bez zmian.

Uzycie:
  python3 src/train_yolo_freq_onfly.py \
    --data data/yolo/synthetic_10k/data.yaml \
    --degrade-root data/yolo/synthetic_10k/images/train \
    --name expB1_blur_noise_onfly_10k_ml \
    --blur-radius 0.4 --noise-sigma 5 --epochs 60

Uwaga: nie uzywac `--cache disk/ram`, bo cache moglby ominac albo utrwalic
zdegradowane obrazy. Domyslnie cache=None.
"""
import argparse
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def path_contains(parent, child):
    parent_s = str(Path(parent).absolute())
    child_s = str(Path(child).absolute())
    return child_s == parent_s or child_s.startswith(parent_s.rstrip("/") + "/")


def degrade_bgr(img, rng, prob, blur_radius, noise_sigma, jpeg_quality_min):
    if img is None or img.ndim < 3:
        return img
    if rng.random() > prob:
        return img

    import cv2

    out = img
    if blur_radius > 0:
        sigma = float(rng.uniform(0.0, blur_radius))
        if sigma > 0:
            out = cv2.GaussianBlur(out, (0, 0), sigmaX=sigma, sigmaY=sigma)

    if noise_sigma > 0:
        sigma = float(rng.uniform(0.0, noise_sigma))
        if sigma > 0:
            arr = out.astype(np.float32)
            arr += rng.normal(0.0, sigma, size=arr.shape)
            out = np.clip(arr, 0, 255).astype(np.uint8)

    if jpeg_quality_min is not None:
        quality = int(rng.integers(jpeg_quality_min, 101))
        ok, enc = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if ok:
            decoded = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            if decoded is not None:
                out = decoded

    return out


@contextmanager
def patch_cv2_imread(degrade_roots, seed, prob, blur_radius, noise_sigma, jpeg_quality_min):
    import cv2

    roots = [Path(p).absolute() for p in degrade_roots]
    rng = np.random.default_rng(seed)
    original_imread = cv2.imread
    counters = {"seen": 0, "degraded": 0}

    def patched_imread(filename, flags=cv2.IMREAD_COLOR):
        img = original_imread(filename, flags)
        counters["seen"] += 1
        try:
            should_degrade = any(path_contains(root, filename) for root in roots)
        except TypeError:
            should_degrade = False
        if should_degrade:
            counters["degraded"] += 1
            return degrade_bgr(img, rng, prob, blur_radius, noise_sigma, jpeg_quality_min)
        return img

    cv2.imread = patched_imread
    try:
        yield counters
    finally:
        cv2.imread = original_imread


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="data.yaml do treningu")
    ap.add_argument("--name", required=True, help="nazwa eksperymentu")
    ap.add_argument("--model", default="yolov10n.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="0")
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--cache", default=None, help="zostaw None dla on-the-fly")
    ap.add_argument("--val-data", default=None, help="opcjonalny data.yaml do finalnej ewaluacji")

    ap.add_argument("--degrade-root", action="append", default=[],
                    help="katalog obrazow, dla ktorych wlaczamy degradacje; mozna podac kilka razy")
    ap.add_argument("--freq-prob", type=float, default=1.0,
                    help="prawdopodobienstwo degradacji wczytanego obrazu train")
    ap.add_argument("--blur-radius", type=float, default=0.0,
                    help="maksymalny sigma GaussianBlur")
    ap.add_argument("--noise-sigma", type=float, default=0.0,
                    help="maksymalne std szumu Gaussa w pikselach 0-255")
    ap.add_argument("--jpeg-quality-min", type=int, default=None,
                    help="jesli podane, JPEG quality losowane z [min, 100]")

    # Zostawiamy kompatybilnosc z train_yolo.py dla ewentualnych runow.
    ap.add_argument("--hsv_h", type=float, default=None)
    ap.add_argument("--hsv_s", type=float, default=None)
    ap.add_argument("--hsv_v", type=float, default=None)
    ap.add_argument("--lr0", type=float, default=None)
    ap.add_argument("--optimizer", default=None)
    ap.add_argument("--cos_lr", action="store_true")
    ap.add_argument("--warmup_epochs", type=float, default=None)
    args = ap.parse_args()

    if args.cache not in (None, "None", ""):
        raise SystemExit("Dla on-the-fly ustaw --cache None; cache disk/ram omija sens augmentacji.")
    if not (0.0 <= args.freq_prob <= 1.0):
        raise SystemExit("--freq-prob musi byc w zakresie [0, 1]")
    if args.jpeg_quality_min is not None and not (1 <= args.jpeg_quality_min <= 100):
        raise SystemExit("--jpeg-quality-min musi byc w zakresie 1..100")

    data_path = Path(args.data)
    degrade_roots = args.degrade_root or [str(data_path.parent / "images" / "train")]
    print("[onfly] degrade_roots:")
    for root in degrade_roots:
        print(f"  - {Path(root).absolute()}")
    print(
        f"[onfly] prob={args.freq_prob} blur_radius<={args.blur_radius} "
        f"noise_sigma<={args.noise_sigma} jpeg_quality_min={args.jpeg_quality_min}",
        flush=True,
    )

    from ultralytics import YOLO

    aug = {}
    for k in ("hsv_h", "hsv_s", "hsv_v"):
        v = getattr(args, k)
        if v is not None:
            aug[k] = v
    if args.lr0 is not None:
        aug["lr0"] = args.lr0
    if args.optimizer is not None:
        aug["optimizer"] = args.optimizer
    if args.cos_lr:
        aug["cos_lr"] = True
    if args.warmup_epochs is not None:
        aug["warmup_epochs"] = args.warmup_epochs

    model = YOLO(args.model)

    with patch_cv2_imread(
        degrade_roots=degrade_roots,
        seed=args.seed,
        prob=args.freq_prob,
        blur_radius=args.blur_radius,
        noise_sigma=args.noise_sigma,
        jpeg_quality_min=args.jpeg_quality_min,
    ) as counters:
        model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            seed=args.seed,
            device=args.device,
            patience=args.patience,
            workers=args.workers,
            cache=None,
            project=str(ROOT / "runs"),
            name=args.name,
            exist_ok=True,
            **aug,
        )

    print(
        f"[onfly] cv2.imread calls={counters['seen']} "
        f"degraded_train_reads={counters['degraded']}",
        flush=True,
    )

    eval_data = args.val_data or args.data
    eval_split = "test" if "test:" in Path(eval_data).read_text() else "val"
    metrics = model.val(
        data=eval_data,
        imgsz=args.imgsz,
        device=args.device,
        project=str(ROOT / "runs"),
        name=f"{args.name}_eval",
        exist_ok=True,
        split=eval_split,
    )

    out = {
        "name": args.name,
        "train_data": args.data,
        "eval_data": eval_data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "seed": args.seed,
        "augmentation": {
            "type": "frequency_onfly",
            "degrade_roots": [str(Path(p).absolute()) for p in degrade_roots],
            "freq_prob": args.freq_prob,
            "blur_radius": args.blur_radius,
            "noise_sigma": args.noise_sigma,
            "jpeg_quality_min": args.jpeg_quality_min,
            "cv2_imread_calls_parent": counters["seen"],
            "degraded_train_reads_parent": counters["degraded"],
            "note": "worker-process counters are not included when workers>0",
        },
        "metrics": {
            "mAP50": float(metrics.box.map50),
            "mAP50-95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
        },
    }

    res_dir = ROOT / "results" / "baselines"
    res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / f"{args.name}.json").write_text(json.dumps(out, indent=2))
    print("\n=== METRYKI ===")
    print(json.dumps(out["metrics"], indent=2))
    print(f"[zapisano] {res_dir / (args.name + '.json')}")


if __name__ == "__main__":
    main()
