"""
Trening detektora YOLO na RarePlanes (reuzywalny dla real i synthetic).

Jednolity protokol dla wszystkich eksperymentow (baseline'y + warianty A/B/C).
Zapisuje wyniki do runs/ (ultralytics) i kopiuje metryki do results/.

Uzycie:
  python3 src/train_yolo.py --data data/yolo/real_aircraft/data.yaml --name real_baseline
  python3 src/train_yolo.py --data data/yolo/synthetic_aircraft/data.yaml --name syn_baseline \
      --val-data data/yolo/real_aircraft/data.yaml   # ewaluacja na realnym tescie
"""
import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="data.yaml do treningu")
    ap.add_argument("--name", required=True, help="nazwa eksperymentu")
    ap.add_argument("--model", default="models/yolov10n.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="0")
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--val-data", default=None,
                    help="opcjonalny data.yaml do finalnej ewaluacji (cross-domain)")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        seed=args.seed,
        device=args.device,
        patience=args.patience,
        project=str(ROOT / "runs"),
        name=args.name,
        exist_ok=True,
        # augmentacje zostawiamy domyslne ultralytics dla baseline'u;
        # warianty A/B/C beda nadpisywac konkretne flagi.
    )

    # finalna ewaluacja: na wlasnym val albo na wskazanej domenie (cross-domain)
    eval_data = args.val_data or args.data
    metrics = model.val(data=eval_data, imgsz=args.imgsz, device=args.device,
                        project=str(ROOT / "runs"), name=f"{args.name}_eval",
                        exist_ok=True, split="test")

    out = {
        "name": args.name,
        "train_data": args.data,
        "eval_data": eval_data,
        "epochs": args.epochs, "imgsz": args.imgsz, "batch": args.batch, "seed": args.seed,
        "metrics": {
            "mAP50": float(metrics.box.map50),
            "mAP50-95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
        },
    }

    # AP per rozmiar obiektu (small/medium/large) — wymog PDF.
    # Ultralytics udostepnia to przez COCO-style stats, gdy dostepne.
    try:
        results_dict = getattr(metrics, "results_dict", {}) or {}
        for k in results_dict:
            kl = k.lower()
            if "small" in kl or "medium" in kl or "large" in kl or "(s)" in kl or "(m)" in kl or "(l)" in kl:
                out["metrics"][k] = float(results_dict[k])
    except Exception as e:
        out["metrics"]["_per_size_note"] = f"per-size niedostepne: {e}"
    res_dir = ROOT / "results" / "baselines"
    res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / f"{args.name}.json").write_text(json.dumps(out, indent=2))
    print("\n=== METRYKI ===")
    print(json.dumps(out["metrics"], indent=2))
    print(f"[zapisano] {res_dir / (args.name + '.json')}")


if __name__ == "__main__":
    main()
