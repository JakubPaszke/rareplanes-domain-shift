"""
Cluster-friendly runner dla eksperymentu B on-the-fly.

Nie tworzy zdegradowanych PNG. Kazdy wariant odpala:
  train_yolo_freq_onfly.py -> eval_per_size.py

Zaklada, ze dane YOLO juz istnieja, np.:
  data/yolo/synthetic_10k/data.yaml
  data/real/PS-RGB_tiled/PS-RGB_tiled/
  data/real/annotations/instances_test_aircraft.json

Przyklady:
  python3 src/run_expB_onfly_cluster.py
  python3 src/run_expB_onfly_cluster.py --variants B1 --epochs 3 --dataset-tag 1k_smoke --src-dataset data/yolo/synthetic_1k
  python3 src/run_expB_onfly_cluster.py --variants B2 B3 --batch 32 --device 0
  python3 src/run_expB_onfly_cluster.py --dry-run

Na SLURM zwykle odpalasz ten plik wewnatrz joba po aktywacji venv/conda.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Variant:
    label: str
    run_prefix: str
    blur_radius: float = 0.0
    noise_sigma: float = 0.0
    jpeg_quality_min: int | None = None


VARIANTS = {
    "B1": Variant(
        label="blur + noise",
        run_prefix="expB1_blur_noise",
        blur_radius=0.4,
        noise_sigma=5.0,
    ),
    "B2": Variant(
        label="noise",
        run_prefix="expB2_noise",
        noise_sigma=8.0,
    ),
    "B3": Variant(
        label="blur + noise + JPEG",
        run_prefix="expB3_blur_noise_jpeg",
        blur_radius=0.6,
        noise_sigma=6.0,
        jpeg_quality_min=75,
    ),
}

SUMMARY_COLS = [
    ("mAP@50", "AP@.5"),
    ("mAP@50:95", "AP@[.5:.95]"),
    ("AP_S", "AP_small"),
    ("AP_M", "AP_medium"),
    ("AP_L", "AP_large"),
    ("AR@100", "AR@100"),
]


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(message: str = "") -> None:
    print(f"[{now()}] {message}", flush=True)


def section(title: str) -> None:
    bar = "=" * 78
    print(f"\n{bar}\n[{now()}] {title}\n{bar}", flush=True)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def resolve_repo_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return ROOT / path


def count_pngs(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for _ in path.glob("*.png"))


def describe_path(label: str, path: Path) -> None:
    exists = path.exists()
    kind = "dir" if path.is_dir() else "file" if path.is_file() else "missing"
    suffix = ""
    if path.is_file():
        suffix = f", size={path.stat().st_size / (1024 ** 2):.1f} MB"
    elif path.is_dir():
        try:
            usage = shutil.disk_usage(path)
            suffix = (
                f", disk_free={usage.free / (1024 ** 3):.1f} GB, "
                f"disk_total={usage.total / (1024 ** 3):.1f} GB"
            )
        except OSError:
            suffix = ""
    log(f"{label}: {rel(path)} [{kind}, exists={exists}{suffix}]")


def run_capture(cmd: list[str]) -> str | None:
    try:
        out = subprocess.check_output(cmd, cwd=ROOT, stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except Exception as exc:
        return f"<nie udalo sie odpalic {' '.join(cmd)}: {exc}>"


def print_environment(args: argparse.Namespace) -> None:
    section("Srodowisko")
    log(f"cwd={ROOT}")
    log(f"python={sys.executable}")
    log(f"python_version={platform.python_version()}")
    log(f"platform={platform.platform()}")
    log(f"SLURM_JOB_ID={os.environ.get('SLURM_JOB_ID', '<brak>')}")
    log(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<brak>')}")

    describe_path("repo", ROOT)
    describe_path("src_dataset", resolve_repo_path(args.src_dataset))
    describe_path("train_images", resolve_repo_path(args.src_dataset) / "images/train")
    describe_path("real_img_dir", resolve_repo_path(args.real_img_dir))
    describe_path("coco_gt", resolve_repo_path(args.coco_gt))
    describe_path("runs_dir", ROOT / "runs")
    describe_path("results_dir", ROOT / "results")

    log("nvidia-smi:")
    nvidia = run_capture(["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"])
    print(nvidia, flush=True)

    log("torch/ultralytics:")
    code = (
        "import importlib.util\n"
        "for name in ['torch','ultralytics','cv2','numpy','pycocotools']:\n"
        "    spec = importlib.util.find_spec(name)\n"
        "    print(f'{name}: ' + ('OK' if spec else 'BRAK'))\n"
        "try:\n"
        "    import torch\n"
        "    print('torch.version=' + str(torch.__version__))\n"
        "    print('cuda_available=' + str(torch.cuda.is_available()))\n"
        "    if torch.cuda.is_available(): print('gpu=' + torch.cuda.get_device_name(0))\n"
        "except Exception as e:\n"
        "    print('torch_check_error=' + repr(e))\n"
    )
    print(run_capture([sys.executable, "-c", code]), flush=True)


def print_plan(args: argparse.Namespace) -> None:
    section("Plan eksperymentu B on-the-fly")
    log(f"variants={' '.join(args.variants)}")
    log(f"src_dataset={args.src_dataset}")
    log(f"dataset_tag={args.dataset_tag}")
    log(f"model={args.model}")
    log(
        f"epochs={args.epochs}, batch={args.batch}, imgsz={args.imgsz}, "
        f"workers={args.workers}, device={args.device}, patience={args.patience}"
    )
    log(f"freq_prob={args.freq_prob}, seed={args.seed}")
    log(f"skip_train={args.skip_train}, skip_eval={args.skip_eval}, dry_run={args.dry_run}")
    for key in args.variants:
        variant = VARIANTS[key]
        log(
            f"{key}: {variant.label}; run={run_name(variant, args.dataset_tag)}; "
            f"blur<={variant.blur_radius}, noise<={variant.noise_sigma}, "
            f"jpeg_quality_min={variant.jpeg_quality_min}"
        )


def run(cmd: list[str], *, dry_run: bool = False) -> None:
    log("[cmd] " + " ".join(cmd))
    if not dry_run:
        start = time.time()
        subprocess.run(cmd, cwd=ROOT, check=True)
        log(f"[cmd done] czas={time.time() - start:.1f}s")
    else:
        log("[dry-run] komenda nie zostala uruchomiona")


def run_name(variant: Variant, dataset_tag: str) -> str:
    return f"{variant.run_prefix}_onfly_{dataset_tag}_ml"


def check_required_paths(args: argparse.Namespace) -> None:
    section("Sprawdzanie danych wejsciowych")
    required = [
        resolve_repo_path(args.src_dataset) / "data.yaml",
        resolve_repo_path(args.src_dataset) / "images/train",
    ]
    if not args.skip_eval:
        required.extend([
            resolve_repo_path(args.real_img_dir),
            resolve_repo_path(args.coco_gt),
        ])

    for path in required:
        describe_path("required", path)

    missing = [path for path in required if not path.exists()]
    if missing:
        msg = "\n".join(f"  - {rel(path)}" for path in missing)
        raise SystemExit(f"Brakuje wymaganych sciezek:\n{msg}")

    train_count = count_pngs(resolve_repo_path(args.src_dataset) / "images/train")
    log(f"synthetic train PNG count={train_count}")
    if train_count == 0:
        raise SystemExit("Brak obrazow PNG w src_dataset/images/train.")

    if not args.skip_eval:
        real_count = count_pngs(resolve_repo_path(args.real_img_dir))
        log(f"real test PNG count={real_count}")
        if real_count == 0:
            raise SystemExit("Brak obrazow PNG w real-img-dir.")

    log("Dane wejsciowe wygladaja OK.")


def train_variant(key: str, args: argparse.Namespace) -> str:
    variant = VARIANTS[key]
    name = run_name(variant, args.dataset_tag)
    data_yaml = f"{args.src_dataset}/data.yaml"
    degrade_root = f"{args.src_dataset}/images/train"

    section(f"{key}: trening on-the-fly")
    log(f"wariant={variant.label}")
    log(f"run_name={name}")
    log(f"data_yaml={data_yaml}")
    log(f"degrade_root={degrade_root}")
    log("Degradacja dzieje sie w RAM podczas czytania obrazow; nie zapisujemy dodatkowych PNG.")
    cmd = [
        sys.executable,
        "-u",
        "src/train_yolo_freq_onfly.py",
        "--data",
        data_yaml,
        "--degrade-root",
        degrade_root,
        "--name",
        name,
        "--model",
        args.model,
        "--epochs",
        str(args.epochs),
        "--batch",
        str(args.batch),
        "--imgsz",
        str(args.imgsz),
        "--seed",
        str(args.seed),
        "--device",
        str(args.device),
        "--workers",
        str(args.workers),
        "--patience",
        str(args.patience),
        "--freq-prob",
        str(args.freq_prob),
        "--val-data",
        data_yaml,
    ]
    if variant.blur_radius > 0:
        cmd += ["--blur-radius", str(variant.blur_radius)]
    if variant.noise_sigma > 0:
        cmd += ["--noise-sigma", str(variant.noise_sigma)]
    if variant.jpeg_quality_min is not None:
        cmd += ["--jpeg-quality-min", str(variant.jpeg_quality_min)]

    run(cmd, dry_run=args.dry_run or args.skip_train)

    best_pt = ROOT / "runs" / name / "weights/best.pt"
    if args.skip_train or args.dry_run:
        log(f"pomijam sprawdzanie checkpointu dla {name}")
    elif best_pt.exists():
        log(f"checkpoint OK: {rel(best_pt)} ({best_pt.stat().st_size / (1024 ** 2):.1f} MB)")
    else:
        raise SystemExit(f"Trening zakonczony, ale nie widze checkpointu: {rel(best_pt)}")
    return name


def eval_variant(name: str, args: argparse.Namespace) -> None:
    section(f"{name}: ewaluacja real holdout")
    log(f"weights=runs/{name}/weights/best.pt")
    log(f"real_img_dir={args.real_img_dir}")
    log(f"coco_gt={args.coco_gt}")
    cmd = [
        sys.executable,
        "src/eval_per_size.py",
        "--weights",
        f"runs/{name}/weights/best.pt",
        "--img-dir",
        args.real_img_dir,
        "--coco-gt",
        args.coco_gt,
        "--device",
        str(args.device),
        "--imgsz",
        str(args.imgsz),
        "--name",
        name,
    ]
    run(cmd, dry_run=args.dry_run or args.skip_eval)

    out_json = ROOT / "results/per_size" / f"{name}.json"
    if args.skip_eval or args.dry_run:
        log(f"pomijam sprawdzanie wyniku eval dla {name}")
    elif out_json.exists():
        log(f"eval JSON OK: {rel(out_json)}")
        try:
            data = json.loads(out_json.read_text())
            metrics = data.get("metrics", {})
            log(
                "metryki: "
                f"AP@.5={format_metric(metrics.get('AP@.5'))}, "
                f"AP@[.5:.95]={format_metric(metrics.get('AP@[.5:.95]'))}, "
                f"AP_small={format_metric(metrics.get('AP_small'))}, "
                f"AP_medium={format_metric(metrics.get('AP_medium'))}, "
                f"AP_large={format_metric(metrics.get('AP_large'))}"
            )
        except Exception as exc:
            log(f"nie udalo sie odczytac metryk z JSON: {exc}")
    else:
        raise SystemExit(f"Ewaluacja zakonczona, ale nie widze wyniku: {rel(out_json)}")


def format_metric(value: object, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def write_summary() -> None:
    section("Podsumowanie wynikow")
    per_size_dir = ROOT / "results/per_size"
    rows: list[dict[str, object]] = []

    for path in sorted(per_size_dir.glob("expB*_onfly*_ml.json")):
        with path.open() as f:
            data = json.load(f)
        metrics = data.get("metrics", {})
        row: dict[str, object] = {
            "run": data.get("name", path.stem),
            "n_det": data.get("n_detections"),
            "file": str(path.relative_to(ROOT)),
        }
        for label, key in SUMMARY_COLS:
            row[label] = metrics.get(key)
        rows.append(row)

    if not rows:
        log("[summary] Brak plikow results/per_size/expB*_onfly*_ml.json")
        return

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "expB_onfly_summary.csv"
    md_path = out_dir / "expB_onfly_summary.md"
    headers = ["run"] + [label for label, _ in SUMMARY_COLS] + ["n_det", "file"]

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    md_lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "---|" * len(headers),
    ]
    for row in rows:
        md_lines.append(
            "| " + " | ".join(format_metric(row.get(header)) for header in headers) + " |"
        )
    md_path.write_text("\n".join(md_lines) + "\n")

    log(f"[zapisano] {rel(csv_path)}")
    log(f"[zapisano] {rel(md_path)}")
    log("Najwazniejsze wiersze:")
    for row in rows:
        log(
            f"{row['run']}: mAP@50={format_metric(row.get('mAP@50'))}, "
            f"mAP@50:95={format_metric(row.get('mAP@50:95'))}, "
            f"AP_S={format_metric(row.get('AP_S'))}, "
            f"AP_M={format_metric(row.get('AP_M'))}, "
            f"AP_L={format_metric(row.get('AP_L'))}"
        )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Runner eksperymentu B on-the-fly na klaster.")
    ap.add_argument("--variants", nargs="+", choices=sorted(VARIANTS), default=["B1", "B2", "B3"])
    ap.add_argument("--src-dataset", default="data/yolo/synthetic_10k")
    ap.add_argument("--dataset-tag", default="10k_onfly")
    ap.add_argument("--real-img-dir", default="data/real/PS-RGB_tiled/PS-RGB_tiled")
    ap.add_argument("--coco-gt", default="data/real/annotations/instances_test_aircraft.json")

    ap.add_argument("--model", default="yolov10n.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--freq-prob", type=float, default=1.0)

    ap.add_argument("--skip-train", action="store_true", help="nie trenuj, tylko wypisz/odpal eval")
    ap.add_argument("--skip-eval", action="store_true", help="nie odpalaj eval_per_size.py")
    ap.add_argument("--no-summary", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="tylko wypisz komendy")
    args = ap.parse_args()

    if not 0 <= args.freq_prob <= 1:
        ap.error("--freq-prob musi byc w zakresie [0, 1]")
    if args.workers < 0:
        ap.error("--workers musi byc >= 0")
    return args


def main() -> None:
    args = parse_args()
    start = time.time()
    section("START run_expB_onfly_cluster.py")
    print_plan(args)
    print_environment(args)

    if not args.dry_run:
        check_required_paths(args)

    for key in args.variants:
        name = train_variant(key, args)
        if not args.skip_eval:
            eval_variant(name, args)

    if not args.no_summary and not args.dry_run and not args.skip_eval:
        write_summary()

    section("KONIEC")
    log(f"caly czas wall-clock={time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
