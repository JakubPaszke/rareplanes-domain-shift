"""
Uruchamia eksperyment B z notebooka colab_expB_frequency_degradation.ipynb.

Pipeline:
  1. Pobiera wymagane adnotacje, real test tiles i synthetic 10k przez HTTPS.
  2. Konwertuje synthetic COCO -> YOLO i tworzy subsety synthetic_10k oraz synthetic_1k.
  3. Dla wariantow B1/B2/B3 trenuje YOLO z degradacja on-the-fly i liczy metryki
     COCO per size na realnym holdoucie.
  4. Zapisuje results/expB_frequency_summary.{csv,md}.

Przyklady:
  python3 src/run_expB_frequency_degradation.py
  python3 src/run_expB_frequency_degradation.py --smoke --epochs 3 --variants B1
  python3 src/run_expB_frequency_degradation.py --skip-prepare --variants B2 B3
  python3 src/run_expB_frequency_degradation.py --mode materialized  # stary wariant: zapis PNG
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RAREPLANES_BASE = "https://rareplanes-public.s3.amazonaws.com"

REAL_TEST_IMG_DIR = ROOT / "data/real/PS-RGB_tiled/PS-RGB_tiled"
REAL_TEST_GT = ROOT / "data/real/annotations/instances_test_aircraft.json"
SYNTHETIC_TRAIN_IMG_DIR = ROOT / "data/synthetic/images/train"


@dataclass(frozen=True)
class Variant:
    label: str
    dataset_suffix: str
    run_prefix: str
    blur_radius: float = 0.0
    noise_sigma: float = 0.0
    jpeg_quality_min: int | None = None


VARIANTS = {
    "B1": Variant(
        label="blur + noise",
        dataset_suffix="b1_blur_noise",
        run_prefix="expB1_blur_noise",
        blur_radius=0.4,
        noise_sigma=5.0,
    ),
    "B2": Variant(
        label="noise",
        dataset_suffix="b2_noise",
        run_prefix="expB2_noise",
        noise_sigma=8.0,
    ),
    "B3": Variant(
        label="blur + noise + JPEG",
        dataset_suffix="b3_blur_noise_jpeg",
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


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    printable = " ".join(cmd)
    print(f"\n[cmd] {printable}", flush=True)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def maybe_install_deps() -> None:
    run([
        sys.executable,
        "-m",
        "pip",
        "install",
        "ultralytics",
        "pycocotools",
        "pillow",
        "pandas",
        "tabulate",
    ])


def ensure_dirs() -> None:
    for path in [
        ROOT / "data/real/annotations",
        ROOT / "data/real/tarballs",
        ROOT / "data/synthetic/annotations",
        SYNTHETIC_TRAIN_IMG_DIR,
        ROOT / "data/real/PS-RGB_tiled",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def download_file(url: str, dst: Path, *, force: bool = False) -> bool:
    if dst.exists() and dst.stat().st_size > 0 and not force:
        print(f"[skip] {rel(dst)} juz istnieje", flush=True)
        return False

    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + ".part")
    if part.exists():
        part.unlink()

    print(f"[download] {url} -> {rel(dst)}", flush=True)
    urllib.request.urlretrieve(url, part)
    os.replace(part, dst)
    return True


def safe_extract_tar(tar_path: Path, dst: Path) -> None:
    dst_resolved = dst.resolve()
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            target = (dst / member.name).resolve()
            try:
                target.relative_to(dst_resolved)
            except ValueError:
                raise RuntimeError(f"Niebezpieczna sciezka w tarballu: {member.name}")
        tar.extractall(dst)


def download_annotations_and_real_test(args: argparse.Namespace) -> None:
    files = [
        (
            f"{RAREPLANES_BASE}/real/metadata_annotations/instances_test_aircraft.json",
            REAL_TEST_GT,
        ),
        (
            f"{RAREPLANES_BASE}/synthetic/metadata_annotations/instances_train_aircraft.json",
            ROOT / "data/synthetic/annotations/instances_train_aircraft.json",
        ),
        (
            f"{RAREPLANES_BASE}/synthetic/metadata_annotations/instances_test_aircraft.json",
            ROOT / "data/synthetic/annotations/instances_test_aircraft.json",
        ),
    ]
    for url, dst in files:
        download_file(url, dst, force=args.force_download)

    tar_path = ROOT / "data/real/tarballs/test.tar.gz"
    download_file(
        f"{RAREPLANES_BASE}/real/tarballs/test/RarePlanes_test_PS-RGB_tiled.tar.gz",
        tar_path,
        force=args.force_download,
    )

    tiles = list(REAL_TEST_IMG_DIR.glob("*.png")) if REAL_TEST_IMG_DIR.exists() else []
    if len(tiles) >= args.min_real_tiles and not args.force_extract:
        print(f"[skip] real test tiles: {len(tiles)}", flush=True)
        return

    print(f"[extract] {rel(tar_path)} -> data/real/PS-RGB_tiled", flush=True)
    safe_extract_tar(tar_path, ROOT / "data/real/PS-RGB_tiled")
    tiles = list(REAL_TEST_IMG_DIR.glob("*.png"))
    print(f"[real] test tiles: {len(tiles)}", flush=True)
    if len(tiles) < args.min_real_tiles:
        raise RuntimeError(
            f"Za malo real test tiles: {len(tiles)} < {args.min_real_tiles}. "
            "Sprawdz pobieranie tarballa."
        )


def synthetic_file_list() -> list[str]:
    files: list[str] = []
    for cfg in [
        ROOT / "configs/synthetic_10k_train_files.txt",
        ROOT / "configs/synthetic_10k_val_files.txt",
    ]:
        with cfg.open() as f:
            files.extend(line.strip() for line in f if line.strip())
    return files


def fetch_synthetic_image(filename: str, *, force: bool = False) -> bool:
    dst = SYNTHETIC_TRAIN_IMG_DIR / filename
    if dst.exists() and dst.stat().st_size > 0 and not force:
        return True

    part = dst.with_name(dst.name + ".part")
    if part.exists():
        part.unlink()
    url = f"{RAREPLANES_BASE}/synthetic/train/images/{filename}"
    try:
        urllib.request.urlretrieve(url, part)
        os.replace(part, dst)
        return True
    except Exception:
        if part.exists():
            part.unlink()
        return False


def download_synthetic_10k(args: argparse.Namespace) -> None:
    files = synthetic_file_list()
    print(f"[synthetic] do pobrania/sprawdzenia: {len(files)}", flush=True)

    ok = 0
    with ThreadPoolExecutor(max_workers=args.synthetic_workers) as pool:
        for i, result in enumerate(
            pool.map(lambda fn: fetch_synthetic_image(fn, force=args.force_download), files),
            start=1,
        ):
            ok += int(result)
            if i % 1000 == 0 or i == len(files):
                print(f"[synthetic] {i}/{len(files)} ok={ok}", flush=True)

    have = len(list(SYNTHETIC_TRAIN_IMG_DIR.glob("*.png")))
    needed = int(len(files) * args.min_synthetic_ratio)
    print(f"[synthetic] ok={ok}/{len(files)}, na dysku={have}, minimum={needed}", flush=True)
    if have < needed:
        raise RuntimeError(
            "Za malo synthetic images. Uruchom skrypt ponownie, pobieranie jest wznawialne."
        )


def prepare_yolo_datasets(args: argparse.Namespace) -> None:
    run([
        sys.executable,
        "src/coco_to_yolo.py",
        "--domain",
        "synthetic",
        "--classes",
        "aircraft",
        "--val-frac",
        str(args.val_frac),
        "--seed",
        str(args.seed),
    ])
    run([
        sys.executable,
        "src/make_subset.py",
        "--n-train",
        "10000",
        "--name",
        "synthetic_10k",
        "--seed",
        str(args.seed),
    ])
    run([
        sys.executable,
        "src/make_subset.py",
        "--n-train",
        "1000",
        "--name",
        "synthetic_1k",
        "--seed",
        str(args.seed),
    ])


def prepare_data(args: argparse.Namespace) -> None:
    ensure_dirs()
    if not args.skip_download:
        download_annotations_and_real_test(args)
        download_synthetic_10k(args)
    prepare_yolo_datasets(args)


def variant_dataset_name(tag: str, variant: Variant) -> str:
    return f"synthetic_{tag}_{variant.dataset_suffix}"


def variant_run_name(tag: str, variant: Variant) -> str:
    return f"{variant.run_prefix}_{tag}_ml"


def variant_onfly_run_name(tag: str, variant: Variant) -> str:
    return f"{variant.run_prefix}_onfly_{tag}_ml"


def run_variant_materialized(key: str, args: argparse.Namespace) -> None:
    variant = VARIANTS[key]
    dataset_name = variant_dataset_name(args.dataset_tag, variant)
    run_name = variant_run_name(args.dataset_tag, variant)
    degraded_yaml = f"data/yolo/{dataset_name}/data.yaml"

    print(f"\n[{key}] {variant.label} (materialized PNG)", flush=True)
    make_cmd = [
        sys.executable,
        "-u",
        "src/make_frequency_degraded_dataset.py",
        "--src",
        args.src_dataset,
        "--name",
        dataset_name,
        "--seed",
        str(args.seed),
        "--overwrite",
    ]
    if variant.blur_radius > 0:
        make_cmd += ["--blur-radius", str(variant.blur_radius)]
    if variant.noise_sigma > 0:
        make_cmd += ["--noise-sigma", str(variant.noise_sigma)]
    if variant.jpeg_quality_min is not None:
        make_cmd += ["--jpeg-quality-min", str(variant.jpeg_quality_min)]
    run(make_cmd)

    train_cmd = [
        sys.executable,
        "src/train_yolo.py",
        "--data",
        degraded_yaml,
        "--name",
        run_name,
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
        "--val-data",
        degraded_yaml,
    ]
    if args.cache is not None:
        train_cmd += ["--cache", args.cache]
    run(train_cmd)

    run([
        sys.executable,
        "src/eval_per_size.py",
        "--weights",
        f"runs/{run_name}/weights/best.pt",
        "--img-dir",
        str(REAL_TEST_IMG_DIR.relative_to(ROOT)),
        "--coco-gt",
        str(REAL_TEST_GT.relative_to(ROOT)),
        "--device",
        str(args.device),
        "--imgsz",
        str(args.imgsz),
        "--name",
        run_name,
    ])

    if args.cleanup_datasets:
        path = ROOT / "data/yolo" / dataset_name
        print(f"[cleanup] {rel(path)}", flush=True)
        shutil.rmtree(path, ignore_errors=True)


def run_variant_onfly(key: str, args: argparse.Namespace) -> None:
    variant = VARIANTS[key]
    run_name = variant_onfly_run_name(args.dataset_tag, variant)
    data_yaml = f"{args.src_dataset}/data.yaml"
    train_img_dir = f"{args.src_dataset}/images/train"

    print(f"\n[{key}] {variant.label} (on-the-fly, bez zapisu PNG)", flush=True)
    train_cmd = [
        sys.executable,
        "-u",
        "src/train_yolo_freq_onfly.py",
        "--data",
        data_yaml,
        "--degrade-root",
        train_img_dir,
        "--name",
        run_name,
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
        train_cmd += ["--blur-radius", str(variant.blur_radius)]
    if variant.noise_sigma > 0:
        train_cmd += ["--noise-sigma", str(variant.noise_sigma)]
    if variant.jpeg_quality_min is not None:
        train_cmd += ["--jpeg-quality-min", str(variant.jpeg_quality_min)]
    run(train_cmd)

    run([
        sys.executable,
        "src/eval_per_size.py",
        "--weights",
        f"runs/{run_name}/weights/best.pt",
        "--img-dir",
        str(REAL_TEST_IMG_DIR.relative_to(ROOT)),
        "--coco-gt",
        str(REAL_TEST_GT.relative_to(ROOT)),
        "--device",
        str(args.device),
        "--imgsz",
        str(args.imgsz),
        "--name",
        run_name,
    ])


def format_metric(value: object, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def write_summary() -> None:
    rows: list[dict[str, object]] = []
    per_size_dir = ROOT / "results/per_size"
    for path in sorted(per_size_dir.glob("expB*_ml.json")):
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
        print("[summary] Brak plikow results/per_size/expB*_ml.json; nie nadpisuje tabel.", flush=True)
        return

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "expB_frequency_summary.csv"
    md_path = out_dir / "expB_frequency_summary.md"

    headers = ["run"] + [label for label, _ in SUMMARY_COLS] + ["n_det", "file"]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    md_lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "---|" * len(headers),
    ]
    for row in rows:
        md_lines.append(
            "| "
            + " | ".join(format_metric(row.get(header)) for header in headers)
            + " |"
        )
    md_path.write_text("\n".join(md_lines) + "\n")

    print(f"\n[zapisano] {rel(csv_path)}", flush=True)
    print(f"[zapisano] {rel(md_path)}", flush=True)
    if rows:
        print("\n=== Podsumowanie ExpB ===", flush=True)
        for row in rows:
            print(
                f"{row['run']}: mAP@50={format_metric(row.get('mAP@50'))}, "
                f"mAP@50:95={format_metric(row.get('mAP@50:95'))}, "
                f"AP_S={format_metric(row.get('AP_S'))}",
                flush=True,
            )
    else:
        print("[summary] Brak plikow results/per_size/expB*_ml.json", flush=True)


def check_gpu(require_gpu: bool) -> None:
    try:
        import torch
    except ImportError:
        print("[gpu] torch nie jest zainstalowany; pomijam sanity check", flush=True)
        if require_gpu:
            raise
        return

    if not torch.cuda.is_available():
        msg = "CUDA niedostepna. Do treningu YOLO wlacz GPU albo podaj --allow-cpu."
        if require_gpu:
            raise RuntimeError(msg)
        print(f"[gpu] {msg}", flush=True)
        return

    print(f"[gpu] CUDA {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}", flush=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Pythonowy odpowiednik notebooks/colab_expB_frequency_degradation.ipynb"
    )
    ap.add_argument("--variants", nargs="+", choices=sorted(VARIANTS), default=["B1", "B2", "B3"])
    ap.add_argument("--smoke", action="store_true", help="uzyj synthetic_1k i domyslnie 3 epok")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default="0")
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--model", default="yolov10n.pt")
    ap.add_argument("--cache", choices=["disk", "ram"], default=None)
    ap.add_argument("--cleanup-datasets", action="store_true")
    ap.add_argument("--mode", choices=["onfly", "materialized"], default="onfly")
    ap.add_argument("--freq-prob", type=float, default=1.0,
                    help="prawdopodobienstwo degradacji obrazu train w trybie onfly")

    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--skip-prepare", action="store_true")
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--install-deps", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true")

    ap.add_argument("--force-download", action="store_true")
    ap.add_argument("--force-extract", action="store_true")
    ap.add_argument("--synthetic-workers", type=int, default=32)
    ap.add_argument("--min-synthetic-ratio", type=float, default=0.99)
    ap.add_argument("--min-real-tiles", type=int, default=2500)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--src-dataset", default=None)
    ap.add_argument("--dataset-tag", default=None)
    args = ap.parse_args()

    if args.epochs is None:
        args.epochs = 3 if args.smoke else 60
    if args.src_dataset is None:
        args.src_dataset = "data/yolo/synthetic_1k" if args.smoke else "data/yolo/synthetic_10k"
    if args.dataset_tag is None:
        if args.mode == "onfly":
            args.dataset_tag = "1k_smoke_onfly" if args.smoke else "10k_onfly"
        else:
            args.dataset_tag = "1k_smoke" if args.smoke else "10k"
    if not 0 < args.min_synthetic_ratio <= 1:
        ap.error("--min-synthetic-ratio musi byc w zakresie (0, 1]")
    if not 0 <= args.freq_prob <= 1:
        ap.error("--freq-prob musi byc w zakresie [0, 1]")
    if args.mode == "onfly" and args.cache is not None:
        ap.error("--cache nie ma sensu w trybie onfly; cache moglby ominac degradacje")

    return args


def main() -> None:
    args = parse_args()

    if args.install_deps:
        maybe_install_deps()

    if args.summary_only:
        write_summary()
        return

    print(
        "[config] "
        f"mode={args.mode} "
        f"variants={' '.join(args.variants)} "
        f"src_dataset={args.src_dataset} dataset_tag={args.dataset_tag} "
        f"epochs={args.epochs} batch={args.batch} workers={args.workers} device={args.device}",
        flush=True,
    )

    if not args.skip_prepare:
        prepare_data(args)
    if args.prepare_only:
        return

    check_gpu(require_gpu=not args.allow_cpu)

    for key in args.variants:
        if args.mode == "onfly":
            run_variant_onfly(key, args)
        else:
            run_variant_materialized(key, args)

    write_summary()


if __name__ == "__main__":
    main()
