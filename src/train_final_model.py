"""
Train the final RarePlanes model by combining the best-tested interventions.

Default recipe, derived from docs/RAPORT.md and notes/06-09:
  C: mixed training with full synthetic train + 25% real train/val,
  D: imgsz=320,
  B: noise-only synthetic degradation, materialized as files,
  A: weak HSV jitter (hsv_s=0.4, hsv_v=0.3),
  architecture: YOLOv10n/CNN.

The real test split is never linked into the training dataset. It is used only
by eval_per_size.py for the final COCO holdout evaluation.

Typical cluster usage:
  python src/train_final_model.py --data-dir /work/$USER/rareplanes-data/data

Smoke test:
  python src/train_final_model.py --smoke --data-dir /work/$USER/rareplanes-data/data

Outputs:
  runs/final_yolov10n_syn45k_noise_real25pct_img320_hsvA1_ml/
  results/baselines/final_yolov10n_syn45k_noise_real25pct_img320_hsvA1_ml.json
  results/per_size/final_yolov10n_syn45k_noise_real25pct_img320_hsvA1_ml.json
  results/final_combined_model_summary.{json,md}
  results/final_combined_benchmark.json
  results/final_combined_run.log
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RAREPLANES_BASE = "https://rareplanes-public.s3.amazonaws.com"
LOG_HANDLE = None


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(message: str = "") -> None:
    line = f"[{now()}] {message}"
    print(line, flush=True)
    if LOG_HANDLE is not None:
        LOG_HANDLE.write(line + "\n")
        LOG_HANDLE.flush()


def section(title: str) -> None:
    bar = "=" * 78
    text = f"\n{bar}\n[{now()}] {title}\n{bar}"
    print(text, flush=True)
    if LOG_HANDLE is not None:
        LOG_HANDLE.write(text + "\n")
        LOG_HANDLE.flush()


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


def run(cmd: list[str], *, dry_run: bool = False) -> None:
    log("[cmd] " + " ".join(cmd))
    if dry_run:
        log("[dry-run] command skipped")
        return

    start = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        if LOG_HANDLE is not None:
            LOG_HANDLE.write(line)
            LOG_HANDLE.flush()
    rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)
    log(f"[cmd done] elapsed={time.time() - start:.1f}s")


def count_pngs(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for _ in path.glob("*.png"))


def uses_default_full_synthetic(args: argparse.Namespace) -> bool:
    return (not args.smoke) and args.synthetic_src == "data/yolo/synthetic_aircraft"


def synthetic_train_files_from_annotations() -> list[str]:
    ann_path = ROOT / "data/synthetic/annotations/instances_train_aircraft.json"
    if not ann_path.exists():
        raise SystemExit(
            "Missing synthetic train annotations. Run without --skip-prepare first, "
            f"or provide the file at {rel(ann_path)}."
        )
    data = json.loads(ann_path.read_text())
    return sorted({img["file_name"] for img in data.get("images", [])})


def fetch_synthetic_train_image(filename: str, *, force: bool) -> bool:
    dst = ROOT / "data/synthetic/images/train" / filename
    if dst.exists() and dst.stat().st_size > 0 and not force:
        return True

    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + ".part")
    if part.exists():
        part.unlink()

    url = f"{RAREPLANES_BASE}/synthetic/train/images/{filename}"
    try:
        urllib.request.urlretrieve(url, part)
        part.replace(dst)
        return True
    except Exception:
        if part.exists():
            part.unlink()
        return False


def final_run_name(args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name
    syn_tag = "syn1k" if args.smoke else "syn45k"
    noise_tag = "clean" if args.no_noise else "noise"
    return (
        f"final_yolov10n_{syn_tag}_{noise_tag}_real{args.real_pct}pct_"
        f"img{args.imgsz}_hsvA1_ml"
    )


def noise_dataset_name(args: argparse.Namespace) -> str:
    if args.noise_dataset_name:
        return args.noise_dataset_name
    syn_tag = "syn1k" if args.smoke else "syn45k"
    return f"final_{syn_tag}_b2_noise"


def mixed_dataset_name(args: argparse.Namespace) -> str:
    if args.mixed_dataset_name:
        return args.mixed_dataset_name
    syn_tag = "syn1k" if args.smoke else "syn45k"
    noise_tag = "clean" if args.no_noise else "noise"
    return f"final_mixed_{syn_tag}_{noise_tag}_real{args.real_pct}pct"


def prepare_data(args: argparse.Namespace) -> None:
    section("Prepare base data")
    if args.skip_prepare:
        log("Skipping prepare step (--skip-prepare)")
        return

    cmd = [
        sys.executable,
        "src/expC.py",
        "--prepare-only",
        "--data-dir",
        args.data_dir,
        "--batch",
        str(args.batch),
        "--workers",
        str(args.workers),
        "--device",
        str(args.device),
        "--imgsz",
        str(args.imgsz),
        "--seed",
        str(args.seed),
    ]
    if args.force_download:
        cmd.append("--force-download")
    if args.force_extract:
        cmd.append("--force-extract")
    run(cmd, dry_run=args.dry_run)


def prepare_full_synthetic(args: argparse.Namespace) -> None:
    section("Prepare full synthetic training set")
    if not uses_default_full_synthetic(args):
        log("Skipping full synthetic preparation; synthetic_src is not the default full dataset")
        return
    if args.skip_full_synthetic_download:
        log("Skipping full synthetic image download (--skip-full-synthetic-download)")
    else:
        ann_path = ROOT / "data/synthetic/annotations/instances_train_aircraft.json"
        if args.dry_run and not ann_path.exists():
            log(f"[dry-run] annotations not present yet: {rel(ann_path)}")
            files = []
        else:
            files = synthetic_train_files_from_annotations()
        dst_dir = ROOT / "data/synthetic/images/train"
        present = sum(1 for fn in files if (dst_dir / fn).exists() and (dst_dir / fn).stat().st_size > 0)
        missing = len(files) - present
        log(
            f"synthetic train images from annotations={len(files)}, "
            f"present={present}, missing={missing}"
        )
        if args.dry_run:
            log("[dry-run] full synthetic download skipped")
        elif missing:
            ok = 0
            with ThreadPoolExecutor(max_workers=args.full_download_workers) as pool:
                iterator = pool.map(
                    lambda fn: fetch_synthetic_train_image(fn, force=args.force_download),
                    files,
                )
                for i, result in enumerate(iterator, start=1):
                    ok += int(result)
                    if i % args.full_progress_every == 0 or i == len(files):
                        log(f"synthetic full download progress {i}/{len(files)} ok={ok}")
            present = sum(
                1 for fn in files
                if (dst_dir / fn).exists() and (dst_dir / fn).stat().st_size > 0
            )
            needed = int(len(files) * args.min_full_synthetic_ratio)
            log(f"synthetic full present={present}/{len(files)} minimum={needed}")
            if present < needed:
                raise RuntimeError(
                    f"Too few full synthetic images: {present}/{len(files)} < {needed}"
                )
        else:
            log("All full synthetic train images are already present")

    cmd = [
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
    ]
    run(cmd, dry_run=args.dry_run)


def check_required_paths(args: argparse.Namespace) -> None:
    section("Check required paths")
    required = [
        resolve_repo_path(args.synthetic_src) / "data.yaml",
        resolve_repo_path(args.synthetic_src) / "images/train",
        resolve_repo_path(args.synthetic_src) / "images/val",
        resolve_repo_path(args.real_src) / "data.yaml",
        resolve_repo_path(args.real_src) / "images/train",
        resolve_repo_path(args.real_src) / "images/val",
        resolve_repo_path(args.real_img_dir),
        resolve_repo_path(args.coco_gt),
    ]
    for path in required:
        kind = "dir" if path.is_dir() else "file" if path.is_file() else "missing"
        log(f"{rel(path)} [{kind}] exists={path.exists()}")
    missing = [path for path in required if not path.exists()]
    if missing:
        msg = "\n".join(f"  - {rel(path)}" for path in missing)
        raise SystemExit(f"Missing required paths:\n{msg}")

    log(f"synthetic train PNG={count_pngs(resolve_repo_path(args.synthetic_src) / 'images/train')}")
    log(f"synthetic val PNG={count_pngs(resolve_repo_path(args.synthetic_src) / 'images/val')}")
    log(f"real train PNG={count_pngs(resolve_repo_path(args.real_src) / 'images/train')}")
    log(f"real val PNG={count_pngs(resolve_repo_path(args.real_src) / 'images/val')}")
    log(f"real holdout PNG={count_pngs(resolve_repo_path(args.real_img_dir))}")

    if uses_default_full_synthetic(args):
        expected = len(synthetic_train_files_from_annotations())
        actual = (
            count_pngs(resolve_repo_path(args.synthetic_src) / "images/train")
            + count_pngs(resolve_repo_path(args.synthetic_src) / "images/val")
        )
        needed = int(expected * args.min_full_synthetic_ratio)
        log(f"full synthetic YOLO train+val={actual}/{expected} minimum={needed}")
        if actual < needed:
            raise SystemExit(
                "Full synthetic dataset is incomplete. Run without "
                "--skip-full-synthetic-download, or lower --min-full-synthetic-ratio."
            )


def make_noise_dataset(args: argparse.Namespace) -> str:
    section("Apply experiment B2: materialized noise-only synthetic")
    if args.no_noise:
        log("Noise disabled (--no-noise); using clean synthetic source")
        return args.synthetic_src

    dataset = noise_dataset_name(args)
    cmd = [
        sys.executable,
        "src/make_frequency_degraded_dataset.py",
        "--src",
        args.synthetic_src,
        "--name",
        dataset,
        "--noise-sigma",
        str(args.noise_sigma),
        "--seed",
        str(args.seed),
        "--overwrite",
    ]
    run(cmd, dry_run=args.dry_run or args.skip_noise)

    out_yaml = ROOT / "data/yolo" / dataset / "data.yaml"
    if not args.dry_run and not args.skip_noise and not out_yaml.exists():
        raise SystemExit(f"Noise dataset was not created: {rel(out_yaml)}")
    if args.skip_noise:
        log("Skipping noise creation; assuming dataset already exists")
    return f"data/yolo/{dataset}"


def make_mixed_dataset(synthetic_src: str, args: argparse.Namespace) -> str:
    section("Apply experiment C: mixed synthetic + real")
    dataset = mixed_dataset_name(args)
    real_frac = args.real_pct / 100.0
    cmd = [
        sys.executable,
        "src/make_mixed_dataset.py",
        "--syn-src",
        synthetic_src,
        "--real-src",
        args.real_src,
        "--name",
        dataset,
        "--real-frac",
        str(real_frac),
        "--seed",
        str(args.seed),
        "--overwrite",
    ]
    run(cmd, dry_run=args.dry_run or args.skip_make)

    out_yaml = ROOT / "data/yolo" / dataset / "data.yaml"
    if not args.dry_run and not args.skip_make and not out_yaml.exists():
        raise SystemExit(f"Mixed dataset was not created: {rel(out_yaml)}")
    if args.skip_make:
        log("Skipping mixed dataset creation; assuming dataset already exists")
    return dataset


def train_final(dataset: str, args: argparse.Namespace) -> str:
    section("Train final model")
    name = final_run_name(args)
    data_yaml = f"data/yolo/{dataset}/data.yaml"
    cmd = [
        sys.executable,
        "src/train_yolo.py",
        "--data",
        data_yaml,
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
        "--hsv_h",
        str(args.hsv_h),
        "--hsv_s",
        str(args.hsv_s),
        "--hsv_v",
        str(args.hsv_v),
        "--val-data",
        data_yaml,
    ]
    if args.cache is not None:
        cmd += ["--cache", args.cache]
    run(cmd, dry_run=args.dry_run or args.skip_train)

    best_pt = ROOT / "runs" / name / "weights/best.pt"
    if not args.dry_run and not args.skip_train and not best_pt.exists():
        raise SystemExit(f"Training finished but best.pt is missing: {rel(best_pt)}")
    return name


def eval_final(name: str, args: argparse.Namespace) -> None:
    section("Evaluate on real holdout")
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
    if not args.dry_run and not args.skip_eval and not out_json.exists():
        raise SystemExit(f"Evaluation JSON is missing: {rel(out_json)}")


def benchmark_final(name: str, args: argparse.Namespace) -> None:
    section("Benchmark final model")
    if args.skip_benchmark or args.dry_run:
        log("Skipping benchmark")
        return

    code = f"""
import json
import time
from pathlib import Path

import torch
from ultralytics import YOLO

root = Path({str(ROOT)!r})
img_dir = root / {args.real_img_dir!r}
files = sorted(img_dir.glob('*.png'))[:{args.benchmark_images}]
model = YOLO(str(root / 'runs/{name}/weights/best.pt'))

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

warmup = files[:min(16, len(files))]
if warmup:
    model.predict(warmup, imgsz={args.imgsz}, device={args.device!r}, verbose=False, batch={args.benchmark_batch})
    if torch.cuda.is_available():
        torch.cuda.synchronize()

start = time.perf_counter()
if files:
    model.predict(files, imgsz={args.imgsz}, device={args.device!r}, verbose=False, batch={args.benchmark_batch})
    if torch.cuda.is_available():
        torch.cuda.synchronize()
elapsed = time.perf_counter() - start
fps = len(files) / elapsed if elapsed > 0 else None
peak_mem_mb = None
if torch.cuda.is_available():
    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

out = {{
    'name': {name!r},
    'weights': f'runs/{name}/weights/best.pt',
    'imgsz': {args.imgsz},
    'n_images': len(files),
    'elapsed_s': elapsed,
    'fps': fps,
    'batch': {args.benchmark_batch},
    'device': {args.device!r},
    'peak_cuda_memory_mb': peak_mem_mb,
}}
(root / 'results').mkdir(exist_ok=True)
(root / 'results/final_combined_benchmark.json').write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
"""
    run([sys.executable, "-c", code], dry_run=False)


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def metric(metrics: dict, key: str) -> object:
    return metrics.get(key)


def write_summary(name: str, synthetic_src: str, dataset: str, args: argparse.Namespace) -> None:
    section("Write final summary")
    if args.dry_run:
        return

    per_size = read_json(ROOT / "results/per_size" / f"{name}.json")
    baseline = read_json(ROOT / "results/baselines" / f"{name}.json")
    benchmark = read_json(ROOT / "results/final_combined_benchmark.json")
    expc_25 = read_json(ROOT / "results/per_size/expC_25pct_real_10k_ml.json")
    real_upper = read_json(ROOT / "results/per_size/real_baseline.json")

    summary = {
        "name": name,
        "recipe": {
            "architecture": args.model,
            "synthetic_base": args.synthetic_src,
            "synthetic_after_expB": synthetic_src,
            "mixed_dataset": f"data/yolo/{dataset}",
            "real_fraction": args.real_pct / 100.0,
            "imgsz": args.imgsz,
            "epochs": args.epochs,
            "batch": args.batch,
            "seed": args.seed,
            "noise_sigma": None if args.no_noise else args.noise_sigma,
            "hsv_h": args.hsv_h,
            "hsv_s": args.hsv_s,
            "hsv_v": args.hsv_v,
        },
        "motivation": {
            "C": "mixed training was the strongest intervention in RAPORT.md",
            "D": "imgsz=320 was best in the scale sweep",
            "B": "noise-only helped; blur variants did not",
            "A": "weak HSV was the best photometric setting",
        },
        "mixed_val_metrics": baseline.get("metrics"),
        "real_holdout_metrics": per_size.get("metrics"),
        "comparison": {
            "best_measured_expC_25pct_img512": expc_25.get("metrics"),
            "real_to_real_upper_reference": real_upper.get("metrics"),
        },
        "benchmark": benchmark,
    }

    out_json = ROOT / "results/final_combined_model_summary.json"
    out_md = ROOT / "results/final_combined_model_summary.md"
    out_json.write_text(json.dumps(summary, indent=2))

    metrics = summary.get("real_holdout_metrics") or {}
    expc_metrics = (summary.get("comparison") or {}).get("best_measured_expC_25pct_img512") or {}
    upper_metrics = (summary.get("comparison") or {}).get("real_to_real_upper_reference") or {}
    bench = summary.get("benchmark") or {}

    md = [
        "# Final combined model summary",
        "",
        f"- run: `{name}`",
        f"- dataset: `data/yolo/{dataset}`",
        f"- recipe: full synthetic + C mixed {args.real_pct}% real + D imgsz={args.imgsz} + "
        f"B2 noise_sigma={None if args.no_noise else args.noise_sigma} + "
        f"A1 HSV=({args.hsv_h}, {args.hsv_s}, {args.hsv_v})",
        "- note: real test holdout is used only for final evaluation.",
        "",
        "| metric | final full | expC 25% 10k measured | real->real upper |",
        "|---|---:|---:|---:|",
    ]
    metric_rows = [
        ("AP@.5", "AP@.5"),
        ("AP@[.5:.95]", "AP@[.5:.95]"),
        ("AP_small", "AP_small"),
        ("AP_medium", "AP_medium"),
        ("AP_large", "AP_large"),
        ("AR@100", "AR@100"),
    ]
    for label, key in metric_rows:
        vals = []
        for source in (metrics, expc_metrics, upper_metrics):
            value = metric(source, key)
            vals.append(f"{value:.4f}" if isinstance(value, (int, float)) else "")
        md.append(f"| {label} | {vals[0]} | {vals[1]} | {vals[2]} |")
    if bench:
        fps = bench.get("fps")
        mem = bench.get("peak_cuda_memory_mb")
        md.extend([
            "",
            "## Benchmark",
            "",
            f"- images: {bench.get('n_images')}",
            f"- fps: {fps:.2f}" if isinstance(fps, (int, float)) else "- fps: ",
            f"- peak CUDA memory MB: {mem:.1f}" if isinstance(mem, (int, float)) else "- peak CUDA memory MB: ",
        ])
    out_md.write_text("\n".join(md) + "\n")
    log(f"[saved] {rel(out_json)}")
    log(f"[saved] {rel(out_md)}")


def print_plan(args: argparse.Namespace) -> None:
    section("Plan")
    log("Final recipe = C(mixed real) + D(imgsz 320) + B2(noise-only files) + A1(weak HSV)")
    log("Real test holdout is used only by eval_per_size.py")
    log(f"data_dir={args.data_dir}")
    log(f"smoke={args.smoke}")
    log(f"run_name={final_run_name(args)}")
    log(f"synthetic_src={args.synthetic_src}")
    log(f"noise_dataset={noise_dataset_name(args) if not args.no_noise else '<disabled>'}")
    log(f"mixed_dataset={mixed_dataset_name(args)}")
    log(f"real_pct={args.real_pct}, noise_sigma={None if args.no_noise else args.noise_sigma}")
    log(f"model={args.model}, epochs={args.epochs}, batch={args.batch}, workers={args.workers}")
    log(f"device={args.device}, imgsz={args.imgsz}, seed={args.seed}, cache={args.cache}")
    log(f"hsv_h={args.hsv_h}, hsv_s={args.hsv_s}, hsv_v={args.hsv_v}")
    log(
        f"full_download_workers={args.full_download_workers}, "
        f"min_full_synthetic_ratio={args.min_full_synthetic_ratio}"
    )
    log(
        f"skip_prepare={args.skip_prepare}, skip_noise={args.skip_noise}, "
        f"skip_make={args.skip_make}, skip_train={args.skip_train}, "
        f"skip_eval={args.skip_eval}, dry_run={args.dry_run}"
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Train the final RarePlanes model from the combined experiment recipe."
    )
    ap.add_argument("--smoke", action="store_true", help="use synthetic_1k and 3 epochs")
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--noise-dataset-name", default=None)
    ap.add_argument("--mixed-dataset-name", default=None)

    ap.add_argument("--model", default="yolov10n.pt")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="0")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--cache", choices=["disk", "ram"], default=None)

    ap.add_argument("--synthetic-src", default=None)
    ap.add_argument("--real-src", default="data/yolo/real_aircraft")
    ap.add_argument("--real-img-dir", default="data/real/PS-RGB_tiled/PS-RGB_tiled")
    ap.add_argument("--coco-gt", default="data/real/annotations/instances_test_aircraft.json")
    ap.add_argument("--real-pct", type=int, default=25)
    ap.add_argument("--val-frac", type=float, default=0.15)

    ap.add_argument("--noise-sigma", type=float, default=8.0)
    ap.add_argument("--no-noise", action="store_true", help="use clean synthetic images")
    ap.add_argument("--hsv_h", type=float, default=0.015)
    ap.add_argument("--hsv_s", type=float, default=0.4)
    ap.add_argument("--hsv_v", type=float, default=0.3)

    ap.add_argument("--benchmark-images", type=int, default=256)
    ap.add_argument("--benchmark-batch", type=int, default=32)
    ap.add_argument("--full-download-workers", type=int, default=32)
    ap.add_argument("--full-progress-every", type=int, default=1000)
    ap.add_argument("--min-full-synthetic-ratio", type=float, default=0.99)
    ap.add_argument(
        "--skip-full-synthetic-download",
        action="store_true",
        help="assume full data/synthetic/images/train is already downloaded",
    )

    ap.add_argument("--force-download", action="store_true")
    ap.add_argument("--force-extract", action="store_true")
    ap.add_argument("--skip-prepare", action="store_true")
    ap.add_argument("--skip-noise", action="store_true")
    ap.add_argument("--skip-make", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--skip-benchmark", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--log-file", default="results/final_combined_run.log")
    args = ap.parse_args()

    if args.epochs is None:
        args.epochs = 3 if args.smoke else 60
    if args.synthetic_src is None:
        args.synthetic_src = "data/yolo/synthetic_1k" if args.smoke else "data/yolo/synthetic_aircraft"
    if not (0 < args.real_pct <= 100):
        ap.error("--real-pct must be in range 1..100")
    if args.noise_sigma < 0:
        ap.error("--noise-sigma must be non-negative")
    if not 0 < args.min_full_synthetic_ratio <= 1:
        ap.error("--min-full-synthetic-ratio must be in range (0, 1]")
    return args


def main() -> None:
    global LOG_HANDLE
    args = parse_args()

    (ROOT / "results").mkdir(parents=True, exist_ok=True)
    log_path = ROOT / args.log_file if not Path(args.log_file).is_absolute() else Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("a", buffering=1) as handle:
        LOG_HANDLE = handle
        start = time.time()
        section("START train_final_model.py")
        print_plan(args)
        prepare_data(args)
        prepare_full_synthetic(args)
        if not args.dry_run:
            check_required_paths(args)
        synthetic_for_mix = make_noise_dataset(args)
        dataset = make_mixed_dataset(synthetic_for_mix, args)
        name = train_final(dataset, args)
        eval_final(name, args)
        benchmark_final(name, args)
        write_summary(name, synthetic_for_mix, dataset, args)
        section("END train_final_model.py")
        log(f"total wall-clock={time.time() - start:.1f}s")
        log(f"log file={rel(log_path)}")


if __name__ == "__main__":
    main()
