"""
Self-contained runner for experiment C (mixed synthetic + real training).

It downloads the required RarePlanes data, prepares YOLO datasets, runs the
mixed-training sweep, evaluates on the real holdout, and writes summary files.

Typical usage on a cluster:
  python src/expC.py --smoke
  python src/expC.py

If your home directory has a small quota, put data on scratch:
  python src/expC.py --data-dir /mnt/storage_2/scratch/$USER/rareplanes-data/data --smoke

Outputs:
  results/per_size/expC*_ml.json
  results/baselines/expC*_ml.json
  results/expC_mixed_summary.csv
  results/expC_mixed_summary.md
  results/expC_run.log
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tarfile
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


def synthetic_file_list() -> list[str]:
    files: list[str] = []
    for cfg in [
        ROOT / "configs/synthetic_10k_train_files.txt",
        ROOT / "configs/synthetic_10k_val_files.txt",
    ]:
        with cfg.open() as f:
            files.extend(line.strip() for line in f if line.strip())
    return files


def ensure_repo_data_dir(data_dir: Path) -> Path:
    data_dir = data_dir.expanduser()
    if not data_dir.is_absolute():
        data_dir = (ROOT / data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    repo_data = ROOT / "data"
    if data_dir.resolve() == repo_data.resolve():
        return data_dir

    if repo_data.exists() or repo_data.is_symlink():
        if repo_data.is_symlink() and repo_data.resolve() == data_dir.resolve():
            return data_dir
        raise SystemExit(
            "Repo already has ./data pointing somewhere else. To use --data-dir, "
            "move/remove ./data first or run without --data-dir.\n"
            f"  ./data -> {repo_data.resolve() if repo_data.exists() else '<broken symlink>'}\n"
            f"  requested -> {data_dir.resolve()}"
        )

    os.symlink(data_dir.resolve(), repo_data, target_is_directory=True)
    log(f"Created symlink: data -> {data_dir.resolve()}")
    return data_dir


def download_file(url: str, dst: Path, *, force: bool = False) -> None:
    if dst.exists() and dst.stat().st_size > 0 and not force:
        log(f"[skip] {rel(dst)} exists")
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + ".part")
    if part.exists():
        part.unlink()

    log(f"[download] {url} -> {rel(dst)}")
    urllib.request.urlretrieve(url, part)
    os.replace(part, dst)


def safe_extract_tar(tar_path: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    dst_resolved = dst.resolve()
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            target = (dst / member.name).resolve()
            try:
                target.relative_to(dst_resolved)
            except ValueError:
                raise RuntimeError(f"Unsafe path in tarball: {member.name}")
        tar.extractall(dst)


def download_annotations(data_dir: Path, force: bool) -> None:
    section("Downloading annotations")
    for domain in ("real", "synthetic"):
        for name in ("instances_train_aircraft.json", "instances_test_aircraft.json"):
            download_file(
                f"{RAREPLANES_BASE}/{domain}/metadata_annotations/{name}",
                data_dir / domain / "annotations" / name,
                force=force,
            )


def download_real_tiles(data_dir: Path, args: argparse.Namespace) -> None:
    section("Downloading/extracting real train+test tiles")
    tar_dir = data_dir / "real/tarballs"
    extract_dir = data_dir / "real/PS-RGB_tiled"
    tile_dir = extract_dir / "PS-RGB_tiled"

    downloads = [
        (
            f"{RAREPLANES_BASE}/real/tarballs/train/RarePlanes_train_PS-RGB_tiled.tar.gz",
            tar_dir / "train.tar.gz",
        ),
        (
            f"{RAREPLANES_BASE}/real/tarballs/test/RarePlanes_test_PS-RGB_tiled.tar.gz",
            tar_dir / "test.tar.gz",
        ),
    ]
    for url, dst in downloads:
        download_file(url, dst, force=args.force_download)

    existing = count_pngs(tile_dir)
    if existing >= args.min_real_tiles and not args.force_extract:
        log(f"[skip extract] real tiles already present: {existing}")
        return

    for tar_path in (tar_dir / "train.tar.gz", tar_dir / "test.tar.gz"):
        log(f"[extract] {rel(tar_path)} -> {rel(extract_dir)}")
        safe_extract_tar(tar_path, extract_dir)

    count = count_pngs(tile_dir)
    log(f"real tiles count={count}")
    if count < args.min_real_tiles:
        raise RuntimeError(f"Too few real tiles: {count} < {args.min_real_tiles}")


def fetch_synthetic_image(dst_dir: Path, filename: str, *, force: bool) -> bool:
    dst = dst_dir / filename
    if dst.exists() and dst.stat().st_size > 0 and not force:
        return True

    part = dst.with_name(dst.name + ".part")
    if part.exists():
        part.unlink()

    try:
        urllib.request.urlretrieve(
            f"{RAREPLANES_BASE}/synthetic/train/images/{filename}",
            part,
        )
        os.replace(part, dst)
        return True
    except Exception:
        if part.exists():
            part.unlink()
        return False


def download_synthetic_10k(data_dir: Path, args: argparse.Namespace) -> None:
    section("Downloading synthetic 10k images")
    dst_dir = data_dir / "synthetic/images/train"
    dst_dir.mkdir(parents=True, exist_ok=True)
    files = synthetic_file_list()
    log(f"synthetic files to check/download={len(files)}")

    ok = 0
    with ThreadPoolExecutor(max_workers=args.download_workers) as pool:
        for i, result in enumerate(
            pool.map(lambda fn: fetch_synthetic_image(dst_dir, fn, force=args.force_download), files),
            start=1,
        ):
            ok += int(result)
            if i % args.progress_every == 0 or i == len(files):
                log(f"synthetic progress {i}/{len(files)} ok={ok}")

    selected_ok = sum(
        1 for fn in files
        if (dst_dir / fn).exists() and (dst_dir / fn).stat().st_size > 0
    )
    needed = int(len(files) * args.min_synthetic_ratio)
    log(f"synthetic selected_ok={selected_ok}/{len(files)} minimum={needed}")
    if selected_ok < needed:
        raise RuntimeError("Too few synthetic images. Re-run the script; download is resumable.")


def prepare_yolo(args: argparse.Namespace) -> None:
    section("Preparing YOLO datasets")
    if args.skip_prepare:
        log("Skipping YOLO preparation (--skip-prepare)")
        return

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
    ], dry_run=args.dry_run)
    run([
        sys.executable,
        "src/coco_to_yolo.py",
        "--domain",
        "real",
        "--classes",
        "aircraft",
        "--val-frac",
        str(args.val_frac),
        "--seed",
        str(args.seed),
    ], dry_run=args.dry_run)
    run([
        sys.executable,
        "src/make_subset.py",
        "--n-train",
        "10000",
        "--name",
        "synthetic_10k",
        "--seed",
        str(args.seed),
    ], dry_run=args.dry_run)
    run([
        sys.executable,
        "src/make_subset.py",
        "--n-train",
        "1000",
        "--name",
        "synthetic_1k",
        "--seed",
        str(args.seed),
    ], dry_run=args.dry_run)


def run_experiment_c(args: argparse.Namespace) -> None:
    section("Running experiment C")
    if args.smoke:
        defaults = {
            "src_dataset": "data/yolo/synthetic_1k",
            "dataset_tag": "1k_smoke",
            "epochs": "3",
            "pcts": ["1"],
            "fracs": ["0.01"],
        }
    else:
        defaults = {
            "src_dataset": "data/yolo/synthetic_10k",
            "dataset_tag": "10k",
            "epochs": "60",
            "pcts": ["1", "5", "10", "25"],
            "fracs": ["0.01", "0.05", "0.10", "0.25"],
        }

    src_dataset = args.src_dataset or defaults["src_dataset"]
    dataset_tag = args.dataset_tag or defaults["dataset_tag"]
    epochs = str(args.epochs if args.epochs is not None else defaults["epochs"])
    pcts = args.pcts or defaults["pcts"]
    fracs = args.fracs or defaults["fracs"]

    cmd = [
        sys.executable,
        "src/run_expC_mixed_cluster.py",
        "--src-dataset",
        src_dataset,
        "--dataset-tag",
        dataset_tag,
        "--real-src",
        "data/yolo/real_aircraft",
        "--real-img-dir",
        "data/real/PS-RGB_tiled/PS-RGB_tiled",
        "--coco-gt",
        "data/real/annotations/instances_test_aircraft.json",
        "--pcts",
        *pcts,
        "--fracs",
        *fracs,
        "--epochs",
        epochs,
        "--batch",
        str(args.batch),
        "--workers",
        str(args.workers),
        "--device",
        str(args.device),
        "--imgsz",
        str(args.imgsz),
        "--model",
        args.model,
        "--patience",
        str(args.patience),
    ]
    if args.skip_train:
        cmd.append("--skip-train")
    if args.skip_eval:
        cmd.append("--skip-eval")
    if args.no_summary:
        cmd.append("--no-summary")
    if args.dry_run:
        cmd.append("--dry-run")

    run(cmd, dry_run=False)


def print_plan(args: argparse.Namespace, data_dir: Path) -> None:
    section("Plan")
    log(f"data_dir={data_dir}")
    log(f"smoke={args.smoke}")
    log(f"download_workers={args.download_workers}")
    log(f"batch={args.batch}, workers={args.workers}, imgsz={args.imgsz}, device={args.device}")
    log(f"model={args.model}, seed={args.seed}")
    log(f"skip_download={args.skip_download}, skip_prepare={args.skip_prepare}, prepare_only={args.prepare_only}")
    log(f"dry_run={args.dry_run}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Self-contained experiment C runner.")
    ap.add_argument("--smoke", action="store_true", help="synthetic_1k, 1%% real, 3 epochs")
    ap.add_argument("--data-dir", default=str(ROOT / "data"), help="where to store/link data")
    ap.add_argument("--src-dataset", default=None, help="override C synthetic YOLO dataset")
    ap.add_argument("--dataset-tag", default=None)
    ap.add_argument("--pcts", nargs="+", default=None, help="e.g. --pcts 1 5 10 25")
    ap.add_argument("--fracs", nargs="+", default=None, help="e.g. --fracs 0.01 0.05 0.10 0.25")

    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default="0")
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--model", default="yolov10n.pt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=20)

    ap.add_argument("--download-workers", type=int, default=32)
    ap.add_argument("--progress-every", type=int, default=1000)
    ap.add_argument("--min-real-tiles", type=int, default=7000)
    ap.add_argument("--min-synthetic-ratio", type=float, default=0.99)
    ap.add_argument("--val-frac", type=float, default=0.15)

    ap.add_argument("--force-download", action="store_true")
    ap.add_argument("--force-extract", action="store_true")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--skip-prepare", action="store_true")
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--no-summary", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--log-file", default="results/expC_run.log")
    args = ap.parse_args()

    if args.pcts is not None and args.fracs is not None and len(args.pcts) != len(args.fracs):
        ap.error("--pcts and --fracs must have the same length")
    if args.download_workers < 1:
        ap.error("--download-workers must be >= 1")
    if not 0 < args.min_synthetic_ratio <= 1:
        ap.error("--min-synthetic-ratio must be in (0, 1]")
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
        section("START expC.py")
        data_dir = ensure_repo_data_dir(Path(args.data_dir))
        print_plan(args, data_dir)

        if not args.skip_download:
            download_annotations(data_dir, args.force_download)
            download_real_tiles(data_dir, args)
            download_synthetic_10k(data_dir, args)
        else:
            log("Skipping downloads (--skip-download)")

        prepare_yolo(args)
        if args.prepare_only:
            section("DONE prepare-only")
            return

        run_experiment_c(args)
        section("KONIEC expC.py")
        log(f"total wall-clock={time.time() - start:.1f}s")
        log(f"log file={rel(log_path)}")


if __name__ == "__main__":
    main()
