"""
Cluster-friendly runner dla eksperymentu C: mixed training synthetic + real.

Kazdy wariant odpala:
  make_mixed_dataset.py -> train_yolo.py -> eval_per_size.py

Tryby:
  --smoke  : synthetic_1k, 1% real, 3 epoki
  domyslnie: synthetic_10k, 1/5/10/25% real, 60 epok na wariant

Przyklady:
  python3 src/run_expC_mixed_cluster.py --smoke
  python3 src/run_expC_mixed_cluster.py
  python3 src/run_expC_mixed_cluster.py --pcts 1 5 --fracs 0.01 0.05 --epochs 30
  python3 src/run_expC_mixed_cluster.py --dry-run

Zaklada, ze istnieja:
  data/yolo/synthetic_10k/data.yaml albo data/yolo/synthetic_1k/data.yaml
  data/yolo/real_aircraft/data.yaml
  data/real/PS-RGB_tiled/PS-RGB_tiled/
  data/real/annotations/instances_test_aircraft.json
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
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

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


def run_capture(cmd: list[str]) -> str:
    try:
        out = subprocess.check_output(cmd, cwd=ROOT, stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except Exception as exc:
        return f"<nie udalo sie odpalic {' '.join(cmd)}: {exc}>"


def run(cmd: list[str], *, dry_run: bool = False) -> None:
    log("[cmd] " + " ".join(cmd))
    if dry_run:
        log("[dry-run] komenda nie zostala uruchomiona")
        return
    start = time.time()
    subprocess.run(cmd, cwd=ROOT, check=True)
    log(f"[cmd done] czas={time.time() - start:.1f}s")


def dataset_name(dataset_tag: str, pct: str) -> str:
    return f"mixed_syn{dataset_tag}_real{pct}pct"


def run_name(dataset_tag: str, pct: str) -> str:
    return f"expC_{pct}pct_real_{dataset_tag}_ml"


def format_metric(value: object, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


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
    describe_path("real_src", resolve_repo_path(args.real_src))
    describe_path("real_img_dir", resolve_repo_path(args.real_img_dir))
    describe_path("coco_gt", resolve_repo_path(args.coco_gt))
    describe_path("runs_dir", ROOT / "runs")
    describe_path("results_dir", ROOT / "results")

    log("nvidia-smi:")
    print(
        run_capture(["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"]),
        flush=True,
    )

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
    section("Plan eksperymentu C")
    log(f"smoke={args.smoke}")
    log(f"src_dataset={args.src_dataset}")
    log(f"real_src={args.real_src}")
    log(f"dataset_tag={args.dataset_tag}")
    log(f"pcts={' '.join(args.pcts)}")
    log(f"fracs={' '.join(str(x) for x in args.fracs)}")
    log(f"model={args.model}")
    log(
        f"epochs={args.epochs}, batch={args.batch}, imgsz={args.imgsz}, "
        f"workers={args.workers}, device={args.device}, patience={args.patience}, seed={args.seed}"
    )
    log(
        f"skip_make={args.skip_make}, skip_train={args.skip_train}, "
        f"skip_eval={args.skip_eval}, dry_run={args.dry_run}"
    )
    for pct, frac in zip(args.pcts, args.fracs):
        log(
            f"C {pct}%: dataset={dataset_name(args.dataset_tag, pct)}, "
            f"run={run_name(args.dataset_tag, pct)}, real_frac={frac}"
        )


def check_required_paths(args: argparse.Namespace) -> None:
    section("Sprawdzanie danych wejsciowych")
    required = [
        resolve_repo_path(args.src_dataset) / "data.yaml",
        resolve_repo_path(args.src_dataset) / "images/train",
        resolve_repo_path(args.src_dataset) / "images/val",
        resolve_repo_path(args.real_src) / "data.yaml",
        resolve_repo_path(args.real_src) / "images/train",
        resolve_repo_path(args.real_src) / "images/val",
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

    syn_train = count_pngs(resolve_repo_path(args.src_dataset) / "images/train")
    syn_val = count_pngs(resolve_repo_path(args.src_dataset) / "images/val")
    real_train = count_pngs(resolve_repo_path(args.real_src) / "images/train")
    real_val = count_pngs(resolve_repo_path(args.real_src) / "images/val")
    log(f"synthetic train={syn_train}, val={syn_val}")
    log(f"real train={real_train}, val={real_val}")
    if min(syn_train, syn_val, real_train, real_val) == 0:
        raise SystemExit("Jeden z wymaganych splitow ma 0 obrazow.")

    if not args.skip_eval:
        real_test = count_pngs(resolve_repo_path(args.real_img_dir))
        log(f"real eval images={real_test}")
        if real_test == 0:
            raise SystemExit("Brak obrazow PNG w real-img-dir.")

    log("Dane wejsciowe wygladaja OK.")


def make_dataset(pct: str, frac: float, args: argparse.Namespace) -> str:
    name = dataset_name(args.dataset_tag, pct)
    section(f"C {pct}%: tworzenie mixed dataset")
    log(f"dataset={name}")
    log(f"real_frac={frac}")
    log(f"synthetic_src={args.src_dataset}")
    log(f"real_src={args.real_src}")

    cmd = [
        sys.executable,
        "src/make_mixed_dataset.py",
        "--syn-src",
        args.src_dataset,
        "--real-src",
        args.real_src,
        "--name",
        name,
        "--real-frac",
        str(frac),
        "--seed",
        str(args.seed),
        "--overwrite",
    ]
    run(cmd, dry_run=args.dry_run or args.skip_make)

    data_yaml = ROOT / "data/yolo" / name / "data.yaml"
    if args.dry_run or args.skip_make:
        log(f"pomijam sprawdzanie mixed dataset: {rel(data_yaml)}")
    elif data_yaml.exists():
        log(f"mixed dataset OK: {rel(data_yaml)}")
        train_count = count_pngs(ROOT / "data/yolo" / name / "images/train")
        val_count = count_pngs(ROOT / "data/yolo" / name / "images/val")
        log(f"mixed train PNG={train_count}, val PNG={val_count}")
    else:
        raise SystemExit(f"Nie widze data.yaml po make_mixed_dataset: {rel(data_yaml)}")
    return name


def train_variant(pct: str, mixed_dataset: str, args: argparse.Namespace) -> str:
    name = run_name(args.dataset_tag, pct)
    data_yaml = f"data/yolo/{mixed_dataset}/data.yaml"
    section(f"C {pct}%: trening")
    log(f"run_name={name}")
    log(f"data={data_yaml}")

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
        "--val-data",
        data_yaml,
    ]
    run(cmd, dry_run=args.dry_run or args.skip_train)

    best_pt = ROOT / "runs" / name / "weights/best.pt"
    if args.dry_run or args.skip_train:
        log(f"pomijam sprawdzanie checkpointu: {rel(best_pt)}")
    elif best_pt.exists():
        log(f"checkpoint OK: {rel(best_pt)} ({best_pt.stat().st_size / (1024 ** 2):.1f} MB)")
    else:
        raise SystemExit(f"Trening zakonczony, ale nie widze checkpointu: {rel(best_pt)}")
    return name


def eval_variant(name: str, args: argparse.Namespace) -> None:
    section(f"{name}: ewaluacja real holdout")
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
    if args.dry_run or args.skip_eval:
        log(f"pomijam sprawdzanie wyniku eval: {rel(out_json)}")
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


def write_summary() -> None:
    section("Podsumowanie wynikow C")
    per_size_dir = ROOT / "results/per_size"
    rows: list[dict[str, object]] = []

    for path in sorted(per_size_dir.glob("expC*_ml.json")):
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
        log("[summary] Brak plikow results/per_size/expC*_ml.json")
        return

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "expC_mixed_summary.csv"
    md_path = out_dir / "expC_mixed_summary.md"
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
        md_lines.append("| " + " | ".join(format_metric(row.get(h)) for h in headers) + " |")
    md_path.write_text("\n".join(md_lines) + "\n")

    log(f"[zapisano] {rel(csv_path)}")
    log(f"[zapisano] {rel(md_path)}")
    for row in rows:
        log(
            f"{row['run']}: mAP@50={format_metric(row.get('mAP@50'))}, "
            f"mAP@50:95={format_metric(row.get('mAP@50:95'))}, "
            f"AP_S={format_metric(row.get('AP_S'))}, "
            f"AP_M={format_metric(row.get('AP_M'))}, "
            f"AP_L={format_metric(row.get('AP_L'))}"
        )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Runner eksperymentu C mixed training na klaster.")
    ap.add_argument("--smoke", action="store_true", help="synthetic_1k, 1%% real, 3 epoki")
    ap.add_argument("--src-dataset", default=None)
    ap.add_argument("--dataset-tag", default=None)
    ap.add_argument("--real-src", default="data/yolo/real_aircraft")
    ap.add_argument("--real-img-dir", default="data/real/PS-RGB_tiled/PS-RGB_tiled")
    ap.add_argument("--coco-gt", default="data/real/annotations/instances_test_aircraft.json")
    ap.add_argument("--pcts", nargs="+", default=None, help="np. --pcts 1 5 10 25")
    ap.add_argument("--fracs", nargs="+", type=float, default=None, help="np. --fracs 0.01 0.05 0.10 0.25")

    ap.add_argument("--model", default="yolov10n.pt")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--patience", type=int, default=20)

    ap.add_argument("--skip-make", action="store_true", help="nie tworz mixed datasetow od nowa")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--no-summary", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="tylko wypisz komendy")
    args = ap.parse_args()

    if args.src_dataset is None:
        args.src_dataset = "data/yolo/synthetic_1k" if args.smoke else "data/yolo/synthetic_10k"
    if args.dataset_tag is None:
        args.dataset_tag = "1k_smoke" if args.smoke else "10k"
    if args.epochs is None:
        args.epochs = 3 if args.smoke else 60
    if args.pcts is None:
        args.pcts = ["1"] if args.smoke else ["1", "5", "10", "25"]
    if args.fracs is None:
        args.fracs = [0.01] if args.smoke else [0.01, 0.05, 0.10, 0.25]

    if len(args.pcts) != len(args.fracs):
        ap.error("--pcts i --fracs musza miec tyle samo elementow")
    if any(frac <= 0 or frac > 1 for frac in args.fracs):
        ap.error("kazdy --fracs musi byc w zakresie (0, 1]")
    if args.workers < 0:
        ap.error("--workers musi byc >= 0")
    return args


def main() -> None:
    args = parse_args()
    start = time.time()
    section("START run_expC_mixed_cluster.py")
    print_plan(args)
    print_environment(args)

    if not args.dry_run:
        check_required_paths(args)

    for pct, frac in zip(args.pcts, args.fracs):
        mixed = make_dataset(pct, frac, args)
        name = train_variant(pct, mixed, args)
        if not args.skip_eval:
            eval_variant(name, args)

    if not args.no_summary and not args.dry_run and not args.skip_eval:
        write_summary()

    section("KONIEC")
    log(f"caly czas wall-clock={time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
