"""
Pobiera dane potrzebne do eksperymentu B, bez treningu i bez ewaluacji.

Pobierane elementy:
  - real test annotations,
  - synthetic train/test annotations,
  - real test PS-RGB tiled tarball + rozpakowanie,
  - synthetic 10k train images z list configs/synthetic_10k_*_files.txt.

Przyklady:
  python3 src/download_expB_data.py
  python3 src/download_expB_data.py --data-dir /mnt/storage_2/scratch/$USER/rareplanes-data/data
  python3 src/download_expB_data.py --skip-synthetic

Jesli pobierasz poza repo, a reszta skryptow ma widziec dane pod ./data,
zrob potem symlink:
  ln -s /mnt/storage_2/scratch/$USER/rareplanes-data/data data
"""
from __future__ import annotations

import argparse
import os
import tarfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RAREPLANES_BASE = "https://rareplanes-public.s3.amazonaws.com"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def download_file(url: str, dst: Path, *, force: bool = False) -> bool:
    if dst.exists() and dst.stat().st_size > 0 and not force:
        print(f"[skip] {rel(dst)} juz istnieje", flush=True)
        return True

    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + ".part")
    if part.exists():
        part.unlink()

    print(f"[download] {url} -> {rel(dst)}", flush=True)
    try:
        urllib.request.urlretrieve(url, part)
        os.replace(part, dst)
        return True
    except Exception:
        if part.exists():
            part.unlink()
        raise


def safe_extract_tar(tar_path: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    dst_resolved = dst.resolve()
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            target = (dst / member.name).resolve()
            try:
                target.relative_to(dst_resolved)
            except ValueError:
                raise RuntimeError(f"Niebezpieczna sciezka w tarballu: {member.name}")
        tar.extractall(dst)


def synthetic_file_list() -> list[str]:
    files: list[str] = []
    for cfg in [
        ROOT / "configs/synthetic_10k_train_files.txt",
        ROOT / "configs/synthetic_10k_val_files.txt",
    ]:
        with cfg.open() as f:
            files.extend(line.strip() for line in f if line.strip())
    return files


def download_real_test(data_dir: Path, args: argparse.Namespace) -> None:
    ann_dir = data_dir / "real/annotations"
    tar_dir = data_dir / "real/tarballs"
    extract_dir = data_dir / "real/PS-RGB_tiled"
    tile_dir = extract_dir / "PS-RGB_tiled"

    download_file(
        f"{RAREPLANES_BASE}/real/metadata_annotations/instances_test_aircraft.json",
        ann_dir / "instances_test_aircraft.json",
        force=args.force,
    )

    tar_path = tar_dir / "test.tar.gz"
    download_file(
        f"{RAREPLANES_BASE}/real/tarballs/test/RarePlanes_test_PS-RGB_tiled.tar.gz",
        tar_path,
        force=args.force,
    )

    if args.no_extract:
        print("[real] pomijam rozpakowanie (--no-extract)", flush=True)
        return

    tiles = list(tile_dir.glob("*.png")) if tile_dir.exists() else []
    if len(tiles) >= args.min_real_tiles and not args.force_extract:
        print(f"[skip] real test tiles: {len(tiles)}", flush=True)
        return

    print(f"[extract] {rel(tar_path)} -> {rel(extract_dir)}", flush=True)
    safe_extract_tar(tar_path, extract_dir)
    tiles = list(tile_dir.glob("*.png"))
    print(f"[real] test tiles: {len(tiles)}", flush=True)
    if len(tiles) < args.min_real_tiles:
        raise RuntimeError(
            f"Za malo real test tiles: {len(tiles)} < {args.min_real_tiles}. "
            "Sprawdz pobieranie/rozpakowanie tarballa."
        )


def download_synthetic_annotations(data_dir: Path, args: argparse.Namespace) -> None:
    ann_dir = data_dir / "synthetic/annotations"
    for name in ["instances_train_aircraft.json", "instances_test_aircraft.json"]:
        download_file(
            f"{RAREPLANES_BASE}/synthetic/metadata_annotations/{name}",
            ann_dir / name,
            force=args.force,
        )


def fetch_synthetic_image(dst_dir: Path, filename: str, *, force: bool = False) -> bool:
    dst = dst_dir / filename
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


def download_synthetic_10k(data_dir: Path, args: argparse.Namespace) -> None:
    dst_dir = data_dir / "synthetic/images/train"
    dst_dir.mkdir(parents=True, exist_ok=True)
    files = synthetic_file_list()
    print(f"[synthetic] do pobrania/sprawdzenia: {len(files)}", flush=True)

    ok = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, result in enumerate(
            pool.map(lambda fn: fetch_synthetic_image(dst_dir, fn, force=args.force), files),
            start=1,
        ):
            ok += int(result)
            if i % args.progress_every == 0 or i == len(files):
                print(f"[synthetic] {i}/{len(files)} ok={ok}", flush=True)

    have_selected = sum(1 for fn in files if (dst_dir / fn).exists() and (dst_dir / fn).stat().st_size > 0)
    needed = int(len(files) * args.min_synthetic_ratio)
    print(
        f"[synthetic] selected_ok={have_selected}/{len(files)}, minimum={needed}",
        flush=True,
    )
    if have_selected < needed:
        raise RuntimeError(
            "Za malo synthetic images. Uruchom skrypt ponownie, pobieranie jest wznawialne."
        )


def print_next_steps(data_dir: Path) -> None:
    repo_data = ROOT / "data"
    print("\n[done] Dane eksperymentu B sa w:", rel(data_dir), flush=True)
    if data_dir.resolve() != repo_data.resolve():
        print("\nUwaga: pozostale skrypty repo domyslnie czytaja ./data.", flush=True)
        print("Jesli chcesz uzyc tego katalogu, zrob symlink w katalogu repo:", flush=True)
        print(f"  ln -s {data_dir.resolve()} data", flush=True)
    print("\nKolejny krok przygotowania YOLO pod B:", flush=True)
    print("  python3 src/coco_to_yolo.py --domain synthetic --classes aircraft --val-frac 0.15", flush=True)
    print("  python3 src/make_subset.py --n-train 10000 --name synthetic_10k", flush=True)
    print("  python3 src/make_subset.py --n-train 1000 --name synthetic_1k", flush=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Pobiera tylko dane wymagane przez eksperyment B.")
    ap.add_argument("--data-dir", default=str(ROOT / "data"), help="gdzie zapisac katalog data")
    ap.add_argument("--workers", type=int, default=32, help="rownolegle pobieranie synthetic PNG")
    ap.add_argument("--progress-every", type=int, default=1000)
    ap.add_argument("--min-real-tiles", type=int, default=2500)
    ap.add_argument("--min-synthetic-ratio", type=float, default=0.99)
    ap.add_argument("--force", action="store_true", help="pobierz ponownie istniejace pliki")
    ap.add_argument("--force-extract", action="store_true", help="rozpakuj real test ponownie")
    ap.add_argument("--no-extract", action="store_true", help="pobierz tarball real test, ale nie rozpakowuj")
    ap.add_argument("--skip-real", action="store_true")
    ap.add_argument("--skip-synthetic", action="store_true")
    args = ap.parse_args()

    if args.workers < 1:
        ap.error("--workers musi byc >= 1")
    if args.progress_every < 1:
        ap.error("--progress-every musi byc >= 1")
    if not 0 < args.min_synthetic_ratio <= 1:
        ap.error("--min-synthetic-ratio musi byc w zakresie (0, 1]")
    return args


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser()
    if not data_dir.is_absolute():
        data_dir = (ROOT / data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"[config] data_dir={data_dir}", flush=True)
    print(f"[config] workers={args.workers}", flush=True)

    if not args.skip_real:
        download_real_test(data_dir, args)
    if not args.skip_synthetic:
        download_synthetic_annotations(data_dir, args)
        download_synthetic_10k(data_dir, args)

    print_next_steps(data_dir)


if __name__ == "__main__":
    main()
