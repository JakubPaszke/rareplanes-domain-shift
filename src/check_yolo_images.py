"""
Validate image files in a YOLO dataset before a long cluster training job.

This catches two failure modes that Ultralytics may only expose after training
starts:
  - broken symlinks / missing image targets,
  - corrupt PNG files, for example libpng IDAT CRC errors.

For RarePlanes synthetic train/val splits, the script can optionally repair bad
files by re-downloading only the affected PNG names from the public S3 mirror.
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BadImage:
    path: Path
    reason: str


def iter_images(dataset: Path, splits: list[str]) -> list[Path]:
    files: list[Path] = []
    for split in splits:
        img_dir = dataset / "images" / split
        if not img_dir.is_dir():
            raise FileNotFoundError(f"Missing image split directory: {img_dir}")
        files.extend(sorted(img_dir.glob("*.png")))
    return files


def check_one(path: Path) -> BadImage | None:
    if not path.exists():
        target = os.readlink(path) if path.is_symlink() else "<not a symlink>"
        return BadImage(path, f"missing target; symlink target={target}")

    try:
        from PIL import Image, ImageFile
    except Exception as exc:  # pragma: no cover - depends on cluster env
        raise RuntimeError("Pillow is required to validate image integrity") from exc

    ImageFile.LOAD_TRUNCATED_IMAGES = False
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            img.load()
    except Exception as exc:
        return BadImage(path, f"{type(exc).__name__}: {exc}")
    return None


def check_many(paths: list[Path], workers: int) -> list[BadImage]:
    if workers <= 1:
        return [bad for path in paths if (bad := check_one(path)) is not None]
    bad: list[BadImage] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(check_one, paths):
            if result is not None:
                bad.append(result)
    return bad


def download_image(filename: str, dst: Path, url_base: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + ".part")
    if part.exists():
        part.unlink()

    url = f"{url_base.rstrip('/')}/{urllib.parse.quote(filename)}"
    print(f"[repair] download {url} -> {dst}", flush=True)
    urllib.request.urlretrieve(url, part)
    os.replace(part, dst)


def repair_bad_images(bad: list[BadImage], repair_dir: Path, url_base: str) -> None:
    seen: set[str] = set()
    for item in bad:
        filename = item.path.name
        if filename in seen:
            continue
        seen.add(filename)
        dst = repair_dir / filename
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        download_image(filename, dst, url_base)


def write_bad_list(path: Path, bad: list[BadImage]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(f"{item.path}\t{item.reason}" for item in bad)
    path.write_text(text + ("\n" if text else ""))
    print(f"[check] bad image list written to {path}", flush=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Validate YOLO image files.")
    ap.add_argument("--dataset", required=True, help="YOLO dataset directory, e.g. data/yolo/synthetic_aircraft")
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--write-bad-list", default=None)
    ap.add_argument("--repair", action="store_true", help="download and replace bad files")
    ap.add_argument("--repair-dir", default=None, help="source image dir to repair, e.g. data/synthetic/images/train")
    ap.add_argument(
        "--url-base",
        default="https://rareplanes-public.s3.amazonaws.com/synthetic/train/images",
    )
    args = ap.parse_args()

    if args.workers < 1:
        ap.error("--workers must be >= 1")
    if args.repair and not args.repair_dir:
        ap.error("--repair requires --repair-dir")
    return args


def main() -> int:
    args = parse_args()
    dataset = Path(args.dataset)
    paths = iter_images(dataset, args.splits)
    print(
        f"[check] dataset={dataset} splits={','.join(args.splits)} images={len(paths)} workers={args.workers}",
        flush=True,
    )

    bad = check_many(paths, args.workers)
    if bad:
        print(f"[check] bad images found: {len(bad)}", flush=True)
        for item in bad[:20]:
            print(f"  BAD {item.path}: {item.reason}", flush=True)
        if len(bad) > 20:
            print(f"  ... {len(bad) - 20} more", flush=True)

    if args.write_bad_list:
        write_bad_list(Path(args.write_bad_list), bad)

    if bad and args.repair:
        repair_bad_images(bad, Path(args.repair_dir), args.url_base)
        print("[check] re-validating repaired images", flush=True)
        bad = check_many([item.path for item in bad], args.workers)
        if args.write_bad_list:
            write_bad_list(Path(args.write_bad_list), bad)

    if bad:
        print("[check] validation failed; fix bad images before training", file=sys.stderr, flush=True)
        return 2

    print("[check] all images OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
