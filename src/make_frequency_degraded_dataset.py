"""
Tworzy wariant datasetu YOLO z obrazami po degradacji czestotliwosciowej.

Uzycie:
  python3 src/make_frequency_degraded_dataset.py \
    --src data/yolo/synthetic_10k \
    --name synthetic_10k_b1_blur_noise \
    --blur-radius 0.4 --noise-sigma 5 --seed 42

Transformacje sa deterministyczne dla zadanego seed. Etykiety sa linkowane
symbolicznie, bo degradacja nie zmienia geometrii bboxow.
"""
import argparse
import io
import os
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parent.parent


def parse_data_yaml(src_yaml):
    """Minimalnie zachowuje nc/names ze zrodlowego data.yaml bez zaleznosci PyYAML."""
    meta = {"nc": "1", "names": "['aircraft']"}
    if not src_yaml.exists():
        return meta
    for line in src_yaml.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("nc:"):
            meta["nc"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("names:"):
            meta["names"] = stripped.split(":", 1)[1].strip()
    return meta


def jpeg_roundtrip(img, quality):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def degrade_image(path, args, rng):
    img = Image.open(path).convert("RGB")

    if args.blur_radius > 0:
        radius = rng.uniform(0.0, args.blur_radius)
        if radius > 0:
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))

    if args.noise_sigma > 0:
        arr = np.asarray(img, dtype=np.float32)
        # random.Random nie generuje tablic, wiec seedujemy lokalny generator numpy
        np_rng = np.random.default_rng(rng.randrange(0, 2**32 - 1))
        sigma = rng.uniform(0.0, args.noise_sigma)
        arr += np_rng.normal(0.0, sigma, size=arr.shape)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    if args.jpeg_quality_min is not None:
        quality = rng.randint(args.jpeg_quality_min, 100)
        img = jpeg_roundtrip(img, quality)

    return img


def link_label(src_label, dst_label):
    dst_label.parent.mkdir(parents=True, exist_ok=True)
    if dst_label.exists() or dst_label.is_symlink():
        dst_label.unlink()
    if src_label.exists():
        try:
            os.symlink(src_label.resolve(), dst_label)
        except OSError:
            # Windows bez uprawnien do symlinkow -> kopiuj
            import shutil
            shutil.copy2(src_label.resolve(), dst_label)
    else:
        dst_label.write_text("")


def process_split(split, src, dst, args, rng):
    src_img_dir = src / "images" / split
    src_lbl_dir = src / "labels" / split
    if not src_img_dir.is_dir():
        print(f"[{split}] pomijam: brak katalogu {src_img_dir}", flush=True)
        return 0

    dst_img_dir = dst / "images" / split
    dst_lbl_dir = dst / "labels" / split
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(src_img_dir.glob("*.png"))
    print(f"[{split}] degradacja obrazow: {len(files)} -> {dst_img_dir}", flush=True)
    for i, img_path in enumerate(files, start=1):
        out_img = dst_img_dir / img_path.name
        degraded = degrade_image(img_path, args, rng)
        degraded.save(out_img)

        src_label = src_lbl_dir / f"{img_path.stem}.txt"
        dst_label = dst_lbl_dir / f"{img_path.stem}.txt"
        link_label(src_label, dst_label)

        if i % 250 == 0 or i == len(files):
            print(f"[{split}] {i}/{len(files)}", flush=True)
    return len(files)


def write_data_yaml(dst, src, splits):
    meta = parse_data_yaml(src / "data.yaml")
    lines = [
        f"# Frequency-degraded YOLO dataset generated from {src}",
        f"path: {dst}",
    ]
    if "train" in splits:
        lines.append("train: images/train")
    if "val" in splits:
        lines.append("val: images/val")
    if "test" in splits:
        lines.append("test: images/test")
    lines.extend([f"nc: {meta['nc']}", f"names: {meta['names']}"])
    (dst / "data.yaml").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="zrodlowy katalog YOLO, np. data/yolo/synthetic_10k")
    ap.add_argument("--name", required=True, help="nazwa katalogu wyjsciowego w data/yolo/")
    ap.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val", "test"])
    ap.add_argument("--blur-radius", type=float, default=0.0,
                    help="maksymalny promien GaussianBlur; per obraz losowo z [0, max]")
    ap.add_argument("--noise-sigma", type=float, default=0.0,
                    help="maksymalne odchylenie std szumu Gaussa w pikselach 0-255")
    ap.add_argument("--jpeg-quality-min", type=int, default=None,
                    help="jesli podane, symuluje kompresje JPEG z quality losowanym z [min, 100]")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    src = (ROOT / args.src).resolve() if not Path(args.src).is_absolute() else Path(args.src)
    dst = ROOT / "data" / "yolo" / args.name

    if not src.is_dir():
        raise SystemExit(f"Brak katalogu zrodlowego: {src}")
    if dst.exists():
        if not args.overwrite:
            raise SystemExit(f"Katalog juz istnieje: {dst} (uzyj --overwrite)")
        shutil.rmtree(dst)

    if args.jpeg_quality_min is not None and not (1 <= args.jpeg_quality_min <= 100):
        raise SystemExit("--jpeg-quality-min musi byc w zakresie 1..100")

    rng = random.Random(args.seed)
    counts = {}
    print(f"[start] src={src}", flush=True)
    print(f"[start] dst={dst}", flush=True)
    print(
        f"[start] blur_radius<={args.blur_radius} "
        f"noise_sigma<={args.noise_sigma} jpeg_quality_min={args.jpeg_quality_min}",
        flush=True,
    )
    for split in args.splits:
        counts[split] = process_split(split, src, dst, args, rng)

    write_data_yaml(dst, src, [s for s, n in counts.items() if n > 0])
    print("\n[zapisano]", dst / "data.yaml")
    for split, n in counts.items():
        print(f"  {split}: {n} obrazow")
    print("parametry:")
    print(f"  blur_radius<= {args.blur_radius}")
    print(f"  noise_sigma<= {args.noise_sigma}")
    print(f"  jpeg_quality_min= {args.jpeg_quality_min}")


if __name__ == "__main__":
    main()
