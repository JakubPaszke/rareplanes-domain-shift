"""
Tworzy dataset YOLO do mixed training: synthetic subset + mala porcja real train.

Uzycie:
  python3 src/make_mixed_dataset.py \
    --syn-src data/yolo/synthetic_10k \
    --real-src data/yolo/real_aircraft \
    --name mixed_syn10k_real5pct \
    --real-frac 0.05 --seed 42

Realny split test nie jest nigdy linkowany do wyjscia. Uzywamy tylko train/val
powstalych z realnego zbioru treningowego przez `src/coco_to_yolo.py`.
"""
import argparse
import os
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def resolve_path(p):
    path = Path(p)
    return path if path.is_absolute() else (ROOT / path).resolve()


def image_files(root, split):
    img_dir = root / "images" / split
    if not img_dir.is_dir():
        return []
    return sorted(img_dir.glob("*.png"))


def sample_real(files, frac, seed):
    if frac <= 0:
        return []
    rng = random.Random(seed)
    files = list(files)
    rng.shuffle(files)
    n = max(1, round(len(files) * frac))
    return sorted(files[:n])


def link_pair(img_path, src_root, dst_root, split, prefix):
    dst_img_dir = dst_root / "images" / split
    dst_lbl_dir = dst_root / "labels" / split
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    out_stem = f"{prefix}__{img_path.stem}"
    dst_img = dst_img_dir / f"{out_stem}.png"
    dst_lbl = dst_lbl_dir / f"{out_stem}.txt"
    src_lbl = src_root / "labels" / split / f"{img_path.stem}.txt"

    for src, dst in ((img_path, dst_img), (src_lbl, dst_lbl)):
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if src.exists():
            os.symlink(src.resolve(), dst)
        else:
            dst.write_text("")


def parse_data_yaml(src_yaml):
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


def write_data_yaml(dst, syn_src):
    meta = parse_data_yaml(syn_src / "data.yaml")
    text = "\n".join([
        "# Mixed RarePlanes YOLO dataset",
        f"path: {dst}",
        "train: images/train",
        "val: images/val",
        f"nc: {meta['nc']}",
        f"names: {meta['names']}",
        "",
    ])
    (dst / "data.yaml").write_text(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--syn-src", required=True, help="synthetic YOLO subset, np. data/yolo/synthetic_10k")
    ap.add_argument("--real-src", required=True, help="real YOLO dataset, np. data/yolo/real_aircraft")
    ap.add_argument("--name", required=True, help="nazwa katalogu wyjsciowego w data/yolo/")
    ap.add_argument("--real-frac", type=float, required=True,
                    help="udzial real train/val do dolaczenia, np. 0.01, 0.05, 0.10, 0.25")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if not (0.0 < args.real_frac <= 1.0):
        raise SystemExit("--real-frac musi byc w zakresie (0, 1]")

    syn_src = resolve_path(args.syn_src)
    real_src = resolve_path(args.real_src)
    dst = ROOT / "data" / "yolo" / args.name

    if not syn_src.is_dir():
        raise SystemExit(f"Brak synthetic YOLO: {syn_src}")
    if not real_src.is_dir():
        raise SystemExit(f"Brak real YOLO: {real_src}")
    if dst.exists():
        if not args.overwrite:
            raise SystemExit(f"Katalog juz istnieje: {dst} (uzyj --overwrite)")
        shutil.rmtree(dst)

    counts = {}
    for split in ("train", "val"):
        syn_files = image_files(syn_src, split)
        real_files_all = image_files(real_src, split)
        real_files = sample_real(real_files_all, args.real_frac, args.seed + (0 if split == "train" else 1))

        for img in syn_files:
            link_pair(img, syn_src, dst, split, "syn")
        for img in real_files:
            link_pair(img, real_src, dst, split, "real")

        counts[split] = {
            "synthetic": len(syn_files),
            "real_selected": len(real_files),
            "real_available": len(real_files_all),
        }

    write_data_yaml(dst, syn_src)

    print("[zapisano]", dst / "data.yaml")
    for split, c in counts.items():
        print(
            f"  {split}: synthetic={c['synthetic']} "
            f"real={c['real_selected']}/{c['real_available']}"
        )
    print("UWAGA: real test nie zostal dolaczony do datasetu mixed.")


if __name__ == "__main__":
    main()
