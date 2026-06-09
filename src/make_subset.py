"""
Tworzy stratyfikowany podzbior YOLO (po lotniskach) do szybkich sweepow.

Z istniejacego data/yolo/synthetic_aircraft/{images,labels}/{train,val} wybiera
N obrazow train (proporcjonalnie wg lotniska z nazwy pliku) + proporcjonalny val.
Linkuje (symlink) do nowego katalogu i pisze data.yaml. Test wskazuje na realny
holdout do ewaluacji cross-domain (osobno, przez eval_per_size).

Uzycie: python3 src/make_subset.py --n-train 10000 --name synthetic_10k
"""
import argparse
import os
import random
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "yolo" / "synthetic_aircraft"


def airport(fname):
    m = re.match(r"([A-Za-z]+_Airport)", fname)
    return m.group(1) if m else "other"


def link_split(split, names, dst):
    img_o = dst / "images" / split
    lbl_o = dst / "labels" / split
    img_o.mkdir(parents=True, exist_ok=True)
    lbl_o.mkdir(parents=True, exist_ok=True)
    for fn in names:
        for sub, ext in (("images", ".png"), ("labels", ".txt")):
            src = SRC / sub / split / (Path(fn).stem + ext)
            dlink = (img_o if sub == "images" else lbl_o) / (Path(fn).stem + ext)
            if src.exists():
                if dlink.exists() or dlink.is_symlink():
                    dlink.unlink()
                os.symlink(src.resolve(), dlink)


def stratified(split, n, seed):
    files = sorted(p.name for p in (SRC / "images" / split).glob("*.png"))
    by_air = defaultdict(list)
    for f in files:
        by_air[airport(f)].append(f)
    rng = random.Random(seed)
    total = len(files)
    picked = []
    for air, fs in by_air.items():
        k = round(n * len(fs) / total)
        rng.shuffle(fs)
        picked.extend(fs[:k])
    rng.shuffle(picked)
    return picked[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=10000)
    ap.add_argument("--name", default="synthetic_10k")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    dst = ROOT / "data" / "yolo" / args.name
    # val skalowany proporcjonalnie (15% jak oryginal)
    n_val = round(args.n_train * 0.15 / 0.85)

    tr = stratified("train", args.n_train, args.seed)
    va = stratified("val", n_val, args.seed)
    link_split("train", tr, dst)
    link_split("val", va, dst)

    yaml = (
        f"# RarePlanes synthetic SUBSET ({args.name}) -> YOLO\n"
        f"path: {dst}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: 1\n"
        f"names: ['aircraft']\n"
    )
    (dst / "data.yaml").write_text(yaml)

    # rozklad lotnisk dla kontroli
    from collections import Counter
    dist = Counter(airport(f) for f in tr)
    print(f"[{args.name}] train={len(tr)} val={len(va)} -> {dst}/data.yaml")
    print("rozklad lotnisk (train):")
    for a, c in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"   {a:24s}: {c}")


if __name__ == "__main__":
    main()
