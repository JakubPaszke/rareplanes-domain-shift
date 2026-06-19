"""
Konwersja adnotacji RarePlanes COCO -> format YOLO (txt per obraz).

YOLO: kazdy obraz ma plik .txt; linia = "cls cx cy w h" znormalizowane do [0,1].
Tworzy strukture:
  <out>/images/{train,val,test}/*.png   (symlinki do oryginalnych kafli)
  <out>/labels/{train,val,test}/*.txt
  <out>/data.yaml

Val wydzielany ze splitu train (deterministycznie wg seed). Test = oryginalny
holdout COCO (instances_test_*). Domyslnie 1 klasa (aircraft); --classes role
przelacza na 3 klasy z instances_*_role.json.

Uzycie:
  python3 src/coco_to_yolo.py --domain real --classes aircraft --val-frac 0.15
  python3 src/coco_to_yolo.py --domain synthetic --classes aircraft   # gdy obrazy pobrane
"""
import argparse
import json
import os
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# zrodla obrazow per domena
IMG_DIRS = {
    "real": {
        "train": ROOT / "data/real/PS-RGB_tiled/PS-RGB_tiled",
        "test": ROOT / "data/real/PS-RGB_tiled/PS-RGB_tiled",
    },
    "synthetic": {
        "train": ROOT / "data/synthetic/images/train",
        "test": ROOT / "data/synthetic/images/test",
    },
}
ANN_DIR = {
    "real": ROOT / "data/real/annotations",
    "synthetic": ROOT / "data/synthetic/annotations",
}


def load_coco(path):
    with open(path) as f:
        return json.load(f)


def build_index(coco):
    """image_id -> (file_name, w, h); image_id -> list[(cls0, bbox)]."""
    images = {im["id"]: im for im in coco["images"]}
    # mapowanie category_id -> indeks 0..N-1 wg posortowanych id
    cat_ids = sorted(c["id"] for c in coco["categories"])
    cat_map = {cid: i for i, cid in enumerate(cat_ids)}
    anns = {}
    for a in coco["annotations"]:
        anns.setdefault(a["image_id"], []).append(
            (cat_map[a["category_id"]], a["bbox"]))
    return images, anns, cat_ids


def yolo_lines(img, ann_list):
    w, h = img["width"], img["height"]
    out = []
    for cls0, (x, y, bw, bh) in ann_list:
        # COCO bbox = [x_min, y_min, w, h] -> YOLO [cx, cy, w, h] / rozmiar
        cx = (x + bw / 2) / w
        cy = (y + bh / 2) / h
        nw = bw / w
        nh = bh / h
        # clamp do [0,1] (kafle moga miec bbox dotykajace brzegu)
        cx, cy = min(max(cx, 0), 1), min(max(cy, 0), 1)
        nw, nh = min(max(nw, 0), 1), min(max(nh, 0), 1)
        if nw <= 0 or nh <= 0:
            continue
        out.append(f"{cls0} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return out


def write_split(split, image_ids, images, anns, img_src_dir, out, link_imgs=True):
    img_out = out / "images" / split
    lbl_out = out / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    n_lbl = 0
    for iid in image_ids:
        img = images[iid]
        fname = img["file_name"]
        src = img_src_dir / fname
        if not src.exists():
            continue
        dst = img_out / fname
        if link_imgs:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            try:
                os.symlink(src.resolve(), dst)
            except OSError:
                # Windows bez uprawnien do symlinkow -> kopiuj plik
                import shutil
                shutil.copy2(src.resolve(), dst)
        # label (pusty plik tez tworzymy -> negatywny przyklad)
        lines = yolo_lines(img, anns.get(iid, []))
        (lbl_out / (Path(fname).stem + ".txt")).write_text("\n".join(lines))
        n_lbl += 1
    return n_lbl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=["real", "synthetic"], required=True)
    ap.add_argument("--classes", choices=["aircraft", "role"], default="aircraft")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ann_dir = ANN_DIR[args.domain]
    train_coco = load_coco(ann_dir / f"instances_train_{args.classes}.json")
    test_coco = load_coco(ann_dir / f"instances_test_{args.classes}.json")

    out = ROOT / "data" / "yolo" / f"{args.domain}_{args.classes}"
    out.mkdir(parents=True, exist_ok=True)

    # --- train/val split (deterministyczny) ---
    tr_images, tr_anns, cat_ids = build_index(train_coco)
    ids = sorted(tr_images.keys())
    rng = random.Random(args.seed)
    rng.shuffle(ids)
    n_val = int(len(ids) * args.val_frac)
    val_ids, train_ids = ids[:n_val], ids[n_val:]

    te_images, te_anns, _ = build_index(test_coco)
    test_ids = sorted(te_images.keys())

    n_tr = write_split("train", train_ids, tr_images, tr_anns,
                        IMG_DIRS[args.domain]["train"], out)
    n_va = write_split("val", val_ids, tr_images, tr_anns,
                       IMG_DIRS[args.domain]["train"], out)
    n_te = write_split("test", test_ids, te_images, te_anns,
                       IMG_DIRS[args.domain]["test"], out)

    names = [c["name"] for c in sorted(train_coco["categories"], key=lambda x: x["id"])]
    yaml = (
        f"# RarePlanes {args.domain} ({args.classes}) -> YOLO\n"
        f"path: {out}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"nc: {len(names)}\n"
        f"names: {names}\n"
    )
    (out / "data.yaml").write_text(yaml)

    print(f"[{args.domain}/{args.classes}] train={n_tr} val={n_va} test={n_te}")
    print(f"  klasy ({len(names)}): {names}")
    print(f"  -> {out}/data.yaml")


if __name__ == "__main__":
    main()
