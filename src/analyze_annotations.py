"""
Analiza adnotacji RarePlanes (COCO, kafelki 512x512).

Liczy statystyki potrzebne do raportu i do zaprojektowania eksperymentow:
- liczba obrazow / instancji, gestosc obiektow na kafel,
- rozklad rozmiarow wg progow COCO (small <32^2, medium 32^2-96^2, large >96^2 px),
- rozklad bokow bbox (wymiary w px), klasy roli, truncation,
- porownanie real vs synthetic na poziomie statystyk adnotacji.

Czyta czyste pliki instances_*_{aircraft,role}.json (kategorie poprawne).
Pliki *_Coco_Annotations_tiled.json maja zepsute pole categories - uzywane tylko
do wyciagniecia bogatych atrybutow (role, wingspan), nie jako zrodlo treningowe.
"""
import json
import sys
from pathlib import Path
from collections import Counter

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REAL_ANN = ROOT / "data" / "real" / "annotations"
SYN_ANN = ROOT / "data" / "synthetic" / "annotations"

# progi powierzchni wg konwencji COCO (w pikselach^2)
SMALL = 32 ** 2
LARGE = 96 ** 2


def load_coco(path):
    with open(path) as f:
        return json.load(f)


def size_bucket(area_px):
    if area_px < SMALL:
        return "small"
    if area_px <= LARGE:
        return "medium"
    return "large"


def summarize(name, coco, has_role=False):
    images = coco["images"]
    anns = coco["annotations"]
    cats = {c["id"]: c["name"] for c in coco.get("categories", []) if "name" in c}

    n_img = len(images)
    n_ann = len(anns)

    # bbox -> szerokosc, wysokosc, powierzchnia w px
    ws = np.array([a["bbox"][2] for a in anns], dtype=float)
    hs = np.array([a["bbox"][3] for a in anns], dtype=float)
    # area: uzyj pola 'area' jesli jest, inaczej w*h
    areas = np.array([a.get("area", a["bbox"][2] * a["bbox"][3]) for a in anns], dtype=float)

    buckets = Counter(size_bucket(a) for a in areas)

    # gestosc: instancji na obraz
    per_img = Counter(a["image_id"] for a in anns)
    dens = np.array(list(per_img.values()), dtype=float)
    empty_imgs = n_img - len(per_img)

    print(f"\n{'='*60}\n{name}\n{'='*60}")
    print(f"  obrazy:        {n_img}")
    print(f"  instancje:     {n_ann}")
    print(f"  obrazy puste (bez adnotacji): {empty_imgs}")
    print(f"  instancji/obraz: mean={dens.mean():.2f}  median={np.median(dens):.0f}  "
          f"min={dens.min():.0f}  max={dens.max():.0f}")
    print(f"  bbox szer. px:  mean={ws.mean():.1f}  median={np.median(ws):.1f}  "
          f"p5={np.percentile(ws,5):.1f}  p95={np.percentile(ws,95):.1f}")
    print(f"  bbox wys.  px:  mean={hs.mean():.1f}  median={np.median(hs):.1f}  "
          f"p5={np.percentile(hs,5):.1f}  p95={np.percentile(hs,95):.1f}")
    print(f"  powierzchnia px^2: mean={areas.mean():.0f}  median={np.median(areas):.0f}")
    print(f"  rozklad rozmiarow COCO:")
    for b in ("small", "medium", "large"):
        c = buckets.get(b, 0)
        print(f"     {b:7s}: {c:6d}  ({100*c/n_ann:5.1f}%)")

    if has_role and cats:
        by_cat = Counter(a["category_id"] for a in anns)
        print(f"  klasy (role):")
        for cid, cnt in sorted(by_cat.items()):
            print(f"     {cats.get(cid, cid):35s}: {cnt:6d}  ({100*cnt/n_ann:5.1f}%)")

    return {
        "name": name, "n_img": n_img, "n_ann": n_ann, "empty_imgs": empty_imgs,
        "dens_mean": float(dens.mean()), "dens_median": float(np.median(dens)),
        "w_mean": float(ws.mean()), "h_mean": float(hs.mean()),
        "area_median": float(np.median(areas)),
        "buckets": dict(buckets),
    }


def main():
    results = {}

    # REAL - 1 klasa (aircraft) i 3 klasy (role)
    for split in ("train", "test"):
        p = REAL_ANN / f"instances_{split}_aircraft.json"
        if p.exists():
            results[f"real_{split}_aircraft"] = summarize(
                f"REAL {split} (aircraft, 1 klasa)", load_coco(p))
        pr = REAL_ANN / f"instances_{split}_role.json"
        if pr.exists():
            results[f"real_{split}_role"] = summarize(
                f"REAL {split} (role, 3 klasy)", load_coco(pr), has_role=True)

    # SYNTHETIC - jesli adnotacje juz pobrane
    for split in ("train", "test"):
        p = SYN_ANN / f"instances_{split}_aircraft.json"
        if p.exists():
            results[f"syn_{split}_aircraft"] = summarize(
                f"SYNTHETIC {split} (aircraft, 1 klasa)", load_coco(p))
        pr = SYN_ANN / f"instances_{split}_role.json"
        if pr.exists():
            results[f"syn_{split}_role"] = summarize(
                f"SYNTHETIC {split} (role, 3 klasy)", load_coco(pr), has_role=True)

    out = ROOT / "results" / "annotation_stats.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[zapisano] {out}")


if __name__ == "__main__":
    main()
