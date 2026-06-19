"""
Grad-CAM (EigenCAM) dla detektorow YOLOv10 — interpretowalnosc (wymog PDF).

Porownuje "uwage" roznych modeli na TYCH SAMYCH realnych kaflach:
pokazuje na co patrzy model real-baseline vs synthetic vs warianty.
Nacisk na MALE obiekty (wymog PDF: "wizualizacja aktywacji dla malych obiektow").

EigenCAM: nie wymaga gradientow wzgledem klasy (dobre dla detektorow end2end);
liczy pierwsza skladowa glowna aktywacji wybranej warstwy backbone.

Uzycie: python3 src/gradcam_compare.py --n 6
Wyniki: results/gradcam/*.png
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "gradcam"
# dane real sa na dysku Windows C: (po przeniesieniu); fallback na WSL jesli sa lokalnie
_C = Path("/mnt/c/rareplanes_win/data/real")
_W = ROOT / "data" / "real"
REAL_BASE = _C if (_C / "PS-RGB_tiled/PS-RGB_tiled").is_dir() else _W
REAL_DIR = REAL_BASE / "PS-RGB_tiled" / "PS-RGB_tiled"
REAL_GT = REAL_BASE / "annotations" / "instances_test_aircraft.json"

# modele do porownania (etykieta -> sciezka best.pt). Pomija te, ktorych brak.
MODELS = {
    "real-baseline (0.974)":      ROOT / "runs/real_baseline_yolov10n/weights/best.pt",
    "synthetic 45k (0.452)":      ROOT / "runs/syn_baseline_full45k_yolov10n/weights/best.pt",
    "A: slaby HSV (0.455)":       ROOT / "runs/expA_final_45k/weights/best.pt",
    "B2: szum (0.490)":           Path("/mnt/c/rareplanes_win/runs/expB2_noise_files_10k_ml/weights/best.pt"),
}


def load_img(path, size=512):
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))
    return img


class _BackboneWrap(torch.nn.Module):
    """Forward tylko przez backbone do warstwy 8 (C2f) — zwraca czysty tensor.
    Glowica detekcji YOLOv10 zwraca tuple, ktorego grad-cam nie obsluguje."""
    def __init__(self, net, stop=8):
        super().__init__()
        self.layers = net.model[: stop + 1]
    def forward(self, x):
        for m in self.layers:
            # warstwy backbone YOLOv10 maja prosty przeplyw (bez polaczen z innych warstw do idx 8)
            x = m(x)
        return x


def cam_for_model(pt_path, rgb_float, size=512):
    """EigenCAM na warstwie 8 (C2f, ostatni blok backbone, czysty tensor)."""
    model = YOLO(str(pt_path))
    net = model.model
    net.eval()
    wrap = _BackboneWrap(net, stop=8)
    target_layers = [wrap.layers[8]]
    inp = torch.from_numpy(rgb_float).permute(2, 0, 1).unsqueeze(0).float()
    try:
        cam = EigenCAM(wrap, target_layers)
        grayscale = cam(inp)[0]
    except Exception as e:
        print(f"  [warn] CAM nieudany dla {pt_path.parent.parent.name}: {e}")
        return np.zeros((size, size), dtype=np.float32)
    return grayscale


def small_object_tiles(n, seed):
    """Wybierz realne kafle z MALYMI obiektami (wymog PDF)."""
    gt = json.load(open(REAL_GT))
    id2name = {im["id"]: im["file_name"] for im in gt["images"]}
    # zlicz male bboxy per obraz (area < 32^2)
    small_per_img = {}
    for a in gt["annotations"]:
        if a.get("area", a["bbox"][2] * a["bbox"][3]) < 32 * 32:
            small_per_img[a["image_id"]] = small_per_img.get(a["image_id"], 0) + 1
    # obrazy z najwieksza liczba malych obiektow
    ranked = sorted(small_per_img.items(), key=lambda x: -x[1])
    names = [id2name[iid] for iid, _ in ranked if (REAL_DIR / id2name[iid]).exists()]
    rng = random.Random(seed)
    pick = names[:max(n * 3, 20)]
    rng.shuffle(pick)
    return pick[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6, help="liczba kafli")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    models = {k: v for k, v in MODELS.items() if v.exists()}
    print(f"modele: {list(models)}")
    tiles = small_object_tiles(args.n, args.seed)
    print(f"kafle (male obiekty): {len(tiles)}")

    ncol = len(models) + 1  # +1 na oryginal
    fig, axes = plt.subplots(len(tiles), ncol, figsize=(3.2 * ncol, 3.2 * len(tiles)))
    if len(tiles) == 1:
        axes = axes[None, :]

    for r, tile in enumerate(tiles):
        rgb = load_img(REAL_DIR / tile)
        rgb_float = rgb.astype(np.float32) / 255.0
        axes[r, 0].imshow(rgb)
        axes[r, 0].set_ylabel(tile[:18], fontsize=7)
        if r == 0:
            axes[r, 0].set_title("oryginal", fontsize=10)
        axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        for c, (label, pt) in enumerate(models.items(), start=1):
            gray = cam_for_model(pt, rgb_float)
            vis = show_cam_on_image(rgb_float, gray, use_rgb=True)
            axes[r, c].imshow(vis)
            if r == 0:
                axes[r, c].set_title(label, fontsize=9)
            axes[r, c].axis("off")

    fig.suptitle("Grad-CAM (EigenCAM) — uwaga modeli na realnych kaflach z malymi obiektami",
                 fontsize=13)
    fig.tight_layout()
    out = OUT / "gradcam_comparison.png"
    fig.savefig(out, dpi=110)
    print(f"[zapisano] {out}")


if __name__ == "__main__":
    main()
