"""
Analiza wygladu domeny: real vs synthetic (RarePlanes).

Realizuje wymog z dodatkowe_wymagadnia.txt:
"color histogram mozna zrobic, transformacje fourriera, szereg analitycznych
narzedzi zeby sprawdzic rozklad obrazow syntetycznych a realnych".

Dla losowej probki kafli z kazdej domeny liczy:
  1. Histogram kolorow (per kanal R/G/B) + statystyki jasnosci/nasycenia.
  2. Usrednione widmo amplitudowe FFT (log) — pokazuje rozklad czestotliwosci
     (synthetic czesto "gladsze"/ostrzejsze niz realne satelitarne).
  3. Radialny profil mocy widma (1D) — zwiezla sygnatura tekstury domeny.

Zapisuje wykresy do results/appearance/ i liczby do results/appearance_stats.json.
Dziala na PROBCE (--n), wiec mozna odpalic zanim splynie caly zbior.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results" / "appearance"

# Domeny: katalog z obrazami (kafelki PNG). Real tiled rozpakowywany pozniej —
# sciezki ustawiamy elastycznie i pomijamy te, ktorych jeszcze nie ma.
DOMAINS = {
    "real": [
        ROOT / "data" / "real" / "PS-RGB_tiled" / "PS-RGB_tiled",
        ROOT / "data" / "real" / "PS-RGB_tiled",
    ],
    "synthetic": [
        ROOT / "data" / "synthetic" / "images" / "train",
    ],
}


def find_dir(candidates):
    for c in candidates:
        if c.is_dir() and any(c.glob("*.png")):
            return c
    return None


def sample_images(d, n, seed=0):
    files = sorted(d.glob("*.png"))
    rng = random.Random(seed)
    if len(files) > n:
        files = rng.sample(files, n)
    return files


def load_rgb(path, size=512):
    img = Image.open(path).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32)


def radial_profile(power):
    """Usredniony 1D profil mocy wzgledem czestotliwosci radialnej."""
    h, w = power.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
    tbin = np.bincount(r.ravel(), power.ravel())
    nr = np.bincount(r.ravel())
    return tbin / np.maximum(nr, 1)


def analyze_domain(name, d, n, seed=0):
    files = sample_images(d, n, seed)
    hist = {c: np.zeros(256) for c in "RGB"}
    fft_accum = None
    radial_accum = None
    bright, sat = [], []

    for f in files:
        arr = load_rgb(f)
        for i, c in enumerate("RGB"):
            h, _ = np.histogram(arr[..., i], bins=256, range=(0, 255))
            hist[c] += h
        bright.append(arr.mean())
        mx, mn = arr.max(axis=2), arr.min(axis=2)
        sat.append((np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)).mean())

        gray = arr.mean(axis=2)
        F = np.fft.fftshift(np.fft.fft2(gray))
        power = np.log1p(np.abs(F))
        fft_accum = power if fft_accum is None else fft_accum + power
        rp = radial_profile(np.abs(F) ** 2)
        if radial_accum is None:
            radial_accum = rp
        else:
            m = min(len(radial_accum), len(rp))
            radial_accum = radial_accum[:m] + rp[:m]

    k = len(files)
    for c in "RGB":
        hist[c] /= hist[c].sum()
    fft_mean = fft_accum / k
    radial_mean = radial_accum / k

    stats = {
        "domain": name, "n_images": k, "dir": str(d),
        "brightness_mean": float(np.mean(bright)),
        "brightness_std": float(np.std(bright)),
        "saturation_mean": float(np.mean(sat)),
        "rgb_mean": {c: float((np.arange(256) * hist[c]).sum()) for c in "RGB"},
    }
    return stats, hist, fft_mean, radial_mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="probka obrazow na domene")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results, hists, ffts, radials = {}, {}, {}, {}

    for name, cands in DOMAINS.items():
        d = find_dir(cands)
        if d is None:
            print(f"[pomijam] {name}: brak obrazow (jeszcze nie pobrane?) w {cands}")
            continue
        print(f"[{name}] katalog: {d}")
        stats, hist, fft, radial = analyze_domain(name, d, args.n, args.seed)
        results[name] = stats
        hists[name], ffts[name], radials[name] = hist, fft, radial
        print(f"  n={stats['n_images']}  jasnosc={stats['brightness_mean']:.1f}  "
              f"nasycenie={stats['saturation_mean']:.3f}  RGB_mean={stats['rgb_mean']}")

    if len(results) < 1:
        print("Brak danych do analizy. Poczekaj na pobranie obrazow.")
        return

    # --- wykres 1: histogramy kolorow ---
    fig, axes = plt.subplots(1, len(hists), figsize=(6 * len(hists), 4), squeeze=False)
    for ax, (name, hist) in zip(axes[0], hists.items()):
        for c, col in zip("RGB", ["r", "g", "b"]):
            ax.plot(hist[c], color=col, label=c, lw=1)
        ax.set_title(f"Histogram kolorow — {name}")
        ax.set_xlabel("wartosc piksela"); ax.set_ylabel("czestosc (norm.)"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT_DIR / "color_histograms.png", dpi=120); plt.close(fig)

    # --- wykres 2: usrednione widmo FFT (log) ---
    if ffts:
        fig, axes = plt.subplots(1, len(ffts), figsize=(5 * len(ffts), 5), squeeze=False)
        for ax, (name, fft) in zip(axes[0], ffts.items()):
            im = ax.imshow(fft, cmap="viridis")
            ax.set_title(f"Srednie widmo FFT (log) — {name}"); ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout(); fig.savefig(OUT_DIR / "fft_spectra.png", dpi=120); plt.close(fig)

    # --- wykres 3: radialny profil mocy (porownanie domen na 1 osi) ---
    if radials:
        fig, ax = plt.subplots(figsize=(7, 5))
        for name, rp in radials.items():
            ax.semilogy(rp, label=name, lw=1.5)
        ax.set_title("Radialny profil mocy widma (sygnatura tekstury domeny)")
        ax.set_xlabel("czestotliwosc radialna"); ax.set_ylabel("moc (log)"); ax.legend()
        fig.tight_layout(); fig.savefig(OUT_DIR / "radial_power.png", dpi=120); plt.close(fig)

    with open(ROOT / "results" / "appearance_stats.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[zapisano] wykresy -> {OUT_DIR}, statystyki -> results/appearance_stats.json")


if __name__ == "__main__":
    main()
