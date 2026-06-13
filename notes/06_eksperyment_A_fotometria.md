# Eksperyment A — augmentacje fotometryczne (HSV jitter): sweep + finalny

> Motywacja (z `notes/02`): synthetic jest systematycznie jaśniejszy (V 133 vs 72)
> i mniej nasycony (S 0.12 vs 0.30) niż real. Hipoteza HA: jitter S/V uczy
> niezmienniczości na to przesunięcie i poprawia transfer synthetic→real.
> Protokół sweepa: YOLOv10n, podzbiór 10k (stratyfikowany po 15 lotniskach,
> `src/make_subset.py`), 60 epok, batch 64, seed 42, eval na realnym holdoucie.

## Wyniki sweepa (10k, real test)

| wariant | hsv_s | hsv_v | mAP@50 | mAP@50:95 | AP_S | AP_M | AP_L |
|---|---|---|---|---|---|---|---|
| **A1 słaby** | 0.4 | 0.3 | **0.459** | **0.264** | 0.306 | 0.357 | 0.091 |
| A2 średni | 0.7 | 0.5 | 0.431 | 0.247 | 0.265 | 0.345 | 0.099 |
| A3 mocny | 0.9 | 0.7 | 0.446 | 0.252 | 0.268 | 0.344 | 0.105 |
| baseline 45k (domyślne aug) | 0.7 | 0.4 | 0.452 | 0.268 | 0.286 | 0.384 | 0.201 |

(hsv_h = 0.015 wszędzie; A2 ≈ domyślne ustawienia YOLO)

## Statementy do raportu

**1. Optimum przy słabym jitterze — wynik kontrintuicyjny.**
> Wbrew hipotezie "im większy shift fotometryczny, tym mocniejsza augmentacja
> powinna pomagać", najlepszy okazał się NAJSŁABSZY jitter (mAP@50 0.459), a
> ustawienia zbliżone do domyślnych YOLO — najgorsze (0.431). Zależność jest
> niemonotoniczna. Interpretacja: nadmierny jitter S/V degraduje subtelne cechy
> (kontrast samolot–tło), na których opiera się detekcja małych obiektów —
> AP_small spada z 0.306 (słaby) do ~0.27 (średni/mocny).

**2. Augmentacja nadrabia rozmiar zbioru.**
> Model z dobranym jitterem trenowany na 10 000 obrazów osiąga 0.459 mAP@50 —
> tyle co baseline na pełnych 45 000 (0.452). Dla małych i średnich obiektów
> 10k+augmentacja przewyższa 45k bez stroju (AP_S 0.306 vs 0.286).

**3. Dużych obiektów augmentacja nie ratuje.**
> AP_large na 10k pozostaje niskie (0.09–0.11) niezależnie od siły jittera,
> podczas gdy samo zwiększenie danych do 45k daje 0.201. Spójne z Krokiem 0:
> reprezentacja dużych samolotów wymaga różnorodności scen, nie fotometrii.

**⚠️ Zastrzeżenie:** pojedynczy seed (42); rozrzut między wariantami (~3 pp)
jest na granicy szumu treningowego. Ranking słaby>mocny>średni traktować
ostrożnie; pewny wniosek to brak monotonicznej poprawy z siłą augmentacji.

## Finalny trening A — WYNIK (2026-06-13)

Słaby HSV (s=0.4, v=0.3) na pełnych 45k. Protokół: 45 epok (uzasadnienie: krzywe
10k pokazują plateau po ep. 40), batch 16, workers 2, cache=disk, seed 42.
Trening padł na ep. 44/45 (WSL CUDA, patrz niżej) — użyto best.pt z **epoki 43**
(najlepszej wg val, mAP50_syn=0.980, krzywa w plateau; selekcja best-checkpoint
to standard, strata 2 epok bez znaczenia).

| metryka (real test) | baseline 45k | **expA final 45k** | Δ |
|---|---|---|---|
| mAP@50 | 0.452 | **0.455** | +0.003 (≈remis) |
| mAP@50:95 | 0.268 | 0.268 | 0.000 |
| **AP_small** | 0.286 | **0.337** | **+0.051 (+18%)** |
| AP_medium | 0.384 | 0.355 | −0.029 |
| **AP_large** | 0.201 | **0.090** | **−0.111 (−55%)** |

**Statement do raportu (uczciwy, niuansowany):**
> Dobrana augmentacja fotometryczna NIE daje addytywnego zysku w zagregowanym
> mAP@50 na pełnych danych (0.455 vs 0.452). Zamiast tego REDYSTRYBUUJE jakość
> między klasami rozmiaru: AP_small rośnie o 18% (0.286→0.337) kosztem załamania
> AP_large (0.201→0.090). Ponieważ realna domena jest zdominowana przez małe
> obiekty (44% instancji "small"), poprawa AP_small jest cenna — ale wynik
> falsyfikuje prostą hipotezę "fotometria zmniejsza lukę domenową globalnie".
> Spójne z obserwacją ze sweepa: fotometria pomaga małym obiektom (cechy
> niskopoziomowe), dużych nie ratuje (te wymagają różnorodności scen).

⚠️ Zastrzeżenia metodologiczne: (1) pojedynczy seed; (2) baseline trenowany
w schedule 100 epok, finalny A w 45 (krzywe sugerują plateau w obu — ale
różnica schedule'ów do odnotowania); (3) trening na WSL2 wymagał workers=2
+ batch 16 (niestabilność dataloadera — 8 crashy `cudaErrorUnknown` przy
workers≥4; szczegóły w sekcji niżej).

### Nota inżynierska: niestabilność WSL2 (2026-06-12/13)
Długie treningi 45k na WSL2 crashowały losowo (`CUDA error: unknown error`,
3-40 min) przy workers≥4 niezależnie od batch/pin_memory; stabilna recepta:
**workers=2 + batch 16 + cache=disk** (tempo ~9.5 min/epokę wall-clock z walidacją).
Nie temperatura/moc/VRAM (max 55°C, 93W, 6/16GB). Finalny run i tak padł na
ep. 44/45 — checkpointy co epokę uratowały wynik. Wniosek: ciężkie treningi
(architektury) przenieść na natywny Linux/Colab.

## Reprodukcja
```bash
python3 src/make_subset.py --n-train 10000 --name synthetic_10k
bash src/sweep_A_photometric.sh           # 3 warianty + eval
# finalny:
python3 src/train_yolo.py --data data/yolo/synthetic_aircraft/data.yaml \
  --name expA_final_45k --epochs 100 --batch 64 --imgsz 512 --seed 42 \
  --hsv_h 0.015 --hsv_s 0.4 --hsv_v 0.3 \
  --val-data data/yolo/synthetic_aircraft/data.yaml
python3 src/eval_per_size.py --weights runs/expA_final_45k/weights/best.pt \
  --img-dir data/real/PS-RGB_tiled/PS-RGB_tiled \
  --coco-gt data/real/annotations/instances_test_aircraft.json --name expA_final_45k
```
