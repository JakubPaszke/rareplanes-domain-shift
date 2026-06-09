# Baseline B — synthetic→real: pomiar luki domenowej (SEDNO PROJEKTU)

> Model: YOLOv10n, ten sam protokół co Baseline A (100 epok, imgsz 512, batch 64, seed 42).
> Trening: SYNTETYCZNE (6460 obr — Atlanta/Basel/Chicago, train 5503 / val 957, 86 250 instancji).
> Ewaluacja: **realny holdout test** (2710 obr, 6812 instancji) — model nigdy nie widział realnych danych.
> Metryki: `results/per_size/syn_to_real_baseline.json` vs `results/per_size/real_baseline.json`.

## ⚠️ Zastrzeżenie o danych
Mamy **6460 / 45000** obrazów syntetycznych (throttling anonimowego S3 do ~0.2 MB/s
zatrzymał pobieranie). To wciąż >real (4943 train) i pokrywa 3 lokalizacje
(Atlanta, Basel, Chicago) — wystarcza na wiarygodny baseline luki domenowej.
Pełne 45k powtórzymy, gdy łącze wróci; spodziewamy się, że większa różnorodność
scen *zmniejszy* lukę, ale nie odwróci wniosku.

## Tabela porównawcza — luka domenowa

| metryka | A: real→real | B: synthetic→real | spadek względny |
|---|---|---|---|
| **mAP@50** | 0.980 | **0.410** | **−58.2%** |
| mAP@50:95 | 0.813 | 0.239 | −70.6% |
| AP@.75 | 0.915 | 0.252 | −72.5% |
| AP_small | 0.759 | 0.262 | −65.5% |
| AP_medium | 0.783 | 0.328 | −58.1% |
| **AP_large** | 0.899 | **0.067** | **−92.5%** |
| liczba detekcji (conf=.001) | 25 763 | **91 490** | ×3.55 |

## Kluczowe statementy do raportu

**1. Drastyczny spadek transferu.** Model trenowany wyłącznie na syntetykach traci
~58% mAP@50 (0.98→0.41) i ~71% mAP@50:95 na realnej domenie. To potwierdza
założenie projektu: naiwny transfer synthetic→real jest niewystarczający.

**2. Paradoks dużych obiektów (najważniejsza obserwacja).** AP_large spada
najmocniej ze wszystkich (−92.5%, do 0.067), choć w domenie real duże obiekty
są najłatwiejsze (Baseline A: AP_large 0.90 — najwyższe). Wyjaśnienie spina się
z analizą adnotacji (`notes/01`): synthetic ma ~38% obiektów "large" i medianę
powierzchni 6776 px² (vs real 1180). Model nauczył się rozpoznawać DUŻE,
renderowane samoloty; realne duże samoloty mają inny wygląd/teksturę/kontekst,
więc reprezentacja "large" zupełnie nie transferuje. Małe i średnie radzą sobie
względnie lepiej (AP_small 0.26, AP_medium 0.33), prawdopodobnie bo cechy
niskopoziomowe (krawędzie) są mniej domenowo-zależne niż wygląd dużych obiektów.

**3. Masowe fałszywe alarmy.** Model syntetyczny generuje 3.55× więcej detekcji
(91 490 vs 25 763). Niska precyzja → halucynacje na elementach tła realnych scen
(pasy, drogi kołowania, budynki), których "czyste" syntetyki nie zawierały.
Wizualizacja: `results/error_analysis_syn_vs_real.png` (góra: model real trafia
w GT; dół: model synthetic rozrzuca boxy i gubi prawdziwe samoloty).

**4. Recall też cierpi.** AR@100 0.37 (vs 0.85 real). Model nie tylko zmyśla,
ale i pomija realne obiekty — szczególnie duże (AR_large 0.07).

## Powiązanie z analizą domenową (zamknięcie pętli)
Luka domenowa ma trzy zmierzone wcześniej źródła, które teraz manifestują się w metrykach:
- **rozmiar/rozkład obiektów** (`notes/01`) → spadek AP, zwłaszcza paradoks large;
- **fotometria** — jasność/nasycenie/balans (`notes/02`) → fałszywe alarmy na tle;
- **częstotliwości** — synthetic gładszy (`notes/02`) → wrażliwość na realny szum/teksturę.

To definiuje hipotezy do fazy poprawiania (eksperymenty A/B/C): augmentacje
fotometryczne (HA), degradacja częstotliwościowa (HB), dopasowanie skali (H1),
oraz fine-tuning na małej liczbie realnych próbek (mixed training).

## Reprodukcja
```bash
python3 src/coco_to_yolo.py --domain synthetic --classes aircraft --val-frac 0.15
python3 src/train_yolo.py --data data/yolo/synthetic_aircraft/data.yaml \
    --name syn_baseline_yolov10n --epochs 100 --batch 64 --imgsz 512 --seed 42 \
    --val-data data/yolo/synthetic_aircraft/data.yaml
python3 src/eval_per_size.py --weights runs/syn_baseline_yolov10n/weights/best.pt \
    --img-dir data/real/PS-RGB_tiled/PS-RGB_tiled \
    --coco-gt data/real/annotations/instances_test_aircraft.json --name syn_to_real_baseline
```
