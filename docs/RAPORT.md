# Detekcja samolotów na zdjęciach lotniczych pod przesunięciem domeny syntetyczne→rzeczywiste

**Projekt 4 — Deep Learning / Computer Vision.** Autorzy: Miłosz, Jakub.
Repozytorium: https://github.com/JakubPaszke/rareplanes-domain-shift

> SZKIC ROBOCZY — wygenerowany z notatek `notes/00–09` i `results/tabela_zbiorcza.md`.
> Do uzupełnienia: wykresy (krzywe few-shot, krzywa skali), Grad-CAM, abstrakt, przegląd literatury.

---

## 1. Wstęp i pytanie badawcze

Detektory obiektów trenowane na danych syntetycznych są tanie w pozyskaniu (nieograniczone
adnotacje, pełna kontrola sceny), ale cierpią na **lukę domenową** — spadek jakości przy
przeniesieniu na dane rzeczywiste. Projekt bada tę lukę na zbiorze **RarePlanes** (satelitarna
detekcja samolotów, warianty rzeczywisty i syntetyczny) i systematycznie testuje metody jej
zmniejszania.

**Pytanie badawcze:** Które architektury i które interwencje (augmentacje, dopasowanie skali,
mieszanie z realnymi danymi) najskuteczniej zmniejszają lukę domenową syntetyczne→rzeczywiste
w detekcji małych obiektów na zdjęciach satelitarnych?

**Hipotezy testowane eksperymentami A/B/C/D:**
- **HA (fotometria):** dopasowanie kolorystyki/jasności syntetyków do real poprawia transfer.
- **HB (częstotliwości):** dodanie szumu/rozmycia upodabnia widmo syntetyków do ziarnistych
  realnych zdjęć → lepszy transfer.
- **HC (mixed training):** dodanie małej liczby realnych próbek do treningu silnie poprawia transfer.
- **HD (skala):** większa rozdzielczość wejścia pomaga widzieć małe realne obiekty.

---

## 2. Dane

**RarePlanes** (Shermeyer et al., 2020; CC BY-SA 4.0). Wariant obrazów: **PS-RGB tiled**
(kafelki 512×512, jedyny RGB porównywalny z syntetycznymi). Zadanie: detekcja klasy `aircraft`.

| zbiór | obrazy | instancje | inst./obraz |
|---|---|---|---|
| real train | 5 815 | 18 393 | 3.2 |
| real test (holdout) | 2 710 | 6 812 | 2.5 |
| synthetic train | 45 000 | 566 143 | 12.6 |

**Zmierzony domain shift (z analizy adnotacji + wyglądu, notes/01–02):**
- **Rozmiar obiektów:** real 44% obiektów "small" (COCO <32²), synthetic tylko ~10%; mediana
  powierzchni bbox real 1180 px² vs synthetic 6776 px² (5.7×). Synthetic = duże samoloty.
- **Gęstość:** synthetic ~12.6 inst/obraz vs real ~3.2 (4×).
- **Fotometria:** real ciemny/nasycony (jasność 72, nasycenie 0.30, ciepły R>G>B); synthetic
  jasny/wyblakły (133, 0.12, chłodny B>G>R).
- **Częstotliwości (FFT):** real ma więcej energii wysokoczęstotliwościowej (szum sensora) —
  synthetic gładszy.

---

## 3. Metoda i protokół

Model bazowy: **YOLOv10n** (2.3M param), pretrained COCO → fine-tune. Protokół jednolity:
imgsz 512 (poza eksp. D), seed 42, ewaluacja **cross-domain** na realnym holdoucie, metryki
COCO (mAP@50, mAP@50:95, AP per rozmiar S/M/L). Sweepy na podzbiorze stratyfikowanym 10k
(15 lotnisk) dla szybkich iteracji; baseline'y i finalne treningi na pełnych 45k.

Konwersja COCO→YOLO (`src/coco_to_yolo.py`), trening (`src/train_yolo.py`), ewaluacja per-size
przez pycocotools (`src/eval_per_size.py`). Pełna reprodukcja w sekcji 7.

---

## 4. Wyniki

### 4.1 Baseline'y (kotwice)

| | mAP@50 | AP_S | AP_M | AP_L |
|---|---|---|---|---|
| **A: real→real** (górny limit) | **0.974** | 0.759 | 0.782 | 0.899 |
| B: synthetic 45k→real (luka) | 0.452 | 0.286 | 0.384 | 0.201 |

Naiwny transfer synthetic→real traci ~54% mAP@50 (0.97→0.45). Sama różnorodność danych
syntetycznych pomaga: 3 lotniska (6460 obr) dały 0.41, pełne 15 lotnisk (45k) — 0.45.

### 4.2 Eksperymenty A/B/C/D (real test, podzbiór 10k)

| metoda | wariant | mAP@50 | Δ vs baseline |
|---|---|---|---|
| **C: mixed training** | 25% real | **0.947** | **+0.495** 🏆 |
| C: mixed training | 10% real | 0.896 | +0.444 |
| C: mixed training | 5% real | 0.852 | +0.400 |
| C: mixed training | 1% real | 0.667 | +0.215 |
| **D: skala** | imgsz 320 | **0.522** | +0.070 |
| **B: częstotliwości** | sam szum | 0.490 | +0.038 |
| **A: fotometria** | słaby HSV | 0.459 | +0.007 |
| *baseline 0% real* | — | 0.452 | — |

### 4.3 Porównanie architektur (10k)

| model | typ | param | mAP@50 (real) |
|---|---|---|---|
| YOLO11l | CNN | 25M | **0.467** |
| YOLOv10n | CNN | 2.3M | 0.459 |
| RT-DETR-x | Transformer | 67M | 0.380 |
| RT-DETR-l | Transformer | 32M | 0.297 |

---

## 5. Analiza i wnioski (weryfikacja hipotez)

**HC potwierdzona — najsilniejszy efekt.** Mixed training to zdecydowanie najskuteczniejsza
metoda: **1% realnych danych → +48% mAP@50** (0.45→0.67), a 25% → 0.947 (prawie poziom
pełnego real). Krzywa few-shot rośnie gwałtownie i nasyca się — kilka procent realnych próbek
daje większy zysk niż jakakolwiek augmentacja syntetyków.

**HD potwierdzona ODWROTNIE (kontrintuicyjnie).** Hipoteza zakładała "większa skala lepsza";
wynik jest monotonicznie odwrotny: imgsz 320 (0.522) > 512 (0.459) > 768 (0.448) > 1024 (0.330).
Niska rozdzielczość treningu działa jak implicytne dopasowanie skali — zmniejsza pozorny rozmiar
dużych syntetycznych samolotów ku małym realnym. Najsilniejsza interwencja czysto-syntetyczna.

**HB potwierdzona NIUANSOWO.** Pomaga sam szum (B2: 0.490, +8%), ale rozmycie szkodzi
(B1/B3: 0.451). Mechanizm zgodny z FFT: szum dodaje brakującą energię wysokoczęstotliwościową,
blur ją usuwa (przeciwskutek).

**HA potwierdzona słabo.** Fotometria daje marginalny zysk (0.459 vs 0.452); dobrana augmentacja
redystrybuuje jakość ku małym obiektom (AP_small +18% na 45k) kosztem dużych, ale nie zmniejsza
luki globalnie. Zależność niemonotoniczna — słaby HSV lepszy niż mocny.

**Architektury:** CNN transferują znacznie lepiej niż transformery (najlepszy CNN 0.467 vs
najlepszy transformer 0.380), mimo większej pojemności transformerów. RT-DETR (end2end bez NMS)
załamuje się na obcej domenie — zwraca 300 detekcji/obraz (wszystkie sloty dekodera). To
architektura, nie pojemność, decyduje o transferze.

**Ranking metod zmniejszania luki:** mixed training ≫ skala > szum > fotometria.

---

## 6. Ograniczenia

- Większość eksperymentów na podzbiorze 10k (nie pełne 45k) dla szybkości iteracji; baseline'y
  i finalny A potwierdzone na 45k. Synthetic 10k = 15 lotnisk (pełna różnorodność scen).
- Pojedynczy seed (42) — rozrzut wariantów rzędu kilku pp bywa na granicy szumu treningowego
  (zwłaszcza eksp. A).
- Architektury i eksp. D liczone na różnym sprzęcie (Colab A100/Blackwell) niż reszta
  (lokalny RTX 5070 Ti) — FPS nieporównywalny między rodzinami; ranking koszt-jakość wymaga
  pomiaru na wspólnym sprzęcie.
- Mixed training (C) to inny reżim (z realnymi danymi) niż czysto-syntetyczny — porównanie
  jest uczciwe co do celu (transfer na real), ale C wymaga dostępu do realnych adnotacji.

---

## 7. Reprodukcja

```bash
# dane (RarePlanes, S3 public)
bash src/download_synthetic.sh           # synthetic (HTTP, ~15 MB/s)
aws s3 cp s3://rareplanes-public/real/tarballs/test/RarePlanes_test_PS-RGB_tiled.tar.gz ...

# pipeline
python3 src/coco_to_yolo.py --domain synthetic --classes aircraft --val-frac 0.15
python3 src/make_subset.py --n-train 10000 --name synthetic_10k

# baseline + eksperymenty (przyklady)
python3 src/train_yolo.py --data data/yolo/real_aircraft/data.yaml --name real_baseline ...
python3 src/eval_per_size.py --weights runs/<name>/weights/best.pt \
    --img-dir data/real/PS-RGB_tiled/PS-RGB_tiled \
    --coco-gt data/real/annotations/instances_test_aircraft.json --name <name>
```
Szczegóły każdego eksperymentu: `notes/03–09`. Tabela wszystkich wyników: `results/tabela_zbiorcza.md`.

## 8. Licencja danych
RarePlanes na CC BY-SA 4.0. Atrybucja: Shermeyer et al., RarePlanes Dataset, June 2020.
Obrazy nie są redystrybuowane w repo (pobranie skryptem).
