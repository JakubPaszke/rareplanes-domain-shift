# Analiza adnotacji RarePlanes (real vs synthetic) — notatki pod raport

> Wygenerowane przez `src/analyze_annotations.py`, surowe liczby w `results/annotation_stats.json`.
> Wszystkie statystyki liczone na adnotacjach COCO (kafelki 512×512 px), bez dotykania obrazów.
> Progi rozmiaru wg konwencji COCO: small <32² px, medium 32²–96² px, large >96² px.

## 1. Struktura zbioru i pliki adnotacji

RarePlanes na S3 (`s3://rareplanes-public`, public, no-sign-request). Adnotacje real:

| plik | klasy | uwaga |
|---|---|---|
| `instances_{split}_aircraft.json` | 1 (`aircraft`) | **czyste COCO, używamy do treningu** |
| `instances_{split}_role.json` | 3 (Small/Medium/Large Civil Transport) | **czyste COCO** |
| `RarePlanes_{Train,Test}_Coco_Annotations_tiled.json` | — | pole `categories` ZEPSUTE (zawiera kopie adnotacji jako "kategorie"). Bogate atrybuty (role, wingspan, num_engines, propulsion, lokalizacja, truncated, partialDec) — używać tylko jako źródło metadanych, NIE jako wejście treningowe |

**Decyzja inżynierska:** trening i ewaluacja na `instances_*` (poprawny schemat COCO). Bogate atrybuty z plików `*_tiled` posłużą do analizy jakościowej (np. AP per rola, per liczba silników).

Kafelki: 512×512 px. Nazewnictwo: `{loc_id}_{cat_id}_tile_{n}.png`.

## 2. Liczności (split train/test)

| zbiór | obrazy | instancje | inst./obraz (mean / median / max) |
|---|---|---|---|
| REAL train (aircraft) | 5 815 | 18 393 | 3.16 / 2 / 45 |
| REAL test (aircraft) | 2 710 | 6 812 | 2.51 / 2 / 36 |
| SYNTHETIC train (role) | 45 000 | 566 143 | **12.58** / 11 / 50 |
| SYNTHETIC test | 5 000 | 62 841 | 12.57 / 11 / 48 |

- **Synthetic train ma ~7.7× więcej obrazów i ~31× więcej instancji niż real train.** Gęstość samolotów na kafel ~4× wyższa (12.6 vs 3.2) — sceny syntetyczne są "zatłoczone" samolotami, real znacznie rzadsze.
- `role` ma mniej obrazów niż `aircraft` (real train 5567 vs 5815) — część samolotów nie ma przypisanej roli. Drobna niespójność do odnotowania przy treningu 3-klasowym.

## 3. KLUCZOWE: rozkład rozmiarów obiektów — twardy dowód domain shift

| zbiór | small | medium | large | mediana powierzchni [px²] |
|---|---|---|---|---|
| REAL train | **43.8%** | 44.0% | 12.2% | 1 180 |
| REAL test | 24.9% | 51.1% | 23.9% | 2 505 |
| SYNTHETIC train | **10.2%** | 51.3% | 38.5% | 6 776 |
| SYNTHETIC test | 10.1% | 51.8% | 38.0% | 6 675 |

**Statement do raportu:**
> W danych syntetycznych samoloty są średnio znacznie większe: mediana powierzchni bbox wynosi ~6.8 tys. px² wobec ~1.2 tys. px² w realnym treningu (≈5.7×). W realnym zbiorze treningowym **44% obiektów to "small" (COCO <32²)**, podczas gdy w syntetycznym tylko **~10%**. Model trenowany na syntetykach widzi głównie duże, dobrze widoczne samoloty — a w realnej domenie dominują obiekty małe. To jeden z głównych mechanizmów spadku transferu syntetyczne→rzeczywiste, jeszcze przed różnicami wyglądu (tekstura, kolor, tło).

Dodatkowo: **wymiary bbox** (px) — real train mean ~58×46, synthetic ~98×89. Synthetic to niemal 2× większe boki.

⚠️ **Uwaga metodologiczna (do uczciwej dyskusji w raporcie):** real train i real test też różnią się rozkładem (test ma mniej small: 25% vs 44%). Czyli sam real ma wewnętrzny shift train→test. Trzeba o tym wspomnieć, żeby nie przypisać całego spadku wyłącznie luce syntetyczne→rzeczywiste.

## 4. Rozkład klas (role) — niezbalansowanie

| klasa | REAL train | REAL test | SYNTHETIC train |
|---|---|---|---|
| Small Civil Transport/Utility | **57.6%** | 40.3% | 12.9% |
| Medium Civil Transport/Utility | 34.3% | 44.6% | 51.4% |
| Large Civil Transport/Utility | 8.1% | 15.1% | 35.7% |

**Statement do raportu:**
> Priory klas są odwrócone między domenami: w realnym treningu dominują małe samoloty (57.6%), w syntetycznym — średnie i duże (51% + 36%). Klasyfikator roli trenowany na syntetykach będzie miał błędny prior. To uzasadnia raportowanie **makro-F1 / AP per-klasa**, nie tylko metryki zagregowanej.

## 5. Wnioski operacyjne dla projektu

1. **Baseline detekcji robimy na `aircraft` (1 klasa)** — czysto mierzy zdolność lokalizacji; rola jako rozszerzenie (3 klasy) do analizy per-class.
2. **Raportować AP osobno dla small/medium/large** (wymóg PDF) — to tu zobaczymy największą różnicę real↔synthetic.
3. Hipotezy do późniejszych eksperymentów A/B/C, podpowiedziane już przez same adnotacje:
   - **H1 (rozmiar):** dopasować skalę/rozdzielczość wejścia lub augmentację skalą, by zniwelować różnicę rozmiarów obiektów.
   - **H2 (gęstość):** synthetic jest gęstszy — sprawdzić wpływ na NMS/recall.
   - **H3 (priory klas):** re-waging klas lub re-sampling przy treningu 3-klasowym.
   - (wygląd: kolor/tekstura/tło — do zbadania osobno color histogramem i FFT na obrazach, kolejny etap.)
