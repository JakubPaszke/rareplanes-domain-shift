# Plan podziału pracy — Miłosz / Jakub

Projekt 4: Detekcja samolotów pod przesunięciem domeny (RarePlanes, synthetic→real).
Realizacja w parze. Cel podziału: **nie dublować roboty**, wykorzystać mocne strony sprzętu.

## Sprzęt
- **Jakub** — mocny GPU lokalny (RTX 5070 Ti 16 GB). → ciężkie/długie treningi, pełne 45k, większe modele, porównanie architektur.
- **Miłosz** — Google Colab (GPU z limitami czasu/pamięci, sesje ~12h, możliwe rozłączenia). → lżejsze treningi, analizy, augmentacje na podzbiorach, wizualizacje.

## Współdzielenie
- **Kod: GitHub** (to repo). Oboje pracują na repo, commitują, pull/push.
- **Dane: każdy pobiera sam** skryptem `src/download_synthetic.sh` (dane NIE w repo — CC BY-SA + rozmiar 145 GB).
  - Miłosz na Colab: sklonować repo, pobrać real (tarballe, ~3 GB) + podzbiór/całość synthetic wg potrzeb zadania.
  - Uwaga Colab: 145 GB synthetic się NIE zmieści na darmowym Colab. Miłosz pracuje na **podzbiorze** (np. 6-10k) lub montuje Google Drive.
- **Wyniki**: każdy commituje swoje `results/<eksperyment>.json` + notatki. Lekkie JSON-y idą do repo, ciężkie wagi (`*.pt`) NIE (gitignore).

## Zasada anty-kolizji
- **Gałęzie git per osoba/eksperyment**: `exp/A-fotometria-jakub`, `exp/C-mixed-milosz` itd. Merge do `main` po skończeniu.
- **Nazewnictwo runów**: prefiks inicjałem — `runEXP_A_jk`, `runEXP_C_ml`. Nie nadpisujemy swoich wyników.
- **Seed wspólny = 42** dla porównywalności. Protokół treningu jednolity (`src/train_yolo.py`).
- **Test set święty**: realny holdout (`instances_test_aircraft.json`) — NIKT nie trenuje na nim, służy tylko do finalnej ewaluacji cross-domain.

---

## PODZIAŁ ZADAŃ

### ✅ Zrobione (Jakub, wspólna baza)
- Pobranie + analiza danych (adnotacje, color histogram, FFT) → `notes/01,02`
- Baseline A (real→real): mAP@50 **0.980** → `notes/03`
- Baseline B (synthetic→real) na 6460 obr: mAP@50 **0.410** (luka domenowa) → `notes/04`
- Pipeline: `coco_to_yolo.py`, `train_yolo.py`, `eval_per_size.py`, `download_synthetic.sh`

### 🔵 JAKUB (mocny GPU — ciężkie treningi na pełnych 45k)
- **Krok 0**: Baseline B na pełnych 45k (15 lotnisk) — w toku (wstrzymany). Ustala punkt odniesienia na komplecie.
- **Eksperyment A — augmentacje fotometryczne**: HSV jitter / histogram matching, trening na 45k. Motywacja: jasność 72 vs 133, nasycenie 0.30 vs 0.12.
- **Eksperyment D — skala/kafelkowanie**: dopasowanie rozmiaru wejścia / multi-scale pod rozkład rozmiarów obiektów (synthetic large vs real small).
- **Porównanie architektur**: RT-DETR / D-FINE vs YOLOv10 na najlepszej konfiguracji (ciężkie, wymaga mocnego GPU). Wymóg PDF.
- **Finalna weryfikacja** najlepszego modelu na pełnym realnym teście + pomiar FPS/pamięci.

### 🟢 MIŁOSZ (Colab — lżejsze, na podzbiorze synthetic ~6-10k)
- **Eksperyment B — degradacja częstotliwościowa**: blur + Gaussian noise / degradacja, by upodobnić widmo synthetic do real. Lekkie augmentacje, mieści się w Colab. Motywacja: real ma więcej energii wysokoczęstotliwościowej (FFT).
- **Eksperyment C — mixed training**: synthetic + mała porcja real (1%, 5%, 10%, 25% realnych próbek). Krzywa few-shot. Mniejsze treningi, idealne na Colab.
- **Analizy per-bbox**: color histogram / FFT liczony TYLKO wewnątrz bboxów (na samolotach, nie tle) — rozdzielenie "shift tła" od "shift obiektu". Czysto CPU/analityczne.
- **Wizualizacje błędów**: Grad-CAM / mapy aktywacji dla wybranych modeli, galerie fałszywych alarmów. Wymóg PDF (interpretowalność).

### 🟡 WSPÓLNE (na koniec)
- Tabela zbiorcza wszystkich eksperymentów (mAP@50, mAP@50:95, AP S/M/L, FPS) vs baseline.
- Raport (struktura artykułu): wstęp, pytanie badawcze, dane, metoda, protokół, wyniki, analiza, ograniczenia, wnioski.
- Wpis do arkusza Google Sheets (numer projektu, repo, raport, dane).
- (opcjonalnie) Notebook `.ipynb` podsumowujący — ładuje `results/*.json`, generuje tabele+wykresy do raportu.

## Kolejność / zależności
1. Jakub kończy **Krok 0** (Baseline B 45k) → ustala referencję na komplecie.
2. Równolegle: Jakub robi A+D, Miłosz robi B+C (różne kierunki, zero kolizji).
3. Najlepsze konfiguracje z A/B/C/D → Jakub odpala porównanie architektur.
4. Wspólnie: synteza wyników + raport.
