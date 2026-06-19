# Eksperyment B — degradacja częstotliwościowa synthetic

> Motywacja (notes/02): synthetic jest "gładszy" niż real — ma mniej energii
> wysokoczęstotliwościowej (analiza FFT). Hipoteza HB: dodanie szumu/rozmycia
> upodabnia widmo synthetic do realnych ziarnistych zdjęć satelitarnych → lepszy transfer.
> YOLOv10n, synthetic 10k, degradacja zapisana jako pliki (NIE on-the-fly — patrz nota),
> eval cross-domain na realnym holdoucie. Trening na natywnym Windows GPU.

## Wyniki (real test)

| wariant | degradacja | mAP@50 | mAP@50:95 | AP_S | AP_M | AP_L | det/obraz |
|---|---|---|---|---|---|---|---|
| **B2** | **sam szum (σ≤8)** | **0.490** | — | 0.283 | 0.382 | 0.119 | 44 |
| B1 | rozmycie(0.4)+szum(5) | 0.451 | — | 0.261 | 0.358 | 0.101 | 30 |
| B3 | rozmycie(0.6)+szum(6)+jpeg(75) | 0.451 | — | 0.294 | 0.356 | 0.101 | 42 |
| baseline 45k (ref) | — | 0.452 | 0.268 | 0.286 | 0.384 | 0.201 | — |
| eksp A1 (10k ref) | — | 0.459 | 0.264 | 0.306 | 0.357 | 0.091 | — |

## GŁÓWNY WNIOSEK

> **Sam Gaussian noise pomaga transferowi (B2: 0.490 vs baseline 0.452, +8%), ale
> rozmycie SZKODZI.** Hipoteza HB potwierdza się częściowo i w sposób niuansowany:
> kluczowy jest SZUM (upodabnia syntetyki do ziarnistych realnych zdjęć satelitarnych),
> a NIE rozmycie. Warianty z blur (B1, B3) wypadają gorzej (0.451) — na poziomie
> baseline, bo rozmycie usuwa drobne szczegóły potrzebne do detekcji małych obiektów.

### Interpretacja (mechanizm)
- **B2 (sam noise) najlepszy:** dodanie szumu Gaussa imituje szum sensora/kompresji
  realnych obrazów satelitarnych. Model uczy się być odporny na tę ziarnistość,
  co transferuje na realną domenę. Spójne z FFT (notes/02): real ma więcej energii
  wysokoczęstotliwościowej, noise ją dodaje do synthetic.
- **Blur szkodzi:** rozmycie Gaussa USUWA wysokie częstotliwości (odwrotnie do celu!)
  — czyli oddala widmo synthetic od real zamiast je upodabniać. Stąd B1/B3 (z blur)
  na poziomie baseline. To koryguje naiwne "blur+noise pomaga" — same te operacje
  mają przeciwny wpływ na widmo.
- **JPEG (B3) neutralny:** dodanie kompresji JPEG do blur+noise nie zmienia wyniku
  vs B1 (oba 0.451) — artefakty JPEG nie pomagają ani nie szkodzą znacząco.

## Statementy do raportu
1. Degradacja częstotliwościowa pomaga transferowi, ale TYLKO szum — nie rozmycie.
   B2 (sam noise) +8% mAP@50 vs baseline; warianty z blur bez zysku.
2. Mechanizm zgodny z analizą FFT: szum dodaje brakującą energię wysokoczęstotliwościową
   (synthetic gładszy niż real), blur ją usuwa (przeciwskutek).
3. To trzecia interwencja po fotometrii (A) i skali (D) — skala (imgsz320, 0.522)
   pozostaje najsilniejsza; noise (0.490) drugi; fotometria (~0.46) i baseline (0.45).

## ⚠️ Nota inżynierska: degradacja on-the-fly vs pliki
Pierwotny skrypt Miłosza (`train_yolo_freq_onfly.py`) degradował obrazy "w locie"
przez monkey-patch `cv2.imread`. NIE DZIAŁA na ultralytics 8.4.69 — biblioteka
czyta obrazy przez własną `ultralytics.utils.patches.imread` (nie cv2.imread),
więc patch jest trafiany 0×. Skutek: 3 warianty trenowały na CZYSTYCH obrazach,
dając identyczny wynik (box_loss 1.5488, mAP 0.463 wszystkie). Wykryte przez
identyczność wyników. Rozwiązanie: **degradacja zapisana jako pliki**
(`make_frequency_degraded_dataset.py`) — pewne, niezależne od wersji ultralytics.
Potwierdzenie: różne box_loss epoki 1 (B1 1.562, B2 1.583, B3 1.562) = degradacja
faktycznie zastosowana. Trening na natywnym Windows GPU (WSL2 niestabilny pod
długim treningiem — cudaErrorUnknown).
