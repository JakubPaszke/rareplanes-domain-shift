# Porównanie architektur pod domain shift: CNN vs Transformer (kwadrat)

> Wszystkie trenowane na synthetic 10k (te same listy plików, 15 lotnisk),
> ewaluacja cross-domain na realnym holdoucie (2710 obr). Transformery na Colab
> (RTX PRO 6000 Blackwell / A100). Wyniki: `results/per_size/*.json`.

## Kwadrat architektur — real test mAP@50

| | **mały** | **duży** |
|---|---|---|
| **CNN** | YOLOv10n (2.3M) **0.459** | YOLO11l (25M) **0.467** |
| **Transformer** | RT-DETR-l (32M) **0.297** | RT-DETR-x (67M) **0.380** |

### Pełne metryki (real test)

| model | typ | param | mAP@50 | mAP@50:95 | AP_S | AP_M | AP_L | det/obraz |
|---|---|---|---|---|---|---|---|---|
| YOLOv10n | CNN | 2.3M | 0.459 | 0.264 | 0.306 | 0.357 | 0.091 | ~27 |
| YOLO11l | CNN | 25M | **0.467** | **0.271** | 0.306 | 0.348 | 0.108 | ~22 |
| RT-DETR-l | Transf | 32M | 0.297 | 0.158 | 0.146 | 0.230 | 0.081 | **300** |
| RT-DETR-x | Transf | 67M | 0.380 | 0.205 | 0.206 | 0.285 | 0.094 | **300** |

(syn val: wszystkie ~0.93-0.97 — każda architektura uczy się synthetic świetnie)

## GŁÓWNE WNIOSKI (do raportu)

**1. CNN transferują cross-domain ZNACZNIE lepiej niż transformery.**
> Oba CNN-y (YOLOv10n, YOLO11l) osiągają ~0.46 mAP@50 na realnym teście, oba
> transformery (RT-DETR-l/-x) wyraźnie mniej (0.30-0.38). Najlepszy CNN bije
> najlepszy transformer o 23% (0.467 vs 0.380), mimo że transformery mają więcej
> parametrów. Wszystkie uczą się synthetic niemal idealnie (~0.95+) — różnica
> jest CZYSTO w generalizacji na obcą domenę.

**2. Korekta hipotezy (uczciwie): pojemność transformera NIE szkodzi liniowo.**
> Wstępna hipoteza brzmiała "większy transformer = większy overfit = gorszy transfer".
> Wynik ją FALSYFIKUJE: RT-DETR-x (67M) transferuje LEPIEJ niż RT-DETR-l (32M) —
> 0.380 vs 0.297. Większy transformer radzi sobie nieco lepiej, nie gorzej. Główna
> teza (CNN > Transformer w transferze) stoi, ale mechanizm "im większy tym gorzej"
> jest błędny. Analogicznie po stronie CNN: YOLO11l (25M) ≈ YOLOv10n (2.3M) — rozmiar
> ma marginalny wpływ na transfer w obu rodzinach. **To architektura, nie pojemność,
> decyduje o transferze.**

**3. Strukturalne załamanie end2end (brak NMS) na domain shift.**
> OBA transformery zwracają dokładnie 300 detekcji/obraz (= max_det) na realnym
> teście — wszystkie zapytania dekodera z niskim confidence. Na obcej domenie model
> nie ma pewnych detekcji, a architektura end2end (bez NMS/progu) wypluwa pełen
> zestaw słabych predykcji. CNN-y z NMS filtrują do ~22-27/obraz. To strukturalna,
> nie tylko jakościowa różnica w odporności na shift — i prawdopodobne źródło
> niskiej precyzji transformerów cross-domain.

**4. Małe obiekty: domena CNN.** AP_small CNN ~0.306 vs transformery 0.15-0.21.
Realna domena to w 44% małe obiekty (notes/01) — stąd przewaga CNN w mAP@50.

## Nota metodologiczna
RT-DETR trudny w treningu: wymaga lr=1e-4 + cosine + warmup (auto-lr 0.002 → nan).
Wstępny RT-DETR-l "0.489" był artefaktem niedotrenowania (ep.2) — po stabilizacji
spadł do prawdziwych 0.297. To samo w sobie obserwacja: transformery wymagają
więcej strojenia (koszt inżynierski w rankingu koszt-jakość).

## TODO
- FPS wszystkich 4 na TYM SAMYM sprzęcie (uczciwy ranking koszt-jakość; teraz YOLO@5070Ti, transf.@Blackwell).
- (opcja) czy transformery z mocniejszym progiem confidence / NMS-postproc nadrabiają precyzję?
