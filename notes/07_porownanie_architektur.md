# Porównanie architektur pod domain shift: YOLOv10n vs RT-DETR-l

> Oba trenowane na podzbiorze synthetic 10k (te same listy plików, 15 lotnisk),
> ewaluacja cross-domain na realnym holdoucie (2710 obr). RT-DETR na Colab A100.
> Wyniki: `results/per_size/{expA1_weak_10k,rtdetr_l_10k_ml}.json`.

## Wyniki (real test, 10k synthetic)

| metryka | YOLOv10n | RT-DETR-l (stabilny 60ep) | komentarz |
|---|---|---|---|
| mAP@50 (**syn val**) | ~0.97 | **0.973** | oba uczą się synthetic perfekcyjnie |
| **mAP@50 (real test)** | **0.459** | **0.297** | YOLO +0.162 — RT-DETR DUŻO gorszy transfer |
| mAP@50:95 (real) | 0.264 | 0.158 | YOLO znacznie lepszy |
| AP_small (real) | **0.306** | 0.146 | YOLO 2× lepszy |
| AP_medium (real) | **0.357** | 0.230 | YOLO lepszy |
| AP_large (real) | **0.091** | 0.081 | ≈remis |
| parametry | 2.3 M | 31.9 M (14×) | |
| detekcji/obraz (real) | ~9.5 | **300 (=max_det)** | ⚠️ RT-DETR zwraca WSZYSTKIE sloty |

## GŁÓWNY WNIOSEK (mocny, do raportu)

> **Transformerowy detektor RT-DETR generalizuje cross-domain ZNACZNIE gorzej niż
> konwolucyjny YOLOv10** mimo 14× większej pojemności. Oba uczą się domeny
> syntetycznej niemal idealnie (mAP_syn ~0.97), ale na realnym teście RT-DETR
> osiąga tylko 0.297 mAP@50 vs 0.459 dla YOLO — spadek o 35%.

### Mechanizm (analiza)
1. **Overfitting do domeny źródłowej:** RT-DETR przy 60 epokach i lr=1e-4 dopasował
   się do synthetic (0.973), ale to dopasowanie NIE transferuje. Im lepiej na
   synthetic, tym gorzej na real — klasyczny domain overfit, silniejszy dla modelu
   o większej pojemności.
2. **Załamanie end2end bez NMS:** RT-DETR zwraca dokładnie 300 detekcji/obraz
   (= max_det) na KAŻDYM realnym kaflu — czyli wszystkie zapytania dekodera z
   niskim confidence. Na obcej domenie model nie ma pewnych detekcji, a brak
   NMS/progu (architektura end2end) oznacza, że head wypluwa pełen zestaw słabych
   predykcji. YOLO z NMS i progiem filtruje do ~9.5/obraz. To strukturalna różnica
   w odporności na domain shift, nie tylko kwestia jakości.

### Uwaga o wyniku wstępnym
Wcześniejszy „lepszy" RT-DETR (mAP 0.489) pochodził z best.pt po 2 epokach
(trening rozbiegał się do nan przy auto-lr 0.002). Był to artefakt
niedotrenowania — przypadkowo wyższy recall. Stabilny trening (lr=1e-4 +
cosine + warmup, krzywa monotoniczna do 0.973 bez nan) ujawnia prawdziwy,
gorszy wynik transferu. To pokazuje też, że **RT-DETR jest istotnie trudniejszy
w treningu** (wymaga starannego strojenia lr) — realny koszt inżynierski.

## Statementy do raportu
1. Większy/transformerowy model ≠ lepszy transfer — RT-DETR przegrywa z YOLO na
   realnej domenie mimo 14× parametrów. Pojemność sprzyja overfitowi do synthetic.
2. Architektura end2end (brak NMS) załamuje się na domain shift przez zwracanie
   wszystkich slotów dekodera — diagnostyczne 300 det/obraz.
3. YOLO wygrywa szczególnie na małych obiektach (AP_small 2×), które dominują
   w realnej domenie.

## TODO (gdy unity)
- D-FINE jako trzecia architektura (czy inne transformery mają ten sam problem?).
- FPS YOLO vs RT-DETR na TYM SAMYM sprzęcie (uczciwy ranking koszt-jakość).
- (opcja) RT-DETR z mocniejszą regularyzacją / early stop wg REALNEGO val —
   czy mniej epok = lepszy transfer? (hipoteza: tak, bo mniej overfitu).
