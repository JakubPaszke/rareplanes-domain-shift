# Baseline A — detektor trenowany i testowany na danych RZECZYWISTYCH

> Model: YOLOv10n (2.27M param), pretrained COCO → fine-tune na RarePlanes real.
> Trening: 100 epok, imgsz 512, batch 64, seed 42, RTX 5070 Ti.
> Dane: `data/yolo/real_aircraft` (train 4943 / val 872 / test 2710 kafli).
> Ewaluacja na holdout **test** (2710 obr, 6812 instancji), klasa: aircraft (1).
> Metryki: `results/baselines/real_baseline_yolov10n.json`, `results/per_size/real_baseline.json`.

## Wyniki — punkt odniesienia "górny" (in-domain)

| metryka | wartość |
|---|---|
| **mAP@50** | **0.980** |
| mAP@50:95 | 0.813 |
| precision | 0.963 |
| recall | 0.949 |
| AP@.75 | 0.915 |

**Spełniony próg z wymagań:** mAP@50 = 0.98 > 0.95. ✅

## AP per rozmiar obiektu (COCO, wymóg PDF)

| rozmiar | AP@[.5:.95] | AR@100 |
|---|---|---|
| small (<32²) | **0.759** | 0.806 |
| medium (32²–96²) | 0.783 | 0.826 |
| large (>96²) | **0.899** | 0.932 |

**Statement do raportu:**
> Nawet w reżimie in-domain (trening i test na danych rzeczywistych) jakość rośnie
> monotonicznie z rozmiarem obiektu: AP small 0.76 < medium 0.78 < large 0.90.
> Małe samoloty są wewnętrznie trudniejsze do detekcji — co jest spójne z analizą
> adnotacji (`notes/01`), gdzie 44% obiektów real to klasa "small". Ten gradient
> rozmiaru będzie punktem odniesienia przy ocenie transferu z syntetyków, gdzie
> małych obiektów jest mało (~10%) i spadek AP_small powinien być najdotkliwszy.

## Koszt obliczeniowy (wymóg PDF: FPS, parametry)

| | wartość |
|---|---|
| parametry | 2.27 M |
| FPS (batch 64, 512px) | **149.6** |
| czas inferencji | 6.69 ms/obraz |
| GPU mem (trening, batch 64) | ~10 GB |
| czas treningu (100 epok) | ~50 min (≈33 s/epokę) |

## Protokół (reprodukcja)
```bash
python3 src/coco_to_yolo.py --domain real --classes aircraft --val-frac 0.15
python3 src/train_yolo.py --data data/yolo/real_aircraft/data.yaml \
    --name real_baseline_yolov10n --epochs 100 --batch 64 --imgsz 512 --seed 42
python3 src/eval_per_size.py --weights runs/real_baseline_yolov10n/weights/best.pt \
    --img-dir data/real/PS-RGB_tiled/PS-RGB_tiled \
    --coco-gt data/real/annotations/instances_test_aircraft.json --name real_baseline
```

## Następny krok
Baseline B: ten sam model+protokół, ale trening na danych SYNTETYCZNYCH, test na
tym samym realnym holdoucie. Oczekiwany duży spadek (zwłaszcza AP_small) — to
zmierzy lukę domenową, którą potem zmniejszamy eksperymentami A/B/C.
