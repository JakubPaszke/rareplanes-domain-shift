# Porównanie architektur pod domain shift: YOLOv10n vs RT-DETR-l

> Oba trenowane na podzbiorze synthetic 10k (te same listy plików, 15 lotnisk),
> ewaluacja cross-domain na realnym holdoucie (2710 obr). RT-DETR liczony na Colab A100.
> Wyniki: `results/per_size/{expA1_weak_10k,rtdetr_l_10k_ml}.json`.

## Wyniki (real test, 10k synthetic)

| metryka | YOLOv10n | RT-DETR-l | komentarz |
|---|---|---|---|
| mAP@50 | 0.459 | **0.489** | RT-DETR +0.030 |
| mAP@50:95 | 0.264 | 0.270 | ≈remis |
| AP_small | **0.306** | 0.238 | YOLO lepszy o 29% |
| AP_medium | 0.357 | 0.335 | YOLO lekko lepszy |
| AP_large | 0.091 | **0.222** | RT-DETR 2.4× lepszy |
| parametry | 2.3 M | 31.9 M | RT-DETR 14× większy |
| inferencja (A100) | — | 3.2 ms | RT-DETR (YOLO mierzony lokalnie 6.7ms@5070Ti) |
| detekcji/obraz (real) | ~9.5 | **~300** | ⚠️ RT-DETR mocno przeszacowuje |

## ⚠️ KRYTYCZNE zastrzeżenie metodologiczne
RT-DETR **NIE wytrenował się stabilnie**: trening oscylował dziko (ep.2 mAP_syn 0.967
→ ep.5 0.014 → ep.13 0.59 → ep.22 **loss=nan**, rozbieżność). EarlyStopping cofnął do
**best.pt z epoki 2**. Optimizer=auto dobrał AdamW lr=0.002 — za dużo dla transformerowego
detektora; RT-DETR potrzebuje lr≈1e-4 + cosine schedule + dłuższego warmupu.

Konsekwencja: wynik RT-DETR pochodzi z modelu po 2 epokach (niedouczony, recall-heavy
— 300 detekcji/obraz vs ~9.5 dla YOLO). mAP@50 0.489 wynika z wysokiego recall przy
NISKIEJ precyzji. **To NIE jest uczciwa górna granica RT-DETR** — to dolna granica
osiągnięta przypadkiem na ep.2. Do raportu: zaznaczyć jako wynik wstępny / pokazać
niestabilność treningu RT-DETR jako osobną obserwację (transformer trudniejszy w
treningu niż YOLO przy tym budżecie i schedule).

## Statementy do raportu (ostrożne)
1. **Transformer (RT-DETR) ma przewagę na dużych obiektach** (AP_large 0.222 vs 0.091)
   — globalna uwaga lepiej modeluje kontekst dużych samolotów. Spójne z naszym
   wątkiem "duże obiekty wymagają kontekstu/różnorodności".
2. **YOLO lepszy na małych obiektach** (AP_small 0.306 vs 0.238) mimo 14× mniej
   parametrów — efektywność konwolucyjnego backbone'u dla drobnych struktur.
3. **RT-DETR jest znacznie trudniejszy w treningu** — wymaga starannego strojenia lr
   (rozbieżność do nan przy auto-lr); YOLOv10 "po prostu działa". To realny koszt
   inżynierski przy wyborze architektury, istotny dla rankingu koszt-jakość.

## TODO (gdy będą unity / stabilny setup)
- Powtórzyć RT-DETR z lr0=1e-4, cos_lr=True, warmup_epochs=5 — uczciwy wynik.
- D-FINE jako trzecia architektura.
- FPS YOLO i RT-DETR na TYM SAMYM sprzęcie (teraz YOLO@5070Ti, RT-DETR@A100 — nieporównywalne).
