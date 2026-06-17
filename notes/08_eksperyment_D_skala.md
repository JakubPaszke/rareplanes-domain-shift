# Eksperyment D — wpływ rozdzielczości wejścia (skala) na transfer

> YOLOv10n, synthetic 10k, trening+ewaluacja w tej samej rozdzielczości, eval
> cross-domain na realnym holdoucie (2710 obr). batch auto (AutoBatch), 45 epok,
> seed 42. Colab A100. Wyniki: `results/per_size/expD_*_10k_ml.json`.

## Krzywa skali (real test)

| imgsz | mAP@50 | mAP@50:95 | AP_S | AP_M | AP_L | det/obraz |
|---|---|---|---|---|---|---|
| **320** | **0.522** | **0.283** | **0.308** | **0.368** | **0.151** | 54 |
| 512 (ref, A1) | 0.459 | 0.264 | 0.306 | 0.357 | 0.091 | ~27 |
| 768 | 0.448 | 0.252 | 0.230 | 0.339 | 0.116 | 15 |
| 1024 | 0.330 | 0.190 | 0.222 | 0.250 | 0.047 | 7 |

## GŁÓWNY WNIOSEK — hipoteza HD SFALSYFIKOWANA (kontrintuicyjnie)

> Hipoteza HD zakładała: większa rozdzielczość wejścia → realne małe obiekty
> lepiej widoczne → lepszy transfer. **Wynik jest dokładnie ODWROTNY i monotoniczny:**
> im MNIEJSZA rozdzielczość, tym LEPSZY transfer. mAP@50 spada monotonicznie
> 0.522 (320) → 0.459 (512) → 0.448 (768) → 0.330 (1024). Najmniejsza testowana
> skala daje najlepszy wynik — o 58% lepszy niż największa (0.522 vs 0.330).

### Interpretacja (mechanizm)
Kluczem jest **luka rozmiarów obiektów** (notes/01): synthetic ma duże samoloty
(mediana 6776 px²), real małe (1180 px²). Trening w wysokiej rozdzielczości sprawia,
że model uczy się cech dużych syntetycznych obiektów w pełnym detalu — a te NIE
transferują na małe realne. **Trening w NISKIEJ rozdzielczości (320) działa jak
implicytne dopasowanie skali:** zmniejsza pozorny rozmiar dużych syntetycznych
samolotów, zmuszając model do uczenia się reprezentacji bliższej małym realnym
obiektom. To "downscaling jako domain adaptation".

Spójne z liczbą detekcji: imgsz 1024 zwraca tylko 7 det/obraz (model "nie widzi"
realnych małych obiektów w wysokiej rozdz.), 320 zwraca 54 (agresywnie wykrywa).

### Powiązanie z innymi eksperymentami
- **Najlepszy wynik całego projektu dotąd: imgsz 320 = mAP@50 0.522** — bije
  baseline 45k (0.452) i eksp A (0.455). Sama redukcja skali to najskuteczniejsza
  pojedyncza interwencja zmniejszająca lukę domenową.
- Łączy się z eksp A (fotometria): tam AP_small rósł kosztem AP_large; tu MAŁA
  skala podnosi OBA (AP_S 0.308, AP_L 0.151) — to mocniejszy, globalny zysk.

## Statementy do raportu
1. Redukcja rozdzielczości treningu (512→320) zmniejsza lukę domenową bardziej niż
   augmentacje fotometryczne — najprostsza interwencja, największy zysk.
2. Wysoka rozdzielczość SZKODZI transferowi synthetic→real (1024: mAP 0.330, AP_L
   0.047) — model uczy się detali dużych syntetycznych obiektów, które nie istnieją
   w małych realnych. Klasyczny przykład, że "więcej szczegółu" ≠ lepsza generalizacja.
3. Mechanizm: niska rozdzielczość = implicytne dopasowanie skali obiektów między domenami.

## TODO / kolejny krok (hipoteza wynikajaca)
- Czy jeszcze mniejsza skala (256, 224) daje jeszcze lepszy transfer, czy jest minimum?
- Łączenie: imgsz 320 + augmentacje A (słaby HSV) + mixed training (eksp C Miłosza)
  — czy zyski się sumują? To kandydat na FINALNY najlepszy model.
- Finalny imgsz 320 na pełnych 45k (nie 10k) — czy zysk się utrzymuje przy skali danych.
