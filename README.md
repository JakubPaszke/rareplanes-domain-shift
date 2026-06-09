# Detekcja samolotów na zdjęciach lotniczych pod przesunięciem domeny (RarePlanes)

Projekt końcowy (Deep Learning / Computer Vision). Badamy transfer detektora obiektów
z danych **syntetycznych** na **rzeczywiste** zdjęcia satelitarne oraz strategie
zmniejszania luki domenowej.

## Pytanie badawcze
Która architektura detekcyjna jest najbardziej odporna na małe obiekty, wysoką
rozdzielczość i różnicę syntetyczne↔rzeczywiste — i jakimi metodami można sukcesywnie
zmniejszyć spadek jakości modelu trenowanego na danych syntetycznych?

## Dane
**RarePlanes** (Shermeyer et al., 2020) — satelitarne zdjęcia lotnisk z samolotami,
w wariancie rzeczywistym (Maxar) i syntetycznym (renderowanym).
- Wariant obrazów: **PS-RGB tiled** (kafelki 512×512) dla real; pełny zbiór syntetyczny.
- Zadanie: detekcja `aircraft` (1 klasa) oraz `role` (3 klasy: Small/Medium/Large Civil Transport).
- Licencja: **CC BY-SA 4.0** — zob. [sekcja Licencja](#licencja).

Statystyki zbioru i analiza domain shift: [`notes/01_analiza_adnotacji.md`](notes/01_analiza_adnotacji.md).

## Struktura repozytorium
```
src/                     kod (analiza, trening, ewaluacja, wizualizacje)
  analyze_annotations.py   statystyki adnotacji COCO (rozmiary, klasy, gęstość)
  analyze_appearance.py    analiza wyglądu domeny (color histogram, FFT)
notes/                   notatki robocze pod raport (markdown, wersjonowane)
results/                 wygenerowane statystyki i wykresy
data/                    DANE — niewersjonowane (.gitignore), pobierane skryptem
docs/                    dokument wynikowy / artykuł
```

## Pobranie danych
RarePlanes jest na publicznym buckecie S3 (bez logowania). Wymaga `aws-cli`.

```bash
# adnotacje (lekkie, ~0.4 GB)
aws s3 cp s3://rareplanes-public/real/metadata_annotations/ data/real/annotations/ --no-sign-request --recursive
aws s3 cp s3://rareplanes-public/synthetic/metadata_annotations/ data/synthetic/annotations/ --no-sign-request --recursive

# obrazy real (PS-RGB tiled, ~2.8 GB)
aws s3 cp s3://rareplanes-public/real/tarballs/train/RarePlanes_train_PS-RGB_tiled.tar.gz data/real/tarballs/ --no-sign-request
aws s3 cp s3://rareplanes-public/real/tarballs/test/RarePlanes_test_PS-RGB_tiled.tar.gz  data/real/tarballs/ --no-sign-request

# obrazy syntetyczne (~145 GB — długo)
aws s3 cp s3://rareplanes-public/synthetic/train/images/ data/synthetic/images/train/ --no-sign-request --recursive
aws s3 cp s3://rareplanes-public/synthetic/test/images/  data/synthetic/images/test/  --no-sign-request --recursive
```

## Reprodukcja analizy
```bash
python3 src/analyze_annotations.py        # -> results/annotation_stats.json
python3 src/analyze_appearance.py --n 300  # -> results/appearance/*.png
```

## Środowisko
Python 3.13, PyTorch 2.9 + CUDA, numpy/pandas/matplotlib/Pillow. Pełna lista: `requirements.txt`.

## Licencja
Zbiór **RarePlanes** na licencji CC BY-SA 4.0. Wymagana atrybucja:

> J. Shermeyer, T. Hossler, A. Van Etten, D. Hogan, R. Lewis, D. Kim.
> In-Q-Tel — CosmiQ Works and AI.Reverie. RarePlanes Dataset, June 2020.

Z uwagi na ShareAlike obrazy nie są redystrybuowane w tym repo — pobiera się je
skryptem z oryginalnego źródła. Kod projektu: zob. `LICENSE` (do ustalenia).
