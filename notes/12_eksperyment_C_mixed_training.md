# Eksperyment C: mixed training synthetic + real

## Cel notatki

Ta notatka opisuje eksperyment C, czyli sprawdzenie, jak niewielka domieszka
realnych obrazow w treningu wplywa na transfer modelu trenowanego glownie na
syntetycznych danych RarePlanes. Jest to najwazniejszy eksperyment w calym
projekcie pod wzgledem jakosci detekcji na realnym holdoucie: wariant
`expC_25pct_real_10k_ml` osiaga najlepszy wynik sposrod eksperymentow
synthetic-to-real przed treningiem finalnym.

Najwazniejsze artefakty:

- `src/run_expC.sh` - skrypt SLURM uzyty do pelnego uruchomienia C na klastrze,
- `src/expC.py` - wrapper pobierajacy dane, przygotowujacy YOLO i odpalajacy C,
- `src/run_expC_mixed_cluster.py` - wlasciwy runner sweepa mixed-training,
- `src/make_mixed_dataset.py` - logika budowania zbiorow synthetic + real,
- `src/expC-71048.out` - log glownego uruchomienia C na 10k syntetyki,
- `results/expC_run.log` - zbiorczy log wrappera,
- `results/expC_mixed_summary.md` i `results/expC_mixed_summary.csv` - tabela wynikow C,
- `results/baselines/expC_*_ml.json` - metryki YOLO na walidacji mieszanej,
- `results/per_size/expC_*_ml.json` - metryki COCO na realnym holdoucie testowym.

## Najkrotszy wniosek

Eksperyment C bardzo mocno potwierdza hipoteze, ze nawet mala liczba realnych
probek jest duzo wazniejsza dla domkniecia luki domenowej niz samo zwiekszanie
syntetyki. Wynik na realnym holdoucie rosnie monotonicznie wraz z liczba
dolaczonych realnych obrazow:

| Wariant | AP@[.5:.95] | AP@.5 | AP_small | AP_medium | AP_large | AR@100 |
|---|---:|---:|---:|---:|---:|---:|
| C 1% real | 0.3957 | 0.6666 | 0.3614 | 0.4535 | 0.3462 | 0.5954 |
| C 5% real | 0.5660 | 0.8516 | 0.5197 | 0.5761 | 0.6108 | 0.7144 |
| C 10% real | 0.6188 | 0.8960 | 0.5404 | 0.6184 | 0.7060 | 0.7349 |
| **C 25% real** | **0.7120** | **0.9466** | **0.6365** | **0.6974** | **0.8166** | **0.7822** |

Najlepszy wariant C, czyli `expC_25pct_real_10k_ml`, dochodzi bardzo blisko real
baseline w `AP@.5`:

| Model | AP@[.5:.95] | AP@.5 | AP_small | AP_medium | AP_large | AR@100 |
|---|---:|---:|---:|---:|---:|---:|
| Synthetic 10k baseline | 0.2388 | 0.4095 | 0.2624 | 0.3284 | 0.0671 | 0.3686 |
| Synthetic 45k baseline | 0.2683 | 0.4525 | 0.2859 | 0.3839 | 0.2008 | 0.5459 |
| **C 25% real 10k** | **0.7120** | **0.9466** | **0.6365** | **0.6974** | **0.8166** | **0.7822** |
| Real baseline | 0.8073 | 0.9737 | 0.7586 | 0.7825 | 0.8991 | 0.8465 |

Glowny statement do raportu:

> Domieszka realnych danych jest najsilniejsza interwencja redukujaca synthetic
> gap. Przy 25% dostepnych realnych obrazow train/val model trenowany nadal
> glownie na syntetyce osiaga `AP@.5=0.9466` na realnym holdoucie, czyli zbliza
> sie do real baseline `0.9737`, mimo ze nie uzywa realnego testu w treningu.

## Co dokladnie oznacza "1/5/10/25% real"

To jest najwazniejsza pulapka interpretacyjna w eksperymencie C.

Parametr `--real-frac` w `src/make_mixed_dataset.py` oznacza procent dostepnego
realnego splitu `train` i `val`, ktory zostaje dolaczony do syntetyki. Nie oznacza
on procentu finalnego zbioru mieszanego.

Kod:

```python
n = max(1, round(len(files) * frac))
```

Dla kazdego splitu:

- `train` losowany jest seedem `42`,
- `val` losowany jest seedem `43`, bo kod uzywa `seed + 1`,
- realny `test` nie jest nigdy linkowany do zbioru mixed.

Dane wejsciowe w glownym sweepie:

| Zbior | Split | Liczba obrazow |
|---|---:|---:|
| synthetic_10k | train | 10 000 |
| synthetic_10k | val | 1 764 |
| real_aircraft | train | 4 943 |
| real_aircraft | val | 872 |
| real holdout COCO | test | 2 710 |

Rzeczywisty sklad zbiorow mixed:

| Wariant | real_frac | Train synthetic | Train real | Real share train | Val synthetic | Val real | Real share val |
|---|---:|---:|---:|---:|---:|---:|---:|
| C 1% | 0.01 | 10 000 | 49 / 4 943 | 0.49% | 1 764 | 9 / 872 | 0.51% |
| C 5% | 0.05 | 10 000 | 247 / 4 943 | 2.41% | 1 764 | 44 / 872 | 2.43% |
| C 10% | 0.10 | 10 000 | 494 / 4 943 | 4.71% | 1 764 | 87 / 872 | 4.70% |
| C 25% | 0.25 | 10 000 | 1 236 / 4 943 | 11.00% | 1 764 | 218 / 872 | 11.00% |

W raporcie warto unikac zdania "25% zbioru treningowego bylo realne". Technicznie
poprawne jest:

> W wariancie C 25% dolaczono 25% dostepnego realnego train/val, co przy 10k
> syntetyki dalo okolo 11% realnych obrazow w finalnym zbiorze treningowym.

To tlumaczy tez pozniejsze zachowanie modelu finalnego 45k: tam ta sama liczba
realnych probek zostala rozcienczona przez 38 250 syntetycznych obrazow
treningowych, wiec realny udzial spadl do okolo 3.13%.

## Pipeline eksperymentu

Eksperyment C mial trzy warstwy uruchomieniowe.

### 1. Skrypt SLURM

Pelne uruchomienie odpowiada logowi `src/expC-71048.out`. Skrypt `src/run_expC.sh`
uruchamial zadanie:

```bash
#SBATCH --job-name=expC-final
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=expC-%j.out
#SBATCH --error=expC-%j.err
```

Faktyczne polecenie w tym skrypcie:

```bash
python expC.py \
  --data-dir /work/$USER/rareplanes-data/data \
  --batch 32 \
  --workers 4 \
  --device 0 \
  --epochs 30
```

W repo istnieje tez podobny `src/runC.sh`, ale aktualne artefakty C z logu
`71048` odpowiadaja konfiguracji `batch=32`, `workers=4`, `imgsz=512`,
`epochs=30`.

### 2. Wrapper `src/expC.py`

Wrapper wykonywal:

1. przygotowanie katalogu danych przez symlink `data -> /work/s473634/rareplanes-data/data`,
2. pobranie lub pominiecie juz istniejacych adnotacji,
3. pobranie lub pominiecie realnych kafli train+test,
4. pobranie lub sprawdzenie listy syntetycznych obrazow 10k,
5. konwersje COCO -> YOLO,
6. utworzenie `synthetic_10k` i `synthetic_1k`,
7. odpalenie `src/run_expC_mixed_cluster.py`.

Z logu glownego uruchomienia:

```text
synthetic selected_ok=11764/11764 minimum=11646
[synthetic/aircraft] train=10000 val=1764 test=0
[real/aircraft] train=4943 val=872 test=2710
[synthetic_10k] train=10000 val=1764
```

### 3. Runner `src/run_expC_mixed_cluster.py`

Runner wykonywal dla kazdego wariantu:

```text
make_mixed_dataset.py -> train_yolo.py -> eval_per_size.py
```

Plan z logu:

```text
C 1%: dataset=mixed_syn10k_real1pct, run=expC_1pct_real_10k_ml, real_frac=0.01
C 5%: dataset=mixed_syn10k_real5pct, run=expC_5pct_real_10k_ml, real_frac=0.05
C 10%: dataset=mixed_syn10k_real10pct, run=expC_10pct_real_10k_ml, real_frac=0.1
C 25%: dataset=mixed_syn10k_real25pct, run=expC_25pct_real_10k_ml, real_frac=0.25
```

## Parametry treningu

Wszystkie glowne warianty C 10k byly trenowane tym samym protokolem:

| Parametr | Wartosc |
|---|---|
| Architektura | `yolov10n.pt` |
| Dataset syntetyczny | `data/yolo/synthetic_10k` |
| Dataset realny | `data/yolo/real_aircraft` |
| Epoki | `30` |
| Batch | `32` |
| Rozmiar obrazu | `512` |
| Workers | `4` |
| Device | `0` |
| Seed | `42` |
| Patience | `20` |
| Cache | `None` |
| Augmentacja | domyslna Ultralytics |

Przykladowe polecenie dla najlepszego wariantu:

```bash
/home/s473634/.conda/envs/rareplanes/bin/python src/train_yolo.py \
  --data data/yolo/mixed_syn10k_real25pct/data.yaml \
  --name expC_25pct_real_10k_ml \
  --model yolov10n.pt \
  --epochs 30 \
  --batch 32 \
  --imgsz 512 \
  --seed 42 \
  --device 0 \
  --workers 4 \
  --patience 20 \
  --val-data data/yolo/mixed_syn10k_real25pct/data.yaml
```

Srodowisko z logu:

| Element | Wartosc |
|---|---|
| Host | `g1n2.cluster.wmi.amu.edu.pl` |
| Python | `3.12.13` |
| Ultralytics | `8.4.71` |
| Torch | `2.5.1+cu121` |
| GPU | `NVIDIA GeForce RTX 4090` |
| VRAM | ok. `24081 MiB` widoczne dla Ultralytics |

## Augmentacje

Eksperyment C nie stroil recznie augmentacji. W plikach
`results/baselines/expC_*_ml.json` pole `augmentation` ma wartosc:

```text
default
```

Z logu Ultralytics wynika, ze byly to domyslne ustawienia treningu YOLO:

| Parametr | Wartosc |
|---|---:|
| `hsv_h` | 0.015 |
| `hsv_s` | 0.7 |
| `hsv_v` | 0.4 |
| `mosaic` | 1.0 |
| `close_mosaic` | 10 |
| `fliplr` | 0.5 |
| `flipud` | 0.0 |
| `scale` | 0.5 |
| `translate` | 0.1 |
| `erasing` | 0.4 |
| `degrees` | 0.0 |
| `shear` | 0.0 |
| `perspective` | 0.0 |
| `mixup` | 0.0 |
| `copy_paste` | 0.0 |
| `auto_augment` | `randaugment` |
| `amp` | `True` |
| `deterministic` | `True` |

To wazne rozroznienie wzgledem eksperymentu A i modelu finalnego:

- C bada przede wszystkim **sklad danych**, a nie sile augmentacji,
- HSV w C jest domyslne (`0.015/0.7/0.4`), a nie wariant A1 (`0.015/0.4/0.3`),
- C nie stosuje degradacji B2, czyli nie materializuje szumu w plikach.

## Walidacja mieszana vs realny holdout

Kazdy wariant byl walidowany dwoma sposobami:

1. przez `train_yolo.py` na mieszanym `data.yaml`,
2. przez `eval_per_size.py` na realnym holdoucie COCO.

Metryki z `results/baselines/expC_*_ml.json` dotycza walidacji mieszanej. Sa
wysokie i bardzo podobne dla wariantow 1/5/10/25%:

| Wariant | Mixed val mAP50 | Mixed val mAP50-95 | Precision | Recall |
|---|---:|---:|---:|---:|
| C 1% | 0.9687 | 0.7953 | 0.9644 | 0.9269 |
| C 5% | 0.9688 | 0.7939 | 0.9670 | 0.9245 |
| C 10% | 0.9687 | 0.7941 | 0.9659 | 0.9231 |
| C 25% | 0.9695 | 0.7929 | 0.9671 | 0.9239 |

Tych liczb nie nalezy uzywac jako glownego dowodu na redukcje luki domenowej,
bo walidacja mieszana jest zdominowana przez syntetyke i zawiera realne probki
z `val` dolaczone w tym samym mechanizmie co trening. Ona potwierdza, ze trening
sie udal, ale nie rozdziela dobrze wariantow.

Glowna miara eksperymentu C to `results/per_size/expC_*_ml.json`, czyli COCO AP
na realnym holdoucie testowym.

## Ewaluacja realnego holdoutu

Ewaluacja byla wykonywana poleceniem:

```bash
python src/eval_per_size.py \
  --weights runs/<run>/weights/best.pt \
  --img-dir data/real/PS-RGB_tiled/PS-RGB_tiled \
  --coco-gt data/real/annotations/instances_test_aircraft.json \
  --device 0 \
  --imgsz 512 \
  --name <run>
```

`run_expC_mixed_cluster.py` wypisuje `real eval images=8525`, bo tyle kafli PNG
znajduje sie w katalogu realnych obrazow po rozpakowaniu train+test. Natomiast
`eval_per_size.py` predykuje tylko obrazy wymienione w pliku COCO GT:

```text
[eval] obrazow do predykcji: 2710 / 2710 w GT
```

Dlatego w JSON-ach wynikowych:

```text
n_images=2710
coco_gt=data/real/annotations/instances_test_aircraft.json
```

Realny test nie jest dolaczany do zbiorow mixed.

## Wyniki glownego sweepa C

| Wariant | Dataset | Train images | Val images | AP@[.5:.95] | AP@.5 | AP@.75 | AP_small | AP_medium | AP_large | AR@100 | Det/img |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C 1% | `mixed_syn10k_real1pct` | 10 049 | 1 773 | 0.3957 | 0.6666 | 0.4308 | 0.3614 | 0.4535 | 0.3462 | 0.5954 | 66.5 |
| C 5% | `mixed_syn10k_real5pct` | 10 247 | 1 808 | 0.5660 | 0.8516 | 0.6648 | 0.5197 | 0.5761 | 0.6108 | 0.7144 | 39.4 |
| C 10% | `mixed_syn10k_real10pct` | 10 494 | 1 851 | 0.6188 | 0.8960 | 0.7242 | 0.5404 | 0.6184 | 0.7060 | 0.7349 | 35.6 |
| **C 25%** | `mixed_syn10k_real25pct` | 11 236 | 1 982 | **0.7120** | **0.9466** | **0.8421** | **0.6365** | **0.6974** | **0.8166** | **0.7822** | **20.8** |

Wynik jest jednoznacznie monotoniczny:

- `AP@.5`: `0.6666 -> 0.8516 -> 0.8960 -> 0.9466`,
- `AP@[.5:.95]`: `0.3957 -> 0.5660 -> 0.6188 -> 0.7120`,
- `AP_large`: `0.3462 -> 0.6108 -> 0.7060 -> 0.8166`,
- `AR@100`: `0.5954 -> 0.7144 -> 0.7349 -> 0.7822`.

Wraz ze wzrostem udzialu realnych probek spada tez liczba detekcji:

| Wariant | Liczba detekcji | Detekcje/obraz |
|---|---:|---:|
| C 1% | 180 329 | 66.5 |
| C 5% | 106 747 | 39.4 |
| C 10% | 96 445 | 35.6 |
| C 25% | 56 483 | 20.8 |

To sugeruje, ze realne probki nie tylko poprawiaja AP, ale tez ucza model
ostrozniejszego zachowania na realnej domenie: mniej falszywych lub
niskojakosciowych kandydatow przechodzi do ewaluacji.

## Porownanie z baseline syntetycznym

### C 25% vs synthetic 10k baseline

| Metryka | Synthetic 10k | C 25% | Zmiana |
|---|---:|---:|---:|
| AP@[.5:.95] | 0.2388 | 0.7120 | +0.4732 |
| AP@.5 | 0.4095 | 0.9466 | +0.5371 |
| AP_small | 0.2624 | 0.6365 | +0.3741 |
| AP_medium | 0.3284 | 0.6974 | +0.3691 |
| AP_large | 0.0671 | 0.8166 | +0.7495 |
| AR@100 | 0.3686 | 0.7822 | +0.4137 |

To jest najmocniejszy dowod eksperymentu: przejscie z czystej syntetyki 10k do
syntetyki 10k z realna domieszka daje ogromny skok w kazdej kategorii rozmiaru,
szczegolnie dla duzych obiektow.

### C 25% vs synthetic 45k baseline

| Metryka | Synthetic 45k | C 25% | Zmiana |
|---|---:|---:|---:|
| AP@[.5:.95] | 0.2683 | 0.7120 | +0.4437 |
| AP@.5 | 0.4525 | 0.9466 | +0.4941 |
| AP_small | 0.2859 | 0.6365 | +0.3505 |
| AP_medium | 0.3839 | 0.6974 | +0.3135 |
| AP_large | 0.2008 | 0.8166 | +0.6158 |
| AR@100 | 0.5459 | 0.7822 | +0.2363 |

To porownanie pokazuje, ze 10k syntetyki + realna domieszka jest znacznie
silniejsze niz 45k samej syntetyki. W raporcie mozna to ujac jako:

> Dla redukcji synthetic-to-real gap wieksze znaczenie miala mala liczba
> rzeczywistych przykladow niz czterokrotne zwiekszenie liczby syntetycznych
> obrazow.

## Porownanie z real baseline

| Metryka | Real baseline | C 25% | Luka |
|---|---:|---:|---:|
| AP@[.5:.95] | 0.8073 | 0.7120 | -0.0952 |
| AP@.5 | 0.9737 | 0.9466 | -0.0271 |
| AP_small | 0.7586 | 0.6365 | -0.1221 |
| AP_medium | 0.7825 | 0.6974 | -0.0850 |
| AP_large | 0.8991 | 0.8166 | -0.0825 |
| AR@100 | 0.8465 | 0.7822 | -0.0643 |

W `AP@.5` C 25% jest bardzo blisko real baseline, ale w `AP@[.5:.95]` luka
pozostaje wyrazna. Interpretacja:

- model nauczy sie dobrze znajdowac samoloty na realnych obrazach,
- precyzja lokalizacji ramek przy ostrzejszych progach IoU nadal odstaje od
  treningu na pelnych realnych danych,
- najwieksza luka wzgledem real baseline zostaje dla `AP_small`.

## Relacja do modelu finalnego 45k

Najlepszy wariant C 10k jest waznym punktem odniesienia dla modelu finalnego
45k, bo finalny model uzywa idei C, ale w innych proporcjach danych.

| Model | AP@[.5:.95] | AP@.5 | AP_small | AP_medium | AP_large | AR@100 |
|---|---:|---:|---:|---:|---:|---:|
| C 25% real 10k | 0.7120 | 0.9466 | 0.6365 | 0.6974 | 0.8166 | 0.7822 |
| Final 45k B2+C+D+A1 | 0.6332 | 0.9031 | 0.5616 | 0.6179 | 0.7523 | 0.7480 |

Finalny model ma wiecej syntetyki i dodatkowe decyzje B2/D/A1, ale slabszy wynik
na realnym holdoucie. Najbardziej prawdopodobne wyjasnienie jest proporcyjne:

- w C 25% na 10k realne obrazy stanowia okolo 11% treningu,
- w finalnym 45k realne obrazy stanowia okolo 3.13% treningu,
- wzrost syntetyki bez zwiekszenia liczby realnych probek rozciencza sygnal
  domeny realnej.

To jest bardzo dobry material do dyskusji koncowej: wynik C nie mowi po prostu
"dodajmy syntetyki", tylko "dodajmy realne przyklady w odpowiedniej proporcji".

## Smoke run

W summary C znajduje sie tez:

```text
expC_1pct_real_1k_smoke_ml
```

To byl techniczny smoke test:

| Parametr | Wartosc |
|---|---|
| Synthetic dataset | `data/yolo/synthetic_1k` |
| Dataset tag | `1k_smoke` |
| Real frac | `0.01` |
| Epoki | `3` |
| Batch | `32` |
| img size | `512` |

Wynik smoke:

| Metryka | Wartosc |
|---|---:|
| AP@[.5:.95] | 0.2644 |
| AP@.5 | 0.4777 |
| AP_small | 0.3034 |
| AP_medium | 0.3411 |
| AP_large | 0.1330 |
| AR@100 | 0.5237 |
| n_det | 813 000 |

Tego wariantu nie nalezy traktowac jako glownego punktu sweepa C. Sluzy jako
potwierdzenie, ze pipeline dziala, zapisuje checkpoint, wykonuje ewaluacje i
generuje JSON-y wynikowe. Jego `n_det=813000` oznacza maksymalne `300`
detekcji na kazdym z 2710 obrazow, czyli model byl jeszcze bardzo slabo
skalibrowany po trzech epokach.

## Koszt obliczeniowy

Pelny sweep 10k z czterema wariantami trwal wedlug wrappera:

```text
caly czas wall-clock=12892.6s
total wall-clock=13002.4s
```

czyli okolo **3.6 h** lacznie z przygotowaniem, treningiem, ewaluacja i zapisem
summary.

Czasy treningow poszczegolnych wariantow:

| Wariant | Czas komendy treningowej | Czas wg Ultralytics |
|---|---:|---:|
| C 1% | 3158.4 s | 0.830 h |
| C 5% | 3168.2 s | 0.834 h |
| C 10% | 3183.2 s | 0.837 h |
| C 25% | 3224.7 s | 0.848 h |

Checkpointy `best.pt` po stripowaniu mialy okolo `5.5 MB`.

## Interpretacja

### 1. Realne probki dzialaja jak kotwica domenowa

Nawet wariant C 1%, czyli tylko 49 realnych obrazow treningowych i 9 walidacyjnych,
poprawia transfer wzgledem synthetic 10k:

| Metryka | Synthetic 10k | C 1% | Zmiana |
|---|---:|---:|---:|
| AP@[.5:.95] | 0.2388 | 0.3957 | +0.1569 |
| AP@.5 | 0.4095 | 0.6666 | +0.2571 |
| AP_large | 0.0671 | 0.3462 | +0.2791 |

To sugeruje, ze model potrzebuje choc niewielkiej liczby realnych przykladow,
zeby przestawic cechy z czystej syntetyki na realna fakture, kolorystyke,
kontrast i typowe tlo.

### 2. Efekt rosnie monotonicznie z real_frac

Najsilniejszy wniosek eksperymentu C to monotonicznosc. W przeciwienstwie do
eksperymentu A, gdzie sila HSV miala efekt niemonotoniczny, tutaj wiecej realnych
danych konsekwentnie poprawia wynik.

### 3. Najwiekszy skok jest miedzy 1% i 5%

Najwiekszy przyrost `AP@.5` zachodzi przy przejsciu z C 1% do C 5%:

| Przejscie | Zmiana AP@.5 | Zmiana AP@[.5:.95] |
|---|---:|---:|
| 1% -> 5% | +0.1850 | +0.1702 |
| 5% -> 10% | +0.0444 | +0.0528 |
| 10% -> 25% | +0.0506 | +0.0933 |

Wniosek: mala liczba realnych danych daje bardzo duzy pierwszy zysk, a dalsze
zwiekszanie real_frac nadal pomaga, ale nie w tak prostym, liniowym tempie.

### 4. Poprawia sie nie tylko recall, ale tez selektywnosc

Spadek liczby detekcji przy wzroscie AP sugeruje, ze model staje sie bardziej
selektywny. Synthetic-only i slabe warianty maja tendencje do generowania wielu
kandydatow na realnych obrazach; wariant C 25% generuje mniej detekcji i osiaga
wyzsza precyzje.

### 5. Walidacja mixed nie wystarcza

Mixed val prawie nie rozroznia wariantow C, mimo ze real holdout pokazuje ogromne
roznice. To jest wazny argument metodologiczny: w projekcie synthetic-to-real
glowna miara musi byc niezalezny realny holdout, a nie walidacja na zbiorze
mieszanym.

## Ograniczenia

1. **Jeden seed.** Wszystkie warianty uzywaja `seed=42`, wiec nie znamy wariancji
   miedzy uruchomieniami.

2. **Nazwy procentow sa mylace.** `real25pct` oznacza 25% dostepnego real train/val,
   a nie 25% calego mixed datasetu.

3. **Real val wchodzi do walidacji mieszanej.** To jest poprawne dla kontroli
   treningu, ale nie powinno byc glowna metryka raportowa.

4. **Brak wariantu 100% real_frac w C.** Nie wiadomo, czy dalsze zwiekszanie
   real_frac nadal poprawialoby wynik i jak blisko doszloby do real baseline.

5. **C nie izoluje augmentacji.** Uzywa domyslnych augmentacji YOLO, wiec nie
   odpowiada na pytanie, czy A1 lub B2 bylyby lepsze w ramach tego samego 10k
   mixed setupu.

6. **Porownanie z real baseline nie jest idealnie symetryczne.** Real baseline
   trenuje na realnym zbiorze, a C na synthetic + probce real. To porownanie jest
   gornym punktem odniesienia, nie kontrola jeden-do-jednego.

## Reprodukcja

Pelny run na klastrze:

```bash
sbatch src/run_expC.sh
```

Bezposrednio:

```bash
python src/expC.py \
  --data-dir /work/$USER/rareplanes-data/data \
  --batch 32 \
  --workers 4 \
  --device 0 \
  --epochs 30
```

Tylko wlasciwy runner C, zakladajac ze dane sa juz przygotowane:

```bash
python src/run_expC_mixed_cluster.py \
  --src-dataset data/yolo/synthetic_10k \
  --dataset-tag 10k \
  --real-src data/yolo/real_aircraft \
  --real-img-dir data/real/PS-RGB_tiled/PS-RGB_tiled \
  --coco-gt data/real/annotations/instances_test_aircraft.json \
  --pcts 1 5 10 25 \
  --fracs 0.01 0.05 0.10 0.25 \
  --epochs 30 \
  --batch 32 \
  --workers 4 \
  --device 0 \
  --imgsz 512 \
  --model yolov10n.pt \
  --patience 20
```

Pojedynczy wariant mozna odtworzyc recznie:

```bash
python src/make_mixed_dataset.py \
  --syn-src data/yolo/synthetic_10k \
  --real-src data/yolo/real_aircraft \
  --name mixed_syn10k_real25pct \
  --real-frac 0.25 \
  --seed 42 \
  --overwrite

python src/train_yolo.py \
  --data data/yolo/mixed_syn10k_real25pct/data.yaml \
  --name expC_25pct_real_10k_ml \
  --model yolov10n.pt \
  --epochs 30 \
  --batch 32 \
  --imgsz 512 \
  --seed 42 \
  --device 0 \
  --workers 4 \
  --patience 20 \
  --val-data data/yolo/mixed_syn10k_real25pct/data.yaml

python src/eval_per_size.py \
  --weights runs/expC_25pct_real_10k_ml/weights/best.pt \
  --img-dir data/real/PS-RGB_tiled/PS-RGB_tiled \
  --coco-gt data/real/annotations/instances_test_aircraft.json \
  --device 0 \
  --imgsz 512 \
  --name expC_25pct_real_10k_ml
```

## Fragmenty gotowe do raportu

Eksperyment C sprawdzal, czy dolaczenie niewielkiej liczby realnych obrazow do
treningu syntetycznego poprawia transfer na realny holdout testowy. Dla kazdego
wariantu trenowano `YOLOv10n` na zbiorze `synthetic_10k` z dolaczonymi odpowiednio
`1%`, `5%`, `10%` i `25%` dostepnych realnych obrazow train/val. Realny split
testowy nie byl dolaczany do danych treningowych ani walidacji mieszanej.

Wyniki jednoznacznie potwierdzaja skutecznosc mixed trainingu. `AP@.5` wzroslo
z `0.4095` dla synthetic 10k baseline do `0.9466` dla wariantu C 25%, a
`AP@[.5:.95]` z `0.2388` do `0.7120`. Poprawa wystapila dla wszystkich rozmiarow
obiektow, szczegolnie dla obiektow duzych, gdzie `AP_large` wzroslo z `0.0671`
do `0.8166`.

Najlepszy wariant C zblizyl sie do real baseline w `AP@.5` (`0.9466` vs `0.9737`),
ale nadal pozostawil luke w `AP@[.5:.95]` (`0.7120` vs `0.8073`). Oznacza to, ze
realna domieszka bardzo skutecznie pomaga modelowi odnajdywac obiekty w realnej
domenie, ale pelny trening na danych realnych nadal daje lepsza lokalizacje ramek
i wyzsza jakosc przy ostrzejszych progach IoU.

Wyniki C sa rowniez kluczowe dla interpretacji modelu finalnego 45k. Finalny model
uzywal tej samej idei dolaczenia 25% real train/val, ale przy znacznie wiekszej
liczbie syntetycznych obrazow. W efekcie realne obrazy stanowily tylko okolo
3.13% finalnego treningu, podczas gdy w C 25% na 10k stanowily okolo 11%. To
tlumaczy, dlaczego C 25% moze przewyzszac finalny model mimo mniejszego zbioru
syntetycznego.

## Co dalej sprawdzic

- wariant C z `real_frac=0.50` i `real_frac=1.00`, zeby zobaczyc, gdzie zaczyna
  sie nasycenie wzgledem real baseline,
- wariant C, w ktorym procent realnych obrazow jest liczony jako procent finalnego
  zbioru mixed, a nie procent dostepnego real train/val,
- C 25% z `imgsz=320` i `imgsz=640`, zeby oddzielic efekt mixed trainingu od
  efektu rozdzielczosci,
- C 25% z A1 HSV (`hsv_s=0.4`, `hsv_v=0.3`),
- C 25% z B2 noise na syntetyce, ale przy zachowaniu podobnego real share,
- powtorzenie najlepszego C na kilku seedach, bo roznice rzedu kilku punktow AP
  moga miec komponent losowy.
