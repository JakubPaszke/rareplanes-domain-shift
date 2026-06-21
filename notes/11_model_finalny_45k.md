# Model finalny 45k: YOLOv10n na pelnym zbiorze syntetycznym z szumem i domieszka real

## Cel notatki

Ta notatka jest punktem wejscia do opisu modelu finalnego trenowanego na pelnym zbiorze
syntetycznym RarePlanes, czyli na 45 tysiacach obrazow syntetycznych po konwersji do formatu
YOLO. Model laczy najwazniejsze decyzje wybrane we wczesniejszych eksperymentach:

- **B2**: degradacja syntetyki szumem gaussowskim `noise_sigma=8.0`,
- **C**: domieszka obrazow realnych do zbioru treningowego,
- **D**: mniejszy rozmiar obrazu `imgsz=320`,
- **A1**: lagodniejsza augmentacja HSV: `hsv_h=0.015`, `hsv_s=0.4`, `hsv_v=0.3`.

Trening zostal uruchomiony skryptem:

- `run_train_final_cluster.sh`

Najwazniejsze artefakty wyjsciowe:

- `train-final-75379.out` - pelny log przygotowania danych, treningu, ewaluacji i benchmarku,
- `results/baselines/final_yolov10n_syn45k_noise_real25pct_img320_hsvA1_ml.json` - metryki Ultralytics
  na zbiorze walidacyjnym uzytym podczas treningu,
- `results/per_size/final_yolov10n_syn45k_noise_real25pct_img320_hsvA1_ml.json` - finalna ewaluacja COCO
  na realnym holdoucie testowym z rozbiciem po rozmiarach obiektow,
- `results/final_combined_model_summary.md`,
- `results/final_combined_model_summary.json`.

## Najkrotszy wniosek

Finalny model bardzo mocno poprawia transfer z syntetyki do realnych zdjec wzgledem bazowego
modelu trenowanego tylko na pelnym zbiorze syntetycznym 45k. Na realnym holdoucie testowym
osiaga:

| Metryka | Wartosc |
|---|---:|
| AP@[.5:.95] | 0.6332 |
| AP@.5 | 0.9031 |
| AP@.75 | 0.7459 |
| AP_small | 0.5616 |
| AP_medium | 0.6179 |
| AP_large | 0.7523 |
| AR@1 | 0.2892 |
| AR@10 | 0.6746 |
| AR@100 | 0.7480 |

Jest to wynik znacznie lepszy od samej syntetyki 45k, ale slabszy od najlepszego wariantu z
eksperymentu C na 10k syntetyki z domieszka 25% realnych danych. To wazny punkt interpretacyjny:
wiekszy zbior syntetyczny nie przelozyl sie liniowo na lepsza jakosc po dodaniu tej samej liczby
realnych probek. W finalnym wariancie nominalne `real25pct` oznacza 25% zbioru realnego, a nie 25%
calego zbioru mieszanego. Przy 38 250 syntetycznych obrazach treningowych realna domieszka
1 236 obrazow stanowila okolo 3.13% treningu.

## Przepis finalny

Nazwa uruchomienia:

```text
final_yolov10n_syn45k_noise_real25pct_img320_hsvA1_ml
```

Receptura zapisana w logu:

```text
Final recipe = C(mixed real) + D(imgsz 320) + B2(noise-only files) + A1(weak HSV)
```

Parametry wysokiego poziomu:

| Element | Wartosc |
|---|---|
| Architektura | `yolov10n.pt` |
| Zbior syntetyczny bazowy | `data/yolo/synthetic_aircraft` |
| Zbior syntetyczny po degradacji | `data/yolo/final_syn45k_b2_noise` |
| Zbior mieszany | `data/yolo/final_mixed_syn45k_noise_real25pct` |
| Epoki | `60` |
| Batch | `64` |
| Rozmiar obrazu | `320` |
| Seed | `42` |
| Device | `0` |
| Workers | `4` |
| Patience | `20` |
| Cache | `None` |
| Model wynikowy | `runs/final_yolov10n_syn45k_noise_real25pct_img320_hsvA1_ml/weights/best.pt` |

## Skad wziely sie parametry

### Eksperyment B: degradacja domenowa syntetyki

Z eksperymentow degradacji wybrano wariant **B2**, czyli dodanie szumu do syntetycznych obrazow.
W finalnym treningu uzyto:

```text
noise_sigma=8.0
blur_radius=0.0
jpeg_quality_min=None
```

Oznacza to, ze finalny model nie byl trenowany na syntetyce rozmytej ani skompresowanej JPEG-em.
Jedyna jawnie materializowana degradacja obrazu to szum. Dane zostaly zapisane jako osobny zbior:

```text
data/yolo/final_syn45k_b2_noise
```

W logu widac, ze degradacja objela caly pelny zbior syntetyczny:

| Split | Liczba obrazow |
|---|---:|
| train | 38 250 |
| val | 6 750 |
| razem | 45 000 |

Sam etap degradacji byl kosztowny czasowo: `28443.8s`, czyli okolo **7.90 h**.

### Eksperyment C: domieszka realnych danych

Z eksperymentu C wzieto pomysl dolaczenia czesci danych realnych do zbioru treningowego. W finalnym
uruchomieniu wykorzystano:

```text
real_frac=0.25
```

To znaczy: pobrano 25% dostepnych obrazow realnych ze splitow `train` i `val`. Nie dolaczono realnego
testu do treningu ani walidacji mieszanej.

Rzeczywisty sklad finalnego zbioru mieszanego:

| Split | Syntetyczne | Realne | Razem | Udzial realnych |
|---|---:|---:|---:|---:|
| train | 38 250 | 1 236 z 4 943 | 39 486 | 3.13% |
| val | 6 750 | 218 z 872 | 6 968 | 3.13% |

To jest bardzo wazne w raporcie: nazwa `real25pct` moze sugerowac, ze jedna czwarta finalnego
zbioru to obrazy realne, ale technicznie oznacza ona 25% dostepnego zbioru realnego. Poniewaz
syntetyka urosla z 10k do 45k, wzgledna waga realnych przykladow spadla.

Realny split testowy zostal zachowany jako holdout:

```text
data/real/annotations/instances_test_aircraft.json
data/real/PS-RGB_tiled/PS-RGB_tiled
```

Liczba obrazow realnego holdoutu testowego w finalnej ewaluacji:

```text
n_images=2710
```

### Eksperyment D: rozmiar obrazu

Z eksperymentu D wybrano `imgsz=320`. Finalny trening uzywa wiec mniejszego rozmiaru wejscia niz
typowe konfiguracje `640`, co skraca trening i inferencje, ale moze ograniczac detekcje bardzo
malych obiektow. W tym projekcie samoloty w kaflach bywaja male, wiec ta decyzja powinna byc w
raporcie opisana jako kompromis miedzy kosztem obliczeniowym a rozdzielczoscia szczegolow.

### Eksperyment A: HSV

Z eksperymentow augmentacyjnych wykorzystano wariant **A1**, czyli lagodne HSV:

```text
hsv_h=0.015
hsv_s=0.4
hsv_v=0.3
```

Te wartosci sa jawnie zapisane w poleceniu treningowym i w pliku wynikowym
`results/baselines/final_yolov10n_syn45k_noise_real25pct_img320_hsvA1_ml.json`.

## Pelne parametry treningu

Trening uruchomiono przez SLURM skryptem `run_train_final_cluster.sh`.

Zasoby zadania w skrypcie:

```bash
#SBATCH --job-name=train-final
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=train-final-%j.out
#SBATCH --error=train-final-%j.err
```

Zmienne domyslne w skrypcie:

```bash
DATA_DIR=/work/${USER}/rareplanes-data/data
DEVICE=0
BATCH=64
WORKERS=4
EPOCHS=60
FULL_DOWNLOAD_WORKERS=32
```

Faktyczne srodowisko z logu:

| Element | Wartosc |
|---|---|
| Host | `g4n1.cluster.wmi.amu.edu.pl` |
| Python | `3.12.13` |
| Ultralytics | `8.4.71` |
| Torch | `2.5.1+cu121` |
| GPU | `NVIDIA GeForce RTX 3090` |
| VRAM | `24124 MiB` |
| CUDA device | `CUDA:0` |

Polecenie treningowe z logu:

```bash
/home/s473634/.conda/envs/rareplanes/bin/python src/train_yolo.py \
  --data data/yolo/final_mixed_syn45k_noise_real25pct/data.yaml \
  --name final_yolov10n_syn45k_noise_real25pct_img320_hsvA1_ml \
  --model yolov10n.pt \
  --epochs 60 \
  --batch 64 \
  --imgsz 320 \
  --seed 42 \
  --device 0 \
  --workers 4 \
  --patience 20 \
  --hsv_h 0.015 \
  --hsv_s 0.4 \
  --hsv_v 0.3 \
  --val-data data/yolo/final_mixed_syn45k_noise_real25pct/data.yaml
```

Najwazniejsze domyslne augmentacje Ultralytics widoczne w logu:

| Parametr | Wartosc |
|---|---:|
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
| `augment` | `False` |
| `amp` | `True` |
| `deterministic` | `True` |

W interpretacji nalezy oddzielic dwie rzeczy:

- degradacja B2 to materialna transformacja plikow syntetycznych przed treningiem,
- HSV A1 oraz standardowe parametry Ultralytics to augmentacje treningowe wykonywane przez pipeline
  uczenia.

## Przygotowanie danych

Pipeline finalny najpierw przygotowal dane przez `src/expC.py`, a potem upewnil sie, ze pelny
zbior syntetyczny jest dostepny.

Kluczowe liczby z logu:

| Zbior | Split | Liczba obrazow |
|---|---:|---:|
| synthetic full | train | 38 250 |
| synthetic full | val | 6 750 |
| synthetic full | razem | 45 000 |
| real | train | 4 943 |
| real | val | 872 |
| real holdout | test | 2 710 |

Log potwierdza, ze brakujace syntetyczne obrazy zostaly pobrane:

```text
synthetic train images from annotations=45000 present=11764 missing=33236
synthetic full present=45000/45000 minimum=44550
```

Limit `min_full_synthetic_ratio=0.99` oznacza, ze pipeline wymagal co najmniej 99% pelnego zbioru
syntetycznego. Finalnie bylo dostepne 100%.

## Przebieg treningu

Model startowal z wag `yolov10n.pt`. Ultralytics przeniosl:

```text
Transferred 493/595 items from pretrained weights
```

W logu modelu:

| Etap | Warstwy | Parametry | GFLOPs |
|---|---:|---:|---:|
| Przed fuse | 224 | 2 707 430 | 8.4 |
| Po fuse | 102 | 2 265 363 | 6.5 |

Trening trwal:

```text
60 epochs completed in 9.421 hours
```

Czas samego polecenia treningowego wedlug wrappera:

```text
elapsed=34281.5s
```

czyli okolo **9.52 h**.

Calkowity czas zadania, lacznie z pobieraniem, przygotowaniem danych, materializacja szumu,
treningiem, ewaluacja i benchmarkiem:

```text
total wall-clock=66939.4s
```

czyli okolo **18.59 h**.

## Wyniki na walidacji mieszanej

Plik:

```text
results/baselines/final_yolov10n_syn45k_noise_real25pct_img320_hsvA1_ml.json
```

Te metryki pochodza z walidacji YOLO na zbiorze mieszanym:

```text
data/yolo/final_mixed_syn45k_noise_real25pct/data.yaml
```

Wyniki:

| Metryka | Wartosc |
|---|---:|
| mAP50 | 0.9234 |
| mAP50-95 | 0.6959 |
| Precision | 0.9495 |
| Recall | 0.8657 |

W logu Ultralytics odpowiada to walidacji na:

```text
6968 images, 86245 instances
```

Ten wynik jest przydatny do kontroli treningu, ale nie powinien byc glowna miara raportowa,
poniewaz walidacja mieszana zawiera przewage syntetyki oraz domieszke realnych walidacyjnych.
Najwazniejsza metryka domenowa to osobna ewaluacja na realnym holdoucie.

## Wyniki na realnym holdoucie testowym

Plik:

```text
results/per_size/final_yolov10n_syn45k_noise_real25pct_img320_hsvA1_ml.json
```

Ewaluacja:

```text
weights=runs/final_yolov10n_syn45k_noise_real25pct_img320_hsvA1_ml/weights/best.pt
coco_gt=data/real/annotations/instances_test_aircraft.json
n_images=2710
n_detections=77988
```

Metryki COCO:

| Metryka | Wartosc |
|---|---:|
| AP@[.5:.95] | 0.6332 |
| AP@.5 | 0.9031 |
| AP@.75 | 0.7459 |
| AP_small | 0.5616 |
| AP_medium | 0.6179 |
| AP_large | 0.7523 |
| AR@1 | 0.2892 |
| AR@10 | 0.6746 |
| AR@100 | 0.7480 |
| AR_small | 0.6671 |
| AR_medium | 0.7431 |
| AR_large | 0.8426 |

Szczegolnie istotne:

- `AP@.5=0.9031` pokazuje dobra skutecznosc detekcji przy luzniejszym progu IoU,
- `AP@[.5:.95]=0.6332` jest wyraznie nizsze, co sugeruje, ze precyzja lokalizacji ramek przy
  ostrzejszych progach nadal ma zapas do poprawy,
- `AP_large=0.7523` jest najlepszym wynikiem po rozmiarach,
- `AP_small=0.5616` pozostaje najslabsza kategoria AP, co jest spodziewane przy `imgsz=320` i
  malych obiektach.

## Porownanie z kluczowymi punktami odniesienia

| Model | Dane treningowe | AP@[.5:.95] | AP@.5 | AP_small | AP_medium | AP_large | AR@100 |
|---|---|---:|---:|---:|---:|---:|---:|
| Synthetic 45k baseline | 45k synthetic | 0.2683 | 0.4525 | 0.2859 | 0.3839 | 0.2008 | 0.5459 |
| ExpC 25% real 10k | 10k synthetic + 25% real train/val | 0.7120 | 0.9466 | 0.6365 | 0.6974 | 0.8166 | 0.7822 |
| Final 45k | 45k synthetic + B2 + 25% real train/val + A1 + imgsz 320 | 0.6332 | 0.9031 | 0.5616 | 0.6179 | 0.7523 | 0.7480 |
| Real baseline | real train/val | 0.8073 | 0.9737 | 0.7586 | 0.7825 | 0.8991 | 0.8465 |

### Final vs synthetic 45k baseline

| Metryka | Synthetic 45k | Final 45k | Zmiana |
|---|---:|---:|---:|
| AP@[.5:.95] | 0.2683 | 0.6332 | +0.3649 |
| AP@.5 | 0.4525 | 0.9031 | +0.4506 |
| AP_small | 0.2859 | 0.5616 | +0.2757 |
| AP_medium | 0.3839 | 0.6179 | +0.2339 |
| AP_large | 0.2008 | 0.7523 | +0.5515 |
| AR@100 | 0.5459 | 0.7480 | +0.2021 |

To najwazniejsze porownanie dla tezy o redukcji luki domenowej. Sama pelna syntetyka 45k transferuje
sie slabo na realne obrazy, natomiast polaczenie syntetyki z szumem i niewielka domieszka realnych
przykladow daje bardzo duzy skok.

Warto tez odnotowac spadek liczby detekcji:

| Model | Liczba detekcji na 2710 obrazach | Srednio detekcji/obraz |
|---|---:|---:|
| Synthetic 45k baseline | 813 000 | 300.0 |
| Final 45k | 77 988 | 28.8 |

Finalny model jest wiec nie tylko skuteczniejszy w AP, ale tez mniej "rozgadany" predykcyjnie:
generuje znacznie mniej kandydatow niz baseline syntetyczny.

### Final vs ExpC 25% real na 10k

| Metryka | ExpC 25% real 10k | Final 45k | Zmiana |
|---|---:|---:|---:|
| AP@[.5:.95] | 0.7120 | 0.6332 | -0.0788 |
| AP@.5 | 0.9466 | 0.9031 | -0.0435 |
| AP_small | 0.6365 | 0.5616 | -0.0749 |
| AP_medium | 0.6974 | 0.6179 | -0.0796 |
| AP_large | 0.8166 | 0.7523 | -0.0643 |
| AR@100 | 0.7822 | 0.7480 | -0.0343 |

To porownanie jest interpretacyjnie najciekawsze. Finalny model ma wiecej syntetycznych danych i
laczy kilka korzystnych decyzji, ale wypada slabiej niz prostszy wariant ExpC na 10k syntetyki.
Najbardziej prawdopodobne wyjasnienia:

- w ExpC na 10k realne obrazy stanowily znacznie wieksza czesc zbioru mieszanego,
- w finalnym 45k te same 1 236 realnych obrazow treningowych zostaly "rozcieczone" przez 38 250
  syntetycznych obrazow,
- dodatnie efekty z eksperymentow B, C, D i A nie musza sumowac sie liniowo,
- `imgsz=320` moglo ograniczyc czesc zysku dla malych i srednich obiektow,
- szum B2 mogl pomoc w redukcji synthetic gap wzgledem baseline, ale niekoniecznie jest optymalny
  po polaczeniu z duza skala syntetyki i mala wzgledna domieszka realnych danych.

Do raportu warto sformulowac to ostroznie: model finalny potwierdza, ze receptura dziala znacznie
lepiej niz czysta syntetyka, ale nie jest globalnie najlepszym wariantem sposrod wszystkich
eksperymentow. Najlepszy znany punkt jakosciowy pozostaje blizej eksperymentu C.

### Final vs real baseline

| Metryka | Real baseline | Final 45k | Luka |
|---|---:|---:|---:|
| AP@[.5:.95] | 0.8073 | 0.6332 | -0.1740 |
| AP@.5 | 0.9737 | 0.9031 | -0.0706 |
| AP_small | 0.7586 | 0.5616 | -0.1970 |
| AP_medium | 0.7825 | 0.6179 | -0.1646 |
| AP_large | 0.8991 | 0.7523 | -0.1468 |
| AR@100 | 0.8465 | 0.7480 | -0.0986 |

Model finalny nadal nie dochodzi do modelu trenowanego na realnych danych. Luka jest szczegolnie
widoczna w `AP@[.5:.95]` i `AP_small`. To wspiera wniosek, ze sama ekspansja syntetyki i proste
dostosowanie domenowe nie zastepuja realnych danych, ale moga znacznie zmniejszyc dystans do
modelu realnego.

## Benchmark inferencji

Benchmark zapisany w `results/final_combined_model_summary.json`:

| Parametr | Wartosc |
|---|---:|
| Liczba obrazow | 256 |
| Batch | 32 |
| FPS | 100.79 |
| Peak CUDA memory | 1340.7 MB |

To sugeruje, ze model `yolov10n` pozostaje bardzo lekki inferencyjnie. W raporcie mozna to
wykorzystac jako argument za praktycznoscia finalnego wariantu: mimo slabszej jakosci niz real
baseline, model jest szybki i tani w uruchomieniu.

## Reprodukcja

Na klastrze trening byl uruchamiany przez:

```bash
sbatch run_train_final_cluster.sh
```

Skrypt uruchamia:

```bash
python src/train_final_model.py \
  --data-dir "$DATA_DIR" \
  --epochs "$EPOCHS" \
  --batch "$BATCH" \
  --workers "$WORKERS" \
  --device "$DEVICE" \
  --full-download-workers "$FULL_DOWNLOAD_WORKERS"
```

Domyslne ustawienia skryptu odpowiadaja finalnemu treningowi:

```bash
EPOCHS=60
BATCH=64
WORKERS=4
DEVICE=0
FULL_DOWNLOAD_WORKERS=32
DATA_DIR=/work/${USER}/rareplanes-data/data
```

Pipeline `src/train_final_model.py` wykonuje po kolei:

1. przygotowanie i sprawdzenie danych RarePlanes,
2. pobranie brakujacej pelnej syntetyki,
3. konwersje YOLO,
4. materializacje wariantu B2 z szumem,
5. utworzenie zbioru mieszanego `synthetic + 25% real train/val`,
6. trening YOLOv10n,
7. ewaluacje per-size na realnym holdoucie,
8. benchmark inferencji,
9. zapis podsumowania finalnego.

## Ograniczenia i ostroznosci do raportu

1. **Nazwa `real25pct` jest latwa do zlego odczytania.** W finalnym 45k oznacza 25% realnego
   train/val, ale realne obrazy stanowia tylko okolo 3.13% mieszanego treningu.

2. **Walidacja mieszana i realny holdout odpowiadaja na inne pytania.** Metryki z
   `results/baselines/...json` pokazuja zachowanie na zbiorze walidacyjnym uzytym w treningu.
   Do wniosku o transferze domenowym wazniejszy jest plik `results/per_size/...json`.

3. **Finalny wariant nie jest najlepszy absolutnie.** Jest bardzo dobry wzgledem synthetic 45k
   baseline, ale slabszy niz ExpC 25% real na 10k. Raport powinien to potraktowac jako wynik:
   wiecej syntetyki bez zwiekszenia wzglednego udzialu realnych danych nie gwarantuje poprawy.

4. **Maly rozmiar obrazu moze ograniczac AP dla malych obiektow.** `imgsz=320` poprawia koszt
   obliczeniowy, ale kategoria `AP_small=0.5616` zostaje wyraznie ponizej real baseline.

5. **Koszt storage i przygotowania danych jest wysoki.** Pelna syntetyka 45k oraz materializowany
   wariant z szumem wymagaja duzo miejsca na `/work`. Sam etap B2 trwal prawie 8 godzin.

6. **Dodatnie efekty eksperymentow nie sumuja sie automatycznie.** B2, C, D i A byly sensowne jako
   kierunki, ale ich polaczenie na pelnym 45k zbiorze zmienilo proporcje danych i zachowanie modelu.

## Fragmenty gotowe do raportu

Model finalny zostal zbudowany jako polaczenie najlepszych decyzji projektowych z eksperymentow
czastkowych: degradacji syntetyki szumem, domieszki realnych danych, zmniejszenia rozmiaru wejscia
do `320` oraz lagodnej augmentacji HSV. Trening przeprowadzono dla architektury `YOLOv10n` przez
60 epok na pelnym zbiorze syntetycznym 45k, po czym model oceniono na niezaleznym realnym holdoucie
testowym liczacym 2710 obrazow.

Wyniki pokazuja duza redukcje luki domenowej wzgledem modelu trenowanego tylko na syntetyce.
`AP@.5` wzroslo z `0.4525` do `0.9031`, a `AP@[.5:.95]` z `0.2683` do `0.6332`. Jednoczesnie model
generowal znacznie mniej detekcji niz syntetyczny baseline, co sugeruje ograniczenie liczby
falszywych kandydatow.

Finalny model nie przewyzszyl jednak wariantu ExpC trenowanego na 10k syntetyki z taka sama liczba
realnych probek. Najbardziej prawdopodobna przyczyna jest spadek wzglednego udzialu danych realnych
w mieszanym zbiorze treningowym po przejsciu z 10k do 45k syntetyki. Wynik ten wskazuje, ze dla
transferu synthetic-to-real kluczowa jest nie tylko liczba syntetycznych obrazow, ale takze proporcja
i rola danych realnych w procesie uczenia.

## Co mozna jeszcze sprawdzic

- powtorzyc finalny wariant z wiekszym udzialem realnych danych liczonym jako procent finalnego
  zbioru mieszanego, a nie procent realnego train/val,
- porownac `imgsz=320` z `imgsz=640` dla finalnej receptury, szczegolnie dla `AP_small`,
- sprawdzic finalny wariant bez B2, ale z pelnym 45k i domieszka realnych, zeby oddzielic efekt
  skali od efektu szumu,
- wykonac analogiczna ewaluacje Grad-CAM lub przeglad jakosciowy FP/FN dla finalnego modelu,
- przeliczyc finalny model przy wiekszym limicie czasu bez koniecznosci ponownego generowania danych,
  jezeli zbiory `final_syn45k_b2_noise` i `final_mixed_syn45k_noise_real25pct` pozostaly na dysku.
