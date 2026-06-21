# Plan raportu finalnego v2

## Cel dokumentu

Ten dokument jest szczegółowym briefem dla agenta, który ma napisać finalny
raport projektu na podstawie szkicu `docs/RAPORT.md` oraz notatek z katalogu
`notes/`.

Raport v2 ma bazować na obecnym szkicu, ale musi go rozszerzyć o wiedzę zebraną
w późniejszych eksperymentach, szczególnie:

- `notes/10_gradcam_interpretowalnosc.md`,
- `notes/11_model_finalny_45k.md`,
- `notes/12_eksperyment_C_mixed_training.md`.

Główna funkcja raportu finalnego: nie tylko pokazać tabelę wyników, ale opisać
proces badawczy, hipotezy, mechanizmy stojące za wynikami oraz praktyczne wnioski
o tym, co rzeczywiście pomaga zmniejszać lukę domenową synthetic-to-real.

## Audyt i krytyka obecnego planu po sprawdzeniu repo

Ten dokument nie powinien być tylko listą sekcji do napisania. Po sprawdzeniu
`docs/RAPORT.md`, notatek `notes/`, skryptów `src/`, tabel `results/` i logów
widać kilka miejsc, w których pierwotny plan wymaga doprecyzowania, żeby finalny
raport był uczciwy metodologicznie.

| Problem w dotychczasowym planie | Dowód w repo | Poprawka wprowadzona do planu | Dlaczego ta zmiana jest potrzebna |
|---|---|---|---|
| Plan dobrze streszcza wyniki, ale za słabo krytykuje własne założenia. | `docs/RAPORT.md` jest szkicem roboczym, a `notes/11` i `notes/12` dopisują późniejsze, częściowo kontrintuicyjne wyniki. | Dodano ten audyt oraz niżej zasady porównywania metryk, protokołów i wariantów. | Raport finalny ma pokazać proces badawczy, a nie tylko ranking. Czytelnik musi zobaczyć, które hipotezy zostały potwierdzone, sfalsyfikowane albo tylko częściowo wsparte. |
| `results/tabela_zbiorcza.md` jest wygodna, ale nie jest pełnym źródłem prawdy. | `src/make_results_table.py` nie dodaje modelu finalnego, a stopka tabeli o RT-DETR jest nieaktualna względem `notes/07_porownanie_architektur.md`. | Ustalono hierarchię: najpierw `results/per_size/*.json`, potem notatki/logi, a `tabela_zbiorcza` tylko jako tabela pomocnicza po weryfikacji. | Bez tej korekty łatwo pominąć model finalny albo przepisać mylący komentarz o RT-DETR. |
| Warianty A/B/D/architektury/C nie są jednym idealnie kontrolowanym eksperymentem. | Notatki i skrypty pokazują różne zbiory, czasy treningu i sprzęt: A sweep 10k/60 epok, D 10k/45 epok, C 10k/30 epok, baseline 45k/100 epok, final 45k/60 epok. | Dodano wymóg jawnego oznaczania datasetu, liczby epok, `imgsz`, sprzętu i tego, czy porównanie jest wewnątrz rodziny eksperymentu, czy między rodzinami. | Zmniejsza ryzyko fałszywych wniosków przy małych różnicach, np. A1 vs baseline, oraz oddziela wynik jakościowy od kosztu obliczeniowego. |
| Punkt `imgsz=512` w eksperymencie D nie ma osobnego pliku `expD_512_*`. | `results/per_size/` zawiera `expD_320_10k_ml`, `expD_768_10k_ml`, `expD_1024_10k_ml`, ale nie zawiera `expD_512_10k_ml`; `notes/08` opisuje 512 jako `ref, A1`. | W sekcji D oznaczono 512 jako punkt referencyjny z przebiegu 10k, a nie pełnoprawny wariant sweepa D. | Wniosek o szkodliwości większej rozdzielczości pozostaje mocny dla 320/768/1024, ale raport nie powinien udawać, że 512 było uruchomione w tym samym dokładnym setupie D. |
| Nazwa `real25pct` jest podatna na błędną interpretację. | `notes/12` i `notes/11` pokazują, że `real_frac=0.25` oznacza 25% dostępnego real train/val, a nie 25% finalnego zbioru mixed. | Wzmocniono opis rzeczywistego udziału realnych danych: ok. 11% w C 25% na 10k i ok. 3.13% w modelu finalnym 45k. | To jest główny mechanizm wyjaśniający, dlaczego final 45k jest słabszy od C 25% 10k mimo większej liczby syntetyków. |
| Model finalny może zostać odczytany jako „najlepszy”, bo łączy najlepsze decyzje. | `results/final_combined_model_summary.md` i `notes/11` pokazują `AP@.5=0.9031`, mniej niż C 25% 10k `0.9466`. | Teza planu zostaje doprecyzowana: final jest najlepszą złożoną recepturą 45k w repo, ale nie najlepszym wynikiem synthetic+real. | To chroni raport przed narracją sukcesu za wszelką cenę. Wynik finalny jest ważny właśnie dlatego, że pokazuje brak liniowej addytywności efektów. |
| Część wyników ma problem kalibracji liczby detekcji, którego same AP nie pokazują. | `eval_per_size.py` zapisuje `n_detections`; synthetic 45k i RT-DETR mają 813000 detekcji, czyli 300/obraz. | `n_detections` / detekcje na obraz awansują z metryki opcjonalnej do obowiązkowej przy baseline, RT-DETR, C i modelu finalnym. | Liczba kandydatów jest dobrym wskaźnikiem nadmiernej „rozgadaności” modelu i pomaga wyjaśnić fałszywe alarmy. |
| Grad-CAM/EigenCAM może zostać potraktowany zbyt ilościowo. | `notes/10` mówi o metodzie jakościowej, a `src/gradcam_compare.py` ma etykietę D `0.515`, podczas gdy główny JSON `expD_320_10k_ml` daje `0.5224`; istnieje też lokalny `expD_320_win=0.5147`. | Sekcja Grad-CAM ma używać figury jakościowo, a liczby tabelaryczne brać z `results/per_size/*.json`. | Figura porównuje wzorce uwagi, nie jest źródłem metryk. Rozbieżne etykiety nie powinny wchodzić do tabel wyników. |
| Wygenerowane figury są częścią repo, ale ich odtwarzalność zależy od danych i wag. | `results/appearance/*.png`, `results/error_analysis_syn_vs_real.png`, `results/gradcam/gradcam_comparison.png` są obecne i śledzone, ale `data/`, `runs/` i `*.pt` są ignorowane. | Plan wskazuje figury jako artefakty raportowe, a reprodukcję jako skrypty/analitykę, nie jako gwarancję pełnego odtworzenia bez pobrania danych i wag. | Finalny raport powinien jasno oddzielić lekkie wyniki w repo od ciężkich danych/checkpointów, których nie redystrybuujemy. |

### Zasada poprawek

Każda zmiana wprowadzona niżej ma trzy elementy: **co poprawić**, **na jakim
artefakcie repo się opiera** oraz **po co ta poprawka jest potrzebna w raporcie**.
Jeśli metryka i opis w notatce różnią się od JSON-a wynikowego, do tabel brać JSON,
a notatkę używać do interpretacji i ograniczeń.

## Źródło prawdy dla metryk

W raporcie trzeba konsekwentnie rozdzielić dwa typy metryk:

| Typ metryk | Pliki | Jak używać |
|---|---|---|
| Ewaluacja COCO na realnym holdoucie | `results/per_size/*.json` | Główne porównania modeli i wnioski synthetic-to-real |
| Walidacja YOLO na zbiorze treningowym lub mieszanym | `results/baselines/*.json` | Pomocniczo, jako kontrola treningu |
| Podsumowania eksperymentów | `results/tabela_zbiorcza.md`, `results/expC_mixed_summary.md`, `results/final_combined_model_summary.md`, `results/final_combined_model_summary.json` | Tabele syntetyczne po sprawdzeniu z JSON-ami źródłowymi |
| Logi | `src/expC-71048.out`, `train-final-75379.out`, `results/expC_run.log`, `results/final_combined_run.log` | Parametry, zasoby, czasy, skład danych |

Do głównych tabel używać:

- `AP@.5`,
- `AP@[.5:.95]`,
- `AP_small`,
- `AP_medium`,
- `AP_large`,
- `AR@100`,
- `n_detections` lub detekcje/obraz jako wskaźnik nadmiernej liczby kandydatów,
  szczególnie dla synthetic baseline, RT-DETR, C i modelu finalnego.

Hierarchia wiarygodności:

1. Do wartości liczbowych w tabelach używać `results/per_size/<run>.json`.
2. Do interpretacji, parametrów i ograniczeń używać notatek oraz logów.
3. `results/tabela_zbiorcza.md` traktować jako wygodne zestawienie robocze, nie
   jako komplet raportu: obecnie nie obejmuje modelu finalnego, a jego stopka
   o RT-DETR wymaga korekty zgodnie z `notes/07_porownanie_architektur.md`.
4. Jeżeli figura lub skrypt ma etykietę metryki różną od JSON-a, w tekście i
   tabelach użyć JSON-a, a figurę traktować jakościowo.

Nie mieszać walidacji YOLO z realnym holdoutem. Przykład:

- `results/baselines/real_baseline_yolov10n.json` daje `mAP50=0.9798` na
  walidacji YOLO,
- `results/per_size/real_baseline.json` daje `AP@.5=0.9737` na realnym holdoucie
  COCO.

Do porównań w raporcie używać wartości z `results/per_size`.

Ważna konsekwencja: metryki z `results/baselines/*.json` mogą być bardzo wysokie,
bo walidują na synthetic albo mixed-val. W projekcie synthetic-to-real nie są
dowodem transferu na realną domenę.

## Główna teza raportu v2

Raport powinien prowadzić czytelnika przez następującą tezę:

> Naiwny transfer detektora z danych syntetycznych RarePlanes na rzeczywiste
> zdjęcia satelitarne załamuje się przez różnice rozmiaru obiektów, gęstości scen,
> fotometrii i tekstury. Sama większa liczba syntetycznych obrazów pomaga tylko
> umiarkowanie. Najsilniejszym sposobem redukcji luki domenowej jest dodanie nawet
> niewielkiej liczby realnych próbek. Interwencje czysto syntetyczne, takie jak
> zmiana skali, szum czy łagodna fotometria, pomagają, ale znacznie słabiej.
> Model finalny 45k potwierdza, że połączenie dobrych decyzji poprawia czystą
> syntetykę, ale nie przebija wariantu C 25% 10k, bo ta sama liczba realnych
> próbek zostaje rozcieńczona przez większy zbiór syntetyczny. To pokazuje, że
> kluczowa jest nie tylko liczba syntetyków, lecz także proporcja i siła sygnału
> realnej domeny.

W raporcie nie budować narracji, że wszystkie efekty A/B/C/D sumują się liniowo.
Lepsza narracja: eksperymenty cząstkowe pozwoliły zrozumieć mechanizmy, a model
finalny był próbą ich połączenia, która ujawniła ograniczenie proporcji danych.

## Docelowa struktura raportu

Raport finalny powinien mieć następujący układ:

1. Tytuł, autorzy, repozytorium.
2. Abstrakt.
3. Wstęp i pytanie badawcze.
4. Dane RarePlanes i licencja.
5. Analiza domain shift: rozmiary, gęstość, role, kolor, histogramy, FFT.
6. Metoda i protokół: YOLOv10n, COCO->YOLO, real holdout, metryki.
7. Baseline real->real.
8. Baseline synthetic->real: 6460 oraz 45k.
9. Eksperyment A: fotometria HSV.
10. Eksperyment B: degradacja częstotliwościowa.
11. Eksperyment C: mixed training.
12. Eksperyment D: skala wejścia.
13. Porównanie architektur.
14. Grad-CAM / EigenCAM.
15. Model finalny 45k.
16. Synteza hipotez i ranking metod.
17. Ograniczenia.
18. Reprodukcja.
19. Licencja i atrybucja.
20. Konkluzja.

## Co dopisać względem `docs/RAPORT.md`

| Obszar | Status w `RAPORT.md` | Zmiana w v2 |
|---|---|---|
| Abstrakt | Brak | Dodać krótkie streszczenie problemu, metod i wyników |
| Dane | Jest skrót | Rozwinąć licencję, wariant PS-RGB tiled i real train/test shift |
| Domain shift | Jest syntetycznie | Dodać role, gęstość, histogramy, FFT i inspekcję wizualną |
| Protokół | Jest skrót | Dopisać źródło prawdy metryk, split holdout i pipeline |
| Porównywalność eksperymentów | Brak | Dodać ostrzeżenie, że A/B/D/arch/C różnią się datasetem, epokami i sprzętem; porównania globalne traktować jako ranking praktyczny, nie pełną izolację przyczynową |
| Baseline'y | Są | Uporządkować real, synthetic 6460 i synthetic 45k |
| A | Jest krótki wniosek | Dodać finalny A 45k i redystrybucję AP_small/AP_large |
| B | Jest krótki wniosek | Dodać caveat o niedziałającym on-the-fly i wersji plikowej |
| C | Jest ranking | Rozwinąć dokładnie real_frac, skład danych i mixed-val vs holdout |
| D | Jest wniosek | Dopisać mechanizm downscaling jako domain adaptation |
| Architektury | Jest | Uściślić, że problemem jest architektura/end-to-end, nie sama pojemność |
| Grad-CAM | Brak | Dodać osobną sekcję z `notes/10` |
| Model finalny | Brak | Dodać przepis B2+C+D+A1, wyniki i interpretację |
| Ograniczenia | Są | Rozwinąć single seed, sprzęt, storage, mixed real_frac, brak pełnej siatki kontrolnej i różne schedule treningowe |
| Reprodukcja | Jest szkic | Dodać konkretne skrypty dla C i modelu finalnego |

## Sekcje raportu finalnego

### 1. Abstrakt

Napisać 150-250 słów. Musi zawierać:

- problem: detekcja `aircraft` w RarePlanes pod domain shift synthetic-to-real,
- model bazowy: YOLOv10n,
- eksperymenty A/B/C/D, architektury, Grad-CAM, model finalny,
- kluczowe wyniki:
  - real baseline: `AP@.5=0.9737`, `AP@[.5:.95]=0.8073`,
  - synthetic 45k baseline: `AP@.5=0.4525`, `AP@[.5:.95]=0.2683`,
  - najlepszy C 25%: `AP@.5=0.9466`, `AP@[.5:.95]=0.7120`,
  - final 45k: `AP@.5=0.9031`, `AP@[.5:.95]=0.6332`,
- główny wniosek: realne próbki są najsilniejszą kotwicą domenową.

### 2. Wstęp i pytanie badawcze

Bazować na sekcji 1 z `docs/RAPORT.md`.

Zawrzeć:

- dlaczego dane syntetyczne są atrakcyjne,
- czym jest luka domenowa,
- dlaczego RarePlanes jest dobrym przypadkiem testowym,
- pytanie badawcze:

```text
Jak skutecznie zmniejszyć spadek jakości detektora trenowanego na syntetycznych
zdjęciach RarePlanes po przeniesieniu na rzeczywiste zdjęcia satelitarne?
```

Hipotezy:

| Hipoteza | Treść | Eksperyment |
|---|---|---|
| HA | Fotometria/HSV zmniejsza różnice kolorystyczne | A |
| HB | Szum/degradacja częstotliwościowa upodabnia synthetic do real | B |
| HC | Mała liczba realnych próbek mocno poprawia transfer | C |
| HD | Zmiana rozdzielczości/skali pomaga małym obiektom | D |
| HArch | Architektura wpływa na odporność na domain shift | porównanie architektur |

W raporcie zaznaczyć, że HD została potwierdzona odwrotnie: pomaga mniejsza, a
nie większa rozdzielczość.

### 3. Dane i analiza domain shift

Źródła:

- `notes/00_dane_i_licencja.md`,
- `notes/01_analiza_adnotacji.md`,
- `notes/02_analiza_wygladu.md`.

#### Dane i licencja

Opisać:

- RarePlanes, Shermeyer et al., 2020,
- licencja CC BY-SA 4.0,
- używany wariant: `PS-RGB_tiled`,
- powód pominięcia MS/PAN: inne kanały, nieporównywalne z RGB synthetic,
- obrazy nie są redystrybuowane w repo.

Tabela:

| Zbiór | Obrazy | Instancje | Inst./obraz |
|---|---:|---:|---:|
| real train | 5 815 | 18 393 | 3.16 |
| real test | 2 710 | 6 812 | 2.51 |
| synthetic train | 45 000 | 566 143 | 12.58 |
| synthetic test | 5 000 | 62 841 | 12.57 |

#### Rozmiar obiektów i gęstość scen

Tabela:

| Zbiór | small | medium | large | Mediana bbox |
|---|---:|---:|---:|---:|
| REAL train | 43.8% | 44.0% | 12.2% | 1 180 px² |
| REAL test | 24.9% | 51.1% | 23.9% | 2 505 px² |
| SYN train | 10.2% | 51.3% | 38.5% | 6 776 px² |
| SYN test | 10.1% | 51.8% | 38.0% | 6 675 px² |

Wnioski:

- synthetic ma dużo większe obiekty,
- real train ma znacznie więcej małych obiektów,
- synthetic ma ok. 12.6 instancji/kafel, real train ok. 3.2,
- real train i real test też mają własny shift, więc nie wszystko wolno
  przypisywać synthetic-to-real.

#### Role klas

Użyć krótko:

- real train jest zdominowany przez małe samoloty,
- synthetic przez średnie i duże,
- projekt finalnie skupia się na jednej klasie `aircraft`, żeby mierzyć
  lokalizację.

#### Kolor, histogramy, FFT

Wstawić lub odwołać się do:

- `results/appearance/color_histograms.png`,
- `results/appearance/fft_spectra.png`,
- `results/appearance/radial_power.png`.

Kluczowe liczby:

| Metryka | Real | Synthetic | Wniosek |
|---|---:|---:|---|
| Jasność średnia | 71.7 | 133.0 | synthetic ok. 1.85x jaśniejszy |
| Nasycenie średnie | 0.303 | 0.122 | real ok. 2.5x bardziej nasycony |
| RGB mean | 82/71/62 | 129/131/139 | real cieplejszy, synthetic chłodniejszy |

Wniosek:

- real jest ciemniejszy, bardziej nasycony i bardziej wysokoczęstotliwościowy,
- synthetic jest jaśniejszy, gładszy i chłodniejszy,
- to motywuje eksperymenty A i B.

### 4. Metoda i protokół

Opisać:

- model bazowy: YOLOv10n, pretrained COCO,
- konwersję COCO->YOLO,
- `synthetic_10k` jako podzbiór do sweepów,
- pełne synthetic 45k jako baseline i final,
- real holdout `instances_test_aircraft.json` jako święty zbiór testowy,
- metryki COCO per-size z `src/eval_per_size.py`.

Uwaga krytyczna do raportu:

- wspólne były przede wszystkim: realny holdout COCO, klasa `aircraft`, seed `42`
  i sposób raportowania `AP/AR`;
- protokół treningowy nie był identyczny dla wszystkich wariantów, więc raport
  ma jawnie podawać dataset, epoki, `batch`, `imgsz`, sprzęt i źródło metryki;
- w eksperymencie C faktyczne pełne uruchomienie miało `epochs=30`, `batch=32`,
  `imgsz=512`, `workers=4`;
- model finalny miał `epochs=60`, `batch=64`, `imgsz=320`, `workers=4`;
- baseline real i baseline synthetic były trenowane w dłuższym reżimie `100 epok`;
- sweep A miał `60 epok`, a finalny A 45k użył `best.pt` z treningu, który według
  notatki zakończył się przed pełnym planowanym końcem;
- D było liczone jako sweep skali na synthetic 10k, ale punkt 512 jest referencją
  z przebiegu 10k, nie osobnym plikiem `expD_512_*`;
- nie przepisywać bezkrytycznie ogólnego `60 epok` ze szkicu, jeśli dany artefakt
  mówi inaczej.

Tabela kontroli protokołu do wstawienia lub streszczenia:

| Grupa wyników | Dane treningowe | Epoki | Batch | `imgsz` | Główne źródło |
|---|---|---:|---:|---:|---|
| Real baseline | real train/val | 100 | 64 | 512 | `notes/03`, `results/per_size/real_baseline.json` |
| Synthetic 6460 | synthetic 6460 | 100 | 64 | 512 | `notes/04`, `results/per_size/syn_to_real_baseline.json` |
| Synthetic 45k | synthetic 45k | 100 | 64 | 512 | `notes/05`, `results/per_size/syn45k_to_real_baseline.json` |
| A sweep | synthetic 10k | 60 | 64 | 512 | `notes/06`, `src/sweep_A_photometric.sh` |
| A final 45k | synthetic 45k + A1 | 45 / best checkpoint | 16 | 512 | `notes/06`, `results/per_size/expA_final_45k.json` |
| B files | synthetic 10k po degradacji | 60 | 64 | 512 | `notes/09`, `src/sweep_B_frequency.sh` |
| C mixed | synthetic 10k + real_frac | 30 | 32 | 512 | `notes/12`, `src/expC-71048.out` |
| D skala | synthetic 10k | 45 | auto | 320/768/1024 | `notes/08`, `results/per_size/expD_*` |
| Architektury | synthetic 10k | różne notebooki/sweep | różne | 512 | `notes/07`, `results/per_size/{yolo11l,rtdetr*}.json` |
| Final 45k | synthetic 45k B2 + 25% real train/val + A1 | 60 | 64 | 320 | `notes/11`, `train-final-75379.out` |

Dlaczego: ta tabela nie ma osłabiać wyników, tylko zabezpieczać interpretację.
Rankingi globalne pokazują najlepsze praktyczne punkty w repo, ale ścisłe wnioski
przyczynowe wyciągać głównie wewnątrz jednej rodziny eksperymentu.

### 5. Baseline'y

Źródła:

- `notes/03_baseline_real.md`,
- `notes/04_baseline_synthetic_luka_domenowa.md`,
- `notes/05_baseline_synthetic_45k.md`,
- `results/per_size/real_baseline.json`,
- `results/per_size/syn_to_real_baseline.json`,
- `results/per_size/syn45k_to_real_baseline.json`.

Tabela:

| Model | AP@.5 | AP@[.5:.95] | AP_small | AP_medium | AP_large | AR@100 | Det/img |
|---|---:|---:|---:|---:|---:|---:|---:|
| Real baseline | 0.9737 | 0.8073 | 0.7586 | 0.7825 | 0.8991 | 0.8465 | 9.5 |
| Synthetic 6460 | 0.4095 | 0.2388 | 0.2624 | 0.3284 | 0.0671 | 0.3686 | 33.8 |
| Synthetic 45k | 0.4525 | 0.2683 | 0.2859 | 0.3839 | 0.2008 | 0.5459 | 300.0 |

Wnioski:

- real baseline spełnia próg jakości i jest górnym punktem odniesienia,
- synthetic 6460 pokazuje dramatyczną lukę domenową,
- synthetic 45k poprawia 6460, szczególnie `AP_large`, ale nadal jest daleko od
  real baseline,
- synthetic 45k osiąga większy recall, ale robi to przy maksymalnych `300`
  detekcjach/obraz, więc poprawę trzeba interpretować razem z kalibracją liczby
  kandydatów,
- sama różnorodność syntetyki pomaga, lecz nie wystarcza.

Wyjaśnienie poprawki: dodanie `Det/img` wynika z `n_detections` w JSON-ach
`results/per_size`. Bez tej kolumny raport nie pokazuje, że część modeli podnosi
AR kosztem ogromnej liczby propozycji.

### 6. Eksperyment A: fotometria HSV

Źródło: `notes/06_eksperyment_A_fotometria.md`.

W raporcie uwzględnić:

- motywacja: różnica jasności, nasycenia i balansu bieli,
- sweep A1/A2/A3 na 10k,
- finalny A1 na 45k.

Tabela sweepa:

| Wariant | hsv_s | hsv_v | AP@.5 | AP@[.5:.95] | AP_small | AP_medium | AP_large |
|---|---:|---:|---:|---:|---:|---:|---:|
| A1 słaby | 0.4 | 0.3 | 0.459 | 0.264 | 0.306 | 0.357 | 0.091 |
| A2 średni | 0.7 | 0.5 | 0.431 | 0.247 | 0.265 | 0.345 | 0.099 |
| A3 mocny | 0.9 | 0.7 | 0.446 | 0.252 | 0.268 | 0.344 | 0.105 |

Ostrożność interpretacyjna:

- sweep A jest na `synthetic_10k`, a baseline 45k jest osobnym punktem odniesienia,
  nie identyczną kontrolą;
- pewny wniosek ze sweepa: silniejszy jitter nie daje monotonicznej poprawy, a
  A1 jest najlepszym testowanym ustawieniem HSV w tej rodzinie;
- porównanie "10k+A1 osiąga ok. poziom 45k baseline" jest wartościowe praktycznie,
  ale nie dowodzi, że fotometria zastępuje 35k obrazów syntetycznych w sensie
  przyczynowym.

Wyjaśnienie poprawki: `notes/06` porównuje warianty A do baseline 45k, ale skrypty
`src/sweep_A_photometric.sh` pokazują, że sweep trenował na `synthetic_10k`. Raport
ma zachować ten ciekawy kontrast, lecz opisać go jako kontrast praktyczny.

Finalny A1 45k:

| Metryka | Synthetic 45k | A1 45k | Zmiana |
|---|---:|---:|---:|
| AP@.5 | 0.4525 | 0.4549 | +0.0024 |
| AP@[.5:.95] | 0.2683 | 0.2684 | ~0 |
| AP_small | 0.2859 | 0.3370 | +0.0511 |
| AP_medium | 0.3839 | 0.3545 | -0.0294 |
| AP_large | 0.2008 | 0.0902 | -0.1106 |

Wniosek:

- HA potwierdzona słabo,
- fotometria nie zmniejsza luki globalnie,
- poprawia `AP_small`, ale kosztuje `AP_large`,
- silniejszy jitter nie daje monotonicznej poprawy.

### 7. Eksperyment B: częstotliwości

Źródło: `notes/09_eksperyment_B_czestotliwosci.md`.

Tabela:

| Wariant | Degradacja | AP@.5 | AP@[.5:.95] | AP_small | AP_medium | AP_large | Det/img |
|---|---|---:|---:|---:|---:|---:|---:|
| B2 | sam szum, sigma <= 8 | 0.490 | 0.280 | 0.283 | 0.382 | 0.119 | 44.2 |
| B1 | blur + szum | 0.451 | 0.259 | 0.261 | 0.358 | 0.101 | 30.0 |
| B3 | blur + szum + JPEG | 0.451 | 0.261 | 0.294 | 0.356 | 0.101 | 41.7 |
| Synthetic 45k ref | brak | 0.452 | 0.268 | 0.286 | 0.384 | 0.201 | 300.0 |

Wniosek:

- pomaga sam szum,
- blur szkodzi albo nie pomaga,
- JPEG neutralny w testowanej konfiguracji,
- mechanizm zgodny z FFT: real ma więcej wysokich częstotliwości, noise je
  dodaje, blur usuwa.

Ostrożność interpretacyjna:

- B1/B2/B3 są porównywalne między sobą jako warianty plikowej degradacji na 10k;
- synthetic 45k w tabeli jest referencją globalną, ale ma inny rozmiar zbioru i
  nie jest kontrolą jeden-do-jednego dla B;
- jeśli raport liczy delty, najczytelniej pisać "B2 jest najlepszym wariantem B"
  oraz osobno "B2 przewyższa praktyczną referencję synthetic 45k w AP@.5, choć nie
  w AP_large".

Wyjaśnienie poprawki: plan pierwotny mógł sugerować proste porównanie B2 10k z
baseline 45k. Zachowujemy ten punkt odniesienia, ale nie udajemy identycznego
kontrolowanego setupu.

Nota inżynierska:

- pierwotna degradacja on-the-fly przez monkey-patch `cv2.imread` nie była
  wiarygodna,
- do raportu używać wyników z wersji plikowej.

### 8. Eksperyment C: mixed training

Źródła:

- `notes/12_eksperyment_C_mixed_training.md`,
- `results/expC_mixed_summary.md`,
- `src/expC-71048.out`,
- `results/per_size/expC_*_ml.json`.

Ta sekcja ma być jedną z najważniejszych w raporcie.

#### Znaczenie `real_frac`

Napisać jasno:

- `real_frac` oznacza procent dostępnego real train/val,
- nie oznacza procentu finalnego zbioru mixed,
- real test nie był użyty w treningu.

Tabela:

| Wariant | Train synthetic | Train real | Real share train | Val synthetic | Val real | Real share val |
|---|---:|---:|---:|---:|---:|---:|
| C 1% | 10 000 | 49 | 0.49% | 1 764 | 9 | 0.51% |
| C 5% | 10 000 | 247 | 2.41% | 1 764 | 44 | 2.43% |
| C 10% | 10 000 | 494 | 4.71% | 1 764 | 87 | 4.70% |
| C 25% | 10 000 | 1 236 | 11.00% | 1 764 | 218 | 11.00% |

#### Parametry

- `YOLOv10n`,
- `epochs=30`,
- `batch=32`,
- `imgsz=512`,
- `workers=4`,
- `seed=42`,
- domyślne augmentacje Ultralytics,
- HSV domyślne: `0.015/0.7/0.4`,
- brak B2 i brak A1.

#### Wyniki

| Wariant | AP@.5 | AP@[.5:.95] | AP_small | AP_medium | AP_large | AR@100 | Det/img |
|---|---:|---:|---:|---:|---:|---:|---:|
| C 1% | 0.6666 | 0.3957 | 0.3614 | 0.4535 | 0.3462 | 0.5954 | 66.5 |
| C 5% | 0.8516 | 0.5660 | 0.5197 | 0.5761 | 0.6108 | 0.7144 | 39.4 |
| C 10% | 0.8960 | 0.6188 | 0.5404 | 0.6184 | 0.7060 | 0.7349 | 35.6 |
| C 25% | 0.9466 | 0.7120 | 0.6365 | 0.6974 | 0.8166 | 0.7822 | 20.8 |

Wnioski:

- HC mocno potwierdzona,
- efekt jest monotoniczny,
- największy skok jest między 1% a 5%,
- C 25% to najlepszy wynik synthetic+real w projekcie,
- C 25% zbliża się do real baseline w `AP@.5`,
- wzrost udziału realnych próbek poprawia też selektywność: detekcje/obraz spadają
  z 66.5 do 20.8,
- mixed-val nie jest główną metryką, bo prawie nie rozróżnia wariantów.

Wyjaśnienie poprawki: `notes/12` i JSON-y `results/per_size/expC_*` pokazują, że
spadek liczby detekcji jest jednym z mechanizmów poprawy. Raport powinien opisać
go obok AP, bo sama krzywa AP nie pokazuje kalibracji modelu na realnej domenie.

### 9. Eksperyment D: skala wejścia

Źródło: `notes/08_eksperyment_D_skala.md`.

Tabela:

| img size | AP@.5 | AP@[.5:.95] | AP_small | AP_medium | AP_large | Det/img | Status punktu |
|---|---:|---:|---:|---:|---:|---:|---|
| 320 | 0.522 | 0.283 | 0.308 | 0.367 | 0.151 | 54.2 | run D / JSON `expD_320_10k_ml` |
| 512 ref | 0.459 | 0.264 | 0.306 | 0.357 | 0.091 | 27.4 | referencja 10k z A1/YOLOv10n, nie osobny `expD_512` |
| 768 | 0.448 | 0.252 | 0.230 | 0.339 | 0.116 | 14.8 | run D / JSON `expD_768_10k_ml` |
| 1024 | 0.330 | 0.190 | 0.222 | 0.250 | 0.047 | 7.4 | run D / JSON `expD_1024_10k_ml` |

Wniosek:

- HD w pierwotnej formie sfalsyfikowana,
- mniejsza rozdzielczość pomaga,
- mechanizm: downscaling dopasowuje pozorny rozmiar dużych syntetycznych samolotów
  do mniejszych realnych.

Wyjaśnienie poprawki:

- w `results/per_size` nie ma osobnego pliku `expD_512_10k_ml.json`, więc 512 musi
  być oznaczone jako punkt referencyjny;
- istnieją dwa wyniki 320: główny `expD_320_10k_ml` (`AP@.5=0.5224`) oraz lokalny
  `expD_320_win` (`AP@.5=0.5147`, użyty m.in. w Grad-CAM). Do tabel używać głównego
  JSON-a `expD_320_10k_ml`, a różnicę traktować jako techniczną rozbieżność artefaktów.

### 10. Porównanie architektur

Źródło: `notes/07_porownanie_architektur.md`.

Tabela:

| Model | Typ | Parametry | AP@.5 | AP@[.5:.95] | AP_small | AP_medium | AP_large | Det/img |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| YOLOv10n | CNN | 2.3M | 0.459 | 0.264 | 0.306 | 0.357 | 0.091 | ~27 |
| YOLO11l | CNN | 25M | 0.467 | 0.271 | 0.306 | 0.348 | 0.108 | ~22 |
| RT-DETR-l | Transformer | 32M | 0.297 | 0.157 | 0.146 | 0.230 | 0.081 | 300 |
| RT-DETR-x | Transformer | 67M | 0.380 | 0.205 | 0.206 | 0.285 | 0.094 | 300 |

Wnioski:

- CNN-y transferują lepiej niż RT-DETR,
- większy transformer był lepszy niż mniejszy, więc nie twierdzić, że problemem
  jest sama pojemność,
- RT-DETR wyrzucał 300 detekcji/obraz na realnym teście,
- architektura i mechanizm selekcji detekcji są ważniejsze niż liczba parametrów.

Korekta względem `results/tabela_zbiorcza.md`:

- nie przepisywać stopki "RT-DETR z best.pt po 2 epokach" jako finalnego opisu;
- `notes/07` mówi, że wcześniejsze `0.489` było artefaktem, a raport ma używać
  ustabilizowanych wartości z `results/per_size/rtdetr_l_10k_ml.json` i
  `results/per_size/rtdetrx_10k_ml.json`;
- fakt krytyczny do zachowania: oba RT-DETR mają `n_detections=813000`, czyli
  dokładnie 300 detekcji na każdy z 2710 obrazów realnego holdoutu.

Wyjaśnienie poprawki: ta zmiana zapobiega sprzeczności między tabelą roboczą a
notatką interpretacyjną. Wniosek architektoniczny ma dotyczyć generalizacji i
mechanizmu end-to-end, nie rzekomego niedotrenowania po dwóch epokach.

### 11. Grad-CAM / EigenCAM

Źródła:

- `notes/10_gradcam_interpretowalnosc.md`,
- `src/gradcam_compare.py`,
- `results/gradcam/gradcam_comparison.png`.

Opisać:

- użyto EigenCAM, nie klasycznego Grad-CAM,
- powód: YOLOv10 jest end-to-end i nie ma wygodnego pojedynczego skalara klasy,
- warstwa: C2f backbone, warstwa 8,
- porównano 6 modeli na tych samych realnych kaflach z małymi obiektami.

Modele w figurze:

1. synthetic 45k,
2. A słaby HSV,
3. B2 szum,
4. D imgsz320,
5. C mixed 25%,
6. real baseline.

Wnioski:

- modele synthetic nie ignorują małych samolotów,
- D ma ostrzejszą uwagę na obiektach,
- C przesuwa uwagę ku wzorcowi real baseline,
- luka domenowa wynika raczej z rozkładu, skali i wyglądu niż z całkowicie
  błędnych cech.

Ostrożność interpretacyjna:

- EigenCAM jest analizą jakościową, nie metryką skuteczności;
- `src/gradcam_compare.py` może etykietować model D jako `0.515`, bo używa
  lokalnego artefaktu `expD_320_win`; w tabelach wyników raportu używać głównego
  `results/per_size/expD_320_10k_ml.json`, czyli `AP@.5=0.5224`;
- nie pisać, że Grad-CAM "dowodzi" mechanizmu przyczynowego. Lepsze sformułowanie:
  wizualizacje są zgodne z mechanizmem wywnioskowanym z metryk i analizy domeny.

Wyjaśnienie poprawki: `notes/10` opisuje figurę jako jakościową, a repo zawiera
dwa bliskie, ale nieidentyczne artefakty D 320. Dzięki tej adnotacji raport nie
pomiesza źródeł liczbowych z wizualizacją.

### 12. Model finalny 45k

Źródła:

- `notes/11_model_finalny_45k.md`,
- `run_train_final_cluster.sh`,
- `train-final-75379.out`,
- `results/final_combined_model_summary.md`.

Przepis:

```text
B2 + C + D + A1
```

Parametry:

- pełne synthetic 45k,
- B2: `noise_sigma=8.0`,
- C: 25% real train/val,
- D: `imgsz=320`,
- A1: HSV `0.015/0.4/0.3`,
- `epochs=60`,
- `batch=64`,
- seed `42`.

Skład danych:

| Split | Synthetic | Real | Real share |
|---|---:|---:|---:|
| train | 38 250 | 1 236 | 3.13% |
| val | 6 750 | 218 | 3.13% |

Tabela:

| Model | AP@.5 | AP@[.5:.95] | AP_small | AP_medium | AP_large | AR@100 | Det/img |
|---|---:|---:|---:|---:|---:|---:|---:|
| C 25% real 10k | 0.9466 | 0.7120 | 0.6365 | 0.6974 | 0.8166 | 0.7822 | 20.8 |
| Final 45k B2+C+D+A1 | 0.9031 | 0.6332 | 0.5616 | 0.6179 | 0.7523 | 0.7480 | 28.8 |
| Real baseline | 0.9737 | 0.8073 | 0.7586 | 0.7825 | 0.8991 | 0.8465 | 9.5 |

Dopisać:

- finalny model bardzo poprawia synthetic 45k baseline,
- nie przebija C 25% 10k,
- realny udział danych realnych spada do ok. 3.13%,
- efekty A/B/C/D nie sumują się liniowo,
- final zmniejsza nadmiar detekcji względem synthetic 45k baseline z 300.0 do
  28.8 detekcji/obraz, ale nadal jest mniej selektywny niż real baseline,
- final benchmark: `100.79 FPS`, peak CUDA memory `1340.7 MB`.

Wyjaśnienie poprawki: pierwotny plan miał poprawną tezę jakościową, ale bez
`Det/img` tracił ważny dowód, że final poprawia nie tylko AP, lecz także kalibrację
predykcji względem czystej syntetyki.

### 13. Synteza hipotez i ranking metod

Tabela:

| Hipoteza | Wynik | Dowód | Interpretacja |
|---|---|---|---|
| HA fotometria | słabo/częściowo | A1 10k `AP@.5=0.459`, A1 45k `0.455` | marginalny zysk, AP_small kosztem AP_large |
| HB częstotliwości | częściowo | B2 `AP@.5=0.490`, B1/B3 ok. `0.451` | pomaga szum, nie blur |
| HC mixed | mocno potwierdzona | C 25% `AP@.5=0.9466` | realne próbki jako kotwica domenowa |
| HD skala | potwierdzona odwrotnie | img 320 `0.522`, 1024 `0.330` | mniejsza rozdzielczość dopasowuje skalę |
| HArch | potwierdzona | CNN `0.459-0.467`, RT-DETR `0.297-0.380` | architektura ważniejsza niż sama pojemność |

Ranking ogólny:

```text
C 25% real 10k > final 45k B2+C+D+A1 > C 10% > C 5% > C 1% > D 320 > B2 > A1 ~= synthetic 45k
```

Ranking czysto syntetyczny:

```text
D imgsz320 > B2 noise > A1 HSV ~= synthetic 45k baseline
```

Ostrożność do dopisania pod rankingiem:

- ranking jest uporządkowany praktycznie według `AP@.5` na tym samym realnym
  holdoucie, ale warianty nie zawsze mają identyczny protokół treningowy;
- dla wniosków przyczynowych ważniejsze są rankingi wewnętrzne: A1>A2/A3,
  B2>B1/B3, C monotonicznie rośnie z `real_frac`, D 320>768>1024;
- final 45k należy interpretować jako test addytywności decyzji, nie jako prostą
  kontynuację C.

Wyjaśnienie poprawki: bez tego akapitu ranking może sugerować większą kontrolę
eksperymentalną, niż faktycznie wynika z notatek i skryptów.

### 14. Ograniczenia

Uwzględnić:

- pojedynczy seed `42`,
- większość sweepów na 10k,
- różne schedule treningowe, batch size i sprzęt między rodzinami eksperymentów,
- różnice sprzętowe,
- mixed training wymaga realnych adnotacji,
- `real25pct` jest mylące,
- mixed-val nie jest główną metryką,
- Grad-CAM/EigenCAM jest jakościowy,
- model finalny ma inną proporcję danych niż C 25%,
- storage i czas przygotowania danych były realnym ograniczeniem,
- brak pełnej siatki kombinacji A/B/C/D,
- brak osobnego punktu `expD_512_*` w sweepie D; 512 jest referencją z przebiegu
  10k, więc krzywą D opisywać z tą adnotacją,
- brak pełnego czystego baseline'u synthetic 10k w tej samej konwencji dla każdego
  sweepa; część porównań używa synthetic 45k jako praktycznej referencji,
- `results/tabela_zbiorcza.md` jest tabelą pochodną i nie obejmuje finalnego modelu,
  więc raport ma ją sprawdzić z JSON-ami przed przepisaniem,
- checkpointy i dane nie są redystrybuowane w repo; odtworzenie Grad-CAM i części
  benchmarków wymaga lokalnych wag oraz danych RarePlanes,
- analiza wyglądu jest liczona na całych kaflach, więc opisuje głównie tło; brak
  domkniętej analizy tylko w obrębie bboxów.

Wyjaśnienie poprawki: te ograniczenia wynikają bezpośrednio z `notes/02`, `notes/08`,
`notes/11`, `notes/12`, `.gitignore` i struktury `results/per_size`. Dodanie ich
chroni raport przed nadmiernie mocnymi twierdzeniami o przyczynowości.

### 15. Reprodukcja

Uwzględnić komendy:

```bash
python3 src/coco_to_yolo.py --domain synthetic --classes aircraft --val-frac 0.15 --seed 42
python3 src/coco_to_yolo.py --domain real --classes aircraft --val-frac 0.15 --seed 42
python3 src/make_subset.py --n-train 10000 --name synthetic_10k --seed 42
```

Eksperyment C:

```bash
sbatch src/run_expC.sh
```

albo:

```bash
python src/run_expC_mixed_cluster.py \
  --src-dataset data/yolo/synthetic_10k \
  --dataset-tag 10k \
  --real-src data/yolo/real_aircraft \
  --pcts 1 5 10 25 \
  --fracs 0.01 0.05 0.10 0.25 \
  --epochs 30 \
  --batch 32 \
  --workers 4 \
  --device 0 \
  --imgsz 512 \
  --model yolov10n.pt
```

Model finalny:

```bash
sbatch run_train_final_cluster.sh
```

albo bezpośrednio:

```bash
python src/train_final_model.py \
  --data-dir /work/$USER/rareplanes-data/data \
  --epochs 60 \
  --batch 64 \
  --workers 4 \
  --device 0 \
  --full-download-workers 32
```

Tabela wyników:

```bash
python src/make_results_table.py
```

Uwaga: obecny `src/make_results_table.py` generuje `results/tabela_zbiorcza.md`,
ale nie dodaje finalnego modelu 45k. Do raportu finalnego trzeba albo dopisać
wiersz finalny ręcznie z `results/per_size/final_yolov10n_syn45k_noise_real25pct_img320_hsvA1_ml.json`,
albo rozszerzyć `ORDER` w `src/make_results_table.py`.

Wyjaśnienie poprawki: bez tej noty osoba pisząca raport może założyć, że tabela
zbiorcza zawiera wszystkie wyniki, choć finalny model jest trzymany osobno w
`results/final_combined_model_summary.*`.

### 16. Licencja i atrybucja

Zawrzeć:

- RarePlanes CC BY-SA 4.0,
- brak redystrybucji obrazów w repo,
- atrybucja:

```text
J. Shermeyer, T. Hossler, A. Van Etten, D. Hogan, R. Lewis, and D. Kim.
In-Q-Tel - CosmiQ Works and AI.Reverie. RarePlanes Dataset, June 2020.
```

## Figury i artefakty do użycia

| Element | Źródło | Sekcja |
|---|---|---|
| Histogramy kolorów | `results/appearance/color_histograms.png` | Dane/domain shift |
| FFT | `results/appearance/fft_spectra.png` | Dane/domain shift |
| Radial power | `results/appearance/radial_power.png` | Dane/domain shift |
| Error analysis | `results/error_analysis_syn_vs_real.png` | Baseline synthetic |
| Grad-CAM/EigenCAM | `results/gradcam/gradcam_comparison.png` | Interpretowalność |
| Krzywa C | `results/expC_mixed_summary.csv` | Eksperyment C |
| Krzywa D | `results/per_size/expD_*` | Eksperyment D |
| Zbiorcza tabela wyników | `results/tabela_zbiorcza.md` | Wyniki cząstkowe; obecnie bez modelu finalnego |
| Final summary | `results/final_combined_model_summary.md` | Model finalny |
| Final per-size JSON | `results/per_size/final_yolov10n_syn45k_noise_real25pct_img320_hsvA1_ml.json` | Liczby modelu finalnego do tabel |

Uwaga o artefaktach:

- figury w `results/appearance`, `results/error_analysis_syn_vs_real.png` i
  `results/gradcam/gradcam_comparison.png` są obecne w repo i mogą iść do raportu;
- ciężkie dane `data/`, checkpointy `*.pt` i `runs/` są ignorowane, więc pełna
  regeneracja figur wymaga pobrania danych i posiadania wag;
- do podpisów pod figurami nie przepisywać ślepo metryk z etykiet obrazka, jeśli
  różnią się od JSON-ów.

## Mapowanie notatek na raport

| Notatka | Gdzie użyć | Najważniejsza treść |
|---|---|---|
| `notes/00_dane_i_licencja.md` | Dane, licencja | RarePlanes, CC BY-SA, PS-RGB tiled |
| `notes/01_analiza_adnotacji.md` | Domain shift | rozmiary, gęstość, role |
| `notes/02_analiza_wygladu.md` | Domain shift, motywacja A/B | kolory, histogramy, FFT |
| `notes/03_baseline_real.md` | Baseline real | górny punkt odniesienia, FPS |
| `notes/04_baseline_synthetic_luka_domenowa.md` | Baseline synthetic | pomiar luki, fałszywe alarmy |
| `notes/05_baseline_synthetic_45k.md` | Synthetic 45k | większa syntetyka pomaga umiarkowanie |
| `notes/06_eksperyment_A_fotometria.md` | Eksperyment A | słaby HSV, AP_small kosztem AP_large |
| `notes/07_porownanie_architektur.md` | Architektury | CNN > RT-DETR, 300 det/obraz |
| `notes/08_eksperyment_D_skala.md` | Eksperyment D | mniejsze imgsz pomaga |
| `notes/09_eksperyment_B_czestotliwosci.md` | Eksperyment B | noise pomaga, blur nie |
| `notes/10_gradcam_interpretowalnosc.md` | Grad-CAM | EigenCAM i uwaga na małych obiektach |
| `notes/11_model_finalny_45k.md` | Model finalny | B2+C+D+A1, final nie najlepszy |
| `notes/12_eksperyment_C_mixed_training.md` | Eksperyment C | real_frac, C 25%, najlepszy wynik |

## Czego nie robić

1. Nie twierdzić, że final 45k jest najlepszym modelem.
2. Nie pisać, że blur pomaga.
3. Nie pisać, że większa rozdzielczość pomaga.
4. Nie pisać, że `real25pct` to 25% finalnego zbioru mixed.
5. Nie traktować mixed-val jako głównej metryki.
6. Nie mieszać `mAP50` z walidacji YOLO z `AP@.5` na realnym holdoucie.
7. Nie pomijać single seed i różnic sprzętowych.
8. Nie redystrybuować danych RarePlanes.
9. Nie pisać, że większy transformer jest gorszy wyłącznie przez pojemność.
10. Nie kończyć raportu samym rankingiem; trzeba wyjaśnić mechanizmy.
11. Nie przedstawiać `results/tabela_zbiorcza.md` jako kompletnej tabeli projektu,
    bo nie zawiera modelu finalnego.
12. Nie przepisywać nieaktualnej stopki o RT-DETR z `tabela_zbiorcza.md`; użyć
    interpretacji z `notes/07`.
13. Nie traktować punktu 512 w eksperymencie D jako osobnego runu `expD_512`,
    bo taki plik nie istnieje w `results/per_size`.
14. Nie porównywać bez komentarza wariantów 10k, 45k, mixed i final jako w pełni
    kontrolowanych eksperymentów jeden-do-jednego.
15. Nie używać metryk z etykiet Grad-CAM jako źródła liczbowego, jeśli JSON-y
    podają inną wartość.
16. Nie pisać, że C 25% oznacza 25% obrazów realnych w finalnym mixed datasecie;
    to 25% dostępnego real train/val.
17. Nie pomijać `n_detections`, gdy model zwraca 300 detekcji/obraz albo gdy
    realne próbki wyraźnie redukują liczbę kandydatów.

## Checklista gotowego raportu

- [ ] Jest abstrakt.
- [ ] Jest pytanie badawcze.
- [ ] Są hipotezy A/B/C/D.
- [ ] Jest opis RarePlanes i licencji.
- [ ] Jest analiza adnotacji.
- [ ] Są histogramy i FFT.
- [ ] Są baseline’y real, synthetic 6460 i synthetic 45k.
- [ ] Są eksperymenty A/B/C/D.
- [ ] Eksperyment C ma opis `real_frac` i rzeczywisty udział realnych danych.
- [ ] Jest porównanie architektur.
- [ ] Jest Grad-CAM/EigenCAM.
- [ ] Jest model finalny 45k.
- [ ] Jest ranking metod.
- [ ] Są ograniczenia.
- [ ] Jest reprodukcja.
- [ ] Jest atrybucja RarePlanes.
- [ ] Główne porównania używają realnego holdoutu COCO.
- [ ] Tabele liczbowe zostały sprawdzone z `results/per_size/*.json`.
- [ ] Model finalny jest dopisany mimo że nie ma go w `results/tabela_zbiorcza.md`.
- [ ] Warianty 10k/45k/final mają jawnie opisany dataset i protokół.
- [ ] `real_frac` jest przetłumaczone na rzeczywisty udział realnych obrazów.
- [ ] D 512 jest opisane jako referencja, nie osobny run.
- [ ] Sekcja Grad-CAM jasno mówi, że to EigenCAM i analiza jakościowa.
- [ ] W raporcie jest wyjaśnienie, dlaczego wprowadzono powyższe korekty względem
      szkicu `docs/RAPORT.md`.
