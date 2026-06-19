# Grad-CAM — interpretowalność modeli (wymóg PDF)

> EigenCAM na warstwie 8 (C2f, ostatni blok backbone YOLOv10n). Porównanie "uwagi"
> 6 modeli na TYCH SAMYCH realnych kaflach z MAŁYMI obiektami (wymóg PDF:
> "wizualizacja aktywacji dla małych obiektów"). Skrypt: `src/gradcam_compare.py`.
> Wynik: `results/gradcam/gradcam_comparison.png`.
>
> Modele ułożone wg rosnącego transferu: synthetic 0.452 → A 0.459 → B2 0.490 →
> D(imgsz320) 0.515 → C(mixed 25%) 0.947 → real-baseline 0.974. Modele C i D
> dotrenowane lokalnie (D na Windows GPU, C od Miłosza z klastra, w `models/`).

## Metoda
EigenCAM (pierwsza składowa główna aktywacji) zamiast klasycznego Grad-CAM, bo
detektory YOLOv10 są end2end i nie mają pojedynczego skalara klasy do liczenia
gradientu. Warstwa 8 backbone (C2f, 256 kanałów, 16×16) — bogata semantycznie,
zachowuje rozdzielczość przestrzenną. Backbone owinięty w `_BackboneWrap`
(forward do warstwy 8), bo pełna głowica YOLOv10 zwraca tuple nieobsługiwany
przez grad-cam.

Modele porównane (6, wg rosnącego transferu): synthetic 45k (0.452), A słaby HSV (0.459),
B2 szum (0.490), D imgsz320 (0.515), C mixed 25% (0.947), real-baseline (0.974).
Kafle wybrane jako te z największą liczbą małych obiektów (area <32²).

## Obserwacje jakościowe (do raportu)

1. **Real-baseline ma rozlaną uwagę** — pokrywa duże obszary sceny włącznie z tłem
   (pasy, teren). Model uczony na realnych danych "rozumie" cały kontekst sceny
   satelitarnej, nie tylko obiekty.

2. **Modele syntetyczne skupiają uwagę na strukturach liniowych** — na kaflach
   z rzędami samolotów (typowy układ na płycie postojowej) uwaga modeli synthetic/B2
   podąża za linią obiektów. Model B2 (z szumem) wykazuje najprecyzyjniejszą
   lokalizację na rzędach małych samolotów — spójne z jego najlepszym wynikiem
   wśród wariantów B (0.490) i z hipotezą, że szum poprawia robustność cech.

3. **Małe obiekty są aktywowane** — wbrew obawom, modele NIE ignorują małych
   samolotów; mapy uwagi reagują na ich pozycje. To istotne dla wątku "co sieci
   uczą się z syntetyków": uczą się obiektu, nie tylko tła/artefaktów renderu.

4. **D (imgsz320) ma ostrzejszą, bardziej skupioną uwagę na obiektach** niż
   synthetic baseline — wizualne potwierdzenie mechanizmu skali: niska
   rozdzielczość dopasowuje pozorny rozmiar obiektów ku małym realnym, model
   wyraźniej lokalizuje samoloty zamiast rozlewać uwagę po tle.

5. **C (mixed 25%) — uwaga przesuwa się ku wzorcowi real-baseline** (bardziej
   rozlana, obejmuje kontekst sceny). To wizualny dowód, że domieszka realnych
   danych "przeciąga" reprezentację modelu do realnej domeny — spójne z jego
   najlepszym wynikiem (0.947, prawie poziom real). Progresja uwagi
   synthetic→A→B→D→C→real jest monotoniczna względem mAP@50.

## Statement do raportu
> Grad-CAM (EigenCAM) potwierdza, że modele — także te trenowane czysto na
> syntetykach — lokalizują uwagę na realnych samolotach, w tym małych. Model
> real-baseline ma uwagę bardziej rozlaną (pełny kontekst sceny), modele
> syntetyczne bardziej skupioną na strukturach obiektów. Nie obserwujemy
> patologicznego skupienia na tle/artefaktach — luka domenowa wynika raczej
> z różnicy rozkładu rozmiarów i wyglądu (notes/01-02), nie z "uczenia się złych
> cech".

## Ograniczenia
- EigenCAM pokazuje aktywację warstwy, nie bezpośrednio "co przesądza o detekcji"
  (brak gradientu klasowego w end2end). Interpretacja jakościowa, nie ilościowa.
- Architektury transformerowe (RT-DETR) nie uwzględnione — inna struktura backbone
  (brak warstwy 8 C2f); wymagałyby osobnego doboru warstwy docelowej.
