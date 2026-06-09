# Analiza wyglądu domeny: real vs synthetic — notatki pod raport

> Wygenerowane przez `src/analyze_appearance.py --n 300`.
> Wykresy: `results/appearance/{color_histograms,fft_spectra,radial_power}.png`.
> Statystyki: `results/appearance_stats.json`. Próbka: 300 kafli 512×512 na domenę.
> Realizuje wymóg z `dodatkowe_wymagadnia.txt` (color histogram, transformacja Fouriera).

## 1. Statystyki kolorów / jasności

| metryka | real (PS-RGB tiled) | synthetic | obserwacja |
|---|---|---|---|
| jasność średnia | **71.7** | **133.0** | synthetic ~1.85× jaśniejszy |
| nasycenie średnie | **0.303** | **0.122** | real ~2.5× bardziej nasycony |
| RGB mean (R/G/B) | 82 / 71 / 62 | 129 / 131 / 139 | real ciepły (R>G>B), synthetic chłodny (B>G>R) |

**Statement do raportu:**
> Domeny różnią się fotometrycznie w sposób systematyczny. Realne kafle satelitarne są ciemne (śr. jasność ~72/255), nasycone i mają ciepłą równowagę bieli (R>G>B). Renderowane syntetyki są wyraźnie jaśniejsze (~133/255), wyblakłe (niskie nasycenie) i chłodne (B>G>R). To globalne przesunięcie kolorystyczne jest dobrym kandydatem do korekty prostą augmentacją (jitter jasności/kontrastu/balansu bieli, dopasowanie histogramu real↔synthetic).

## 2. Histogramy kolorów (`color_histograms.png`)
- **Real:** rozkład skośny ku ciemnym tonom — wyraźne piki w okolicy 25–50, długi ogon w jasnych. Pik na 255 w kanale R (przepalenia / brzegi kafli / artefakty maski nodata).
- **Synthetic:** rozkład zbliżony do symetrycznego, skupiony wokół 100–140, brak ciemnej masy i ogonów skrajnych. "Studyjne" oświetlenie renderu.

Wniosek: kształt histogramów, nie tylko średnie, różni się jakościowo — dopasowanie samej średniej nie wystarczy; rozważyć histogram matching / CDF.

## 3. Widmo Fouriera i profil radialny mocy (`fft_spectra.png`, `radial_power.png`)
- Radialny profil mocy: **real ma konsekwentnie więcej energii w wysokich częstotliwościach** (linia real powyżej synthetic dla f≳60). Synthetic jest **gładszy** — mniej drobnej tekstury/szumu sensora.

**Statement do raportu:**
> W dziedzinie częstotliwości realne obrazy satelitarne zawierają więcej energii wysokoczęstotliwościowej (drobna tekstura, szum sensora, kompresja), podczas gdy rendery syntetyczne są gładsze. Detektor uczony na "czystych" syntetykach może być wrażliwy na realny szum i utratę ostrości. To uzasadnia hipotezę augmentacyjną: dodanie szumu / lekkiego blur / degradacji do syntetyków, by upodobnić ich widmo do realnego.

## 4. Hipotezy augmentacyjne wynikające z analizy wyglądu (do eksperymentów A/B/C)
- **HA (fotometria):** color jitter (jasność↓, nasycenie↑, balans bieli), ewentualnie histogram matching real→synthetic. Cel: zniwelować globalne przesunięcie kolorów.
- **HB (częstotliwości):** Gaussian noise + lekki blur / degradacja JPEG na syntetykach, by dorównać widmu real.
- **HC (łącznie z analizą adnotacji):** te augmentacje + dopasowanie skali obiektów (z `notes/01`) — sprawdzić, czy łączenie fotometrii, częstotliwości i skali daje addytywny zysk transferu.

## 5. Inspekcja wizualna (`sample_tiles_real_vs_syn.png`)
Siatka 6×2 losowych kafli potwierdza jakościowo:
- real: ciemne, kontrastowe, rzadkie (często 0–1 samolot/kafel, dużo tła: pasy, teren);
- synthetic: jaśniejsze, wyblakłe, zatłoczone samolotami w regularnych układach na płytach.
Zgodne z gęstością 12.6 vs 3.2 inst/kafel (zob. `notes/01`).

## Uwaga metodologiczna
Analiza na całych kaflach (tło + obiekty). Tło dominuje powierzchniowo, więc statystyki opisują głównie różnicę teł/oświetlenia. W kolejnym kroku warto policzyć te same metryki tylko w obrębie bboxów (na samych samolotach), by rozdzielić "shift tła" od "shift obiektu".
