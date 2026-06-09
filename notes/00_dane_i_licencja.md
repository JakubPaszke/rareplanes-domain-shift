# Dane RarePlanes — źródło, licencja, warianty

## Źródło
- Bucket S3 publiczny: `s3://rareplanes-public` (dostęp `--no-sign-request`, bez logowania).
- Struktura: `real/`, `synthetic/`, `weights/` (gotowe Faster R-CNN / Mask R-CNN z oryginalnego baseline'u), `LICENSE.txt`.

## Licencja — CC BY-SA 4.0 (Attribution-ShareAlike)
Zbiór na licencji **Creative Commons Attribution-ShareAlike 4.0 International**.
Wymagana atrybucja:

> J. Shermeyer, T. Hossler, A. Van Etten, D. Hogan, R. Lewis, and D. Kim.
> In-Q-Tel - CosmiQ Works and AI.Reverie. RarePlanes Dataset, June 2020.

```bibtex
@misc{RarePlanes_Dataset,
    title={RarePlanes Dataset},
    author={Shermeyer, Jacob and Hossler, Thomas and Van Etten, Adam and Hogan, Daniel and Lewis, Ryan and Kim, Daeil},
    organization={In-Q-Tel - CosmiQ Works and AI.Reverie},
    month={June}, year={2020}
}
```

**Konsekwencja ShareAlike:** jeśli redystrybuujemy przetworzone dane/adnotacje, muszą być na tej samej licencji. W repo: NIE wrzucamy obrazów do gita (tylko skrypty pobierające), w README sekcja o licencji + atrybucja. Kopia licencji: `data/LICENSE_rareplanes.txt`.

## Warianty obrazów (real) — co i dlaczego pobieramy
RarePlanes daje ten sam teren w wielu produktach Maxar:
- **PS-RGB** — pansharpened RGB (3 kanały). Jedyny porównywalny z syntetykami (też RGB) → color histogram / FFT mają sens tylko tu.
- **MS** (~8 pasm multispektralnych), **PAN** (1 kanał) — inne pasma, nieużyteczne dla detektora RGB i do porównania z synthetic. Pomijamy.
- **`tiled` vs surowe sceny** — `tiled` = gotowe kafelki 512×512 dopasowane do COCO `tiled`; surowe = wielkie sceny do samodzielnego cięcia.

**Wybór (potwierdzony):** `PS-RGB_tiled` dla real (train 1.9 GB + test 889 MB). Surowe PS-RGB tylko gdyby robić własny eksperyment z rozmiarem kafla.

## Rozmiary (do planowania pobrań; łącze ~2 MB/s)
- real PS-RGB_tiled: train 1.9 GB, test 889 MB
- real pozostałe warianty: ~105 GB (pomijamy)
- synthetic train: 45 000 obrazów / 130 GB; test: 5 000 / 14.5 GB
- adnotacje: real ~75 MB, synthetic ~330 MB (pobrane)
