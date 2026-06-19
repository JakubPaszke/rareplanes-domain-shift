# Final combined model summary

- run: `final_yolov10n_syn1k_noise_real25pct_img320_hsvA1_ml`
- dataset: `data/yolo/final_mixed_syn1k_noise_real25pct`
- recipe: full synthetic + C mixed 25% real + D imgsz=320 + B2 noise_sigma=8.0 + A1 HSV=(0.015, 0.4, 0.3)
- note: real test holdout is used only for final evaluation.

| metric | final full | expC 25% 10k measured | real->real upper |
|---|---:|---:|---:|
| AP@.5 | 0.8426 | 0.9466 | 0.9737 |
| AP@[.5:.95] | 0.5291 | 0.7120 | 0.8073 |
| AP_small | 0.4392 | 0.6365 | 0.7586 |
| AP_medium | 0.5286 | 0.6974 | 0.7825 |
| AP_large | 0.6622 | 0.8166 | 0.8991 |
| AR@100 | 0.7236 | 0.7822 | 0.8465 |
