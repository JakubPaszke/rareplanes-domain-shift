| Wariant | mAP@50 | mAP@50:95 | AP_S | AP_M | AP_L | AR@100 |
|---|---|---|---|---|---|---|
| Baseline A: real->real (gorny ref) | 0.974 | 0.807 | 0.759 | 0.782 | 0.899 | 0.847 |
| Baseline B: synth 6460 -> real | 0.410 | 0.239 | 0.262 | 0.328 | 0.067 | 0.369 |
| Baseline B: synth 45k -> real | 0.452 | 0.268 | 0.286 | 0.384 | 0.201 | 0.546 |
| Eksp A: slaby HSV (10k) | 0.459 | 0.264 | 0.306 | 0.357 | 0.091 | 0.397 |
| Eksp A: sredni HSV (10k) | 0.431 | 0.247 | 0.265 | 0.345 | 0.099 | 0.393 |
| Eksp A: mocny HSV (10k) | 0.446 | 0.252 | 0.268 | 0.344 | 0.105 | 0.394 |
| Eksp A: slaby HSV (45k, finalny) | 0.455 | 0.268 | 0.337 | 0.355 | 0.090 | 0.404 |
| Arch: RT-DETR-l (10k, 60ep stabilny) | 0.297 | 0.157 | 0.146 | 0.230 | 0.081 | 0.308 |

RT-DETR-l: stabilny trening 60 epok (lr=1e-4+cosine), mAP_syn 0.973 ale transfer na real slaby (overfit do synthetic) — patrz notes/07
