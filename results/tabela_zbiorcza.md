| Wariant | mAP@50 | mAP@50:95 | AP_S | AP_M | AP_L | AR@100 |
|---|---|---|---|---|---|---|
| Baseline A: real->real (gorny ref) | 0.974 | 0.807 | 0.759 | 0.782 | 0.899 | 0.847 |
| Baseline B: synth 6460 -> real | 0.410 | 0.239 | 0.262 | 0.328 | 0.067 | 0.369 |
| Baseline B: synth 45k -> real | 0.452 | 0.268 | 0.286 | 0.384 | 0.201 | 0.546 |
| Eksp A: slaby HSV (10k) | 0.459 | 0.264 | 0.306 | 0.357 | 0.091 | 0.397 |
| Eksp A: sredni HSV (10k) | 0.431 | 0.247 | 0.265 | 0.345 | 0.099 | 0.393 |
| Eksp A: mocny HSV (10k) | 0.446 | 0.252 | 0.268 | 0.344 | 0.105 | 0.394 |
| Eksp A: slaby HSV (45k, finalny) | 0.455 | 0.268 | 0.337 | 0.355 | 0.090 | 0.404 |
| Arch: YOLO11l CNN duzy (10k) | 0.467 | 0.271 | 0.306 | 0.348 | 0.108 | 0.392 |
| Arch: RT-DETR-l Transf maly (10k) | 0.297 | 0.157 | 0.146 | 0.230 | 0.081 | 0.308 |
| Arch: RT-DETR-x Transf duzy (10k) | 0.380 | 0.205 | 0.206 | 0.285 | 0.094 | 0.373 |
| Eksp D: imgsz 320 (10k) | 0.522 | 0.283 | 0.308 | 0.367 | 0.151 | 0.436 |
| Eksp D: imgsz 768 (10k) | 0.448 | 0.252 | 0.230 | 0.339 | 0.116 | 0.374 |
| Eksp D: imgsz 1024 (10k) | 0.330 | 0.190 | 0.222 | 0.250 | 0.047 | 0.284 |

*RT-DETR z best.pt po 2 epokach (trening rozbiezny, patrz notes/07)
