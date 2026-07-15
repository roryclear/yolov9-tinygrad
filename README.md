tinygrad implementation of: https://github.com/WongKinYiu/yolov9

## Setup:
```
pip install -r requirements.txt
```

## Inference on single image:
```
python yolov9.py {link to an image} {model variant}
```

## Live WebGPU inference
```
python compile_to_webgpu.py
python -m http.server 8080
```
open localhost:8080

## Testing performance
```
PYTHONPATH=. python test/test_jit.py
```
### for faster inference use tinygrad's BEAM search:
```
PYTHONPATH=. BEAM=2 python test/test_jit.py
```
this will result in a longer initial run time as the searches are performed and cached. For visibility on the process use:
```
PYTHONPATH=. BEAM=2 DEBUG=2 python test/test_jit.py
```

# Speed
## with BEAM=2:
| Model | Resolution | FPS (M3 Macbook Air) | FPS (RX7600) |
|-------|------------|----------------------|--------------|
| t | 320 | 198.56 | 278.13 |
| t | 640 | 78.08 | 180.91 |
| t | 960 | 39.41 | 97.21 |
| t | 1280 | 25.27 | 55.46 |
| t | 1536 | 16.48 | 39.69 |
| s | 320 | 97.11 | 159.05 |
| s | 640 | 33.31 | 70.24 |
| s | 960 | 17.85 | 35.44 |
| s | 1280 | 12.24 | 22.35 |
| s | 1536 | 7.94 | 17.99 |
| m | 320 | 46.09 | 75.92 |
| m | 640 | 15.81 | 29.93 |
| m | 960 | 7.73 | 7.71 |
| m | 1280 | 5.01 | 10.14 |
| m | 1536 | 3.37 | 1.41 |
| c | 320 | 35.72 | 68.32 |
| c | 640 | 13.55 | 22.21 |
| c | 960 | 5.82 | 9.25 |
| c | 1280 | 4.22 | 7.31 |
| c | 1536 | 2.58 | 4.12 |
| e | 320 | 20.36 | 36.28 |
| e | 640 | 7.49 | 10.20 |
| e | 960 | 3.25 | 5.63 |
| e | 1280 | 2.25 | 4.15 |
| e | 1536 | 1.43 | 2.37 |

## without BEAM=2:
| Model | Resolution | FPS (M3 Macbook Air) | FPS (RX7600) |
|-------|------------|----------------------|--------------|
| t | 320 | 170.45 | 212.63 |
| t | 640 | 76.62 | 124.35 |
| t | 960 | 23.75 | 50.55 |
| t | 1280 | 17.39 | 34.88 |
| t | 1536 | 10.21 | 26.23 |
| s | 320 | 64.32 | 111.30 |
| s | 640 | 27.36 | 51.14 |
| s | 960 | 10.06 | 22.67 |
| s | 1280 | 6.01 | 11.92 |
| s | 1536 | 3.50 | 10.14 |
| m | 320 | 27.94 | 51.23 |
| m | 640 | 9.46 | 14.66 |
| m | 960 | 3.58 | 8.24 |
| m | 1280 | 1.51 | 3.31 |
| m | 1536 | 0.89 | 2.81 |
| c | 320 | 18.23 | 24.90 |
| c | 640 | 6.70 | 12.57 |
| c | 960 | 2.74 | 5.84 |
| c | 1280 | 1.24 | 2.95 |
| c | 1536 | 0.75 | 2.30 |
| e | 320 | 8.73 | 9.03 |
| e | 640 | 3.56 | 6.08 |
| e | 960 | 1.49 | 2.94 |
| e | 1280 | 0.75 | 1.77 |
| e | 1536 | 0.44 | 1.28 |
