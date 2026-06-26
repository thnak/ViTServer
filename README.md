# ViTServer — End-to-End NMS-Free Object Detection Platform

Real-time object detection built end-to-end without non-maximum suppression. Trained with Hungarian matching for 1-to-1 assignment, then served through a dual-backend C++ inference server that supports both CPU (ONNX Runtime) and GPU (TensorRT) deployment. 100% MIT/Apache-2.0 — no AGPL obligations.

## Architecture Overview

```
ViTServer/
├── training/           # Python/PyTorch — CNN-Transformer hybrid, NMS-Free
│   ├── models/         # backbone.py, mfe.py, transformer.py, detector.py
│   ├── losses/         # hungarian.py, bbox_loss.py, focal_loss.py
│   ├── datasets/       # coco_dataset.py, transforms.py (albumentations)
│   ├── configs/        # YAML training configs
│   ├── scripts/        # download_coco.py, create_smoke_dataset.py
│   ├── train.py
│   └── export.py
└── inference-server/   # C++17 — Boost.Beast HTTP/WebSocket, ORT + TRT backends
    ├── include/        # engine.hpp (IEngine), engine_ort.hpp, engine_trt.hpp
    ├── src/            # engine_ort.cpp, engine_trt.cpp, engine_factory.cpp, main.cpp
    ├── scripts/        # test_client.py
    └── CMakeLists.txt
```

---

## Part 1 — Training (PyTorch)

### Network Design

| Component | Detail |
|---|---|
| Input | Configurable (default 1280×1280×3) |
| CNN Backbone | C2f cross-stage-partial blocks, stride-2 Conv downsampling |
| MFE | Multi-Scale Feature Embedding: P3/P4/P5 projected to `embed_dim` tokens |
| Transformer Decoder | 6 layers, cross-attention with learnable object queries |
| Output | `pred_boxes [B, Q, 4]` + `pred_scores [B, Q, C]` — **no NMS** |

### Model Variants

All variants trained on COCO 2017 (80 classes) unless noted. GFLOPs measured at the listed input resolution with a batch size of 1.

| Variant | `base_ch` | `embed_dim` | Params | GFLOPs | Input |
|---|---|---|---|---|---|
| **smoke** *(dev only)* | 8 | 32 | 0.2 M | 0.003 | 64×64 |
| **nano** | 16 | 64 | 0.8 M | 0.70 | 640×640 |
| **small** | 32 | 128 | 3.6 M | 2.80 | 640×640 |
| **medium** | 48 | 256 | 12.9 M | 9.01 | 640×640 |
| **large** | 64 | 256 | 16.5 M | 12.17 | 640×640 |
| **xlarge** | 96 | 512 | 51.1 M | 35.37 | 640×640 |
| **xlarge** | 96 | 512 | 51.1 M | 112.63 | 1280×1280 |

> GFLOPs = 2 × MACs, counted with [thop](https://github.com/Lyken17/pytorch-OpCounter).  
> thop undercounts parameters (~25%) for custom attention layers; the Params column uses `sum(p.numel())`.

### Loss Functions

| Loss | Purpose |
|---|---|
| CIoU + L1 | Box regression |
| Focal Loss | Classification |
| Hungarian Matching | 1-to-1 assignment — eliminates duplicate predictions |

### Quick Start

```bash
cd training

# Set up Python environment (requires uv)
uv venv && uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
uv pip install albumentations pycocotools onnx onnxscript

# 1. Download COCO 2017 (~19 GB total)
uv run python scripts/download_coco.py            # → data/coco/
uv run python scripts/download_coco.py --dest /data/coco  # custom path

# 2. Train (--data_path defaults to data/coco)
uv run python train.py --config configs/custom_model.yaml
uv run python train.py --no-val        # skip validation for faster iteration

# 3. Export to ONNX
uv run python export.py \
    --checkpoint runs/<name>/best.pt \
    --config configs/custom_model.yaml \
    --output runs/<name>/model.onnx
```

### Smoke Test (no GPU, no dataset download)

```bash
cd training

# Generate a tiny synthetic dataset (8 train + 4 val images, no external deps)
uv run python scripts/create_smoke_dataset.py

# Train for 2 epochs with a 8-channel micro-model on 64×64 images
uv run python train.py --config configs/smoke_test.yaml --device cpu

# Export to ONNX
uv run python export.py \
    --checkpoint runs/smoke_test/best.pt \
    --config configs/smoke_test.yaml \
    --output runs/smoke_test/smoke.onnx
```

---

## Part 2 — Inference Server (C++)

### Backend Selection

The server supports two interchangeable inference backends selected at **compile time** via CMake options and at **runtime** by model file extension:

| Backend | Flag | Extension | Hardware |
|---|---|---|---|
| ONNX Runtime | `-DVIT_USE_ORT=ON` (default) | `.onnx` | CPU (any) |
| TensorRT | `-DVIT_USE_TRT=ON` | `.trt` / `.engine` | NVIDIA GPU |

Both backends implement the same `IEngine` interface — no changes to `main.cpp` needed when switching.

### Build

```bash
cd inference-server

# CPU build (ONNX Runtime, no GPU required)
cmake -S . -B build -DVIT_USE_ORT=ON -DVIT_USE_TRT=OFF
cmake --build build --parallel $(nproc)

# GPU build (TensorRT — requires CUDA ≥ 12.0 and TensorRT ≥ 10.0)
cmake -S . -B build -DVIT_USE_ORT=OFF -DVIT_USE_TRT=ON \
      -DTENSORRT_ROOT=/usr/local/tensorrt
cmake --build build --parallel $(nproc)

# Both backends in one binary (selects at runtime from file extension)
cmake -S . -B build -DVIT_USE_ORT=ON -DVIT_USE_TRT=ON
cmake --build build --parallel $(nproc)

# Enable RTSP streaming support (requires OpenCV videoio + GStreamer/FFmpeg)
cmake -S . -B build -DVIT_USE_ORT=ON -DVIT_USE_RTSP=ON
```

**OpenCV** is built from source automatically (core, imgproc, imgcodecs only) — no system-wide install required.  
**ONNX Runtime** 1.27.0 pre-built binary must be present at `third_party/onnxruntime/`:

```bash
ORT_VER=1.27.0
mkdir -p inference-server/third_party/onnxruntime
curl -fL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VER}/onnxruntime-linux-x64-${ORT_VER}.tgz" \
  | tar -xz -C inference-server/third_party/onnxruntime --strip-components=1
```

### Run

```bash
# ORT backend — pass any .onnx file
./build/bin/InferenceServer --engine runs/smoke_test/smoke.onnx --port 8080

# TRT backend — convert first, then run
trtexec --onnx=model.onnx --saveEngine=model.trt --fp16 \
        --optShapes=images:1x3x1280x1280
./build/bin/InferenceServer --engine model.trt --port 8080

# With config file for score threshold and RTSP URLs
./build/bin/InferenceServer --engine model.onnx --config config.json --port 8080
```

The server auto-detects the model's input dimensions from the ONNX/TRT metadata.

### API

#### REST

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `POST` | `/infer` | Single-image inference (raw JPEG/PNG bytes in body) |

```bash
# Health check
curl http://localhost:8080/health
# → {"status":"ok"}

# Infer from an image file
curl -X POST http://localhost:8080/infer \
     --data-binary @image.jpg \
     -H "Content-Type: application/octet-stream"
# → {"count":3,"timestamp_ms":1700000000000,"boxes":[...]}
```

#### WebSocket (Binary, frame-based)

Connect to `ws://host:port/` — upgrade from HTTP.

**Client → Server frame** (raw BGR image):

| Offset | Type | Field |
|---|---|---|
| 0–3 | `uint32_t` | Frame width (px) |
| 4–7 | `uint32_t` | Frame height (px) |
| 8 + | `uint8_t[]` | BGR pixel data (width × height × 3) |

**Server → Client frame** (detections):

| Offset | Type | Field |
|---|---|---|
| 0–7 | `uint64_t` | Timestamp (Unix ms) |
| 8–9 | `uint16_t` | Detection count N |
| 10 + 10×i | `Box` | 10-byte struct × N |

```cpp
#pragma pack(push, 1)
struct Box {
    uint16_t x1, y1, x2, y2; // normalised coords × 65535
    uint8_t  class_id;
    uint8_t  score;           // 0–100
};
#pragma pack(pop)
```

#### Python Test Client

```bash
cd inference-server
python scripts/test_client.py --port 8080 --image /path/to/image.jpg
```

### Config File (`config.json`)

```json
{
  "score_thresh": 0.3,
  "img_size": 1280,
  "rtsp_urls": [
    "rtsp://camera1/stream",
    "rtsp://camera2/stream"
  ]
}
```

### GPU Performance (TensorRT, Ampere+)

| Feature | Detail |
|---|---|
| CUDA Graphs | Full compute graph captured once → zero kernel-launch overhead |
| Pinned memory | Host↔GPU transfers via `cudaMallocHost` for maximum throughput |
| `enqueueV3` | TensorRT dynamic-batch async inference |
| FP16 | Enabled with `--fp16` during `trtexec` conversion |

---

## Licence

MIT / Apache 2.0 — free for commercial closed-source use, no copyleft obligations.
