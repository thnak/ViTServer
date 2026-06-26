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
| MFE | Multi-Scale Feature Embedding: P3/P4/P5 projected to `embed_dim` tokens with sine PE + level embeddings |
| Encoder | Configurable — `none` (default), `window` (per-scale windowed attention), or `full` (legacy O(N²)) |
| Transformer Decoder | 6 layers, cross-attention with learnable object queries over all P3+P4+P5 tokens |
| Output | `pred_boxes [B, Q, 4]` + `pred_scores [B, Q, C]` — **no NMS** |

#### Encoder modes

| `encoder_type` | Cost at 640 px | Scales seen | Notes |
|---|---|---|---|
| `none` (default) | 0 | P3 + P4 + P5 | Decoder cross-attention routes to all scales directly. Fastest; matches encoder performance with ≥ 4 decoder layers. |
| `window` | O(N × ws²) ≈ 0.7 M ops | P3 + P4 + P5 | ws=8 windows per scale; 104× cheaper than full. Adds local intra-scale context before the decoder. |
| `full` | O(N²) ≈ 70 M ops @ 640 px | P3 + P4 + P5 | Original design. Practical only for nano/small at 640 px. |

### Model Variants

All variants target COCO 2017 (80 classes). GFLOPs at batch size 1, split into CNN backbone and transformer attention costs (decoder only — `encoder_type: none` by default so encoder cost is 0).

> **Note on GFLOPs:** with `encoder_type: none` the encoder contributes 0 GFLOPs — all attention cost is from the 6-layer decoder cross-attention + self-attention over the object queries. `thop` has no hook for `nn.MultiheadAttention` so backbone-only GFLOPs are what profilers typically report; decoder attention was computed analytically. If `encoder_type: window` is used, add ≈ 0.7 G per encoder layer at 640 px (≈ 11 G per layer at 1280 px).

| Variant | `base_ch` | `embed_dim` | Params | CNN GFLOPs | Attn GFLOPs | **Total GFLOPs** | Input |
|---|---|---|---|---|---|---|---|
| **smoke** *(dev only)* | 8 | 32 | 0.2 M | 0.003 | 0.003 | **0.006** | 64×64 |
| **nano** | 16 | 64 | 0.8 M | 0.70 | 19.0 | **19.7** | 640×640 |
| **small** | 32 | 128 | 3.6 M | 2.80 | 41.0 | **43.8** | 640×640 |
| **medium** | 48 | 256 | 12.9 M | 9.01 | 94.6 | **103.6** | 640×640 |
| **large** | 64 | 256 | 16.5 M | 12.17 | 94.6 | **106.8** | 640×640 |
| **xlarge** | 96 | 512 | 51.1 M | 35.4 | 201.7 | **237** | 640×640 |
| **xlarge** | 96 | 512 | 51.1 M | 112.6 | 2 515 | **2 628** | 1280×1280 |

### Deployment Planning

Recommended serving hardware per variant. FPS estimates assume TensorRT FP16, single stream, sustained throughput (not peak burst).

| Variant | Total GFLOPs | Target Platform | Runtime | Priority | Estimated FPS |
|---|---|---|---|---|---|
| **xlarge @1280** | 2 628 G | ☁️ Cloud — A100 / H100 80 GB | TRT FP16, batch ≥ 4 | Accuracy-first, async / offline | ~10–30 |
| **xlarge @640** | 237 G | ☁️ Cloud — A100 / RTX 4090 | TRT FP16 | High accuracy, low latency | ~60–120 |
| **large @640** | 107 G | 🖥️ Server — RTX 3080 / 4080 | TRT FP16 | Balanced | ~60–100 |
| **medium @640** | 104 G | 🖥️ Server — RTX 3060 / 4070 | TRT FP16 | Production sweet spot | ~50–80 |
| **small @640** | 43.8 G | 📦 Edge server — Jetson AGX Orin | TRT INT8 | Edge, near-real-time | ~20–40 |
| **nano @640** | 19.7 G | 📦 Edge — Jetson Orin NX / Xavier | TRT INT8 | Ultra-edge, power-constrained | ~15–30 |
| **medium @640** | 104 G | ☁️ Google Colab — TPU v5e-1 | torch_xla bfloat16 | Cloud training (not serving) | ~30–60 |
| **any** | — | 💻 CPU only (dev / test) | ORT FP32 | Development and CI only | <5 |

> FPS figures are directional estimates; actual throughput depends on batch size, memory bandwidth, driver version, and encoder architecture (see note above). Benchmark with `trtexec --iterations=100` for your specific hardware.

### Loss Functions

| Loss | Purpose |
|---|---|
| CIoU + L1 | Box regression |
| Focal Loss | Classification |
| Hungarian Matching | 1-to-1 assignment — eliminates duplicate predictions |

### Quick Start

```bash
cd training

# Install dependencies
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU dev
# or for CUDA 12.1:
# pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 1. Download COCO 2017 (~19 GB total)
python scripts/download_coco.py               # → data/coco/
python scripts/download_coco.py --dest /data/coco  # custom path

# 2. Train
python train.py --config configs/custom_model.yaml --device cuda
python train.py --no-val        # skip validation for faster iteration

# 3. Export to ONNX
python export.py \
    --checkpoint runs/<name>/best.pt \
    --config configs/custom_model.yaml \
    --output runs/<name>/model.onnx
```

### Smoke Test (no GPU, no dataset download)

```bash
cd training

# Generate a tiny synthetic dataset (8 train + 4 val images, no external deps)
python scripts/create_smoke_dataset.py

# Train for 2 epochs with an 8-channel micro-model on 64×64 images
python train.py --config configs/smoke_test.yaml --device cpu

# Export to ONNX
python export.py \
    --checkpoint runs/smoke_test/epoch_1.pt \
    --config configs/smoke_test.yaml \
    --output runs/smoke_test/smoke.onnx
```

### Google Colab — TPU v5e

Select **TPU v5e** runtime: Runtime → Change runtime type → TPU v5e.

```python
# Cell 1 — install dependencies
!pip install torch_xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html
!git clone https://github.com/thnak/ViTServer.git
%cd ViTServer/training
!pip install -r requirements.txt

# Cell 2 — download COCO (or mount Drive with a pre-downloaded copy)
!python scripts/download_coco.py --dest data/coco

# Cell 3 — train on TPU v5e-1
!python train.py \
    --config configs/custom_model.yaml \
    --device tpu \
    --data_path data/coco
```

`--device tpu` automatically:
- Loads `torch_xla` and acquires the XLA device (`xla:0` on a v5e-1 chip)
- Disables AMP and GradScaler (not supported on XLA; TPU uses bfloat16 natively)
- Calls `xm.mark_step()` after each optimizer step to flush the lazy evaluation graph

**Tips for Colab TPU:**
- Use `batch_size: 8` or larger — TPU v5e-1 has 16 GB HBM and favors large batches
- Set `num_workers: 0` in the data config (Colab TPU workers don't support `fork`)
- Export is still done on CPU/GPU after training (`python export.py --device cpu ...`)

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
