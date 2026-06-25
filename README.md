# ViTServer — End-to-End NMS-Free Object Detection Platform

Real-time object detection system optimised end-to-end: from deep-learning training to hardware-compiled edge inference. 100 % proprietary IP — built on open-architecture components to avoid AGPL licence constraints.

## Architecture Overview

```
ViTServer/
├── training/          # Python / PyTorch — CNN-Transformer hybrid, Hungarian-Matching NMS-Free
└── inference-server/  # C++ / TensorRT — RTSP + HTTP/WebSocket, CUDA Graphs, Zero-Copy
```

---

## Part 1 — Training (PyTorch)

### Network Design

| Component | Detail |
|---|---|
| Input | 1280 × 1280 × 3, NHWC (Channels-Last) |
| CNN Embedding | Conv2d (kernel=2, stride=2) spatial downsampling — no Slice ops |
| MFE | P3 → 6 400 tokens · P4 → 1 600 tokens · P5 → 400 tokens = **8 400 total** |
| Decoder | Object Queries + Intra-Scale + Cross-Scale Attention |
| Output | Bounding-box coordinates + class logits, **no NMS** |

### Loss Functions

| Loss | Purpose |
|---|---|
| CIoU + L1 | Box regression |
| Focal Loss | Classification over 8 400 background tokens |
| Hungarian Matching | 1-to-1 assignment — eliminates duplicate predictions |

### Quick Start

```bash
cd training
pip install -r requirements.txt

# Train
python train.py --config configs/custom_model.yaml \
                --data_path /path/to/dataset \
                --img_size 1280

# Export ONNX (dynamic shapes + NHWC)
python export.py --weights best.pt --format onnx --dynamic
```

---

## Part 2 — Inference Server (C++ / TensorRT)

### Core Optimisations

| Feature | Detail |
|---|---|
| TensorRT + CUDA Streams | Async GPU load, `enqueueV3` dynamic batching |
| Zero-Copy Processing | Resize / Normalize / RGB → CUDA Kernel, never touches CPU RAM |
| CUDA Graphs | Full compute graph captured → zero kernel-launch overhead → stable 60 FPS |
| NMS-Free Post-processing | Single O(N) confidence-threshold scan |

### API & Binary Protocol

Server accepts:
- **RTSP streams** — async frame capture, drop-frame on full queue
- **Single-image HTTP** — REST endpoint, synchronous response

Binary WebSocket payload:

| Offset | Type | Field |
|---|---|---|
| 0–7 | `uint64_t` | Timestamp (Unix ms) |
| 8–9 | `uint16_t` | Detection count N |
| 10 + 10×i | `Box` | 10-byte box × N |

```cpp
#pragma pack(push, 1)
struct Box {
    uint16_t x1, y1, x2, y2; // normalised coords × 65535
    uint8_t  class_id;
    uint8_t  score;           // 0–100
};
#pragma pack(pop)
```

### Requirements

- NVIDIA GPU — Ampere architecture or newer
- CUDA Toolkit ≥ 12.0
- TensorRT ≥ 10.0
- OpenCV ≥ 4.5 (CUDA build)
- CMake ≥ 3.18
- Boost ≥ 1.82 (Asio + Beast)

### Build & Run

```bash
cd inference-server
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)

# Convert ONNX → TensorRT engine
/usr/src/tensorrt/bin/trtexec \
    --onnx=model.onnx \
    --saveEngine=model.trt \
    --fp16 \
    --optShapes=images:1x3x1280x1280

# Run server
./InferenceServer --engine=model.trt --port=8080 --config=../config.json
```

---

## Licence

MIT / Apache 2.0 — free for commercial closed-source use, no copyleft obligations.
