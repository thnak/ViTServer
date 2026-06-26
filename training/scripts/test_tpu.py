#!/usr/bin/env python3
"""TPU smoke-test — run this before full training to verify XLA is healthy.

Usage (Colab TPU runtime):
    python scripts/test_tpu.py
    python scripts/test_tpu.py --model   # also tests a tiny NMSFreeDetector forward pass
"""

import argparse
import sys
import time

import torch


# ── helpers ──────────────────────────────────────────────────────────────────

def _sync(xm):
    """Block until all queued XLA ops finish."""
    xm.mark_step()
    xm.wait_device_ops()


def _timed(label: str, fn, xm, warmup: int = 1, runs: int = 5):
    """Run fn() 'warmup' times (discarded), then 'runs' times and report avg ms."""
    for _ in range(warmup):
        fn()
        _sync(xm)

    t0 = time.perf_counter()
    for _ in range(runs):
        fn()
        _sync(xm)
    elapsed = (time.perf_counter() - t0) / runs * 1000
    print(f"  {label}: {elapsed:.1f} ms / step  (avg over {runs} runs)")
    return elapsed


# ── stages ───────────────────────────────────────────────────────────────────

def stage_device(xm):
    print("\n[1] Device")
    dev = torch.device(xm.xla_device() if hasattr(xm, "xla_device") else "xla:0")
    # newer API
    try:
        import torch_xla
        dev = torch_xla.device()
    except Exception:
        pass
    print(f"    device  : {dev}")
    print(f"    dtype   : bfloat16 (TPU native)")
    return dev


def stage_tensor_ops(dev, xm):
    print("\n[2] Tensor ops")

    # small matmul
    a = torch.randn(512, 512, device=dev, dtype=torch.bfloat16)
    b = torch.randn(512, 512, device=dev, dtype=torch.bfloat16)
    _timed("512×512 matmul", lambda: torch.mm(a, b), xm)

    # larger matmul (closer to real decoder attention)
    a2 = torch.randn(1, 300, 256, device=dev, dtype=torch.bfloat16)
    b2 = torch.randn(1, 256, 8400, device=dev, dtype=torch.bfloat16)
    _timed("bmm 300×256 × 256×8400", lambda: torch.bmm(a2, b2), xm)

    mem = xm.get_memory_info(dev)
    used_mb = mem.get("bytes_used", 0) / 1024 ** 2
    total_mb = mem.get("bytes_limit", 0) / 1024 ** 2
    print(f"  memory  : {used_mb:.0f} MiB / {total_mb:.0f} MiB")


def stage_model(dev, xm):
    print("\n[3] NMSFreeDetector forward pass (tiny config)")
    sys.path.insert(0, ".")
    from models import NMSFreeDetector

    model = NMSFreeDetector(
        num_classes=80,
        base_channels=16,
        embed_dim=64,
        num_heads=2,
        num_encoder_layers=0,
        num_decoder_layers=2,
        num_queries=100,
        encoder_type="none",
    ).to(dev).to(torch.bfloat16)
    model.eval()

    dummy = torch.randn(1, 3, 320, 320, device=dev, dtype=torch.bfloat16)

    print("  compiling graph (first call — may take 1-3 min) …", flush=True)
    t_compile = time.perf_counter()
    with torch.no_grad():
        model(dummy)
    _sync(xm)
    print(f"  compilation done in {time.perf_counter() - t_compile:.1f}s")

    with torch.no_grad():
        _timed("forward 320px batch=1", lambda: model(dummy), xm)

    # check outputs are finite
    with torch.no_grad():
        out = model(dummy)
        _sync(xm)
    boxes  = out["pred_boxes"].float().cpu()
    logits = out["pred_logits"].float().cpu()
    ok = boxes.isfinite().all() and logits.isfinite().all()
    print(f"  outputs finite: {ok}")
    if not ok:
        print("  WARNING: NaN/Inf in model output — check model or dtype handling")


def stage_train_step(dev, xm):
    print("\n[4] Train step (forward + backward + optimizer)")
    sys.path.insert(0, ".")
    from models import NMSFreeDetector
    from losses import HungarianCriterion
    from losses.hungarian import HungarianMatcher

    model = NMSFreeDetector(
        num_classes=3,
        base_channels=8,
        embed_dim=32,
        num_heads=2,
        num_encoder_layers=0,
        num_decoder_layers=2,
        num_queries=20,
        encoder_type="none",
    ).to(dev).to(torch.bfloat16)

    matcher = HungarianMatcher(2.0, 5.0, 2.0)
    criterion = HungarianCriterion(3, matcher, 2.0, 5.0, 2.0, 0.25, 2.0).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    B = 2
    imgs = torch.randn(B, 3, 64, 64, device=dev, dtype=torch.bfloat16)
    targets = [
        {
            "boxes":   torch.tensor([[0.5, 0.5, 0.2, 0.2]], device=dev),
            "labels":  torch.tensor([0], device=dev),
            "image_id": torch.tensor(i),
            "orig_size": torch.tensor([64, 64]),
        }
        for i in range(B)
    ]

    def step():
        opt.zero_grad()
        out = model(imgs)
        loss = criterion(out, targets)["total"]
        loss.backward()
        opt.step()
        xm.mark_step()

    print("  compiling train graph (first call — may take 2-5 min) …", flush=True)
    t0 = time.perf_counter()
    step()
    xm.wait_device_ops()
    print(f"  compilation done in {time.perf_counter() - t0:.1f}s")

    _timed("train step 64px batch=2", step, xm)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", action="store_true", help="Also run model forward + train-step tests")
    args = p.parse_args()

    try:
        import torch_xla.core.xla_model as xm
    except ImportError:
        sys.exit(
            "torch_xla not found.\n"
            "  pip install torch_xla[tpu] "
            "-f https://storage.googleapis.com/libtpu-releases/index.html"
        )

    print("=" * 55)
    print("  ViTServer TPU diagnostic")
    print("=" * 55)

    dev = stage_device(xm)
    stage_tensor_ops(dev, xm)

    if args.model:
        stage_model(dev, xm)
        stage_train_step(dev, xm)

    print("\n[OK] TPU is healthy\n")


if __name__ == "__main__":
    main()
