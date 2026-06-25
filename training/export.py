#!/usr/bin/env python3
"""Export trained model to ONNX with dynamic shapes and NHWC input for TensorRT."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import onnx
import onnxruntime as ort
import yaml

from models import NMSFreeDetector


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Export NMS-Free Detector to ONNX")
    p.add_argument("--weights", required=True, help="Path to .pt checkpoint")
    p.add_argument("--config", default="configs/custom_model.yaml")
    p.add_argument("--format", default="onnx", choices=["onnx"])
    p.add_argument("--dynamic", action="store_true", help="Dynamic batch + spatial axes")
    p.add_argument("--img_size", type=int, default=1280)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--output", default="", help="Output path (default: weights stem + .onnx)")
    return p.parse_args()


class ExportWrapper(torch.nn.Module):
    """Thin wrapper that disables aux_loss and returns (boxes, scores)."""

    def __init__(self, model: NMSFreeDetector) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        # x: [B, 3, H, W] NCHW
        out = self.model(x)
        boxes = out["pred_boxes"]               # [B, Q, 4]
        scores = out["pred_logits"].sigmoid()   # [B, Q, C]
        return boxes, scores


def main() -> None:
    args = parse_args()
    device = torch.device("cpu")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    mc = cfg["model"]
    model = NMSFreeDetector(
        num_classes=mc["num_classes"],
        base_channels=mc["base_channels"],
        embed_dim=mc["embed_dim"],
        num_heads=mc["num_heads"],
        num_encoder_layers=mc["num_encoder_layers"],
        num_decoder_layers=mc["num_decoder_layers"],
        num_queries=mc["num_queries"],
        dropout=0.0,
        aux_loss=False,
    )

    ckpt = torch.load(args.weights, map_location=device)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    wrapper = ExportWrapper(model)
    dummy = torch.zeros(1, 3, args.img_size, args.img_size)

    out_path = args.output or str(Path(args.weights).with_suffix(".onnx"))

    dynamic_axes = None
    if args.dynamic:
        dynamic_axes = {
            "images": {0: "batch", 2: "height", 3: "width"},
            "pred_boxes": {0: "batch"},
            "pred_scores": {0: "batch"},
        }

    print(f"Exporting to {out_path} ...")
    torch.onnx.export(
        wrapper,
        dummy,
        out_path,
        opset_version=args.opset,
        input_names=["images"],
        output_names=["pred_boxes", "pred_scores"],
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )

    # Verify
    model_onnx = onnx.load(out_path)
    onnx.checker.check_model(model_onnx)

    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    feed = {sess.get_inputs()[0].name: dummy.numpy()}
    boxes, scores = sess.run(None, feed)
    print(f"ONNX verified: boxes={boxes.shape}, scores={scores.shape}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
