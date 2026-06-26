#!/usr/bin/env python3
"""Export trained model to ONNX with dynamic shapes and NHWC input for TensorRT."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import onnx
import onnx.shape_inference
import onnxruntime as ort
import yaml

from models import NMSFreeDetector
from models.transformer import MultiheadAttentionONNX


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Export NMS-Free Detector to ONNX")
    p.add_argument("--weights", "--checkpoint", required=True, dest="weights",
                   help="Path to .pt checkpoint")
    p.add_argument("--config", default="configs/custom_model.yaml")
    p.add_argument("--format", default="onnx", choices=["onnx"])
    p.add_argument("--dynamic", action="store_true", help="Dynamic batch + spatial axes")
    p.add_argument("--img_size", type=int, default=1280)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--output", default="", help="Output path (default: weights stem + .onnx)")
    p.add_argument("--no-simplify", action="store_true",
                   help="Skip onnxsim graph simplification")
    return p.parse_args()


def _replace_mha_for_onnx(model: torch.nn.Module) -> torch.nn.Module:
    """Swap nn.MultiheadAttention → MultiheadAttentionONNX and migrate weights.

    PyTorch 2.9+ fuses MHA into aten::_native_multi_head_attention which has
    no ONNX symbolic.  We replace each MHA with our bmm-based implementation
    and copy the fused in_proj_weight (Q||K||V stacked) into separate Linear
    layers so no retraining is needed.
    """
    for name, child in list(model.named_children()):
        if isinstance(child, torch.nn.MultiheadAttention):
            D = child.embed_dim
            onnx_attn = MultiheadAttentionONNX(D, child.num_heads, child.dropout)
            w = child.in_proj_weight.data
            onnx_attn.q_proj.weight.data.copy_(w[:D])
            onnx_attn.k_proj.weight.data.copy_(w[D : 2 * D])
            onnx_attn.v_proj.weight.data.copy_(w[2 * D :])
            if child.in_proj_bias is not None:
                b = child.in_proj_bias.data
                onnx_attn.q_proj.bias.data.copy_(b[:D])
                onnx_attn.k_proj.bias.data.copy_(b[D : 2 * D])
                onnx_attn.v_proj.bias.data.copy_(b[2 * D :])
            onnx_attn.out_proj.weight.data.copy_(child.out_proj.weight.data)
            onnx_attn.out_proj.bias.data.copy_(child.out_proj.bias.data)
            setattr(model, name, onnx_attn)
        else:
            _replace_mha_for_onnx(child)
    return model


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

    _replace_mha_for_onnx(model)   # swap fused MHA → bmm-based (ONNX opset 17 compatible)

    wrapper = ExportWrapper(model)
    wrapper.eval()
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
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            out_path,
            opset_version=args.opset,
            input_names=["images"],
            output_names=["pred_boxes", "pred_scores"],
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            dynamo=False,          # use TorchScript path; torch.export fails on 1×1 spatial dims
        )

    # Post-processing: simplify + shape inference
    model_onnx = onnx.load(out_path)
    onnx.checker.check_model(model_onnx)
    n_before = len(model_onnx.graph.node)

    if not args.no_simplify:
        try:
            import onnxsim
            print("Running onnxsim ...")
            model_onnx, ok = onnxsim.simplify(model_onnx)
            if ok:
                print(f"  nodes: {n_before} → {len(model_onnx.graph.node)}")
            else:
                print("  onnxsim check failed — keeping original graph")
        except ImportError:
            print("  onnxsim not installed, skipping (uv pip install onnxsim)")

    # Shape inference fills type/shape on every intermediate value so Netron
    # can display tensor shapes throughout the network.
    print("Running ONNX shape inference ...")
    model_onnx = onnx.shape_inference.infer_shapes(model_onnx)
    onnx.save(model_onnx, out_path)

    # ORT round-trip verification
    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    feed = {sess.get_inputs()[0].name: dummy.numpy()}
    boxes, scores = sess.run(None, feed)
    print(f"ONNX verified: boxes={boxes.shape}, scores={scores.shape}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
