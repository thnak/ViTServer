"""Diagnose decoder query/output collapse: does the model's prediction actually
respond to where an object is in the image, or has the decoder learned to ignore
the visual memory and emit a near-constant output regardless of input?

Draws a bright rectangle at several different positions and checks:
  1. Do backbone/MFE features differ meaningfully between positions? (sanity check
     that the CNN is at least extracting position-dependent signal)
  2. Does the decoder's final pred_boxes/pred_logits actually move with the rectangle,
     or stay frozen near a learned default regardless of where the object is?

If (1) varies but (2) barely does, the decoder's cross-attention hasn't started
localizing yet — a well-known slow-to-converge characteristic of global (non-deformable)
attention trained from scratch, exacerbated by an under-trained backbone or a missing
encoder stage. Not necessarily a bug, but worth tracking across checkpoints.

Usage: python test_decoder_sensitivity.py --config configs/coco_3cls.yaml --checkpoint runs/coco_3cls_v3/last.pt
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from models import NMSFreeDetector

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def make_img(img_size: int, cx: int, cy: int, size: int = 60) -> torch.Tensor:
    img = np.full((img_size, img_size, 3), 114, dtype=np.uint8)
    cv2.rectangle(img, (cx - size // 2, cy - size // 2), (cx + size // 2, cy + size // 2), (220, 30, 30), -1)
    t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return ((t - MEAN) / STD).unsqueeze(0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    mc = cfg["model"]
    S = mc["img_size"]

    model = NMSFreeDetector(
        num_classes=mc["num_classes"], base_channels=mc["base_channels"],
        embed_dim=mc["embed_dim"], num_heads=mc["num_heads"],
        num_encoder_layers=mc["num_encoder_layers"], num_decoder_layers=mc["num_decoder_layers"],
        num_queries=mc["num_queries"], dropout=mc["dropout"], aux_loss=False,
        encoder_type=mc.get("encoder_type", "none"), window_size=mc.get("window_size", 8),
    )
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"Loaded {args.checkpoint} (epoch {ck.get('epoch')})\n")

    margin = S // 5
    positions = [(margin, margin), (S - margin, margin), (margin, S - margin), (S - margin, S - margin), (S // 2, S // 2)]

    with torch.no_grad():
        # 1. Where does the top-confidence query place its box, per position?
        print("--- Does the top prediction track the rectangle? ---")
        for cx, cy in positions:
            img = make_img(S, cx, cy)
            out = model(img)
            boxes = out["pred_boxes"][0]
            scores = out["pred_logits"][0].sigmoid()
            real, cls = scores[:, :-1].max(dim=-1)
            noobj = scores[:, -1]
            top = (real * (1 - noobj)).argmax().item()
            pb = boxes[top]
            print(f"  rect@({cx:3d},{cy:3d}) -> query {top:2d}: pred_cxcy=({pb[0]*S:6.1f},{pb[1]*S:6.1f}) "
                  f"cls={cls[top].item()} real_score={real[top]:.3f} noobj={noobj[top]:.3f}")

        # 2. Isolate backbone/MFE variation vs. decoder-output variation for two extreme positions
        img_a = make_img(S, positions[0][0], positions[0][1])
        img_b = make_img(S, positions[3][0], positions[3][1])
        p3a, p4a, p5a = model.backbone(img_a)
        p3b, p4b, p5b = model.backbone(img_b)
        mem_a, _ = model.mfe(p3a, p4a, p5a)
        mem_b, _ = model.mfe(p3b, p4b, p5b)
        dec_a = model.decoder(mem_a)
        dec_b = model.decoder(mem_b)

        mem_diff = (mem_a - mem_b).abs().mean().item()
        mem_std = mem_a.std().item()
        box_diff = (dec_a["pred_boxes"] - dec_b["pred_boxes"]).abs().mean().item()
        logit_diff = (dec_a["pred_logits"] - dec_b["pred_logits"]).abs().mean().item()

        print("\n--- Backbone/MFE signal vs. decoder-output sensitivity (opposite corners) ---")
        print(f"  MFE memory |diff| mean: {mem_diff:.4f}  (memory std: {mem_std:.4f}, ratio {mem_diff/mem_std:.3f})")
        print(f"  decoder pred_boxes |diff| mean:  {box_diff:.5f}")
        print(f"  decoder pred_logits |diff| mean: {logit_diff:.5f}")

        if mem_diff / mem_std > 0.05 and box_diff < 0.01:
            print("\n  -> Backbone IS picking up positional signal, but the decoder's cross-attention")
            print("     is not yet using it — output is still ~constant regardless of object position.")
        elif box_diff >= 0.01:
            print("\n  -> Decoder output is responding to object position. Good sign.")
        else:
            print("\n  -> Backbone itself shows little positional signal — check backbone training first.")


if __name__ == "__main__":
    main()
