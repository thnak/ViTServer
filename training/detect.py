#!/usr/bin/env python3
"""Run detection on images using a trained .pt checkpoint.

Usage:
    # Single image
    python detect.py --checkpoint runs/nano/best.pt --config configs/nano.yaml --source image.jpg

    # Directory of images
    python detect.py --checkpoint runs/nano/last.pt --config configs/nano.yaml --source images/

    # Custom threshold and output dir
    python detect.py --checkpoint runs/medium/best.pt --config configs/medium.yaml \
                     --source images/ --score-thresh 0.4 --output runs/medium/detections/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from models import NMSFreeDetector
from datasets.transforms import build_val_transforms

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]

# Distinct BGR colours, one per class (cycles if num_classes > len)
_PALETTE = [
    (255,  56,  56), (255, 157,  151), (255, 112,  31), (255, 178,  29),
    (207, 210,  49), ( 72, 249,  10), ( 146, 204,  23), ( 61, 219, 134),
    ( 26, 147,  52), (  0, 212, 187), ( 44, 153, 168), (  0, 194, 255),
    ( 52,  69, 147), (100,  45, 255), (142,  46, 196), (215,  39, 133),
]


def _colour(cls_id: int):
    return _PALETTE[cls_id % len(_PALETTE)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("ViTServer Detector")
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    p.add_argument("--config",     required=True, help="YAML config used during training")
    p.add_argument("--source",     required=True, help="Image file or directory")
    p.add_argument("--score-thresh", type=float, default=0.3, help="Detection confidence threshold")
    p.add_argument("--output",     default="runs/detect", help="Output directory")
    p.add_argument("--device",     default="cuda")
    p.add_argument("--class-names", default=None,
                   help="Path to a text file with one class name per line "
                        "(defaults to COCO 80-class names)")
    return p.parse_args()


def load_model(cfg: dict, checkpoint: str, device: torch.device) -> NMSFreeDetector:
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
        encoder_type=mc.get("encoder_type", "none"),
        window_size=mc.get("window_size", 8),
    ).to(device)

    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt.get("model", ckpt)   # handle both raw state_dict and our checkpoint format
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def letterbox_params(orig_h: int, orig_w: int, img_size: int):
    """Return (scale, pad_top, pad_left) for the val letterbox transform."""
    scale = img_size / max(orig_h, orig_w)
    new_h = round(orig_h * scale)
    new_w = round(orig_w * scale)
    pad_top  = (img_size - new_h) // 2
    pad_left = (img_size - new_w) // 2
    return scale, pad_top, pad_left


def boxes_to_orig(boxes: torch.Tensor, orig_h: int, orig_w: int, img_size: int) -> np.ndarray:
    """Convert normalised cx,cy,w,h → pixel x1,y1,x2,y2 in original image space."""
    scale, pad_top, pad_left = letterbox_params(orig_h, orig_w, img_size)
    b = boxes.cpu().numpy() * img_size          # → pixel space of padded image
    cx, cy, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    x1 = (cx - w / 2 - pad_left) / scale
    y1 = (cy - h / 2 - pad_top)  / scale
    x2 = (cx + w / 2 - pad_left) / scale
    y2 = (cy + h / 2 - pad_top)  / scale
    x1 = np.clip(x1, 0, orig_w)
    y1 = np.clip(y1, 0, orig_h)
    x2 = np.clip(x2, 0, orig_w)
    y2 = np.clip(y2, 0, orig_h)
    return np.stack([x1, y1, x2, y2], axis=1).astype(int)


def draw_detections(
    img_bgr: np.ndarray,
    boxes_px: np.ndarray,       # [N, 4] x1,y1,x2,y2 int
    scores: np.ndarray,         # [N]
    class_ids: np.ndarray,      # [N]
    class_names: list[str],
) -> np.ndarray:
    out = img_bgr.copy()
    for (x1, y1, x2, y2), score, cls_id in zip(boxes_px, scores, class_ids):
        colour = _colour(cls_id)
        label  = f"{class_names[cls_id] if cls_id < len(class_names) else cls_id} {score:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 4), (x1 + tw, y1), colour, -1)
        cv2.putText(out, label, (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


@torch.no_grad()
def detect_image(
    model: NMSFreeDetector,
    img_bgr: np.ndarray,
    transforms,
    img_size: int,
    score_thresh: float,
    class_names: list[str],
    device: torch.device,
) -> np.ndarray:
    orig_h, orig_w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    result = transforms(image=img_rgb, bboxes=[], labels=[])
    tensor = result["image"].unsqueeze(0).to(device)   # [1, 3, H, W]

    out = model(tensor)
    scores_all = out["pred_logits"][0].sigmoid()        # [Q, C]
    boxes_all  = out["pred_boxes"][0]                   # [Q, 4]

    scores, class_ids = scores_all.max(dim=-1)          # [Q]
    keep = scores >= score_thresh

    if keep.sum() == 0:
        return img_bgr

    boxes_px = boxes_to_orig(boxes_all[keep], orig_h, orig_w, img_size)
    return draw_detections(
        img_bgr,
        boxes_px,
        scores[keep].cpu().numpy(),
        class_ids[keep].cpu().numpy(),
        class_names,
    )


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dev_str = "cuda" if args.device.lower() == "gpu" else args.device
    device = torch.device(dev_str if torch.cuda.is_available() or dev_str == "cpu" else "cpu")
    img_size = cfg["model"]["img_size"]

    class_names = COCO_CLASSES
    if args.class_names:
        class_names = Path(args.class_names).read_text().strip().splitlines()

    print(f"  Checkpoint  {args.checkpoint}")
    print(f"  Config      {args.config}")
    print(f"  Device      {device}  │  img_size {img_size}  │  thresh {args.score_thresh}")

    model      = load_model(cfg, args.checkpoint, device)
    transforms = build_val_transforms(img_size)

    source  = Path(args.source)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(source.rglob("*")) if source.is_dir() else [source]
    paths = [p for p in paths if p.suffix.lower() in IMG_EXTS]

    if not paths:
        raise SystemExit(f"No images found at '{source}'")

    print(f"  Images      {len(paths)}  →  {out_dir}\n")

    for path in paths:
        img = cv2.imread(str(path))
        if img is None:
            print(f"  [skip] cannot read {path.name}")
            continue

        result = detect_image(model, img, transforms, img_size, args.score_thresh, class_names, device)

        out_path = out_dir / path.name
        cv2.imwrite(str(out_path), result)
        print(f"  {path.name:40s} → {out_path}")

    print(f"\nDone. Results saved to {out_dir}/")


if __name__ == "__main__":
    main()
