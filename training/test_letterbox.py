"""Verify the letterbox round-trip: dataset normalisation → model space → decode back
to original-image pixels must recover the original box. Also shows numerically how far
off the old (pre-fix) naive scaling was, for a non-square image.

No GPU / dataset download required — synthesises one non-square image in-memory.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from datasets.transforms import build_val_transforms
from detect import boxes_to_orig  # same math now used in utils/metrics.py

IMG_SIZE = 416
ORIG_H, ORIG_W = 1080, 640         # strongly non-square (portrait) — big letterbox padding
GT_BOX_XYWH = [20.0, 30.0, 80.0, 60.0]  # near the padded edge, x, y, w, h in original pixels

img = np.zeros((ORIG_H, ORIG_W, 3), dtype=np.uint8)

transforms = build_val_transforms(IMG_SIZE)
result = transforms(image=img, bboxes=[GT_BOX_XYWH], labels=[0])
tb_x, tb_y, tb_w, tb_h = result["bboxes"][0]

# Same normalisation CocoDetection.__getitem__ performs (datasets/coco_dataset.py:58-65)
cx = (tb_x + tb_w / 2) / IMG_SIZE
cy = (tb_y + tb_h / 2) / IMG_SIZE
nw = tb_w / IMG_SIZE
nh = tb_h / IMG_SIZE
pred_boxes = torch.tensor([[cx, cy, nw, nh]])  # pretend the model predicted the GT exactly

# --- NEW (fixed) decode path: same math as utils/metrics.py MeanAveragePrecision.update() ---
decoded = boxes_to_orig(pred_boxes, ORIG_H, ORIG_W, IMG_SIZE)[0]
x1, y1, x2, y2 = decoded
new_box_xywh = [x1, y1, x2 - x1, y2 - y1]

# --- OLD (buggy) decode path: naive scale by original (w, h), no unletterbox ---
boxes_xyxy = torch.tensor([cx - nw / 2, cy - nh / 2, cx + nw / 2, cy + nh / 2])
old_x1, old_y1, old_x2, old_y2 = (boxes_xyxy * torch.tensor([ORIG_W, ORIG_H, ORIG_W, ORIG_H])).tolist()
old_box_xywh = [old_x1, old_y1, old_x2 - old_x1, old_y2 - old_y1]


def iou_xywh(a, b) -> float:
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


new_iou = iou_xywh(new_box_xywh, GT_BOX_XYWH)
old_iou = iou_xywh(old_box_xywh, GT_BOX_XYWH)

print(f"Original image: {ORIG_W}x{ORIG_H} (non-square)")
print(f"GT box (pixels):        {[round(v, 1) for v in GT_BOX_XYWH]}")
print(f"Decoded — NEW (fixed):  {[round(v, 1) for v in new_box_xywh]}   IoU={new_iou:.4f}")
print(f"Decoded — OLD (buggy):  {[round(v, 1) for v in old_box_xywh]}   IoU={old_iou:.4f}")
print()

assert new_iou > 0.98, f"FAIL: fixed decode should reproduce the GT box almost exactly, got IoU={new_iou:.4f}"
assert old_iou < 0.5, f"expected the old naive scaling to be visibly wrong on a non-square image, got IoU={old_iou:.4f}"
print("Test PASSED — fixed decode recovers the box; old naive scaling was measurably wrong.")
