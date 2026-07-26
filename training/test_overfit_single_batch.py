"""End-to-end sanity check: dataset -> model -> Hungarian loss -> decode, all on one
synthetic non-square image, on CPU.

If the whole pipeline (letterbox transform, normalisation, matcher/loss, and the
letterbox-aware decode in utils/metrics.py) is wired correctly, overfitting a tiny
model on a single image should drive the loss down AND produce a decoded box that
tightly overlaps the known ground-truth box in *original* pixel coordinates.

This is the strongest verification: it would have failed loudly under the old
metrics.py bug (loss goes down, but decoded IoU stays low) and catches other
possible pipeline bugs (matcher assignment, normalisation, box parameterisation)
that the narrower test_letterbox.py cannot.
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from datasets.coco_dataset import CocoDetection, collate_fn
from models import NMSFreeDetector
from losses.hungarian import HungarianMatcher, HungarianCriterion
from losses.bbox_loss import box_cxcywh_to_xyxy
from detect import boxes_to_orig

torch.manual_seed(0)

IMG_SIZE = 64
ORIG_H, ORIG_W = 300, 120           # non-square, portrait — exercises letterbox padding
GT_BOX_XYWH = [30.0, 40.0, 50.0, 80.0]  # x, y, w, h in original pixels
NUM_CLASSES = 3
GT_LABEL = 1  # category_id (1-indexed, COCO style)
STEPS = 400

tmpdir = Path(tempfile.mkdtemp(prefix="vit_overfit_"))
try:
    img_dir = tmpdir / "images"
    img_dir.mkdir()
    img = np.full((ORIG_H, ORIG_W, 3), 114, dtype=np.uint8)
    x, y, w, h = (int(v) for v in GT_BOX_XYWH)
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), -1)
    img_path = img_dir / "img0.png"
    cv2.imwrite(str(img_path), img)

    ann = {
        "images": [{"id": 1, "file_name": "img0.png", "height": ORIG_H, "width": ORIG_W}],
        "annotations": [{
            "id": 1, "image_id": 1, "category_id": GT_LABEL,
            "bbox": GT_BOX_XYWH, "area": w * h, "iscrowd": 0,
        }],
        "categories": [{"id": i + 1, "name": f"c{i}"} for i in range(NUM_CLASSES)],
    }
    ann_path = tmpdir / "ann.json"
    ann_path.write_text(json.dumps(ann))

    dataset = CocoDetection(str(img_dir), str(ann_path), img_size=IMG_SIZE, train=False)
    image, target = dataset[0]
    # duplicate to batch=2 — this backbone collapses P5 to 1x1 spatial at IMG_SIZE=64,
    # and BatchNorm needs >1 element per channel (batch*H*W) while in train() mode
    batch_images, batch_targets = collate_fn([(image, target), (image, target)])

    model = NMSFreeDetector(
        num_classes=NUM_CLASSES, base_channels=8, embed_dim=32, num_heads=2,
        num_encoder_layers=0, num_decoder_layers=2, num_queries=10,
        dropout=0.0, aux_loss=False,
    )
    matcher = HungarianMatcher(2.0, 5.0, 2.0)
    criterion = HungarianCriterion(num_classes=NUM_CLASSES, matcher=matcher, no_obj_weight=0.5)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)

    model.train()
    first_loss, last_loss = None, None
    for step in range(STEPS):
        out = model(batch_images)
        losses = criterion(out, batch_targets)
        loss = losses["total"]
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == 0:
            first_loss = loss.item()
        last_loss = loss.item()
        if step % 100 == 0 or step == STEPS - 1:
            print(f"  step {step:4d}  loss={loss.item():.4f}")

    print(f"\nLoss: {first_loss:.4f} -> {last_loss:.4f}")
    assert last_loss < first_loss * 0.3, (
        f"FAIL: loss did not drop sharply on a single overfit sample "
        f"({first_loss:.4f} -> {last_loss:.4f}) — matcher/loss pipeline is suspect."
    )

    # Decode predictions the same way validate()/MeanAveragePrecision does.
    model.eval()
    with torch.no_grad():
        out = model(batch_images)
    scores = out["pred_logits"].sigmoid()[0, :, :-1]      # [Q, C] exclude ∅
    best_score, best_label = scores.max(dim=-1)
    top_q = best_score.argmax().item()

    pred_box = out["pred_boxes"][0, top_q:top_q + 1]       # [1, 4] normalised, padded-square space
    decoded = boxes_to_orig(pred_box, ORIG_H, ORIG_W, IMG_SIZE)[0]
    dx1, dy1, dx2, dy2 = decoded
    pred_box_xywh = [float(dx1), float(dy1), float(dx2 - dx1), float(dy2 - dy1)]

    def iou_xywh(a, b) -> float:
        ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
        bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = a[2] * a[3] + b[2] * b[3] - inter
        return inter / union if union > 0 else 0.0

    iou = iou_xywh(pred_box_xywh, GT_BOX_XYWH)
    print(f"\nTop query: idx={top_q} score={best_score[top_q]:.3f} label={best_label[top_q].item()}")
    print(f"GT box (orig px):        {[round(v, 1) for v in GT_BOX_XYWH]}")
    print(f"Decoded pred (orig px):  {[round(v, 1) for v in pred_box_xywh]}")
    print(f"IoU vs GT: {iou:.4f}")

    assert iou > 0.5, (
        f"FAIL: after overfitting a single sample, the decoded top prediction should "
        f"tightly overlap the GT box (IoU>0.5), got IoU={iou:.4f} — decode/letterbox "
        f"pipeline is broken."
    )
    print("\nTest PASSED — data transform, loss, and letterbox-aware decode are consistent end-to-end.")
finally:
    shutil.rmtree(tmpdir, ignore_errors=True)
