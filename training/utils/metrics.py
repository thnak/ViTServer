"""COCO-style mAP evaluation via pycocotools."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch
from torch import Tensor
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from losses.bbox_loss import box_cxcywh_to_xyxy


class MeanAveragePrecision:
    def __init__(self, ann_file: str, img_size: int) -> None:
        self.gt_coco = COCO(ann_file)
        self.img_size = img_size
        self.results: list[dict] = []
        # Reverse the dataset's cat2idx: contiguous model label → real COCO category id
        cats = sorted(self.gt_coco.cats.keys())
        self.idx2cat: dict[int, int] = {i: c for i, c in enumerate(cats)}

    def update(
        self,
        pred_boxes: Tensor,     # [B, Q, 4] cx,cy,w,h normalised to the padded img_size square
        pred_scores: Tensor,    # [B, Q, C] sigmoid scores
        image_ids: list[int],
        orig_sizes: Tensor,     # [B, 2] H, W (pre-letterbox)
    ) -> None:
        B = pred_boxes.shape[0]
        for b in range(B):
            h, w = orig_sizes[b].tolist()
            scores, labels = pred_scores[b][:, :-1].max(dim=-1)  # [Q], [Q] — exclude ∅ channel
            boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes[b])  # [Q, 4] normalised, padded-square space

            # Undo LongestMaxSize + PadIfNeeded (see datasets/transforms.py) to recover
            # original-image pixel coords — must match detect.py's boxes_to_orig().
            scale = self.img_size / max(h, w)
            new_h, new_w = round(h * scale), round(w * scale)
            pad_top  = (self.img_size - new_h) // 2
            pad_left = (self.img_size - new_w) // 2

            boxes_pad_px = boxes_xyxy * self.img_size  # → pixel space of the padded square
            offset = torch.tensor(
                [pad_left, pad_top, pad_left, pad_top], device=boxes_pad_px.device, dtype=torch.float32
            )
            boxes_px = (boxes_pad_px - offset) / scale
            boxes_px[:, [0, 2]] = boxes_px[:, [0, 2]].clamp(0, w)
            boxes_px[:, [1, 3]] = boxes_px[:, [1, 3]].clamp(0, h)

            keep = scores > 0.01
            for box, score, label in zip(
                boxes_px[keep].tolist(), scores[keep].tolist(), labels[keep].tolist()
            ):
                x1, y1, x2, y2 = box
                self.results.append({
                    "image_id": image_ids[b],
                    "category_id": self.idx2cat[label],
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": round(score, 4),
                })

    def compute(self) -> dict[str, float]:
        if not self.results:
            return {"mAP": 0.0, "mAP50": 0.0}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(self.results, f)
            tmp = f.name

        dt_coco = self.gt_coco.loadRes(tmp)
        eval_ = COCOeval(self.gt_coco, dt_coco, "bbox")
        eval_.evaluate()
        eval_.accumulate()
        eval_.summarize()
        Path(tmp).unlink()

        return {
            "mAP": float(eval_.stats[0]),
            "mAP50": float(eval_.stats[1]),
            "mAP75": float(eval_.stats[2]),
            "mAP_s": float(eval_.stats[3]),
            "mAP_m": float(eval_.stats[4]),
            "mAP_l": float(eval_.stats[5]),
        }

    def reset(self) -> None:
        self.results.clear()
