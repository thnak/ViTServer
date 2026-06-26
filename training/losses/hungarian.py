"""Hungarian Matching criterion — 1-to-1 assignment, eliminates NMS in training."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from scipy.optimize import linear_sum_assignment

from .bbox_loss import ciou_loss, l1_loss, box_cxcywh_to_xyxy, box_iou
from .focal_loss import sigmoid_focal_loss

# Must match MAX_TARGETS in datasets/coco_dataset.py
_MAX_T = 100


class HungarianMatcher(nn.Module):
    """Compute optimal bipartite matching between predictions and padded targets.

    Returns fixed-size (pi, ti, valid) tuples — [_MAX_T] CPU tensors — so the
    downstream criterion only ever sees constant-shape index tensors on device.
    """

    def __init__(
        self,
        cls_weight: float = 2.0,
        bbox_weight: float = 5.0,
        giou_weight: float = 2.0,
    ) -> None:
        super().__init__()
        self.cls_w  = cls_weight
        self.bbox_w = bbox_weight
        self.giou_w = giou_weight

    @torch.no_grad()
    def forward(
        self,
        pred_logits: Tensor,   # [B, Q, C]
        pred_boxes:  Tensor,   # [B, Q, 4]
        targets: list[dict],   # padded: labels/boxes [MAX_T], valid [MAX_T]
    ) -> list[tuple[Tensor, Tensor, Tensor]]:
        # Entire cost computation on CPU — single host boundary, no mid-graph syncs.
        pred_logits = pred_logits.detach().float().cpu()
        pred_boxes  = pred_boxes.detach().float().cpu()
        B, Q, C = pred_logits.shape
        indices = []

        for b in range(B):
            pi_pad = torch.zeros(_MAX_T, dtype=torch.long)
            ti_pad = torch.zeros(_MAX_T, dtype=torch.long)
            v_pad  = torch.zeros(_MAX_T, dtype=torch.bool)

            valid_cpu = targets[b]["valid"].cpu()              # [MAX_T]
            tgt_cls   = targets[b]["labels"].cpu()[valid_cpu]  # [M]
            tgt_boxes = targets[b]["boxes"].cpu()[valid_cpu]   # [M, 4]
            M = len(tgt_cls)

            if M > 0:
                p        = pred_logits[b].sigmoid()
                cls_cost = -p[:, tgt_cls]
                l1_cost  = torch.cdist(pred_boxes[b], tgt_boxes, p=1)

                p_xyxy = box_cxcywh_to_xyxy(
                    pred_boxes[b].unsqueeze(1).expand(-1, M, -1).reshape(-1, 4))
                t_xyxy = box_cxcywh_to_xyxy(
                    tgt_boxes.unsqueeze(0).expand(Q, -1, -1).reshape(-1, 4))
                iou, union = box_iou(p_xyxy, t_xyxy)
                enc_area = (
                    (torch.max(p_xyxy[:, 2:], t_xyxy[:, 2:])
                     - torch.min(p_xyxy[:, :2], t_xyxy[:, :2]))
                    .clamp(0).prod(dim=1)
                )
                giou      = iou - (enc_area - union) / enc_area.clamp(1e-6)
                giou_cost = -giou.reshape(Q, M)

                cost     = self.cls_w * cls_cost + self.bbox_w * l1_cost + self.giou_w * giou_cost
                row, col = linear_sum_assignment(cost.numpy())
                k = len(row)
                pi_pad[:k] = torch.as_tensor(row, dtype=torch.long)
                ti_pad[:k] = torch.as_tensor(col, dtype=torch.long)
                v_pad[:k]  = True

            indices.append((pi_pad, ti_pad, v_pad))

        return indices


class HungarianCriterion(nn.Module):
    """Full loss with Hungarian matching + CIoU + Focal.

    Targets are expected to be pre-padded to MAX_TARGETS by the dataloader's
    collate_fn.  All loss ops use fixed-shape [MAX_T] index tensors so XLA
    compiles the graph exactly once, regardless of how many GT boxes each
    image contains.
    """

    def __init__(
        self,
        num_classes: int,
        matcher: HungarianMatcher,
        cls_weight:   float = 2.0,
        bbox_weight:  float = 5.0,
        giou_weight:  float = 2.0,
        focal_alpha:  float = 0.25,
        focal_gamma:  float = 2.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.matcher     = matcher
        self.cls_w       = cls_weight
        self.bbox_w      = bbox_weight
        self.giou_w      = giou_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    # ── internal helpers ────────────────────────────────────────────────────

    def _loss_labels(
        self,
        logits:  Tensor,
        targets: list[dict],
        indices: list[tuple[Tensor, Tensor, Tensor]],
    ) -> Tensor:
        B, Q, C = logits.shape
        dev     = logits.device
        tgt_cls = logits.new_zeros(B, Q, C)  # [B, Q, C] on device — always same shape

        for b, (pi, ti, valid) in enumerate(indices):
            pi_d    = pi.to(dev)                         # [MAX_T]
            ti_d    = ti.to(dev)                         # [MAX_T]
            valid_f = valid.to(dev).to(logits.dtype)     # [MAX_T]

            labs     = targets[b]["labels"][ti_d]        # [MAX_T] — fixed-shape gather
            flat_idx = pi_d * C + labs                   # [MAX_T] — flat (query, class) index
            tgt_cls[b].view(-1).scatter_add_(0, flat_idx, valid_f)

        return sigmoid_focal_loss(logits, tgt_cls, self.focal_alpha, self.focal_gamma)

    def _loss_boxes(
        self,
        pred_boxes: Tensor,
        targets:    list[dict],
        indices:    list[tuple[Tensor, Tensor, Tensor]],
    ) -> tuple[Tensor, Tensor]:
        dev        = pred_boxes.device
        total_l1   = pred_boxes.new_tensor(0.0)
        total_ciou = pred_boxes.new_tensor(0.0)

        for b, (pi, ti, valid) in enumerate(indices):
            pi_d    = pi.to(dev)                     # [MAX_T]
            ti_d    = ti.to(dev)                     # [MAX_T]
            valid_f = valid.float().to(dev)          # [MAX_T]

            p = pred_boxes[b, pi_d]                  # [MAX_T, 4] — fixed-shape gather
            t = targets[b]["boxes"][ti_d]            # [MAX_T, 4] — fixed-shape gather
            total_l1   = total_l1   + (l1_loss(p, t)   * valid_f).sum()
            total_ciou = total_ciou + (ciou_loss(p, t) * valid_f).sum()

        return total_l1, total_ciou

    def _compute(
        self,
        pred_logits: Tensor,
        pred_boxes:  Tensor,
        targets:     list[dict],
        indices:     list[tuple[Tensor, Tensor, Tensor]],
        num_boxes:   int,
        prefix:      str = "",
    ) -> dict[str, Tensor]:
        l1, ciou = self._loss_boxes(pred_boxes, targets, indices)
        cls      = self._loss_labels(pred_logits, targets, indices)
        p = f"{prefix}_" if prefix else ""
        return {
            f"{p}loss_cls":  cls  / num_boxes * self.cls_w,
            f"{p}loss_l1":   l1   / num_boxes * self.bbox_w,
            f"{p}loss_ciou": ciou / num_boxes * self.giou_w,
        }

    # ── public ──────────────────────────────────────────────────────────────

    def forward(self, outputs: dict, targets: list[dict]) -> dict[str, Tensor]:
        indices   = self.matcher(outputs["pred_logits"], outputs["pred_boxes"], targets)
        num_boxes = max(1, sum(int(v.sum()) for _, _, v in indices))

        losses = self._compute(
            outputs["pred_logits"], outputs["pred_boxes"], targets, indices, num_boxes
        )

        if "aux_outputs" in outputs:
            for i, aux in enumerate(outputs["aux_outputs"]):
                idx   = self.matcher(aux["pred_logits"], aux["pred_boxes"], targets)
                aux_n = max(1, sum(int(v.sum()) for _, _, v in idx))
                losses.update(self._compute(
                    aux["pred_logits"], aux["pred_boxes"],
                    targets, idx, aux_n, prefix=f"aux_{i}",
                ))

        losses["total"] = sum(losses.values())
        return losses
