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
    """Bipartite matching on CPU.  Returns fixed-size [_MAX_T] index tensors so
    every downstream transfer and gather has the same shape every step."""

    def __init__(
        self,
        cls_weight:  float = 2.0,
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
        targets: list[dict],   # padded targets: labels/boxes [MAX_T], valid [MAX_T]
    ) -> list[tuple[Tensor, Tensor, Tensor]]:
        """Returns list of (pi, ti, valid) — all [_MAX_T] on CPU."""
        pred_logits = pred_logits.detach().float().cpu()
        pred_boxes  = pred_boxes.detach().float().cpu()
        B, Q, C = pred_logits.shape
        indices = []

        for b in range(B):
            pi_pad = torch.zeros(_MAX_T, dtype=torch.long)
            ti_pad = torch.zeros(_MAX_T, dtype=torch.long)
            v_pad  = torch.zeros(_MAX_T, dtype=torch.bool)

            valid_cpu = targets[b]["valid"].cpu()
            tgt_cls   = targets[b]["labels"].cpu()[valid_cpu]   # [M]
            tgt_boxes = targets[b]["boxes"].cpu()[valid_cpu]    # [M, 4]
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
    """Loss = Focal + L1 + CIoU with Hungarian assignment.

    Targets arrive pre-padded to MAX_TARGETS (from collate_fn).  All loss ops
    are fully batched — no Python loops over the batch dimension in forward — so
    XLA compiles the graph once and reuses it every step.

    Configure via YAML loss block; no device flag needed: fixed-shape batch
    ops work identically on CUDA and XLA.
    """

    def __init__(
        self,
        num_classes:  int,
        matcher:      HungarianMatcher,
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

    # ── batched loss kernels ────────────────────────────────────────────────
    # All inputs are on-device tensors with fixed shapes — same every step.

    def _loss_labels(
        self,
        logits:  Tensor,   # [B, Q, C]
        labels:  Tensor,   # [B, MAX_T] — pre-padded target labels on device
        pi:      Tensor,   # [B, MAX_T] — matched query indices
        ti:      Tensor,   # [B, MAX_T] — matched target indices
        valid:   Tensor,   # [B, MAX_T] bool
    ) -> Tensor:
        B, Q, C = logits.shape
        labs     = labels.gather(1, ti)              # [B, MAX_T] — batched gather
        flat_idx = pi * C + labs                     # [B, MAX_T] — flat (q, cls) index
        valid_f  = valid.to(logits.dtype)            # [B, MAX_T]
        tgt_cls  = logits.new_zeros(B, Q * C)        # [B, Q*C]
        tgt_cls.scatter_add_(1, flat_idx, valid_f)   # single batched scatter
        return sigmoid_focal_loss(logits, tgt_cls.view(B, Q, C), self.focal_alpha, self.focal_gamma)

    def _loss_boxes(
        self,
        pred_boxes: Tensor,   # [B, Q, 4]
        boxes:      Tensor,   # [B, MAX_T, 4] — pre-padded target boxes on device
        pi:         Tensor,   # [B, MAX_T]
        ti:         Tensor,   # [B, MAX_T]
        valid:      Tensor,   # [B, MAX_T] bool
    ) -> tuple[Tensor, Tensor]:
        B = pred_boxes.shape[0]
        pi_exp = pi.unsqueeze(-1).expand(-1, -1, 4)  # [B, MAX_T, 4]
        ti_exp = ti.unsqueeze(-1).expand(-1, -1, 4)  # [B, MAX_T, 4]
        p      = pred_boxes.gather(1, pi_exp).view(B * _MAX_T, 4)  # batched gather → flat
        t      = boxes.gather(1, ti_exp).view(B * _MAX_T, 4)
        vf     = valid.to(pred_boxes.dtype).view(B * _MAX_T)
        return (l1_loss(p, t) * vf).sum(), (ciou_loss(p, t) * vf).sum()

    # ── orchestration ────────────────────────────────────────────────────────

    def _compute(
        self,
        pred_logits: Tensor,
        pred_boxes:  Tensor,
        targets:     list[dict],
        indices:     list[tuple[Tensor, Tensor, Tensor]],
        num_boxes:   int,
        prefix:      str = "",
    ) -> dict[str, Tensor]:
        dev = pred_logits.device

        # Stack CPU index tensors → one [B, MAX_T] transfer each
        pi    = torch.stack([p      for p, _, _ in indices]).to(dev)
        ti    = torch.stack([t      for _, t, _ in indices]).to(dev)
        valid = torch.stack([v      for _, _, v in indices]).to(dev)

        # Stack on-device target tensors — batched, fixed shape, no data copy
        labels = torch.stack([t["labels"] for t in targets])   # [B, MAX_T]
        boxes  = torch.stack([t["boxes"]  for t in targets])   # [B, MAX_T, 4]

        cls      = self._loss_labels(pred_logits, labels, pi, ti, valid)
        l1, ciou = self._loss_boxes(pred_boxes, boxes, pi, ti, valid)

        p = f"{prefix}_" if prefix else ""
        return {
            f"{p}loss_cls":  cls  / num_boxes * self.cls_w,
            f"{p}loss_l1":   l1   / num_boxes * self.bbox_w,
            f"{p}loss_ciou": ciou / num_boxes * self.giou_w,
        }

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
