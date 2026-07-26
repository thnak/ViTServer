"""Hungarian Matching criterion — 1-to-1 assignment, eliminates NMS in training.

Key design decisions (DETR-style):
  1. num_classes + 1 output channels — the extra channel is "no-object" (∅).
     During matching, the ∅ class is excluded from cls_cost so queries are never
     matched to a real object based on predicting ∅.
  2. Focal loss on the first C channels only.  Unmatched queries are trained to
     predict 0 for all real classes (which the focal loss handles via the
     (1-alpha) * (1-p)^gamma * log(1-p) term for negative targets).
  3. Classification loss normalized by num_boxes (matched pairs), NOT by B*Q.
     Normalizing by B*Q dilutes the signal ~25x, collapsing the classifier.
  4. For the ∅ channel we apply a separate binary focal loss so unmatched
     queries learn to predict "no object" (target=1 for ∅, target=0 for all
     real classes).  Matched queries predict target=0 for ∅.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from scipy.optimize import linear_sum_assignment

from .bbox_loss import ciou_loss, l1_loss, box_cxcywh_to_xyxy, box_iou
from .focal_loss import sigmoid_focal_loss

# Must match MAX_TARGETS in datasets/coco_dataset.py
_MAX_T = 100


class HungarianMatcher(nn.Module):
    """Bipartite matching.  Computes costs batched on GPU, then runs Hungarian
    on CPU per sample (scipy).  Returns fixed-size [_MAX_T] index tensors."""

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
    def _compute_costs_batched(
        self,
        pred_logits: Tensor,  # [B, Q, C+1]
        pred_boxes:  Tensor,  # [B, Q, 4]
        targets: list[dict],
    ) -> tuple[Tensor, list[int]]:
        """Compute all matching costs batched on GPU, return (cost, valid_counts)."""
        B, Q, C = pred_logits.shape
        # Find max valid targets across batch
        valid_counts = [int(t["valid"].sum().item()) for t in targets]
        max_M = max(valid_counts) if valid_counts else 0

        if max_M == 0:
            return torch.empty(B, Q, 0, device=pred_logits.device), valid_counts

        # Build padded target tensors on device (no CPU sync per sample)
        tgt_cls   = torch.zeros(B, max_M, dtype=torch.long, device=pred_logits.device)
        tgt_boxes = torch.zeros(B, max_M, 4, device=pred_logits.device)
        for b in range(B):
            m = valid_counts[b]
            if m > 0:
                valid_mask = targets[b]["valid"]
                tgt_cls[b, :m]   = targets[b]["labels"][valid_mask][:m]
                tgt_boxes[b, :m] = targets[b]["boxes"][valid_mask][:m]

        # ── Classification cost ──────────────────────────────────────────
        # prob[b, q, c] = sigmoid(logits[b, q, c])  for c in real classes
        probs = pred_logits[:, :, :-1].sigmoid()        # [B, Q, C-1]
        # For each target, score is prob[b, q, target_cls]
        # -torch.bmm(probs, one_hot_targets) gives -prob[b, q, target_c] per column
        tgt_onehot = F.one_hot(tgt_cls, num_classes=C - 1).float()  # [B, max_M, C-1]
        cls_cost = -torch.bmm(probs, tgt_onehot.transpose(1, 2))     # [B, Q, max_M]

        # ── L1 cost ─────────────────────────────────────────────────────
        l1_cost = torch.cdist(pred_boxes, tgt_boxes, p=1)            # [B, Q, max_M]

        # ── GIoU cost ───────────────────────────────────────────────────
        # Expand all pairs for batched GIoU
        p_exp = pred_boxes.unsqueeze(2).expand(-1, -1, max_M, -1).reshape(-1, 4)   # [B*Q*max_M, 4]
        t_exp = tgt_boxes.unsqueeze(1).expand(-1, Q, -1, -1).reshape(-1, 4)        # [B*Q*max_M, 4]
        p_xyxy = box_cxcywh_to_xyxy(p_exp)
        t_xyxy = box_cxcywh_to_xyxy(t_exp)
        iou, union = box_iou(p_xyxy, t_xyxy)
        enc_area = (
            (torch.max(p_xyxy[:, 2:], t_xyxy[:, 2:])
             - torch.min(p_xyxy[:, :2], t_xyxy[:, :2]))
            .clamp(0).prod(dim=1)
        )
        giou = iou - (enc_area - union) / enc_area.clamp(1e-6)      # [B*Q*max_M]
        giou_cost = -giou.reshape(B, Q, max_M)

        cost = (self.cls_w * cls_cost + self.bbox_w * l1_cost + self.giou_w * giou_cost)
        return cost, valid_counts

    @torch.no_grad()
    def forward(
        self,
        pred_logits: Tensor,   # [B, Q, C+1] — last channel is ∅ (no-object)
        pred_boxes:  Tensor,   # [B, Q, 4]
        targets: list[dict],   # padded targets: labels/boxes [MAX_T], valid [MAX_T]
    ) -> list[tuple[Tensor, Tensor, Tensor]]:
        """Returns list of (pi, ti, valid) — all [_MAX_T] on CPU."""
        B, Q = pred_logits.shape[:2]
        cost_gpu, valid_counts = self._compute_costs_batched(pred_logits, pred_boxes, targets)

        # Single CPU sync — move full cost tensor at once
        cost_cpu = cost_gpu.cpu()
        indices = []

        for b in range(B):
            pi_pad = torch.zeros(_MAX_T, dtype=torch.long)
            ti_pad = torch.zeros(_MAX_T, dtype=torch.long)
            v_pad  = torch.zeros(_MAX_T, dtype=torch.bool)
            M = valid_counts[b]

            if M > 0:
                c = cost_cpu[b, :, :M].numpy()
                if not np.isfinite(c).all():
                    # A non-finite pred_box/logit (e.g. AMP fp16 overflow) reached the
                    # cost matrix — scipy raises on this. Clamp to a large-but-finite
                    # penalty so matching degrades gracefully instead of crashing the
                    # whole run; the loss for this sample will still reflect the bad
                    # prediction (and train_one_epoch skips the step if it's non-finite).
                    print(f"[HungarianMatcher] non-finite cost entries in batch item {b} — clamping")
                    c = np.nan_to_num(c, nan=1e6, posinf=1e6, neginf=-1e6)
                row, col = linear_sum_assignment(c)
                k = len(row)
                pi_pad[:k] = torch.as_tensor(row, dtype=torch.long)
                ti_pad[:k] = torch.as_tensor(col, dtype=torch.long)
                v_pad[:k]  = True

            indices.append((pi_pad, ti_pad, v_pad))

        return indices


class HungarianCriterion(nn.Module):
    """Loss = Focal (C real classes + ∅) + L1 + CIoU with Hungarian assignment.

    Targets arrive pre-padded to MAX_TARGETS (from collate_fn).  All loss ops
    are fully batched — no Python loops over the batch dimension in forward — so
    XLA compiles the graph once and reuses it every step.
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
        no_obj_weight: float = 0.5,  # weight for the "no-object" channel
    ) -> None:
        super().__init__()
        self.num_classes  = num_classes
        self.matcher      = matcher
        self.cls_w        = cls_weight
        self.bbox_w       = bbox_weight
        self.giou_w       = giou_weight
        self.focal_alpha  = focal_alpha
        self.focal_gamma  = focal_gamma
        self.no_obj_weight = no_obj_weight

    def _loss_labels(
        self,
        logits:  Tensor,   # [B, Q, C+1]
        labels:  Tensor,   # [B, MAX_T] — pre-padded target labels on device
        pi:      Tensor,   # [B, MAX_T] — matched query indices
        ti:      Tensor,   # [B, MAX_T] — matched target indices
        valid:   Tensor,   # [B, MAX_T] bool
    ) -> tuple[Tensor, Tensor]:
        """Returns (real_cls_loss, no_obj_loss)."""
        B, Q, C = logits.shape  # C = num_classes + 1
        n_real = C - 1

        # ── Real-class focal loss (channels 0..C-2) ──
        real_logits = logits[:, :, :n_real]           # [B, Q, n_real]
        labs     = labels.gather(1, ti)               # [B, MAX_T]
        flat_idx = pi * n_real + labs                 # [B, MAX_T]
        valid_f  = valid.to(logits.dtype)             # [B, MAX_T]
        tgt_cls  = logits.new_zeros(B, Q * n_real)   # [B, Q * n_real]
        tgt_cls.scatter_add_(1, flat_idx, valid_f)    # matched queries get target=1 for their class
        cls_focal = sigmoid_focal_loss(
            real_logits, tgt_cls.view(B, Q, n_real),
            self.focal_alpha, self.focal_gamma,
        )

        # ── No-object channel ──
        # Matched queries → target 0 (don't predict ∅, predict a real class)
        # Unmatched queries → target 1 (predict ∅)
        no_obj_logits = logits[:, :, -1:]            # [B, Q, 1]
        no_obj_target = logits.new_ones(B, Q, 1)     # default: predict ∅

        matched_mask = torch.zeros(B, Q, dtype=torch.bool, device=logits.device)
        matched_mask.scatter_(1, pi, valid)           # matched positions
        no_obj_target[matched_mask] = 0.0             # matched: don't predict ∅

        no_obj_loss = sigmoid_focal_loss(
            no_obj_logits, no_obj_target,
            self.focal_alpha, self.focal_gamma,
        )

        return cls_focal, no_obj_loss

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
        p      = pred_boxes.gather(1, pi_exp).view(B * _MAX_T, 4)
        t      = boxes.gather(1, ti_exp).view(B * _MAX_T, 4)
        vf     = valid.to(pred_boxes.dtype).view(B * _MAX_T)
        return (l1_loss(p, t) * vf).sum(), (ciou_loss(p, t) * vf).sum()

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

        pi    = torch.stack([p      for p, _, _ in indices]).to(dev)
        ti    = torch.stack([t      for _, t, _ in indices]).to(dev)
        valid = torch.stack([v      for _, _, v in indices]).to(dev)

        labels = torch.stack([t["labels"] for t in targets])
        boxes  = torch.stack([t["boxes"]  for t in targets])

        cls_focal, no_obj_loss = self._loss_labels(pred_logits, labels, pi, ti, valid)
        l1, ciou = self._loss_boxes(pred_boxes, boxes, pi, ti, valid)

        B, Q, _ = pred_logits.shape
        p = f"{prefix}_" if prefix else ""
        return {
            f"{p}loss_cls":   cls_focal   / num_boxes * self.cls_w,
            f"{p}loss_noobj": no_obj_loss / num_boxes * self.no_obj_weight,
            f"{p}loss_l1":    l1          / num_boxes * self.bbox_w,
            f"{p}loss_ciou":  ciou        / num_boxes * self.giou_w,
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
