"""Hungarian Matching criterion — 1-to-1 assignment, eliminates NMS in training."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from scipy.optimize import linear_sum_assignment

from .bbox_loss import ciou_loss, l1_loss, box_cxcywh_to_xyxy, box_iou
from .focal_loss import sigmoid_focal_loss


class HungarianMatcher(nn.Module):
    """Compute optimal bipartite matching between predictions and targets."""

    def __init__(
        self,
        cls_weight: float = 2.0,
        bbox_weight: float = 5.0,
        giou_weight: float = 2.0,
    ) -> None:
        super().__init__()
        self.cls_w = cls_weight
        self.bbox_w = bbox_weight
        self.giou_w = giou_weight

    @torch.no_grad()
    def forward(
        self,
        pred_logits: Tensor,   # [B, Q, C]
        pred_boxes: Tensor,    # [B, Q, 4]
        targets: list[dict],   # list of {"labels": [M], "boxes": [M, 4]}
    ) -> list[tuple[Tensor, Tensor]]:
        B, Q, C = pred_logits.shape
        indices = []

        for b in range(B):
            tgt = targets[b]
            M = len(tgt["labels"])
            if M == 0:
                indices.append((torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)))
                continue

            # Classification cost: focal-style, [Q, M]
            p = pred_logits[b].sigmoid()        # [Q, C]
            tgt_cls = tgt["labels"]             # [M]
            cls_cost = -p[:, tgt_cls]           # [Q, M]

            # L1 box cost: [Q, M]
            tgt_boxes = tgt["boxes"].to(pred_boxes.device)   # [M, 4]
            l1_cost = torch.cdist(pred_boxes[b], tgt_boxes, p=1)  # [Q, M]

            # GIoU cost: [Q, M]
            p_xyxy = box_cxcywh_to_xyxy(pred_boxes[b].unsqueeze(1).expand(-1, M, -1).reshape(-1, 4))
            t_xyxy = box_cxcywh_to_xyxy(tgt_boxes.unsqueeze(0).expand(Q, -1, -1).reshape(-1, 4))
            iou, union = box_iou(p_xyxy, t_xyxy)
            enc_area = (
                (torch.max(p_xyxy[:, 2:], t_xyxy[:, 2:]) - torch.min(p_xyxy[:, :2], t_xyxy[:, :2]))
                .clamp(0).prod(dim=1)
            )
            giou = iou - (enc_area - union) / enc_area.clamp(1e-6)
            giou_cost = -giou.reshape(Q, M)

            cost = self.cls_w * cls_cost + self.bbox_w * l1_cost + self.giou_w * giou_cost
            row, col = linear_sum_assignment(cost.cpu().numpy())
            indices.append((
                torch.as_tensor(row, dtype=torch.long),
                torch.as_tensor(col, dtype=torch.long),
            ))

        return indices


class HungarianCriterion(nn.Module):
    """Full loss with Hungarian matching + CIoU + Focal."""

    def __init__(
        self,
        num_classes: int,
        matcher: HungarianMatcher,
        cls_weight: float = 2.0,
        bbox_weight: float = 5.0,
        giou_weight: float = 2.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.cls_w = cls_weight
        self.bbox_w = bbox_weight
        self.giou_w = giou_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def _loss_labels(
        self,
        logits: Tensor,
        targets: list[dict],
        indices: list[tuple[Tensor, Tensor]],
    ) -> Tensor:
        B, Q, C = logits.shape
        tgt_cls = torch.zeros(B, Q, C, device=logits.device)
        for b, (pi, ti) in enumerate(indices):
            if len(pi):
                tgt_cls[b, pi, targets[b]["labels"][ti]] = 1.0
        return sigmoid_focal_loss(logits, tgt_cls, self.focal_alpha, self.focal_gamma)

    def _loss_boxes(
        self,
        pred_boxes: Tensor,
        targets: list[dict],
        indices: list[tuple[Tensor, Tensor]],
    ) -> tuple[Tensor, Tensor]:
        preds, gts = [], []
        for b, (pi, ti) in enumerate(indices):
            if len(pi):
                preds.append(pred_boxes[b, pi])
                gts.append(targets[b]["boxes"][ti].to(pred_boxes.device))
        if not preds:
            z = pred_boxes.new_tensor(0.0)
            return z, z
        preds = torch.cat(preds)
        gts = torch.cat(gts)
        return l1_loss(preds, gts).sum(), ciou_loss(preds, gts).sum()

    def forward(
        self,
        outputs: dict,
        targets: list[dict],
    ) -> dict[str, Tensor]:
        indices = self.matcher(outputs["pred_logits"], outputs["pred_boxes"], targets)
        num_boxes = max(1, sum(len(t["labels"]) for t in targets))

        l1, ciou = self._loss_boxes(outputs["pred_boxes"], targets, indices)
        cls = self._loss_labels(outputs["pred_logits"], targets, indices)

        losses: dict[str, Tensor] = {
            "loss_cls": cls / num_boxes * self.cls_w,
            "loss_l1": l1 / num_boxes * self.bbox_w,
            "loss_ciou": ciou / num_boxes * self.giou_w,
        }

        if "aux_outputs" in outputs:
            for i, aux in enumerate(outputs["aux_outputs"]):
                idx = self.matcher(aux["pred_logits"], aux["pred_boxes"], targets)
                l1a, cioua = self._loss_boxes(aux["pred_boxes"], targets, idx)
                clsa = self._loss_labels(aux["pred_logits"], targets, idx)
                losses[f"aux_{i}_loss_cls"] = clsa / num_boxes * self.cls_w
                losses[f"aux_{i}_loss_l1"] = l1a / num_boxes * self.bbox_w
                losses[f"aux_{i}_loss_ciou"] = cioua / num_boxes * self.giou_w

        losses["total"] = sum(losses.values())
        return losses
