"""Bounding-box regression losses: CIoU and L1."""

from __future__ import annotations

import torch
from torch import Tensor


def box_cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def box_area(boxes: Tensor) -> Tensor:
    return (boxes[:, 2] - boxes[:, 0]).clamp(0) * (boxes[:, 3] - boxes[:, 1]).clamp(0)


def box_iou(boxes1: Tensor, boxes2: Tensor) -> tuple[Tensor, Tensor]:
    a1, a2 = box_area(boxes1), box_area(boxes2)
    inter = (
        torch.min(boxes1[:, 2:], boxes2[:, 2:]) - torch.max(boxes1[:, :2], boxes2[:, :2])
    ).clamp(0).prod(dim=1)
    union = a1 + a2 - inter
    iou = inter / union.clamp(min=1e-6)
    return iou, union


def ciou_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Complete IoU loss on [N, 4] cx,cy,w,h tensors."""
    p_xyxy = box_cxcywh_to_xyxy(pred)
    t_xyxy = box_cxcywh_to_xyxy(target)

    iou, _ = box_iou(p_xyxy, t_xyxy)

    # Enclosing box
    enc_x1 = torch.min(p_xyxy[:, 0], t_xyxy[:, 0])
    enc_y1 = torch.min(p_xyxy[:, 1], t_xyxy[:, 1])
    enc_x2 = torch.max(p_xyxy[:, 2], t_xyxy[:, 2])
    enc_y2 = torch.max(p_xyxy[:, 3], t_xyxy[:, 3])
    c2 = (enc_x2 - enc_x1).pow(2) + (enc_y2 - enc_y1).pow(2) + 1e-7

    # Centre distance
    rho2 = (pred[:, 0] - target[:, 0]).pow(2) + (pred[:, 1] - target[:, 1]).pow(2)

    # Aspect-ratio term
    v = (4 / (torch.pi ** 2)) * (
        torch.atan(target[:, 2] / target[:, 3].clamp(1e-7))
        - torch.atan(pred[:, 2] / pred[:, 3].clamp(1e-7))
    ).pow(2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + 1e-7)

    return 1 - iou + rho2 / c2 + alpha * v


def l1_loss(pred: Tensor, target: Tensor) -> Tensor:
    return torch.abs(pred - target).sum(dim=-1)
