"""Sigmoid Focal Loss for multi-label classification."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def sigmoid_focal_loss(
    logits: Tensor,
    targets: Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "sum",
) -> Tensor:
    """
    Args:
        logits:  [N, C] raw (un-sigmoid'd) scores
        targets: [N, C] binary targets {0, 1}
    """
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * (1 - p_t) ** gamma * ce

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss
