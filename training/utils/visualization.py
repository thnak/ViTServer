"""Simple box visualisation utilities."""

from __future__ import annotations

import numpy as np
import cv2
from torch import Tensor


PALETTE = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
    (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
    (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
    (52, 69, 147), (100, 115, 255), (0, 24, 236), (132, 56, 255),
    (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 199),
]


def draw_boxes(
    image: np.ndarray,
    boxes: Tensor,      # [N, 4] x1,y1,x2,y2 pixel coords
    labels: Tensor,     # [N] int
    scores: Tensor,     # [N] float
    class_names: list[str] | None = None,
    score_thresh: float = 0.3,
) -> np.ndarray:
    img = image.copy()
    for box, label, score in zip(boxes.tolist(), labels.tolist(), scores.tolist()):
        if score < score_thresh:
            continue
        x1, y1, x2, y2 = map(int, box)
        color = PALETTE[int(label) % len(PALETTE)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        name = class_names[label] if class_names else str(label)
        cv2.putText(
            img, f"{name} {score:.2f}",
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
        )
    return img
