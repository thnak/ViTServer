"""Detection-safe augmentation pipeline using albumentations."""

from __future__ import annotations

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


def build_train_transforms(img_size: int = 1280) -> A.Compose:
    return A.Compose(
        [
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT, fill=114),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.7),
            A.GaussNoise(p=0.2),
            A.Blur(blur_limit=3, p=0.1),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="coco", label_fields=["labels"], min_visibility=0.1),
    )


def build_val_transforms(img_size: int = 1280) -> A.Compose:
    return A.Compose(
        [
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT, fill=114),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="coco", label_fields=["labels"], min_visibility=0.0),
    )
