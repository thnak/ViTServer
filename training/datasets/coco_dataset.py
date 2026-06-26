"""COCO-format detection dataset."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from pycocotools.coco import COCO

from .transforms import build_train_transforms, build_val_transforms


class CocoDetection(Dataset):
    def __init__(
        self,
        img_dir: str,
        ann_file: str,
        img_size: int = 1280,
        train: bool = True,
    ) -> None:
        self.img_dir = Path(img_dir)
        self.coco = COCO(ann_file)
        self.ids = sorted(self.coco.imgs.keys())
        self.transforms = (
            build_train_transforms(img_size) if train else build_val_transforms(img_size)
        )
        self.img_size = img_size
        # Map category IDs to contiguous indices
        cats = sorted(self.coco.cats.keys())
        self.cat2idx = {c: i for i, c in enumerate(cats)}

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> tuple[Tensor, dict]:
        img_id = self.ids[idx]
        info = self.coco.imgs[img_id]
        path = self.img_dir / info["file_name"]
        img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)

        # Drop degenerate boxes (zero w or h) — albumentations rejects them
        anns = [a for a in anns if a["bbox"][2] > 0 and a["bbox"][3] > 0]
        bboxes = [a["bbox"] for a in anns]          # [x,y,w,h] pixel, COCO fmt
        labels = [self.cat2idx[a["category_id"]] for a in anns]

        result = self.transforms(image=img, bboxes=bboxes, labels=labels)
        image = result["image"]                     # [3, H, W] float32 tensor

        # Convert pixel COCO boxes → normalised cx,cy,w,h
        out_boxes: list[list[float]] = []
        for bx, by, bw, bh in result["bboxes"]:
            cx = (bx + bw / 2) / self.img_size
            cy = (by + bh / 2) / self.img_size
            nw = bw / self.img_size
            nh = bh / self.img_size
            out_boxes.append([cx, cy, nw, nh])

        target = {
            "boxes": torch.as_tensor(out_boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(result["labels"], dtype=torch.long),
            "image_id": torch.tensor(img_id),
            "orig_size": torch.tensor([h, w]),
        }
        return image, target


MAX_TARGETS = 100  # COCO max ≈ 93; must match _MAX_T in losses/hungarian.py


def collate_fn(batch: list) -> tuple[Tensor, list[dict]]:
    images = []
    targets = []
    for img, t in batch:
        images.append(img)
        M = len(t["labels"])
        n = min(M, MAX_TARGETS)
        labels = torch.zeros(MAX_TARGETS, dtype=torch.long)
        boxes  = torch.zeros(MAX_TARGETS, 4)
        valid  = torch.zeros(MAX_TARGETS, dtype=torch.bool)
        if n > 0:
            labels[:n] = t["labels"][:n]
            boxes[:n]  = t["boxes"][:n]
            valid[:n]  = True
        targets.append({
            "image_id":  t["image_id"],
            "orig_size": t["orig_size"],
            "labels": labels,
            "boxes":  boxes,
            "valid":  valid,
        })
    return torch.stack(images), targets


def build_dataloader(
    img_dir: str,
    ann_file: str,
    img_size: int = 1280,
    batch_size: int = 4,
    num_workers: int = 8,
    train: bool = True,
    pin_memory: bool = True,
) -> DataLoader:
    dataset = CocoDetection(img_dir, ann_file, img_size, train)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=train,
    )
