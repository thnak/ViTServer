#!/usr/bin/env python3
"""Generate a minimal synthetic COCO-format dataset for smoke testing.

Creates:
  <dest>/
    images/train2017/   — 8 synthetic PNG images
    images/val2017/     — 4 synthetic PNG images
    annotations/
      instances_train2017.json
      instances_val2017.json

Usage:
    python scripts/create_smoke_dataset.py                 # → data/smoke/
    python scripts/create_smoke_dataset.py --dest /tmp/sd
"""

from __future__ import annotations

import argparse
import json
import random
import struct
from pathlib import Path


def write_png(path: Path, width: int, height: int, color: tuple[int, int, int]) -> None:
    """Write a minimal solid-colour PNG using only stdlib (no Pillow/cv2)."""
    import zlib, struct as st

    def png_chunk(name: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(name + data) & 0xFFFFFFFF
        return st.pack(">I", len(data)) + name + data + st.pack(">I", crc)

    # One scanline: filter byte 0 + RGB × width
    row = bytes([0] + list(color) * width)
    raw = row * height
    idat = zlib.compress(raw)

    png = (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", st.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + png_chunk(b"IDAT", idat)
        + png_chunk(b"IEND", b"")
    )
    path.write_bytes(png)


def make_split(
    dest: Path,
    split: str,
    n_images: int,
    n_classes: int,
    img_w: int = 100,
    img_h: int = 100,
) -> None:
    img_dir = dest / "images" / split
    ann_dir = dest / "annotations"
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(42 if split == "train2017" else 7)
    palette = [(200, 50, 50), (50, 200, 50), (50, 50, 200), (200, 200, 50)]

    categories = [{"id": i + 1, "name": f"class_{i}", "supercategory": "object"} for i in range(n_classes)]
    images = []
    annotations = []
    ann_id = 1

    for img_idx in range(n_images):
        img_id = img_idx + 1
        fname = f"{img_id:012d}.png"
        color = palette[img_idx % len(palette)]
        write_png(img_dir / fname, img_w, img_h, color)

        images.append({"id": img_id, "file_name": fname, "width": img_w, "height": img_h})

        # 1–3 boxes per image
        for _ in range(rng.randint(1, 3)):
            x = rng.randint(0, img_w - 20)
            y = rng.randint(0, img_h - 20)
            w = rng.randint(10, min(40, img_w - x))
            h = rng.randint(10, min(40, img_h - y))
            cat_id = rng.randint(1, n_classes)
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0,
            })
            ann_id += 1

    coco = {"info": {}, "licenses": [], "categories": categories, "images": images, "annotations": annotations}
    ann_name = split.replace("2017", "") + "2017"  # e.g. train2017
    out_file = ann_dir / f"instances_{ann_name}.json"
    out_file.write_text(json.dumps(coco, indent=2))
    print(f"  [{split}] {n_images} images, {ann_id - 1} annotations → {out_file}")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic COCO smoke dataset")
    p.add_argument("--dest", default="data/smoke", help="Output root (default: data/smoke)")
    p.add_argument("--classes", type=int, default=3, help="Number of categories (default: 3)")
    args = p.parse_args()

    dest = Path(args.dest).resolve()
    print(f"Creating smoke dataset at: {dest}")
    make_split(dest, "train2017", n_images=8, n_classes=args.classes)
    make_split(dest, "val2017", n_images=4, n_classes=args.classes)
    print(f"\nDone. Use with:")
    print(f"  python train.py --config configs/smoke_test.yaml --data_path {dest} --device cpu --no-val")


if __name__ == "__main__":
    main()
