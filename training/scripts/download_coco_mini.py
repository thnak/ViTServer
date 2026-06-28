#!/usr/bin/env python3
"""Download COCO val2017 and split it into a small train / val dataset.

Total download: ~1.2 GB (val2017 images + annotations).
No COCO train2017 (~18 GB) is needed.

The output mirrors the layout expected by train.py:
  <dest>/
    images/
      train2017/   ← first --train images from val2017
      val2017/     ← remaining --val images from val2017  (symlinks on Linux/Mac;
                                                           copies on Windows)
    annotations/
      instances_train2017.json
      instances_val2017.json

Usage (Colab / local):
    python scripts/download_coco_mini.py                        # 4000 train / 1000 val
    python scripts/download_coco_mini.py --train 500 --val 100  # tiny quick-check
    python scripts/download_coco_mini.py --dest /content/coco_mini
    python scripts/download_coco_mini.py --keep-zips

Then train:
    python train.py --config configs/coco_mini.yaml --data_path <dest> --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

try:
    from tqdm import tqdm as _tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# Official COCO 2017 URLs
_VAL_IMAGES_URL  = "http://images.cocodataset.org/zips/val2017.zip"
_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


# ── download helpers ──────────────────────────────────────────────────────────

def _reporthook(pbar):
    last = [0]
    def hook(count, block, total):
        if HAS_TQDM and pbar is not None:
            if total > 0:
                pbar.total = total
            delta = count * block - last[0]
            last[0] += delta
            pbar.update(max(0, delta))
    return hook


def _download(url: str, dest: Path, label: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nDownloading {label}")
    print(f"  {url}")
    print(f"  → {dest}")
    if HAS_TQDM:
        with _tqdm(unit="B", unit_scale=True, unit_divisor=1024, miniters=1) as pbar:
            urllib.request.urlretrieve(url, dest, reporthook=_reporthook(pbar))
    else:
        urllib.request.urlretrieve(url, dest)
    print(f"  {dest.stat().st_size / 1e9:.2f} GB saved")


def _extract(zip_path: Path, out_dir: Path) -> None:
    print(f"Extracting {zip_path.name} …")
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.namelist()
        if HAS_TQDM:
            for m in _tqdm(members, unit="file"):
                zf.extract(m, out_dir)
        else:
            zf.extractall(out_dir)
    print(f"  → {out_dir}")


# ── annotation split ──────────────────────────────────────────────────────────

def _split_annotations(src_json: Path, n_train: int, n_val: int, seed: int = 42) -> tuple[dict, dict]:
    """Split a COCO annotation file into train and val dicts."""
    import random
    with open(src_json) as f:
        full = json.load(f)

    rng = random.Random(seed)
    images = full["images"][:]
    rng.shuffle(images)

    total = len(images)
    n_train = min(n_train, total)
    n_val   = min(n_val,   total - n_train)

    train_imgs = images[:n_train]
    val_imgs   = images[n_train : n_train + n_val]

    def _subset(imgs):
        ids = {img["id"] for img in imgs}
        return {
            "info":        full.get("info", {}),
            "licenses":    full.get("licenses", []),
            "categories":  full["categories"],
            "images":      imgs,
            "annotations": [a for a in full["annotations"] if a["image_id"] in ids],
        }

    return _subset(train_imgs), _subset(val_imgs)


# ── image layout ──────────────────────────────────────────────────────────────

def _link_or_copy(src: Path, dst: Path) -> None:
    """Symlink on POSIX, copy on Windows (or when symlinks not supported)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    try:
        dst.symlink_to(src)
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)


def _build_image_dirs(all_images_dir: Path, dest: Path, train_ann: dict, val_ann: dict) -> None:
    """Create train2017/ and val2017/ under dest/images/ pointing at the right files."""
    train_dir = dest / "images" / "train2017"
    val_dir   = dest / "images" / "val2017"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    print(f"Linking {len(train_ann['images'])} train images …")
    for img in train_ann["images"]:
        src = all_images_dir / img["file_name"]
        _link_or_copy(src.resolve(), train_dir / img["file_name"])

    print(f"Linking {len(val_ann['images'])} val images …")
    for img in val_ann["images"]:
        src = all_images_dir / img["file_name"]
        _link_or_copy(src.resolve(), val_dir / img["file_name"])


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download COCO val2017 mini dataset")
    p.add_argument("--dest",       default="data/coco_mini",
                   help="Output root (default: data/coco_mini)")
    p.add_argument("--train",      type=int, default=4000,
                   help="Images to use for training (default: 4000)")
    p.add_argument("--val",        type=int, default=1000,
                   help="Images to use for validation (default: 1000)")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--keep-zips",  action="store_true",
                   help="Keep .zip files after extraction")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dest = Path(args.dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    print(f"COCO mini destination: {dest}")
    print(f"Split: {args.train} train  +  {args.val} val  (from 5 000 val2017 images)")

    # ── step 1: download ──────────────────────────────────────────────────────
    tmp = dest / "_download"
    tmp.mkdir(exist_ok=True)

    val_zip = tmp / "val2017.zip"
    ann_zip = tmp / "annotations_trainval2017.zip"

    if not (dest / "images" / "val2017_all").exists():
        if not val_zip.exists():
            _download(_VAL_IMAGES_URL, val_zip, "val2017 images (~1 GB)")
        _extract(val_zip, tmp)
        (dest / "images").mkdir(parents=True, exist_ok=True)
        shutil.move(str(tmp / "val2017"), str(dest / "images" / "val2017_all"))
        if not args.keep_zips:
            val_zip.unlink()
    else:
        print("\n[SKIP] val2017 images already extracted")

    all_images_dir = dest / "images" / "val2017_all"

    raw_ann = dest / "annotations" / "_instances_val2017_raw.json"
    if not raw_ann.exists():
        if not ann_zip.exists():
            _download(_ANNOTATIONS_URL, ann_zip, "annotations (~241 MB)")
        _extract(ann_zip, tmp)
        ann_src = tmp / "annotations" / "instances_val2017.json"
        raw_ann.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(ann_src), str(raw_ann))
        # clean up unneeded annotation files
        shutil.rmtree(tmp / "annotations", ignore_errors=True)
        if not args.keep_zips:
            ann_zip.unlink()
    else:
        print("\n[SKIP] annotations already extracted")

    # ── step 2: split annotations ─────────────────────────────────────────────
    print(f"\nSplitting annotations (seed={args.seed}) …")
    train_ann, val_ann = _split_annotations(raw_ann, args.train, args.val, args.seed)

    ann_dir = dest / "annotations"
    train_json = ann_dir / "instances_train2017.json"
    val_json   = ann_dir / "instances_val2017.json"
    with open(train_json, "w") as f:
        json.dump(train_ann, f)
    with open(val_json, "w") as f:
        json.dump(val_ann, f)
    print(f"  train: {len(train_ann['images'])} images, {len(train_ann['annotations'])} annotations")
    print(f"  val:   {len(val_ann['images'])} images,   {len(val_ann['annotations'])} annotations")

    # ── step 3: image directories ─────────────────────────────────────────────
    print()
    _build_image_dirs(all_images_dir, dest, train_ann, val_ann)

    # ── step 4: verify ────────────────────────────────────────────────────────
    checks = [
        dest / "images"      / "train2017",
        dest / "images"      / "val2017",
        dest / "annotations" / "instances_train2017.json",
        dest / "annotations" / "instances_val2017.json",
    ]
    print("\nLayout check:")
    ok = True
    for p in checks:
        exists = p.exists()
        print(f"  {'OK    ' if exists else 'MISS  '} {p}")
        ok = ok and exists

    if ok:
        print(f"""
Dataset ready at: {dest}

Train:
  python train.py --config configs/coco_mini.yaml \\
                  --data_path {dest} \\
                  --device cuda

Quick 500-image run:
  python scripts/download_coco_mini.py --train 500 --val 100 --dest data/coco_tiny
""")
    else:
        print("\nSome paths are missing — check the log above.")


if __name__ == "__main__":
    main()
