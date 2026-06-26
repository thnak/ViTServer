#!/usr/bin/env python3
"""Create a small COCO subset by sampling N images from an existing annotation file.

The output is a valid COCO JSON — no images are copied, the image files are
read from the same directory as the full dataset.

Usage:
    # 5 000 train + 1 000 val (default)
    python scripts/create_coco_subset.py --data_path data/coco

    # Custom size
    python scripts/create_coco_subset.py --data_path data/coco --train 2000 --val 500
"""

import argparse
import json
import random
from pathlib import Path


def sample_coco(src_json: Path, n: int, seed: int) -> dict:
    print(f"  loading {src_json} …")
    with open(src_json) as f:
        full = json.load(f)

    rng = random.Random(seed)
    images = rng.sample(full["images"], min(n, len(full["images"])))
    keep_ids = {img["id"] for img in images}
    anns = [a for a in full["annotations"] if a["image_id"] in keep_ids]

    return {
        "info": full.get("info", {}),
        "licenses": full.get("licenses", []),
        "categories": full["categories"],
        "images": images,
        "annotations": anns,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default="data/coco")
    p.add_argument("--train", type=int, default=5000, help="Train images to keep")
    p.add_argument("--val",   type=int, default=1000, help="Val images to keep")
    p.add_argument("--seed",  type=int, default=42)
    p.add_argument("--out_dir", default=None,
                   help="Output dir (default: <data_path>/annotations/)")
    args = p.parse_args()

    root = Path(args.data_path)
    ann_dir = root / "annotations"
    out_dir = Path(args.out_dir) if args.out_dir else ann_dir

    pairs = [
        (ann_dir / "instances_train2017.json", out_dir / "instances_train2017_mini.json", args.train),
        (ann_dir / "instances_val2017.json",   out_dir / "instances_val2017_mini.json",   args.val),
    ]

    for src, dst, n in pairs:
        if not src.exists():
            print(f"  skip {src.name} (not found)")
            continue
        subset = sample_coco(src, n, args.seed)
        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(dst, "w") as f:
            json.dump(subset, f)
        print(f"  {dst.name}: {len(subset['images'])} images, {len(subset['annotations'])} annotations")

    print("Done. Use configs/tpu_dev.yaml to train on the subset.")


if __name__ == "__main__":
    main()
