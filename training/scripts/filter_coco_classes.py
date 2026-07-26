#!/usr/bin/env python3
"""Filter COCO mini annotations to keep only N classes.

Creates a new annotation directory with filtered train/val JSONs,
reusing the same images from coco_mini.

Usage:
    python scripts/filter_coco_classes.py --data_path data/coco_mini --classes 1 3 2
"""

import argparse
import json
from pathlib import Path
from collections import Counter


def filter_split(src_json: Path, dst_json: Path, keep_cat_ids: set[int], split_name: str):
    print(f"  reading {src_json.name} …")
    with open(src_json) as f:
        data = json.load(f)

    # Filter categories
    old_cats = data["categories"]
    keep_cats = [c for c in old_cats if c["id"] in keep_cat_ids]
    # Map old cat_id → contiguous index (starting at 1 for COCO convention)
    old_to_new = {c["id"]: i + 1 for i, c in enumerate(keep_cats)}
    new_cats = [{"id": old_to_new[c["id"]], "name": c["name"], "supercategory": c.get("supercategory", "object")}
                for c in keep_cats]

    # Filter annotations
    anns = [a for a in data["annotations"] if a["category_id"] in keep_cat_ids]
    for a in anns:
        a["category_id"] = old_to_new[a["category_id"]]

    # Keep only images that have at least one annotation after filtering
    keep_img_ids = {a["image_id"] for a in anns}
    images = [img for img in data["images"] if img["id"] in keep_img_ids]

    # Report
    cat_counts = Counter(a["category_id"] for a in anns)
    cat_name = {c["id"]: c["name"] for c in new_cats}
    print(f"  [{split_name}] {len(images)} images, {len(anns)} annotations")
    for cid, cnt in cat_counts.most_common():
        print(f"      {cat_name[cid]}: {cnt}")

    out = {
        "info": data.get("info", {}),
        "licenses": data.get("licenses", []),
        "categories": new_cats,
        "images": images,
        "annotations": anns,
    }
    dst_json.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_json, "w") as f:
        json.dump(out, f)
    print(f"  wrote {dst_json}")


def main():
    p = argparse.ArgumentParser("Filter COCO annotations to N classes")
    p.add_argument("--data_path", default="data/coco_mini")
    p.add_argument("--classes", nargs="+", type=int,
                   default=[1, 3, 2],
                   help="COCO category IDs to keep (default: person=1, car=3, bicycle=2)")
    p.add_argument("--out_dir", default=None,
                   help="Output dir (default: <data_path>/annotations_3cls/)")
    args = p.parse_args()

    root = Path(args.data_path)
    ann_dir = root / "annotations"
    out_dir = Path(args.out_dir) if args.out_dir else root / "annotations_3cls"
    keep_ids = set(args.classes)

    pairs = [
        (ann_dir / "instances_train2017.json", out_dir / "instances_train2017.json", "train"),
        (ann_dir / "instances_val2017.json", out_dir / "instances_val2017.json", "val"),
    ]
    for src, dst, name in pairs:
        if not src.exists():
            print(f"  skip {src.name} (not found)")
            continue
        filter_split(src, dst, keep_ids, name)

    print(f"\nDone. Use with --data_path {root} and update config to point to:")
    print(f"  train_ann: annotations_3cls/instances_train2017.json")
    print(f"  val_ann:   annotations_3cls/instances_val2017.json")


if __name__ == "__main__":
    main()
