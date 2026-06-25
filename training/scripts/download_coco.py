#!/usr/bin/env python3
"""Download and extract the COCO 2017 dataset.

Usage:
    python scripts/download_coco.py                   # → data/coco/
    python scripts/download_coco.py --dest /data/coco
    python scripts/download_coco.py --no-train        # val + annotations only
    python scripts/download_coco.py --keep-zips       # don't delete .zip after extract
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# COCO 2017 official download URLs + SHA-256 digests
# ---------------------------------------------------------------------------
ASSETS = [
    {
        "name": "train2017 images (~18 GB)",
        "url": "http://images.cocodataset.org/zips/train2017.zip",
        "sha256": None,   # skip checksum for large file (slow to hash 18 GB)
        "extract_to": "images",
        "skip_if": "images/train2017",
    },
    {
        "name": "val2017 images (~1 GB)",
        "url": "http://images.cocodataset.org/zips/val2017.zip",
        "sha256": None,
        "extract_to": "images",
        "skip_if": "images/val2017",
    },
    {
        "name": "annotations (~241 MB)",
        "url": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        "sha256": None,
        "extract_to": ".",
        "skip_if": "annotations/instances_train2017.json",
    },
]


def _progress_hook(pbar):
    last = [0]

    def update(count, block_size, total_size):
        if pbar is None:
            return
        if total_size > 0:
            pbar.total = total_size
        delta = count * block_size - last[0]
        last[0] += delta
        pbar.update(max(0, delta))

    return update


def download_file(url: str, dest_path: Path, name: str) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n  Downloading {name}")
    print(f"  {url}")
    print(f"  → {dest_path}")

    if tqdm is not None:
        with tqdm(unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc="  Progress") as pbar:
            urllib.request.urlretrieve(url, dest_path, reporthook=_progress_hook(pbar))
    else:
        urllib.request.urlretrieve(url, dest_path)

    print(f"  Saved {dest_path.stat().st_size / 1e9:.2f} GB")


def verify_sha256(path: Path, expected: str) -> bool:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest() == expected


def extract_zip(zip_path: Path, dest_dir: Path) -> None:
    print(f"  Extracting {zip_path.name} …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        if tqdm is not None:
            for member in tqdm(members, desc="  Extracting", unit="file"):
                zf.extract(member, dest_dir)
        else:
            zf.extractall(dest_dir)
    print(f"  Done → {dest_dir}")


def process_asset(asset: dict, dest: Path, keep_zips: bool) -> None:
    skip_marker = dest / asset["skip_if"]
    if skip_marker.exists():
        print(f"\n  [SKIP] {asset['name']} — already present at {skip_marker}")
        return

    zip_name = asset["url"].rsplit("/", 1)[-1]
    zip_path = dest / zip_name

    if not zip_path.exists():
        download_file(asset["url"], zip_path, asset["name"])
    else:
        print(f"\n  [CACHED] {zip_name} already downloaded, skipping download")

    if asset["sha256"]:
        print("  Verifying checksum …")
        if not verify_sha256(zip_path, asset["sha256"]):
            raise RuntimeError(f"SHA-256 mismatch for {zip_name}. Re-download and retry.")
        print("  Checksum OK")

    extract_dir = dest / asset["extract_to"]
    extract_dir.mkdir(parents=True, exist_ok=True)
    extract_zip(zip_path, extract_dir)

    if not keep_zips:
        zip_path.unlink()
        print(f"  Removed {zip_name}")


def verify_layout(dest: Path) -> None:
    expected = [
        "images/train2017",
        "images/val2017",
        "annotations/instances_train2017.json",
        "annotations/instances_val2017.json",
        "annotations/instances_train2017.json",
    ]
    ok = True
    print("\nVerifying layout:")
    for rel in expected:
        p = dest / rel
        exists = p.exists()
        status = "OK" if exists else "MISSING"
        print(f"  [{status}] {p}")
        ok = ok and exists
    if ok:
        print("\nAll files present. Dataset ready.")
        print(f"Pass --data_path {dest} to train.py.\n")
    else:
        print("\nSome files are missing — check the download log above.\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download COCO 2017 dataset")
    p.add_argument("--dest", default="data/coco", help="Destination directory (default: data/coco)")
    p.add_argument("--no-train", action="store_true", help="Skip train2017 images (~18 GB)")
    p.add_argument("--keep-zips", action="store_true", help="Keep .zip files after extraction")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dest = Path(args.dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    print(f"COCO 2017 download destination: {dest}")

    assets = ASSETS
    if args.no_train:
        assets = [a for a in assets if "train2017 images" not in a["name"]]
        print("Skipping train2017 images (--no-train)")

    for asset in assets:
        process_asset(asset, dest, keep_zips=args.keep_zips)

    verify_layout(dest)


if __name__ == "__main__":
    main()
