#!/usr/bin/env python3
"""Quick smoke test for the ViTServer HTTP API.

Usage:
    python scripts/test_client.py                      # uses localhost:8080
    python scripts/test_client.py --port 9000
    python scripts/test_client.py --image /path/to.jpg
"""

import argparse
import struct
import sys
import time
import urllib.request
import urllib.error

def check_health(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=5) as r:
            body = r.read().decode()
            print(f"[health] {r.status} {body}")
            return r.status == 200
    except urllib.error.URLError as e:
        print(f"[health] FAILED: {e}")
        return False

def infer_image(base_url: str, image_path: str) -> None:
    with open(image_path, "rb") as f:
        data = f.read()
    req = urllib.request.Request(
        f"{base_url}/infer",
        data=data,
        method="POST",
        headers={"Content-Type": "application/octet-stream"},
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            body = r.read().decode()
            elapsed = (time.perf_counter() - t0) * 1000
            print(f"[infer ] {r.status} ({elapsed:.1f} ms)")
            print(f"         {body}")
    except urllib.error.HTTPError as e:
        print(f"[infer ] HTTP {e.code}: {e.read().decode()}")
    except urllib.error.URLError as e:
        print(f"[infer ] FAILED: {e}")

def make_synthetic_bgr(w: int = 320, h: int = 240) -> bytes:
    """Create a minimal JPEG-encoded BGR image using OpenCV if available, else skip."""
    try:
        import cv2
        import numpy as np
        img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        ok, buf = cv2.imencode(".jpg", img)
        if ok:
            return bytes(buf)
    except ImportError:
        pass
    return b""

def main() -> None:
    p = argparse.ArgumentParser(description="ViTServer smoke test")
    p.add_argument("--host",  default="localhost")
    p.add_argument("--port",  default=8080, type=int)
    p.add_argument("--image", default="", help="Path to a JPEG/PNG image (optional)")
    args = p.parse_args()

    base = f"http://{args.host}:{args.port}"
    print(f"Target: {base}\n")

    # 1. Health check
    ok = check_health(base)
    if not ok:
        sys.exit(1)

    # 2. Infer
    image_path = args.image
    if not image_path:
        print("[infer ] No --image given; generating a synthetic BGR frame...")
        img_bytes = make_synthetic_bgr()
        if not img_bytes:
            print("[infer ] OpenCV not available and no --image provided — skipping infer test")
            return
        # Write to a temp file then read it back (urllib needs a file path or bytes)
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(img_bytes)
            image_path = tmp.name
        try:
            infer_image(base, image_path)
        finally:
            os.unlink(image_path)
    else:
        infer_image(base, image_path)

if __name__ == "__main__":
    main()
