#!/usr/bin/env python3
"""
Download and unpack the COLMAP "South Building" sample dataset (stdlib only).

  data/south-building/
  ├── images/       128 JPEGs, 3072x2304, one camera
  ├── sparse/       reference sparse model made by COLMAP (cameras/images/points3D.txt)
  └── database.db   features + verified matches (can be reused to run only the mapper)

Usage:
  python download_data.py                 # -> data/south-building
  python download_data.py --keep-zip      # keep data/south-building.zip after extracting
  python download_data.py --url <zip>     # any zip with <name>/images[/sparse] inside
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
import urllib.request
import zipfile

SOUTH_BUILDING_URL = "https://github.com/colmap/colmap/releases/download/3.11.1/south-building.zip"

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")


def download(url: str, dst: str) -> None:
    tmp = dst + ".part"
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done, t0, last = 0, time.time(), 0.0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            now = time.time()
            if now - last > 0.5 or done == total:
                last = now
                mb, speed = done / 1e6, done / 1e6 / max(now - t0, 1e-6)
                if total:
                    print(f"  {mb:7.1f} / {total / 1e6:.1f} MB  ({100 * done / total:5.1f} %)  {speed:5.1f} MB/s", flush=True)
                else:
                    print(f"  {mb:7.1f} MB  {speed:5.1f} MB/s", flush=True)
    os.replace(tmp, dst)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=SOUTH_BUILDING_URL)
    ap.add_argument("--out", default=DATA_DIR, help="directory the zip is extracted into")
    ap.add_argument("--keep-zip", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-download even if the dataset exists")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    zip_path = os.path.join(args.out, os.path.basename(args.url))
    name = os.path.splitext(os.path.basename(args.url))[0]
    target = os.path.join(args.out, name)

    if os.path.isdir(os.path.join(target, "images")) and not args.force:
        print(f"{target} already exists (use --force to re-download)")
        return

    if not os.path.exists(zip_path) or args.force:
        print(f"Downloading {args.url}", flush=True)
        download(args.url, zip_path)
    print(f"Extracting {zip_path}", flush=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(args.out)
    if not args.keep_zip:
        os.remove(zip_path)

    images = os.path.join(target, "images")
    if not os.path.isdir(images):
        sys.exit(f"Expected {images} after extracting, check the archive layout")
    # The release zip ships macOS metadata folders in some versions.
    shutil.rmtree(os.path.join(args.out, "__MACOSX"), ignore_errors=True)
    print(f"Done: {target} ({len(os.listdir(images))} images)")


if __name__ == "__main__":
    main()
