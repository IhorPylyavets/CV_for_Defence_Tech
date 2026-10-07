#!/usr/bin/env python3
"""
Sparse reconstruction (structure from motion) with pycolmap:

  1. feature extraction   SIFT keypoints + descriptors for every image      -> database.db
  2. feature matching     exhaustive or sequential pairs + geometric check  -> database.db
  3. mapping              incremental (COLMAP) or global (GLOMAP) SfM       -> sparse/<k>/

Output layout (one folder per run):

  data/<dataset>/runs/<run>/
  ├── params.json         the options this run was started with
  ├── database.db
  ├── snapshots/<ts>/     intermediate models written while incremental mapping registers images
  ├── sparse/0/           final model(s): cameras.bin, images.bin, points3D.bin, ...
  └── sparse/0/points3D.ply

Lines starting with "@@ " on stdout are JSON progress events, used by server.py.
COLMAP's own log (glog, stderr) passes through unchanged.

Usage:
  python reconstruct.py                                     # best quality: all images, full res, exhaustive
  python reconstruct.py --count 40 --matcher sequential     # quick run on the first 40 images
  python reconstruct.py --reuse-db                          # mapping only, from data/<dataset>/database.db
  python reconstruct.py --mapper global                     # GLOMAP-style global SfM
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections.abc import Sequence
from typing import Any

import pycolmap

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
CAMERA_MODELS = ["SIMPLE_RADIAL", "SIMPLE_PINHOLE", "PINHOLE", "RADIAL", "OPENCV"]


def event(**kw: Any) -> None:
    print("@@ " + json.dumps(kw), flush=True)


def list_images(image_dir: str) -> list[str]:
    return sorted(f for f in os.listdir(image_dir) if f.lower().endswith(IMAGE_EXT))


def select_images(names: list[str], first: int, count: int, step: int) -> list[str]:
    sel = names[first::max(step, 1)]
    return sel[:count] if count > 0 else sel


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="south-building", help="folder in data/ containing images/")
    ap.add_argument("--run", default=None, help="run name (default: timestamp)")
    g = ap.add_argument_group("images")
    g.add_argument("--first", type=int, default=0, help="index of the first image (sorted by name)")
    g.add_argument("--count", type=int, default=0, help="number of images, 0 = all")
    g.add_argument("--step", type=int, default=1, help="take every k-th image")
    g = ap.add_argument_group("features")
    g.add_argument("--reuse-db", action="store_true",
                   help="copy data/<dataset>/database.db and run only the mapper")
    g.add_argument("--camera-model", default="SIMPLE_RADIAL", choices=CAMERA_MODELS)
    g.add_argument("--camera-mode", default="single", choices=["single", "per-image", "auto"],
                   help="share one set of intrinsics between all images or not")
    g.add_argument("--max-image-size", type=int, default=3200,
                   help="images are downscaled to this for SIFT (3200 = full resolution for South Building)")
    g.add_argument("--max-features", type=int, default=8192)
    g = ap.add_argument_group("matching")
    g.add_argument("--matcher", default="exhaustive", choices=["exhaustive", "sequential"])
    g.add_argument("--overlap", type=int, default=10, help="sequential: match each image to the next N")
    g.add_argument("--loop-detection", action="store_true", help="sequential: needs a vocab tree (downloaded)")
    g.add_argument("--guided-matching", action="store_true")
    g = ap.add_argument_group("mapping")
    g.add_argument("--mapper", default="incremental", choices=["incremental", "global"])
    g.add_argument("--min-num-matches", type=int, default=15)
    g.add_argument("--no-refine-focal", action="store_true")
    g.add_argument("--no-refine-extra", action="store_true", help="keep distortion parameters fixed")
    g.add_argument("--snapshot-every", type=int, default=5,
                   help="incremental: write a snapshot every N registered images (0 = off)")
    ap.add_argument("--num-threads", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=0)
    return ap


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    ds_dir = os.path.join(DATA_DIR, args.dataset)
    image_dir = os.path.join(ds_dir, "images")
    if not os.path.isdir(image_dir):
        sys.exit(f"{image_dir} not found. Run: python download_data.py")

    run = args.run or time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(ds_dir, "runs", run)
    sparse_dir = os.path.join(run_dir, "sparse")
    snap_dir = os.path.join(run_dir, "snapshots")
    db_path = os.path.join(run_dir, "database.db")
    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)
    os.makedirs(sparse_dir)

    names = select_images(list_images(image_dir), args.first, args.count, args.step)
    if len(names) < 2:
        sys.exit("Need at least 2 images")
    with open(os.path.join(run_dir, "params.json"), "w") as f:
        json.dump({**vars(args), "run": run, "images": names, "pycolmap": pycolmap.__version__,
                   "started": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=1)
    event(stage="start", run=run, run_dir=os.path.relpath(run_dir, DATA_DIR), num_images=len(names))
    pycolmap.set_random_seed(args.seed)
    device = pycolmap.Device.auto
    t_all = time.time()

    # ---------------------------------------------------------------- 1 + 2: features and matches
    if args.reuse_db:
        src = os.path.join(ds_dir, "database.db")
        if not os.path.exists(src):
            sys.exit(f"--reuse-db: {src} does not exist")
        shutil.copy(src, db_path)
        event(stage="features", skipped=True)
        event(stage="matching", skipped=True)
    else:
        t = time.time()
        event(stage="features", total=len(names))
        ext = pycolmap.FeatureExtractionOptions()
        ext.max_image_size = args.max_image_size
        ext.num_threads = args.num_threads
        ext.sift.max_num_features = args.max_features
        reader = pycolmap.ImageReaderOptions()
        reader.camera_model = args.camera_model
        mode = {"single": pycolmap.CameraMode.SINGLE, "per-image": pycolmap.CameraMode.PER_IMAGE,
                "auto": pycolmap.CameraMode.AUTO}[args.camera_mode]
        pycolmap.extract_features(db_path, image_dir, image_names=names, camera_mode=mode,
                                  reader_options=reader, extraction_options=ext, device=device)
        event(stage="features", done=True, seconds=round(time.time() - t, 2))

        t = time.time()
        event(stage="matching", total=len(names), matcher=args.matcher)
        mopt = pycolmap.FeatureMatchingOptions()
        mopt.num_threads = args.num_threads
        mopt.guided_matching = args.guided_matching
        if args.matcher == "exhaustive":
            pycolmap.match_exhaustive(db_path, matching_options=mopt, device=device)
        else:
            popt = pycolmap.SequentialPairingOptions()
            popt.overlap = args.overlap
            popt.loop_detection = args.loop_detection
            pycolmap.match_sequential(db_path, matching_options=mopt, pairing_options=popt, device=device)
        event(stage="matching", done=True, seconds=round(time.time() - t, 2))

    db = pycolmap.Database.open(db_path)
    event(stage="database", images=db.num_images(), keypoints=db.num_keypoints(),
          matched_pairs=db.num_matched_image_pairs(), verified_pairs=db.num_verified_image_pairs(),
          inlier_matches=db.num_inlier_matches())
    db.close()

    # ---------------------------------------------------------------- 3: mapping
    t = time.time()
    event(stage="mapping", total=len(names), mapper=args.mapper)
    if args.mapper == "incremental":
        opt = pycolmap.IncrementalPipelineOptions()
        opt.image_names = names
        opt.min_num_matches = args.min_num_matches
        opt.ba_refine_focal_length = not args.no_refine_focal
        opt.ba_refine_extra_params = not args.no_refine_extra
        opt.num_threads = args.num_threads
        opt.random_seed = args.seed
        if args.snapshot_every > 0:
            os.makedirs(snap_dir)
            opt.snapshot_path = snap_dir
            opt.snapshot_frames_freq = args.snapshot_every
        n_reg = [0]

        def on_initial_pair() -> None:
            n_reg[0] = 2
            event(stage="mapping", registered=2, total=len(names), initial_pair=True)

        def on_next_image() -> None:
            n_reg[0] += 1
            event(stage="mapping", registered=n_reg[0], total=len(names))

        recs = pycolmap.incremental_mapping(db_path, image_dir, sparse_dir, options=opt,
                                            initial_image_pair_callback=on_initial_pair,
                                            next_image_callback=on_next_image)
    else:
        opt = pycolmap.GlobalPipelineOptions()
        opt.image_names = names
        opt.min_num_matches = args.min_num_matches
        opt.num_threads = args.num_threads
        opt.random_seed = args.seed
        recs = pycolmap.global_mapping(db_path, image_dir, sparse_dir, options=opt)

    if not recs:
        event(stage="mapping", done=True, failed=True, seconds=round(time.time() - t, 2))
        sys.exit("Mapping failed: no model could be initialised (try more images, more overlap or exhaustive matching)")

    models: list[dict[str, Any]] = []
    for idx, rec in sorted(recs.items(), key=lambda kv: -kv[1].num_reg_images()):
        out = os.path.join(sparse_dir, str(idx))
        os.makedirs(out, exist_ok=True)
        rec.write(out)
        rec.export_PLY(os.path.join(out, "points3D.ply"))
        models.append({"path": os.path.relpath(out, DATA_DIR), "images": rec.num_reg_images(),
                       "points": rec.num_points3D(), "error": round(rec.compute_mean_reprojection_error(), 4),
                       "track": round(rec.compute_mean_track_length(), 3)})
        print(rec.summary(), flush=True)
    event(stage="mapping", done=True, seconds=round(time.time() - t, 2))
    event(stage="finished", models=models, seconds=round(time.time() - t_all, 2))


if __name__ == "__main__":
    main()
