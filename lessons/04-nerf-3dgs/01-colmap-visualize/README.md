# COLMAP Sparse Explorer

Run structure from motion with [pycolmap](https://github.com/colmap/pycolmap) on COLMAP's
[South Building](https://github.com/colmap/colmap/releases/download/3.11.1/south-building.zip) sample, and
inspect the result in the browser: camera poses, the sparse point cloud, keypoints and point tracks.
You can also open the reference sparse model that ships with the dataset without computing anything.

```
01-colmap-visualize/
├── download_data.py   downloads + unpacks south-building.zip into data/   (stdlib only)
├── reconstruct.py     features -> matching -> incremental / global mapping (pycolmap), also a CLI
├── server.py          serves the viewer, reads models with pycolmap, runs reconstruct.py as a job
└── viewer/            WebGL2 app (index.html, app.js, style.css)

data/south-building/               (git-ignored)
├── images/                        128 photos, 3072×2304, one camera
├── sparse/                        reference model made by COLMAP (text format)
├── database.db                    reference features + verified matches
└── runs/<run>/                    your reconstructions
    ├── params.json, log.txt, database.db
    ├── snapshots/<ts>/            partial models written during incremental mapping
    └── sparse/0/                  final model (.bin) + points3D.ply
```

## Run

Set up the shared environment first (see the [root README](../README.md#setup)), then:

```bash
cd 01-colmap-visualize
python download_data.py                # ≈420 MB; the UI also has a button for this
python server.py                       # opens http://127.0.0.1:8002
```

The CLI works without the UI. Each run is written to `data/<dataset>/runs/<run>/`:

```bash
python reconstruct.py --count 30 --matcher sequential --max-image-size 1000   # ~15 s on 16 CPU cores
python reconstruct.py                                  # best quality: all images, full res, exhaustive (~16 min)
python reconstruct.py --reuse-db                       # mapping only, from the shipped database.db (~2.5 min)
python reconstruct.py --mapper global --count 40       # GLOMAP-style global SfM
python reconstruct.py --help
```

The PyPI pycolmap wheels are CPU-only. SIFT extraction, matching and bundle adjustment all run on the CPU.

## UI

* **Data**: choose the dataset and a model: the *reference* model shipped with the dataset, or one of your runs.
  For runs that wrote snapshots, **Mapping replay** steps through the incremental mapper image by image.
  Runs can be deleted from here.
* **Sparse reconstruction**: choose a preset and press *Run*. *What will run* lists the exact settings.

  | preset | what it does | time (16-core CPU) |
  |---|---|---|
  | **Best quality** (default) | all images, SIFT at full resolution (≤ 8192 features), exhaustive matching | ~16 min |
  | Quick preview | every 2nd image, SIFT at 1600 px, exhaustive matching | ~3 min |
  | Mapping only | reuses the features and matches in the dataset's `database.db` | ~3 min |

  All presets use one shared SIMPLE_RADIAL camera (the photos come from one camera) and the incremental mapper,
  which refines focal length and distortion. That is COLMAP's standard, most robust setup. More options
  (sequential matching, global mapper, camera models, ...) are on the CLI: `python reconstruct.py --help`.

  The job runs in a subprocess, so the UI stays responsive and the job can be cancelled. Progress bars are parsed from
  COLMAP's log. With **Live preview**, each new snapshot is loaded while mapping runs. The finished model is opened
  automatically.
* **Display**: colour by RGB, reprojection error, track length or height. Filter by max error and min track length.
  Set point and camera size, and toggle image planes (textured frustums), the camera path and the ground grid.
  *Align up axis* rotates the scene so the average camera "up" points up. COLMAP's own frame is arbitrary.
* **Inspect**
  * Click a **camera** to see its photo with keypoints (green = triangulated). Its 3D points are highlighted.
    **V** flies into that camera with matching intrinsics and blends the photo over the render (*Photo overlay*), so
    you can check the reprojection by eye. **← / →** step through the images.
  * Click a **3D point**, or a green keypoint in the photo, to see its track: rays to every camera that observes it,
    and the image patch around the keypoint in each of those images.
* **Statistics**: registered images, points, observations, mean track length and reprojection error, plus
  histograms of track length and error. The current filters are shown in the histograms.

Mouse: drag to orbit · right/shift-drag to pan · wheel to zoom · double-click to orbit around a point. Keys: **R** reset,
**A** auto-rotate, **C** cameras on/off, **Esc** clear the selection.

## Notes

* Developed against pycolmap 4.2 (COLMAP with rigs/frames). `server.py` handles `cam_from_world` both as a property
  (≤ 3.11) and as a method (≥ 3.12). `reconstruct.py` uses the 4.x option names (`FeatureExtractionOptions`, `snapshot_frames_freq`, ...).
* The shipped `sparse/` is in COLMAP's old text format. pycolmap 4 reads it fine. The shipped `database.db` also
  works with the current mapper (`--reuse-db`), which gives a model very close to the reference (128/128 images, ~61k points, 0.51 px).
* Keypoint coordinates are always in original image pixels, even when SIFT ran on a downscaled image (`--max-image-size`).
* Thumbnails for the inspector are cached in `data/<dataset>/.thumbs/`. Without Pillow, full-size images are sent instead.
