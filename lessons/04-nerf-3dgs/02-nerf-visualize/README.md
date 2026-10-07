# NeRF Voxel Explorer

This project bakes the pretrained NeRFs from `nerf-pytorch/pretrained` (from [krrish94/nerf-pytorch](https://github.com/krrish94/nerf-pytorch))
into 128³ voxel grids of density σ and colour, then renders them interactively in the browser by alpha
compositing. The MLP is never called at render time.

```
02-nerf-visualize/
├── nerf-pytorch/        git submodule (checkpoints in pretrained/*-lowres/), see the root README to fetch it
├── bake_voxels.py       MLP -> voxel grid (σ + spherical-harmonics colour)
├── validate_bake.py     renders one view with the MLP and with the grid, prints PSNR
├── serve.py             tiny static server for the viewer (stdlib only)
└── viewer/              WebGL2 app (index.html, app.js, style.css)
    └── scenes/<name>/   meta.json, density.f16, sh.f16   <- written by bake_voxels.py
```

## Run

Set up the shared environment and fetch the `nerf-pytorch` submodule first (see the [root README](../README.md#setup)), then:

```bash
cd 02-nerf-visualize
python bake_voxels.py                 # all 7 scenes, ~20 s each on CPU (GPU is used if available)
python bake_voxels.py --scenes lego --res 128 --sh-degree 2 --supersample 2   # options
python validate_bake.py --scene lego  # optional: grid vs MLP comparison
python serve.py                       # opens http://127.0.0.1:8000
```

## What is stored

For every voxel centre the fine MLP is evaluated once:

* **σ**: `relu(fc_alpha(...))`, stored as float16 (`density.f16`, `[z][y][x]`, x fastest).
* **colour**: depends on view direction, so it is evaluated for 64 directions and least-squares fitted with
  real spherical harmonics (degree 1 by default, so 4 coefficients × RGB). Stored as `sh.f16`, `[coef][rgb][z][y][x]`.
  Degree 0 gives plain diffuse colour and a 4× smaller file.

The viewer uploads these as 3D half-float textures and, per pixel, marches the ray through the grid box with
step δ = diagonal / samples:

    α_i = 1 − exp(−σ_i δ),  T_i = Π_{j<i}(1 − α_j),  C = Σ T_i α_i c_i + T_N · C_bg

These are the same equations as `nerf/volume_rendering_utils.py`.

## Viewer features

* Orbit / pan / zoom camera (perspective or orthographic), auto-rotate.
* Render modes: RGB, expected depth, opacity (1 − T), max-σ "X-ray".
* Knobs: samples per ray, σ scale, σ cutoff (removes floaters), resolution, FOV, view-dependent colour on/off, jitter.
* **Voxel slice**: an X/Y/Z plane showing the raw stored σ with voxel grid lines.
* **Ray inspector**: click any pixel to plot σ, α, T and w = T·α for every sample along its ray, see the colour
  strips c_i and w_i·c_i, and the final composited colour. Orbit afterwards to see the ray and its weighted
  samples in 3D.

## Notes / findings

* The pretrained checkpoints **do not match the `config.yml` next to them**. For example, lego's config says
  8 layers × 256, but every checkpoint is 4 × 128 without skip connections (fern uses 6 PE frequencies instead
  of 10). `bake_voxels.py` therefore builds the network from the state-dict shapes. It re-implements the forward
  pass and does not import the repo, so none of the repo's requirements (torchsearchsorted etc.) are needed.
* The bounding box is fitted automatically to where σ > 10 (on a 96³ pre-pass), so the 128 voxels cover the
  object rather than the whole [-1.5, 1.5]³ cube. Hotdog and ship stay at full size because their training
  cameras never see below the object, so the MLP fills that region with density.
* **fern** is an LLFF scene trained in NDC space. It is baked in the NDC cube [-1,1]³, where rays are parallel,
  so the viewer uses an orthographic camera for it.
* Measured grid-vs-MLP PSNR on the 200×200 test view: lego 27.3, materials 28.1, chair 26.4, drums 25.8,
  hotdog 24.9, ship 22.3 dB. The loss comes from 128³ resolution and the SH fit.
