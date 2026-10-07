# 03-image-3dgs-fitting: interactive 2D Gaussian splatting

A browser-based, real-time take on [bobarna/tiny-2d-gaussian-splatting](https://github.com/bobarna/tiny-2d-gaussian-splatting).
It fits N anisotropic 2D Gaussians to an image and lets you watch and tweak the optimisation live.
There's no PyTorch: the forward pass, the analytic backward pass and Adam are written in plain
JavaScript and run in a Web Worker.

```bash
cd 03-image-3dgs-fitting
python serve.py          # opens http://127.0.0.1:8001  (any static server works; a server
                         # is needed only because browsers block module workers on file://)
```

**Requirements:** a browser with WebGL2 and any Python 3 (`serve.py` uses only the standard library).
No packages are needed; see the [root README](../README.md#setup) for the general setup.

## Files

| file | what it does |
|---|---|
| `engine.js` | the model: tile-binned rasteriser, analytic gradients, Adam, clone/split/prune |
| `worker.js` | training loop in a Web Worker, streams parameters and images to the UI |
| `gl_render.js` | WebGL2 splat renderer for display, at screen resolution |
| `app.js` | UI: controls, overlays, hover inspection, PSNR chart, export |
| `images.js` | procedural sample targets (no image files needed) |
| `gradcheck.html` | checks the analytic gradients against finite differences |

## Model

Each Gaussian has a centre μ, log-scales log σx, log σy, rotation θ, an RGB colour c and an
opacity logit (used only in alpha mode). Σ = R(θ) diag(σ²) R(θ)ᵀ and Gᵢ(p) = exp(−½ (p−μ)ᵀ Σ⁻¹ (p−μ)).

* **Additive** (the reference notebook): I(p) = Σᵢ cᵢ Gᵢ(p)
* **Alpha (3DGS)**: I(p) = Σᵢ cᵢ αᵢ Tᵢ + T_N·bg, with αᵢ = min(0.99, oᵢ Gᵢ(p)) and Tᵢ = Πⱼ₍ⱼ<ᵢ₎ (1−αⱼ).
  Gaussian index acts as depth.

Differences from the notebook:
* Gaussians are evaluated within 3.5σ through 16×16 tile binning, as in 3DGS, instead of densely
  for every pixel. That is what makes it real-time.
* σ is optimised in log space, and parameters are in units of image width, so the training
  resolution can change mid-run.
* Learning rates are set per parameter group (the notebook has this as a commented-out `lr_dict`).
* Optional adaptive density control, which the notebook mentions as a TODO: every k iterations the
  Gaussians with the largest mean positional gradient are split (if σ is above the threshold) or
  cloned. Transparent or colourless ones are pruned.

`gradcheck.html` compares every analytic gradient with central differences; the median relative error is about 1e-5.

## UI

* **Target**: procedural samples, upload, drag-and-drop or paste, or a **live webcam** feed, which the
  Gaussians keep tracking. Training resolution (48–512 px) can be changed while training.
* **Model**: additive or alpha compositing, background, N, initialisation (random as in the notebook,
  colours sampled from the target, or a grid), initial σ, initial opacity, random rotation.
* **Optimiser**: L1, L2 or a mix; global and per-group learning rates, each with a freeze checkbox;
  exponential LR decay; stop-at-iteration.
* **Density control**: interval, growth per event, max N, split threshold, prune threshold.
* **Views**:
  * *Render*: WebGL at screen resolution. Gaussians are continuous, so this is sharper than the training resolution.
  * *Splats*: each Gaussian drawn as a flat 1σ ellipse.
  * *Error*: |I − target|.
  * *Exact*: the CPU render the loss actually sees.
* **Overlays**: ellipse outlines at k·σ, as in the notebook's plots, and centres.
* **Hover** over the reconstruction to see the Gaussian contributing most to that pixel (μ, σ, θ, colour, opacity, weight).
* PSNR and N chart, densification log, export of Gaussians as JSON, and save PNG.

Keys: **Space** train/pause · **S** step · **R** reset.
