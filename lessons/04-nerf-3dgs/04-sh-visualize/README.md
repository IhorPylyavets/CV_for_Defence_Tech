# 04-sh-visualize: spherical harmonics explorer

A small interactive page that shows how 3DGS (and the SH colour in [`02-nerf-visualize/`](../02-nerf-visualize/README.md)) stores
**view-dependent colour** with real spherical harmonics. Plain WebGL2 and no dependencies.

```bash
xdg-open 04-sh-visualize/index.html      # or just open it in a browser. No server and no build step needed.
```

## Files

| file | what it does |
|---|---|
| `index.html` | layout |
| `app.js` | SH basis (JS + GLSL), sphere mesh, the two views, UI |
| `style.css` | styles (same look as [`03-image-3dgs-fitting/`](../03-image-3dgs-fitting/README.md)) |

## What it shows

Colour is a function of the view direction d (a unit vector):

    c(d) = max(0, 0.5 + Σ_k f_k Y_k(d)),   k = 0 … (deg+1)² − 1,   one f_k per RGB channel

with the same constants, order and signs as 3DGS (`utils/sh_utils.py`) and `02-nerf-visualize/bake_voxels.py`.

* **Left view**: c(d) painted on the unit sphere of directions (*Sphere*), or the same thing with
  radius = brightness (*Lobe*), so you can see which directions are bright.
* **Right view**: the 16 basis functions Y_l^m for l = 0…3, painted on the unit sphere with a diverging
  colour map (orange positive, blue negative, light gray zero, normalised per function). The dark lines are
  the nodal lines where Y = 0: band l has l of them. Bands above the current degree are dimmed. Click one to edit its coefficient.
* **Degree 0–3**: truncates the expansion. Degree 0 is a single constant colour (diffuse). Degree 3 is
  what 3DGS uses: 16 coefficients × RGB = 48 floats per Gaussian (`f_dc` 3 + `f_rest` 45).
* **Coefficient editor**: R/G/B sliders for the selected f_k, and a table of all 16 (click a row to select it).
* **Presets**: gray, diffuse only, lit from above (l = 1, 2 along z), x→R y→G z→B (the three l = 1
  functions are just −y, z, −x), a band-limited "specular" highlight, and random.
* **What a 3DGS camera sees**: 3DGS gives each Gaussian a single colour per frame, c(d) with
  d = normalize(centre − camera). The swatch shows that colour for the current orbit camera next to
  the DC-only colour, so you can see view dependence while you orbit.

Mouse: drag to orbit (both views share the camera), wheel to zoom the left view. Axes: x red, y green, z blue, z up.
