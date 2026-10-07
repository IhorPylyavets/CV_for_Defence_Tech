# rd-nerf-3dgs
Materials for Lecture 4

| project | what it is | needs |
|---|---|---|
| [`01-colmap-visualize/`](01-colmap-visualize/README.md) | sparse SfM with pycolmap on COLMAP's South Building, with a browser UI to run, replay and inspect reconstructions | `pycolmap`, `numpy`, `pillow` |
| [`02-nerf-visualize/`](02-nerf-visualize/README.md) | pretrained NeRFs baked into voxel grids and rendered by alpha compositing in WebGL2 | `torch`, `numpy`, `pillow` (bake / validate only) |
| [`03-image-3dgs-fitting/`](03-image-3dgs-fitting/README.md) | interactive 2D Gaussian splatting: fits Gaussians to an image, trained in the browser | nothing (browser + stdlib) |
| [`04-sh-visualize/`](04-sh-visualize/README.md) | spherical harmonics basis and view-dependent SH colour, as used by 3DGS | nothing (open `index.html`) |

## Setup

All projects share one Python environment, created once at the repo root. The per-folder READMEs assume
it is already active.

### Requirements

* **git**, to clone the repo together with its submodule.
* **Python 3.11+** (3.14 recommended). It must be a version that has PyPI wheels for `pycolmap` and `torch`.
* **A browser with WebGL2**: a recent Chrome, Firefox or Edge, or Safari 15+. All viewers run in the browser.
* Disk: about 420 MB for the South Building dataset (`01-colmap-visualize`) plus about 2 GB for the
  PyTorch install.
* No GPU is needed. `pycolmap` from PyPI is CPU-only. The NeRF bake runs on CPU and uses CUDA or MPS
  automatically when available.

The `nerf-pytorch` submodule is not imported, so its own requirements (`torchsearchsorted`, `tensorboard`, ...) are not needed.

### 1. Clone

```bash
git clone --recurse-submodules <this repo>     # 02-nerf-visualize/nerf-pytorch is a git submodule
cd rd-nerf-3dgs
# or, in an existing clone:
git submodule update --init
```

The submodule contains the pretrained NeRF checkpoints used by `02-nerf-visualize`.

### 2. Create the environment

With conda:

```bash
conda create -n rd-nerf-3dgs python=3.14
conda activate rd-nerf-3dgs
pip install -r requirements.txt
```

Or with venv:

```bash
python3 -m venv .venv
source .venv/bin/activate                      # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

`pip install torch` installs the default build for your platform. That is a CUDA build on Linux, which is
large. The CPU build is enough here (a 128³ bake takes about 20–40 s). To install it instead:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 3. Check

```bash
python -c "import numpy, torch, PIL, pycolmap; print('torch', torch.__version__, '| pycolmap', pycolmap.__version__)"
```

Then follow the README in each project folder. Activate the environment in every new terminal first.
