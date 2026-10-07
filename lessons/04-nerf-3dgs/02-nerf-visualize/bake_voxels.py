#!/usr/bin/env python3
"""
Bake a pretrained NeRF (krrish94/nerf-pytorch checkpoints) into a dense voxel grid.

For every voxel centre we run the MLP once and store
  * sigma  - volume density (raw, after ReLU), float16
  * colour - view-dependent RGB, fitted to real spherical harmonics (degree 0..2), float16

The viewer then does plain alpha compositing through the grid with trilinear
sampling - no network evaluation at render time.

Note: the pretrained checkpoints in the repo do NOT match the config.yml shipped
next to them (e.g. lego's config says 8x256, the weights are 4x128 without skip
connections). Therefore the architecture is inferred from the state-dict shapes.

Usage:
  python bake_voxels.py                              # bake all scenes in nerf-pytorch/pretrained
  python bake_voxels.py --scenes lego chair --res 128 --sh-degree 1
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import time
from typing import Any, Literal, TypedDict, cast

import numpy as np
import torch

SceneType = Literal["blender", "llff", "deepvoxels"]


class SceneConfig(TypedDict):
    type: SceneType
    white_background: bool
    near: float
    far: float


HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PRETRAINED = os.path.join(HERE, "nerf-pytorch", "pretrained")
DEFAULT_OUT = os.path.join(HERE, "viewer", "scenes")


# --------------------------------------------------------------------------- #
# Checkpoint / config loading
# --------------------------------------------------------------------------- #
def read_config(path: str) -> SceneConfig:
    """Minimal reader for the few config values we need (avoids a pyyaml dependency)."""
    cfg: SceneConfig = {"type": "blender", "white_background": True, "near": 2.0, "far": 6.0}
    if not os.path.exists(path):
        return cfg
    text = open(path).read()
    m = re.search(r"^\s*type:\s*(blender|llff|deepvoxels)\s*$", text, re.M)
    if m:
        cfg["type"] = cast(SceneType, m.group(1))
    m = re.search(r"^\s*white_background:\s*(\w+)", text, re.M)
    if m:
        cfg["white_background"] = m.group(1).lower() == "true"
    for key in ("near", "far"):
        m = re.search(rf"^\s*{key}:\s*([-\d.eE]+)", text, re.M)
        if m:
            cfg[key] = float(m.group(1))
    return cfg


def load_checkpoint(path: str) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # very old torch without weights_only
        return torch.load(path, map_location="cpu")


class BakedNeRF(torch.nn.Module):
    """Re-implementation of nerf-pytorch's FlexibleNeRFModel (with use_viewdirs=True),
    built from a state dict, so that the repo itself does not need to be importable
    (its forward() also references a non-existent self.linear_layers on skip layers)."""

    def __init__(self, sd: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.hidden: int = sd["layer1.weight"].shape[0]
        self.dim_xyz: int = sd["layer1.weight"].shape[1]
        n_xyz = len([k for k in sd if re.match(r"layers_xyz\.\d+\.weight", k)])
        if "fc_alpha.weight" not in sd:
            raise ValueError("Only checkpoints trained with use_viewdirs=True are supported")
        self.dim_dir: int = sd["layers_dir.0.weight"].shape[1] - self.hidden

        self.layer1 = torch.nn.Linear(self.dim_xyz, self.hidden)
        self.layers_xyz = torch.nn.ModuleList()
        self.skip: list[bool] = []
        for i in range(n_xyz):
            in_dim = sd[f"layers_xyz.{i}.weight"].shape[1]
            self.skip.append(in_dim != self.hidden)  # skip layer takes concat(x, xyz_enc)
            self.layers_xyz.append(torch.nn.Linear(in_dim, self.hidden))
        self.fc_alpha = torch.nn.Linear(self.hidden, 1)
        self.fc_feat = torch.nn.Linear(self.hidden, self.hidden)
        self.layers_dir = torch.nn.ModuleList(
            [torch.nn.Linear(self.dim_dir + self.hidden, self.hidden // 2)]
        )
        self.fc_rgb = torch.nn.Linear(self.hidden // 2, 3)
        self.load_state_dict(sd)

        self.include_input_xyz, self.n_freq_xyz = infer_encoding(self.dim_xyz)
        self.include_input_dir, self.n_freq_dir = infer_encoding(self.dim_dir)

    def describe(self) -> str:
        return (f"{1 + len(self.layers_xyz)} x {self.hidden} trunk, "
                f"skip={any(self.skip)}, PE xyz L={self.n_freq_xyz}, dir L={self.n_freq_dir}")

    def geometry(self, pts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """pts (P,3) -> sigma (P,), feat (P,hidden)"""
        enc = positional_encoding(pts, self.n_freq_xyz, self.include_input_xyz)
        x = self.layer1(enc)
        # NB: the original forward has no activation after layer1, keep it identical.
        for layer, skip in zip(self.layers_xyz, self.skip):
            if skip:
                x = torch.cat((x, enc), dim=-1)
            x = torch.relu(layer(x))
        sigma = torch.relu(self.fc_alpha(x)[..., 0])
        feat = torch.relu(self.fc_feat(x))
        return sigma, feat

    def macs_per_sample(self) -> int:
        """Multiply-accumulates of one full MLP query (what the grid saves per sample)."""
        return sum(m.weight.numel() for m in self.modules() if isinstance(m, torch.nn.Linear))

    def split_dir_layer(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """layers_dir.0 acts on concat(feat, dir_enc): W = [W_feat | W_dir].
        Splitting it lets us evaluate many view directions per point cheaply."""
        w = self.layers_dir[0].weight
        return w[:, : self.hidden], w[:, self.hidden:], self.layers_dir[0].bias


def infer_encoding(dim: int) -> tuple[bool, int]:
    """dim = 3*include_input + 6*L"""
    if (dim - 3) % 6 == 0:
        return True, (dim - 3) // 6
    if dim % 6 == 0:
        return False, dim // 6
    raise ValueError(f"Cannot infer positional encoding from input dim {dim}")


def positional_encoding(x: torch.Tensor, n_freq: int, include_input: bool = True) -> torch.Tensor:
    # Matches nerf_helpers.positional_encoding with log_sampling=True:
    # [x, sin(2^0 x), cos(2^0 x), sin(2^1 x), cos(2^1 x), ...]
    out: list[torch.Tensor] = [x] if include_input else []
    for i in range(n_freq):
        f = 2.0 ** i
        out.append(torch.sin(x * f))
        out.append(torch.cos(x * f))
    return torch.cat(out, dim=-1)


# --------------------------------------------------------------------------- #
# Spherical harmonics
# --------------------------------------------------------------------------- #
def sh_basis(d: torch.Tensor, degree: int) -> torch.Tensor:
    """Real SH basis (same constants/order as Plenoxels / 3DGS). d: (K,3) unit vectors."""
    x, y, z = d[:, 0], d[:, 1], d[:, 2]
    out = [torch.full_like(x, 0.28209479177387814)]
    if degree >= 1:
        c1 = 0.4886025119029199
        out += [-c1 * y, c1 * z, -c1 * x]
    if degree >= 2:
        out += [
            1.0925484305920792 * x * y,
            -1.0925484305920792 * y * z,
            0.31539156525252005 * (2.0 * z * z - x * x - y * y),
            -1.0925484305920792 * x * z,
            0.5462742152960396 * (x * x - y * y),
        ]
    return torch.stack(out, dim=-1)  # (K, n_coef)


def fibonacci_sphere(n: int) -> torch.Tensor:
    i = torch.arange(n, dtype=torch.float32) + 0.5
    phi = torch.acos(1.0 - 2.0 * i / n)
    theta = math.pi * (1.0 + 5.0 ** 0.5) * i
    return torch.stack(
        [torch.cos(theta) * torch.sin(phi), torch.sin(theta) * torch.sin(phi), torch.cos(phi)], -1
    ).float()


def fitting_directions(scene_type: SceneType, n: int) -> torch.Tensor:
    d = fibonacci_sphere(n * 4 if scene_type == "llff" else n)
    if scene_type == "llff":
        # Forward-facing capture: cameras look roughly along world -z.
        d = d[d[:, 2] < -0.5][:n]
    return d


# --------------------------------------------------------------------------- #
# Baking
# --------------------------------------------------------------------------- #
def grid_points(bbox_min: torch.Tensor, bbox_max: torch.Tensor, res: int, device: torch.device | str) -> torch.Tensor:
    """Voxel centres, ordered [z][y][x] (x fastest) to match WebGL texImage3D."""
    axes = [
        bbox_min[a] + (torch.arange(res, device=device, dtype=torch.float32) + 0.5)
        * (bbox_max[a] - bbox_min[a]) / res
        for a in range(3)
    ]
    zz, yy, xx = torch.meshgrid(axes[2], axes[1], axes[0], indexing="ij")
    return torch.stack([xx, yy, zz], -1).reshape(-1, 3)


@torch.no_grad()
def query_sigma(model: BakedNeRF, pts: torch.Tensor, chunk: int) -> torch.Tensor:
    return torch.cat([model.geometry(pts[i:i + chunk])[0] for i in range(0, len(pts), chunk)])


@torch.no_grad()
def find_bbox(model: BakedNeRF, search_min: torch.Tensor, search_max: torch.Tensor, threshold: float,
              device: torch.device | str, chunk: int, res: int = 96, pad: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    pts = grid_points(search_min, search_max, res, device)
    sigma = query_sigma(model, pts, chunk).reshape(res, res, res)  # [z,y,x]
    occ = (sigma > threshold).nonzero()
    if len(occ) == 0:
        return search_min, search_max
    lo = occ.min(0).values.flip(0).float()  # -> x,y,z
    hi = occ.max(0).values.flip(0).float() + 1
    size = (search_max - search_min) / res
    bmin = torch.maximum(search_min, search_min + (lo - pad) * size)
    bmax = torch.minimum(search_max, search_min + (hi + pad) * size)
    return bmin, bmax


@torch.no_grad()
def bake(model: BakedNeRF, bbox_min: torch.Tensor, bbox_max: torch.Tensor, res: int, dirs: torch.Tensor, sh_degree: int,
         supersample: int, device: torch.device | str, chunk: int) -> tuple[np.ndarray, np.ndarray]:
    n_vox = res ** 3
    w_feat, w_dir, b_dir = model.split_dir_layer()
    dirs = dirs.to(device)
    dir_enc = positional_encoding(dirs, model.n_freq_dir, model.include_input_dir)
    dir_term = dir_enc @ w_dir.T + b_dir  # (K, hidden/2) - precomputed once per direction

    basis = sh_basis(dirs, sh_degree).float()  # (K, C)
    # ridge-regularised least squares projector: coef = P @ rgb_k
    lam = 1e-3
    proj = torch.linalg.solve(
        basis.T @ basis + lam * torch.eye(basis.shape[1], dtype=torch.float32, device=device), basis.T
    ).float().to(device)  # (C, K)

    voxel = (bbox_max - bbox_min) / res
    offsets = torch.zeros(1, 3, device=device)
    if supersample > 1:
        s = (torch.arange(supersample, device=device) + 0.5) / supersample - 0.5
        offsets = torch.stack(torch.meshgrid(s, s, s, indexing="ij"), -1).reshape(-1, 3) * voxel

    pts_all = grid_points(bbox_min, bbox_max, res, device)
    sigma_out = torch.empty(n_vox, dtype=torch.float16)
    sh_out = torch.empty(basis.shape[1], 3, n_vox, dtype=torch.float16)

    step = max(256, chunk // (len(offsets) * max(1, len(dirs) // 4)))
    t0 = time.time()
    for i in range(0, n_vox, step):
        p = pts_all[i:i + step]
        P = len(p)
        sub = (p[:, None, :] + offsets[None]).reshape(-1, 3)
        sigma, feat = model.geometry(sub)
        sigma = sigma.reshape(P, -1).mean(1)
        feat = feat.reshape(P, -1, feat.shape[-1]).mean(1)

        h = torch.relu((feat @ w_feat.T)[:, None, :] + dir_term[None])  # (P, K, hidden/2)
        rgb = torch.sigmoid(model.fc_rgb(h))  # (P, K, 3)
        coef = torch.einsum("ck,pkd->cdp", proj, rgb)  # (C, 3, P)

        sigma_out[i:i + P] = sigma.clamp(max=60000).half().cpu()
        sh_out[:, :, i:i + P] = coef.half().cpu()
        done = (i + P) / n_vox
        el = time.time() - t0
        print(f"\r    baking {done * 100:5.1f}%  {el:5.1f}s  (eta {el / done - el:5.1f}s)", end="")
    print()
    return sigma_out.numpy(), sh_out.numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pretrained", default=DEFAULT_PRETRAINED, help="folder with <scene>-lowres/ subfolders")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--scenes", nargs="*", default=None, help="e.g. lego chair (default: all)")
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--sh-degree", type=int, default=2, choices=[0, 1, 2],
                    help="0 = diffuse colour, 1/2 = view-dependent colour (bigger files)")
    ap.add_argument("--n-dirs", type=int, default=64, help="view directions used to fit SH")
    ap.add_argument("--model", choices=["fine", "coarse"], default="fine")
    ap.add_argument("--supersample", type=int, default=1, help="NxNxN MLP samples averaged per voxel")
    ap.add_argument("--bbox", type=float, nargs=6, default=None, metavar=("X0", "Y0", "Z0", "X1", "Y1", "Z1"),
                    help="manual bounding box (otherwise auto-fitted to the occupied region)")
    ap.add_argument("--bbox-threshold", type=float, default=10.0, help="sigma threshold for auto bbox")
    ap.add_argument("--chunk", type=int, default=1 << 18)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    if args.device == "auto":
        device: str = "cuda" if torch.cuda.is_available() else (
            "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    else:
        device = args.device
    print(f"device: {device}")
    os.makedirs(args.out, exist_ok=True)

    scene_dirs = sorted(glob.glob(os.path.join(args.pretrained, "*")))
    scene_dirs = [d for d in scene_dirs if glob.glob(os.path.join(d, "*.ckpt"))]
    for sdir in scene_dirs:
        name = os.path.basename(sdir).replace("-lowres", "")
        if args.scenes and name not in args.scenes:
            continue
        ckpt_path = sorted(glob.glob(os.path.join(sdir, "*.ckpt")))[-1]
        cfg = read_config(os.path.join(sdir, "config.yml"))
        print(f"\n== {name}  ({cfg['type']})  {os.path.relpath(ckpt_path, HERE)}")

        ckpt = load_checkpoint(ckpt_path)
        sd = ckpt.get(f"model_{args.model}_state_dict") or ckpt["model_coarse_state_dict"]
        model = BakedNeRF(sd).to(device).eval()
        print(f"    model: {model.describe()}")

        if cfg["type"] == "llff":
            # LLFF scenes are trained in normalised device coordinates: the scene lives in [-1,1]^3
            search_min, search_max = torch.tensor([-1.0, -1.0, -1.0]), torch.tensor([1.0, 1.0, 1.0])
        else:
            search_min, search_max = torch.tensor([-1.5, -1.5, -1.5]), torch.tensor([1.5, 1.5, 1.5])
        search_min, search_max = search_min.to(device), search_max.to(device)

        if args.bbox:
            bmin = torch.tensor(args.bbox[:3], device=device)
            bmax = torch.tensor(args.bbox[3:], device=device)
        elif cfg["type"] == "llff":
            bmin, bmax = search_min, search_max
        else:
            bmin, bmax = find_bbox(model, search_min, search_max, args.bbox_threshold, device, args.chunk)
        print(f"    bbox: {bmin.cpu().numpy().round(3)} -> {bmax.cpu().numpy().round(3)}")

        dirs = fitting_directions(cfg["type"], args.n_dirs)
        sigma, sh = bake(model, bmin, bmax, args.res, dirs, args.sh_degree, args.supersample, device, args.chunk)

        out_dir = os.path.join(args.out, name)
        os.makedirs(out_dir, exist_ok=True)
        sigma.tofile(os.path.join(out_dir, "density.f16"))
        sh.tofile(os.path.join(out_dir, "sh.f16"))  # layout [coef][rgb][z][y][x]

        voxel_size = ((bmax - bmin) / args.res).cpu().tolist()
        meta: dict[str, Any] = {
            "name": name,
            "type": cfg["type"],
            "resolution": [args.res] * 3,
            "bbox_min": bmin.cpu().tolist(),
            "bbox_max": bmax.cpu().tolist(),
            "voxel_size": voxel_size,
            "sh_degree": args.sh_degree,
            "white_background": cfg["white_background"],
            "up": [0, 1, 0] if cfg["type"] == "llff" else [0, 0, 1],
            # LLFF models live in NDC space: rays there are parallel to +z (orthographic camera),
            # the x axis is mirrored w.r.t. world, and view directions were world-space (-z forward).
            "ndc": cfg["type"] == "llff",
            "sigma_max": float(sigma.max()),
            "sigma_p99": float(np.percentile(sigma[sigma > 0.1], 99)) if (sigma > 0.1).any() else 1.0,
            "occupancy": float((sigma > args.bbox_threshold).mean()),
            "model": model.describe(),
            "mlp_macs_per_sample": model.macs_per_sample(),
            "checkpoint": os.path.relpath(ckpt_path, HERE),
            "psnr": float(ckpt["psnr"]) if "psnr" in ckpt else None,
            "iter": int(ckpt["iter"]) if "iter" in ckpt else None,
            "files": {"density": "density.f16", "sh": "sh.f16"},
        }
        json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=2)
        mb = (sigma.nbytes + sh.nbytes) / 2 ** 20
        print(f"    saved {out_dir}  ({mb:.1f} MB, occupancy {meta['occupancy'] * 100:.1f}%)")

    # index for the viewer
    scenes = sorted(d for d in os.listdir(args.out) if os.path.exists(os.path.join(args.out, d, "meta.json")))
    json.dump({"scenes": scenes}, open(os.path.join(args.out, "index.json"), "w"), indent=2)
    print(f"\nindex: {scenes}")


if __name__ == "__main__":
    main()
