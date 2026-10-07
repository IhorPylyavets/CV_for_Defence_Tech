#!/usr/bin/env python3
"""
Sanity check for a bake: render the same camera with
  (a) the original MLP  (fine network, dense samples, i.e. real NeRF inference)
  (b) the baked voxel grid (trilinear lookups + the viewer's alpha compositing)
and report PSNR between them plus timing. Writes viewer/scenes/<scene>/validate.png.

  python validate_bake.py --scene lego
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from bake_voxels import BakedNeRF, DEFAULT_OUT, DEFAULT_PRETRAINED, load_checkpoint, positional_encoding


def camera_rays(H: int, W: int, fov_x: float, radius: float, yaw: float, pitch: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Blender-style orbit camera looking at the origin, z up."""
    pos = torch.tensor([math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch)]) * radius
    fwd = -pos / pos.norm()
    right = torch.linalg.cross(fwd, torch.tensor([0.0, 0.0, 1.0]))
    right = right / right.norm()
    up = torch.linalg.cross(right, fwd)
    tan = math.tan(fov_x / 2)
    j, i = torch.meshgrid(torch.arange(H) + 0.5, torch.arange(W) + 0.5, indexing="ij")
    x = (i / W * 2 - 1) * tan
    y = (1 - j / H * 2) * tan * H / W
    d = fwd + x[..., None] * right + y[..., None] * up
    d = d / d.norm(dim=-1, keepdim=True)
    return pos.expand_as(d).reshape(-1, 3), d.reshape(-1, 3)


def composite(sigma: torch.Tensor, rgb: torch.Tensor, dists: torch.Tensor, bg: float) -> torch.Tensor:
    alpha = 1 - torch.exp(-sigma * dists)
    T = torch.cumprod(torch.cat([torch.ones_like(alpha[:, :1]), 1 - alpha + 1e-10], -1), -1)[:, :-1]
    w = T * alpha
    return (w[..., None] * rgb).sum(1) + (1 - w.sum(1, keepdim=True)) * bg


@torch.no_grad()
def render_mlp(model: BakedNeRF, ro: torch.Tensor, rd: torch.Tensor, near: float, far: float, n: int, bg: float,
               chunk: int = 4096) -> torch.Tensor:
    out: list[torch.Tensor] = []
    t = torch.linspace(near, far, n)
    for i in range(0, len(ro), chunk):
        o, d = ro[i:i + chunk], rd[i:i + chunk]
        pts = (o[:, None] + d[:, None] * t[None, :, None]).reshape(-1, 3)
        sigma, feat = model.geometry(pts)
        denc = positional_encoding(d, model.n_freq_dir, model.include_input_dir)
        denc = denc[:, None].expand(-1, n, -1).reshape(-1, denc.shape[-1])
        h = torch.relu(model.layers_dir[0](torch.cat([feat, denc], -1)))
        rgb = torch.sigmoid(model.fc_rgb(h))
        dists = torch.full((len(o), n), (far - near) / (n - 1))
        out.append(composite(sigma.reshape(-1, n), rgb.reshape(-1, n, 3), dists, bg))
    return torch.cat(out)


@torch.no_grad()
def render_grid(meta: dict[str, Any], sigma: torch.Tensor, sh: torch.Tensor, ro: torch.Tensor, rd: torch.Tensor, n: int,
                bg: float, chunk: int = 8192) -> torch.Tensor:
    bmin, bmax = torch.tensor(meta["bbox_min"]), torch.tensor(meta["bbox_max"])
    res = meta["resolution"]
    vol = torch.cat([sigma[None], sh.reshape(-1, *sigma.shape)], 0)[None]  # (1, 1+3C, z, y, x)
    dt = (bmax - bmin).norm().item() / n
    out: list[torch.Tensor] = []
    for i in range(0, len(ro), chunk):
        o, d = ro[i:i + chunk], rd[i:i + chunk]
        ta, tb = (bmin - o) / d, (bmax - o) / d
        t0 = torch.minimum(ta, tb).max(-1).values.clamp(min=0)
        t1 = torch.maximum(ta, tb).min(-1).values
        t = t0[:, None] + dt * (torch.arange(n) + 0.5)[None]
        valid = (t <= t1[:, None]).float()
        pts = o[:, None] + d[:, None] * t[..., None]
        g = ((pts - bmin) / (bmax - bmin)) * 2 - 1  # align_corners=False matches GL texel centres
        v = F.grid_sample(vol, g[None, None], mode="bilinear", padding_mode="border", align_corners=False)
        v = v[0, :, 0]  # (C, rays, n)
        s = v[0] * valid
        x, y, z = d[:, 0:1], d[:, 1:2], d[:, 2:3]
        Y: list[torch.Tensor] = [torch.full_like(x, 0.28209479)]
        if meta["sh_degree"] >= 1:
            Y += [-0.48860251 * y, 0.48860251 * z, -0.48860251 * x]
        if meta["sh_degree"] >= 2:
            Y += [1.09254843 * x * y, -1.09254843 * y * z, 0.31539157 * (2 * z * z - x * x - y * y),
                  -1.09254843 * x * z, 0.54627422 * (x * x - y * y)]
        coef = v[1:].reshape(len(Y), 3, len(o), n)
        rgb = sum(Yk[None] * coef[k] for k, Yk in enumerate(Y)).clamp(0, 1).permute(1, 2, 0)
        out.append(composite(s, rgb, torch.full_like(s, dt), bg))
    return torch.cat(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="lego")
    ap.add_argument("--size", type=int, default=200)
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--yaw", type=float, default=-135.0)
    ap.add_argument("--pitch", type=float, default=30.0)
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    sdir = os.path.join(DEFAULT_OUT, args.scene)
    meta: dict[str, Any] = json.load(open(os.path.join(sdir, "meta.json")))
    if meta.get("ndc"):
        raise SystemExit("validation camera is only implemented for blender (360°) scenes")
    R = meta["resolution"]
    sigma = torch.from_numpy(np.fromfile(os.path.join(sdir, "density.f16"), np.float16).astype(np.float32)).reshape(R[2], R[1], R[0])
    sh = torch.from_numpy(np.fromfile(os.path.join(sdir, "sh.f16"), np.float16).astype(np.float32))

    ckpt = sorted(glob.glob(os.path.join(DEFAULT_PRETRAINED, f"{args.scene}-lowres", "*.ckpt")))[-1]
    model = BakedNeRF(load_checkpoint(ckpt)["model_fine_state_dict"]).eval()

    H = W = args.size
    ro, rd = camera_rays(H, W, 0.6911112070083618, 4.0311, math.radians(args.yaw), math.radians(args.pitch))
    bg = 1.0 if meta["white_background"] else 0.0

    t = time.time(); img_mlp = render_mlp(model, ro, rd, 2.0, 6.0, args.samples, bg); t_mlp = time.time() - t
    t = time.time(); img_grid = render_grid(meta, sigma, sh, ro, rd, args.samples, bg); t_grid = time.time() - t
    mse = ((img_mlp - img_grid) ** 2).mean().item()
    print(f"{args.scene}: PSNR(grid vs MLP) = {-10 * math.log10(mse):.2f} dB   "
          f"MLP {t_mlp:.2f}s, grid {t_grid:.2f}s (both CPU torch, {H}x{W}, {args.samples} samples/ray)")
    try:
        from PIL import Image
        both = torch.cat([img_mlp.reshape(H, W, 3), img_grid.reshape(H, W, 3)], 1)
        out = os.path.join(sdir, "validate.png")
        Image.fromarray((both.clamp(0, 1).numpy() * 255).astype(np.uint8)).save(out)
        print(f"left: MLP, right: voxel grid -> {out}")
    except ImportError:
        print("(install pillow to also write a comparison image)")


if __name__ == "__main__":
    main()
