"""
dino_features — бібліотека для Демо 1 (Заняття 5).

DINOv2 / DINOv2 + registers / DINOv3 на VisDrone-VID без жодної мітки:
  1. PCA патч-фіч, норми токенів і карта уваги [CLS]       (слайди 17, 26–28)
  2. подібність патчів за кліком: one-shot-локалізація      (слайди 26, 22)
  3. label efficiency: kNN / linear probe на кропах, випадковий vs кластерний добір міток (слайд 30)
  4. архів як актив: карта сцен, дублікати й витік даних при розбитті train/test

Усі вектори L2-нормалізовані: скалярний добуток = косинус.
"""
from __future__ import annotations

import hashlib
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import aerial_search as A   # discover_visdrone, make_synthetic_visdrone, шляхи репозиторію

REPO_ROOT = A.REPO_ROOT
VISDRONE_DIR = A.VISDRONE_DIR
MODELS_DIR = A.ARTIFACTS / "models" / "dino"    # tools/fetch_dino_models.py
HF_HOME = MODELS_DIR / "hf_home"

# --------------------------------------------------------------------------------------
# Зоопарк моделей: усі через transformers.AutoModel, тож інтерфейс один.
# --------------------------------------------------------------------------------------
DINO_ZOO: Dict[str, Dict[str, str]] = {
    # DINOv2 (Oquab et al., 2023): patch 14, вивчені абсолютні позиції (інтерполюються під кадр), Apache 2.0.
    "dinov2": {"small": "facebook/dinov2-small", "base": "facebook/dinov2-base", "large": "facebook/dinov2-large"},
    # DINOv2 + 4 register-токени (Darcet et al., 2024): прибирають високонормні артефакти на фоні.
    "dinov2reg": {"small": "facebook/dinov2-with-registers-small", "base": "facebook/dinov2-with-registers-base",
                  "large": "facebook/dinov2-with-registers-large"},
    # DINOv3 (Siméoni et al., 2025): patch 16, RoPE, 4 регістри, Gram anchoring. Gated, DINOv3 License.
    "dinov3": {"small": "facebook/dinov3-vits16-pretrain-lvd1689m", "base": "facebook/dinov3-vitb16-pretrain-lvd1689m",
               "large": "facebook/dinov3-vitl16-pretrain-lvd1689m"},
    # DINOv3, навчений на супутникових знімках SAT-493M. Лише ViT-L (і 7B).
    "dinov3sat": {"large": "facebook/dinov3-vitl16-pretrain-sat493m"},
}
GATED = {"dinov3", "dinov3sat"}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Категорії VisDrone: 0 ignored, 1 pedestrian, 2 people, 3 bicycle, 4 car, 5 van, 6 truck,
# 7 tricycle, 8 awning-tricycle, 9 bus, 10 motor, 11 others.
CROP_CLASSES: Dict[str, List[int]] = {
    "pedestrian": [1], "car": [4], "van": [5], "truck": [6], "bus": [9], "motor": [10],
}
ANN_COLS = ["frame", "tid", "x", "y", "w", "h", "score", "cat", "trunc", "occ"]


def repo_id(name: str, size: str) -> str:
    if name not in DINO_ZOO:
        raise KeyError(f"невідома модель {name!r}; є: {list(DINO_ZOO)}")
    if size not in DINO_ZOO[name]:
        raise KeyError(f"{name} має лише розміри {list(DINO_ZOO[name])}")
    return DINO_ZOO[name][size]


def pick_device(device: Optional[str] = None) -> str:
    """"cuda" | "cpu" | None (авто). Якщо попросили cuda, а її немає, — попереджаємо й падаємо на cpu."""
    import torch

    if device in (None, "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        warnings.warn("DEVICE='cuda', але CUDA недоступна (torch без CUDA або немає драйвера) — працюємо на CPU")
        return "cpu"
    return device


def _tiny_config(name: str):
    """Крихітна архітектура з випадковими вагами — лише для smoke-тесту без мережі."""
    import transformers as T

    kw = dict(hidden_size=64, num_hidden_layers=2, num_attention_heads=4, intermediate_size=128)
    if name == "dinov2":
        return T.Dinov2Config(patch_size=14, **kw)
    if name == "dinov2reg":
        return T.Dinov2WithRegistersConfig(patch_size=14, num_register_tokens=4, **kw)
    return T.DINOv3ViTConfig(patch_size=16, num_register_tokens=4, **kw)


class DinoEncoder:
    """
    Єдиний інтерфейс до DINOv2 / DINOv2-reg / DINOv3.
    Вихід трансформера: [CLS, регістри..., патчі] — регістри відкидаємо, патчі складаємо в сітку h×w.
    """

    def __init__(self, name: str, size: str = "base", device: Optional[str] = None,
                 random_weights: bool = False):
        import torch
        import transformers as T

        self.name, self.size = name, size
        self.repo = repo_id(name, size)
        self.device = pick_device(device)
        self.random_weights = random_weights
        t0 = time.time()
        if random_weights:
            self.model = T.AutoModel.from_config(_tiny_config(name))
            mean, std = IMAGENET_MEAN, IMAGENET_STD
        else:
            self.model = T.AutoModel.from_pretrained(self.repo)
            # Нормалізацію беремо з препроцесора моделі: у dinov3sat вона своя, не ImageNet.
            proc = T.AutoImageProcessor.from_pretrained(self.repo)
            mean, std = tuple(proc.image_mean), tuple(proc.image_std)
        self.load_s = time.time() - t0
        self.model.eval().to(self.device)
        cfg = self.model.config
        self.patch = int(cfg.patch_size if isinstance(cfg.patch_size, int) else cfg.patch_size[0])
        self.n_reg = int(getattr(cfg, "num_register_tokens", 0) or 0)
        self.prefix = 1 + self.n_reg
        self.dim = int(cfg.hidden_size)
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self._mean = torch.tensor(mean).view(1, 3, 1, 1)
        self._std = torch.tensor(std).view(1, 3, 1, 1)
        self.img_per_s: Optional[float] = None

    # ------------------------------------------------------------------ допоміжне
    def grid_size(self, w: int, h: int, max_side: int) -> Tuple[int, int]:
        """Розмір входу (W, H), кратний патчу, з довшою стороною ≈ max_side (слайд 17: 1280×720 -> 1274×714 для P=14)."""
        s = max_side / max(w, h)
        p = self.patch
        return max(p, int(round(w * s / p)) * p), max(p, int(round(h * s / p)) * p)

    def _tensor(self, ims: Sequence[Image.Image], W: int, H: int):
        import torch

        arr = np.stack([np.asarray(im.convert("RGB").resize((W, H), Image.BICUBIC)) for im in ims])
        x = torch.from_numpy(arr).permute(0, 3, 1, 2).float().div_(255)
        return ((x - self._mean) / self._std).to(self.device)

    def _autocast(self):
        import contextlib

        import torch

        if self.device.startswith("cuda"):
            return torch.autocast("cuda", dtype=torch.float16)
        return contextlib.nullcontext()

    def _forward(self, x, attentions: bool = False):
        import torch

        with torch.inference_mode(), self._autocast():
            if not attentions:
                return self.model(pixel_values=x)
            # sdpa не віддає ваги уваги: на один прохід перемикаємось на eager.
            self.model.set_attn_implementation("eager")
            try:
                return self.model(pixel_values=x, output_attentions=True)
            finally:
                self.model.set_attn_implementation("sdpa")

    # ------------------------------------------------------------------ щільні фічі
    def features(self, im: Image.Image, max_side: int = 896) -> dict:
        """CLS (D,) і патчі (h, w, D) одного кадру; float32 на CPU."""
        W, H = self.grid_size(*im.size, max_side)
        out = self._forward(self._tensor([im], W, H))
        t = out.last_hidden_state[0].float().cpu().numpy()
        gh, gw = H // self.patch, W // self.patch
        return dict(cls=t[0], patches=t[self.prefix:].reshape(gh, gw, -1), input=(W, H), grid=(gh, gw))

    def cls_attention(self, im: Image.Image, max_side: int = 672) -> np.ndarray:
        """Увага [CLS] -> патчі в останньому шарі, середнє по головах: (h, w).
        Eager-увага зберігає матриці N×N усіх шарів, тому рахуємо на меншому кадрі (O(N²), слайд 19)."""
        W, H = self.grid_size(*im.size, max_side)
        out = self._forward(self._tensor([im], W, H), attentions=True)
        a = out.attentions[-1][0, :, 0, self.prefix:].float().mean(0).cpu().numpy()
        return a.reshape(H // self.patch, W // self.patch)

    # ------------------------------------------------------------------ глобальні ембедінги
    def embed(self, ims: Sequence[Image.Image], size: Tuple[int, int] = (224, 224), batch: int = 64) -> np.ndarray:
        """CLS-ембедінги (N, D), L2-нормалізовані. size=(W, H) округлюється до кратного патчу."""
        p = self.patch
        W, H = max(p, round(size[0] / p) * p), max(p, round(size[1] / p) * p)
        out, t0 = [], time.time()
        for i in range(0, len(ims), batch):
            o = self._forward(self._tensor(ims[i:i + batch], W, H))
            out.append(o.last_hidden_state[:, 0].float().cpu().numpy())
        E = np.concatenate(out) if out else np.zeros((0, self.dim), np.float32)
        if len(ims):
            self.img_per_s = len(ims) / max(time.time() - t0, 1e-6)
        return l2n(E)

    def cache_key(self, *parts) -> str:
        h = hashlib.sha1(json.dumps([self.repo, self.random_weights, *map(str, parts)]).encode()).hexdigest()[:10]
        return f"{self.name}-{self.size}_{h}"

    def __repr__(self) -> str:
        return (f"DinoEncoder({self.repo}{' RANDOM' if self.random_weights else ''}, patch={self.patch}, "
                f"registers={self.n_reg}, dim={self.dim}, {self.n_params / 1e6:.0f}M, {self.device})")


def load_encoders(models: Sequence[str], size: str, device: Optional[str] = None,
                  random_weights: bool = False) -> Dict[str, DinoEncoder]:
    """Вантажить моделі; gated DINOv3 без доступу пропускає з інструкцією, а не падає."""
    encs = {}
    for name in models:
        sz = size if size in DINO_ZOO[name] else next(iter(DINO_ZOO[name]))
        try:
            encs[name] = DinoEncoder(name, sz, device=device, random_weights=random_weights)
            print(" ", encs[name])
        except Exception as e:  # noqa: BLE001
            msg = f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"
            if name in GATED:
                print(f"  ✗ {name}: {msg}\n"
                      f"    DINOv3 на HF — gated. Прийміть ліцензію на https://huggingface.co/{repo_id(name, sz)},\n"
                      f"    потім `hf auth login` (або HF_TOKEN) і `python tools/fetch_dino_models.py --models {name}`.\n"
                      f"    Поки що демо йде без {name}.")
            else:
                print(f"  ✗ {name}: {msg}")
    return encs


def l2n(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


# --------------------------------------------------------------------------------------
# Дані VisDrone
# --------------------------------------------------------------------------------------
def video_id(seq: str) -> str:
    """uav0000013_01073_v -> uav0000013: кілька послідовностей VisDrone вирізано з одного відео."""
    return seq.split("_")[0]


def list_sequences(root: str | Path = VISDRONE_DIR) -> pd.DataFrame:
    rows = []
    for s in A.discover_visdrone(root):
        split_dir, seq = s.source_id.split("/")
        frames = A._image_files(Path(s.path))
        if not frames:
            continue
        with Image.open(frames[0]) as im:
            w, h = im.size
        rows.append(dict(split=split_dir.replace("VisDrone2019-VID-", ""), seq=seq, vid=video_id(seq),
                         n_frames=len(frames), width=w, height=h, path=s.path, ann_path=s.ann_path))
    return pd.DataFrame(rows)


def frame_path(seq_row, idx: int) -> Path:
    """Кадр за 0-based індексом (у VisDrone файли 0000001.jpg..., у розмітці frame 1-based).
    seq_row — рядок seqs: Series, dict або namedtuple з itertuples()."""
    p = seq_row["path"] if isinstance(seq_row, (dict, pd.Series)) else seq_row.path
    return Path(p) / f"{idx + 1:07d}.jpg"


def load_ann(ann_path: str) -> pd.DataFrame:
    return pd.read_csv(ann_path, header=None, usecols=range(10), names=ANN_COLS)


def make_synthetic(root: str | Path, seed: int = 0) -> Path:
    """Синтетичні train / val / test-dev у форматі VisDrone (для smoke-тесту без даних і ваг)."""
    import shutil

    root = Path(root)
    off = 0
    for i, (split, n_seq) in enumerate([("train", 4), ("val", 2), ("test-dev", 2)]):
        dst = root / f"VisDrone2019-VID-{split}"
        if dst.exists():
            off += n_seq
            continue
        tmp = root / f"_tmp_{split}"
        src = A.make_synthetic_visdrone(tmp, n_seq=n_seq, n_frames=60, seed=seed + i)
        for s in range(n_seq):
            old, new = f"uav{s:07d}_00000_v", f"uav{off + s:07d}_00000_v"
            (src / "sequences" / old).rename(src / "sequences" / new)
            (src / "annotations" / f"{old}.txt").rename(src / "annotations" / f"{new}.txt")
        src.rename(dst)
        shutil.rmtree(tmp, ignore_errors=True)
        off += n_seq
    return root


# --------------------------------------------------------------------------------------
# Блок 1 — що бачить модель: PCA, норми, увага [CLS]
# --------------------------------------------------------------------------------------
def pca_rgb(patches: np.ndarray, q: float = 1.0) -> np.ndarray:
    """Перші 3 головні компоненти патч-фіч як RGB (h, w, 3) у [0, 1]. Знак і порядок компонент довільні."""
    gh, gw, d = patches.shape
    X = patches.reshape(-1, d).astype(np.float32)
    X = X - X.mean(0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    P = X @ Vt[:3].T
    lo, hi = np.percentile(P, q, axis=0), np.percentile(P, 100 - q, axis=0)
    return np.clip((P - lo) / np.maximum(hi - lo, 1e-6), 0, 1).reshape(gh, gw, 3)


def high_norm_tokens(patches: np.ndarray, k: float = 3.0) -> Tuple[np.ndarray, int]:
    """Норми патч-токенів і кількість «викидів» (> медіана + k·MAD) — артефакти, які прибирають регістри."""
    n = np.linalg.norm(patches, axis=-1)
    med = np.median(n)
    mad = np.median(np.abs(n - med)) + 1e-6
    return n, int((n > med + k * 1.4826 * mad).sum())


def show_feature_maps(im: Image.Image, encs: Dict[str, DinoEncoder], max_side: int = 896,
                      attn_max_side: int = 672, title: str = ""):
    """Рядок на модель: кадр | PCA патчів | норми токенів | увага [CLS]."""
    import matplotlib.pyplot as plt

    n = len(encs)
    fig, axes = plt.subplots(n, 4, figsize=(17, 2.75 * n + 0.6), squeeze=False)
    ext = (0, im.width, im.height, 0)
    for r, (name, enc) in enumerate(encs.items()):
        f = enc.features(im, max_side)
        gh, gw = f["grid"]
        norms, n_out = high_norm_tokens(f["patches"])
        att = enc.cls_attention(im, attn_max_side)
        axes[r, 0].imshow(im)
        axes[r, 0].set_title(f"{name} · вхід {f['input'][0]}×{f['input'][1]} · patch {enc.patch} → {gw}×{gh}"
                             f" = {gh * gw} ток.", fontsize=9)
        axes[r, 1].imshow(pca_rgb(f["patches"]), extent=ext, interpolation="nearest")
        axes[r, 1].set_title("PCA патч-фіч (3 компоненти = RGB)", fontsize=9)
        axes[r, 2].imshow(norms, extent=ext, cmap="magma", interpolation="nearest")
        axes[r, 2].set_title(f"‖токен‖ · високонормних: {n_out}", fontsize=9)
        axes[r, 3].imshow(im, alpha=0.35)
        axes[r, 3].imshow(att, extent=ext, cmap="viridis", alpha=0.75, interpolation="nearest",
                          vmax=np.percentile(att, 99.5))
        axes[r, 3].set_title(f"увага [CLS], останній шар (≤{attn_max_side}px)", fontsize=9)
        for ax in axes[r]:
            ax.set_xlim(0, im.width)
            ax.set_ylim(im.height, 0)
            ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    plt.show()


def feature_maps_panel(seqs: pd.DataFrame, encs: Dict[str, DinoEncoder], max_side: int = 896,
                       attn_max_side: int = 672):
    """Інтерактив до блоку 1: послідовність, кадр, роздільність входу."""
    try:
        import ipywidgets as W
        from IPython.display import display
    except ImportError:
        print("ipywidgets не встановлено — викликайте show_feature_maps напряму")
        return None
    opts = [(f"{r.split}/{r.seq} ({r.width}×{r.height})", i) for i, r in seqs.iterrows()]
    seq_dd = W.Dropdown(options=opts, description="послідовність", layout=W.Layout(width="60%"))
    pos_sl = W.FloatSlider(value=0.5, min=0, max=1, step=0.05, description="позиція", continuous_update=False)
    side_sl = W.IntSlider(value=max_side, min=224, max=1344, step=112, description="max side",
                          continuous_update=False)
    btn = W.Button(description="Порахувати", button_style="primary", icon="play")
    out = W.Output()

    def go(_=None):
        r = seqs.loc[seq_dd.value]
        idx = int(round(pos_sl.value * (r.n_frames - 1)))
        with out:
            out.clear_output(wait=True)
            show_feature_maps(Image.open(frame_path(r, idx)), encs, side_sl.value, attn_max_side,
                              f"{r.split}/{r.seq} · кадр {idx + 1}/{r.n_frames}")

    btn.on_click(go)
    display(W.VBox([W.HBox([seq_dd, pos_sl]), W.HBox([side_sl, btn]),
                    W.HTML("<i>max side кратний 112 = 7·16 = 8·14: однакова довша сторона для patch 14 і 16. "
                           "Великі значення на CPU рахуються секунди.</i>"), out]))
    go()


# --------------------------------------------------------------------------------------
# Блок 2 — подібність патчів за кліком
# --------------------------------------------------------------------------------------
def largest_box(seq_row, idx: int, cats: Sequence[int] = (4,)) -> Optional[Tuple[float, float, float, float]]:
    """Найбільший bbox потрібних категорій на кадрі — стартова точка панелі, щоб перший кадр був змістовним."""
    if not seq_row.get("ann_path"):
        return None
    a = load_ann(seq_row["ann_path"])
    a = a[(a.frame == idx + 1) & a.cat.isin(cats) & (a.score == 1)]
    if a.empty:
        return None
    b = a.loc[(a.w * a.h).idxmax()]
    return float(b.x), float(b.y), float(b.w), float(b.h)


def pick_frames(seqs: pd.DataFrame, splits: Sequence[str], n: int, exclude_vids: Sequence[str] = (),
                pos: float = 0.5, seed: int = 0) -> List[Tuple[str, Path]]:
    """По одному кадру (позиція pos у послідовності) з n послідовностей різних відео."""
    cand = seqs[seqs.split.isin(splits) & ~seqs.vid.isin(list(exclude_vids))].drop_duplicates("vid")
    cand = cand.sample(frac=1.0, random_state=seed).head(n)
    return [(f"{r.split}/{r.seq}", frame_path(r, int(pos * (r.n_frames - 1)))) for r in cand.itertuples()]


def _overlay(im: Image.Image, sim: np.ndarray, vmin: float, vmax: float, cmap: str = "inferno") -> np.ndarray:
    import matplotlib

    s = np.clip((sim - vmin) / max(vmax - vmin, 1e-6), 0, 1)
    s = cv2.resize(s.astype(np.float32), im.size, interpolation=cv2.INTER_NEAREST)
    heat = matplotlib.colormaps[cmap](s)[..., :3]
    base = np.asarray(im.convert("RGB"), dtype=np.float32) / 255
    return 0.45 * base + 0.55 * heat


class PatchSimilarityPanel:
    """
    Клік по кадру-запиту -> косинус фічі цього патча (або окілу 3×3) до всіх патчів запиту й галереї.
    Рядок на модель; сітки патчів у моделей різні (14 vs 16), клік перераховується в кожну окремо.
    """
    PREVIEW_W = 760

    def __init__(self, encs: Dict[str, DinoEncoder], queries: List[Tuple[str, Path]],
                 gallery: List[Tuple[str, Path]], max_side: int = 896, start_xy: Optional[Tuple[float, float]] = None):
        import ipywidgets as W

        self.W, self.encs, self.queries, self.gallery, self.max_side = W, encs, queries, gallery, max_side
        self.g_ims = [Image.open(p).convert("RGB") for _, p in gallery]
        print("Фічі галереї…", end=" ")
        self.g_feats = {m: [l2n(e.features(im, max_side)["patches"]) for im in self.g_ims] for m, e in encs.items()}
        print("готово")
        self.q_im: Optional[Image.Image] = None
        self.q_feats: Dict[str, np.ndarray] = {}
        self.xy = start_xy

        L = W.Layout
        self.q_dd = W.Dropdown(options=[(lbl, i) for i, (lbl, _) in enumerate(queries)], description="запит",
                               layout=L(width="55%"))
        self.rad = W.ToggleButtons(options=[("1 патч", 0), ("окіл 3×3", 1)], value=0)
        self.preview = W.Image(format="jpeg", layout=L(width=f"{self.PREVIEW_W}px"))
        self.info = W.HTML()
        self.out = W.Output()
        try:
            from ipyevents import Event

            self.ev = Event(source=self.preview, watched_events=["click"], prevent_default_action=True)
            self.ev.on_dom_event(self._on_click)
            hint = "Клікніть по об'єкту на кадрі-запиті."
        except ImportError:
            hint = "ipyevents не встановлено: точку задавайте через panel.set_point(x, y) у пікселях кадру."
        self.q_dd.observe(lambda ch: self._load(ch["new"]), names="value")
        self.rad.observe(lambda _: self._render(), names="value")
        self.ui = W.VBox([W.HBox([self.q_dd, self.rad]), W.HTML(f"<i>{hint}</i>"), self.preview, self.info, self.out])
        self._load(0, keep_xy=start_xy is not None)

    def _load(self, i: int, keep_xy: bool = False):
        self.q_im = Image.open(self.queries[i][1]).convert("RGB")
        self.q_feats = {m: l2n(e.features(self.q_im, self.max_side)["patches"]) for m, e in self.encs.items()}
        if not keep_xy:
            self.xy = (self.q_im.width / 2, self.q_im.height / 2)
        self._render()

    def _on_click(self, ev):
        rw, rh = ev.get("boundingRectWidth"), ev.get("boundingRectHeight")
        if rw and rh and self.q_im is not None:
            self.set_point(ev.get("relativeX", 0) / rw * self.q_im.width, ev.get("relativeY", 0) / rh * self.q_im.height)

    def set_point(self, x: float, y: float):
        self.xy = (float(x), float(y))
        self._render()

    def _query_vec(self, F: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        gh, gw, _ = F.shape
        x, y = self.xy
        r, c = min(int(y / self.q_im.height * gh), gh - 1), min(int(x / self.q_im.width * gw), gw - 1)
        k = self.rad.value
        v = F[max(r - k, 0):r + k + 1, max(c - k, 0):c + k + 1].reshape(-1, F.shape[-1]).mean(0)
        return l2n(v), (r, c)

    def _render(self):
        import io

        import matplotlib.pyplot as plt
        from PIL import ImageDraw

        pv = self.q_im.copy()
        x, y = self.xy
        ImageDraw.Draw(pv).ellipse([x - 9, y - 9, x + 9, y + 9], outline=(0, 255, 255), width=3)
        sc = self.PREVIEW_W / pv.width
        pv = pv.resize((self.PREVIEW_W, round(pv.height * sc)))
        buf = io.BytesIO()
        pv.save(buf, "JPEG", quality=85)
        self.preview.value = buf.getvalue()

        n_m, n_g = len(self.encs), len(self.g_ims)
        with self.out:
            self.out.clear_output(wait=True)
            fig, axes = plt.subplots(n_m, n_g + 1, figsize=(2.9 * (n_g + 1), 2.0 * n_m + 0.5), squeeze=False)
            for r, m in enumerate(self.encs):
                q, (pr, pc) = self._query_vec(self.q_feats[m])
                s_q = self.q_feats[m] @ q
                s_g = [F @ q for F in self.g_feats[m]]
                # Шкала — сирі cos від медіани до максимуму. Окремо для запиту (там сам об'єкт дає ~1)
                # і спільна для всієї галереї рядка, щоб кадри галереї порівнювались чесно.
                s_q_wo = s_q.copy()
                s_q_wo[pr, pc] = np.nan
                qmin, qmax = np.nanpercentile(s_q_wo, 50), np.nanmax(s_q_wo)
                pool = np.concatenate([s.ravel() for s in s_g])
                vmin, vmax = np.percentile(pool, 50), pool.max()
                axes[r, 0].imshow(_overlay(self.q_im, s_q, qmin, qmax))
                axes[r, 0].set_title(f"{m} · запит · галерея: cos {vmin:.2f}…{vmax:.2f}", fontsize=8)
                for j, (im, s) in enumerate(zip(self.g_ims, s_g)):
                    axes[r, j + 1].imshow(_overlay(im, s, vmin, vmax))
                    axes[r, j + 1].set_title(f"{self.gallery[j][0].split('/')[-1][:18]} · max {s.max():.2f}", fontsize=8)
                for ax in axes[r]:
                    ax.axis("off")
            fig.tight_layout()
            plt.show()
        self.info.value = f"точка ({x:.0f}, {y:.0f}) px · максимальні cos у галереї — у заголовках"


def patch_similarity_panel(encs, queries, gallery, max_side: int = 896, start_xy=None):
    try:
        from IPython.display import display
    except ImportError:
        return None
    p = PatchSimilarityPanel(encs, queries, gallery, max_side, start_xy)
    display(p.ui)
    return p


# --------------------------------------------------------------------------------------
# Блок 3 — кропи об'єктів і label efficiency
# --------------------------------------------------------------------------------------
def extract_crops(seqs: pd.DataFrame, classes: Dict[str, List[int]] = CROP_CLASSES, frame_step: int = 10,
                  min_px: int = 8, max_occ: int = 1, per_track: int = 2, cap_per_class_seq: int = 40,
                  seed: int = 0) -> pd.DataFrame:
    """
    Кропи з bbox VisDrone. Беремо кожен frame_step-й кадр (кадри поруч майже однакові), з кожного треку —
    не більше per_track кропів, з кожної пари (послідовність, клас) — не більше cap_per_class_seq.
    """
    cat2cls = {c: k for k, cs in classes.items() for c in cs}
    rng = np.random.default_rng(seed)
    rows = []
    for r in seqs.itertuples():
        if not r.ann_path:
            continue
        a = load_ann(r.ann_path)
        a = a[(a.score == 1) & a.cat.isin(list(cat2cls)) & (a.trunc == 0) & (a.occ <= max_occ)
              & (np.minimum(a.w, a.h) >= min_px) & ((a.frame - 1) % frame_step == 0)]
        if a.empty:
            continue
        a = a.assign(cls=a.cat.map(cat2cls))
        picked = []
        for _, g in a.groupby("tid"):
            g = g.sort_values("frame")
            picked.append(g.iloc[np.unique(np.linspace(0, len(g) - 1, min(per_track, len(g))).round().astype(int))])
        a = pd.concat(picked)
        for _, g in a.groupby("cls"):
            if len(g) > cap_per_class_seq:
                g = g.iloc[rng.choice(len(g), cap_per_class_seq, replace=False)]
            for b in g.itertuples():
                rows.append(dict(split=r.split, seq=r.seq, vid=r.vid, frame=int(b.frame) - 1, tid=int(b.tid),
                                 cls=b.cls, x=int(b.x), y=int(b.y), w=int(b.w), h=int(b.h),
                                 short=int(min(b.w, b.h)), path=str(frame_path(r, int(b.frame) - 1))))
    df = pd.DataFrame(rows)
    return df.sort_values(["split", "seq", "frame"]).reset_index(drop=True) if len(df) else df


def crop_image(im: Image.Image, x: int, y: int, w: int, h: int, pad: float = 0.25, size: int = 224) -> Image.Image:
    """Квадрат навколо bbox з контекстом pad з кожного боку; за межами кадру — чорне поле."""
    cx, cy, side = x + w / 2, y + h / 2, max(w, h) * (1 + 2 * pad)
    box = tuple(int(round(v)) for v in (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2))
    return im.crop(box).resize((size, size), Image.BICUBIC)


def materialize_crops(crops: pd.DataFrame, cache_dir: str | Path, size: int = 224, pad: float = 0.25) -> pd.DataFrame:
    """Вирізає кропи один раз у JPEG (кадр 4K декодується довго, а моделей кілька). Повертає crops з колонкою crop_path."""
    cache_dir = Path(cache_dir)
    key = hashlib.sha1(pd.util.hash_pandas_object(crops[["path", "x", "y", "w", "h"]], index=False).values.tobytes()
                       + f"{size}-{pad}".encode()).hexdigest()[:10]
    d = cache_dir / f"crops_{key}"
    d.mkdir(parents=True, exist_ok=True)
    crops = crops.assign(crop_path=[str(d / f"{i:06d}.jpg") for i in range(len(crops))])
    todo = crops[[not Path(p).exists() for p in crops.crop_path]]
    for path, g in todo.groupby("path"):
        im = Image.open(path).convert("RGB")
        for b in g.itertuples():
            crop_image(im, b.x, b.y, b.w, b.h, pad, size).save(b.crop_path, quality=92)
    return crops


def embed_crops(enc: DinoEncoder, crops: pd.DataFrame, cache_dir: str | Path, batch: int = 128) -> np.ndarray:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    f = cache_dir / f"crops_{enc.cache_key(len(crops), crops.crop_path.iloc[0], crops.crop_path.iloc[-1])}.npy"
    if f.exists():
        return np.load(f)
    out = []
    for i in range(0, len(crops), batch):
        out.append(enc.embed([Image.open(p) for p in crops.crop_path.iloc[i:i + batch]], batch=batch))
    E = np.concatenate(out)
    np.save(f, E)
    return E


def knn_predict(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, n_cls: int, k: int = 20, T: float = 0.07,
                chunk: int = 4096) -> np.ndarray:
    """Зважений kNN по косинусу, як в оцінці DINO: голос сусіда = exp(cos / T)."""
    k = max(1, min(k, len(Xtr)))
    pred = np.empty(len(Xte), dtype=np.int64)
    for i in range(0, len(Xte), chunk):
        S = Xte[i:i + chunk] @ Xtr.T
        idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
        w = np.exp(np.take_along_axis(S, idx, axis=1) / T)
        votes = np.zeros((len(S), n_cls))
        np.add.at(votes, (np.repeat(np.arange(len(S)), k), ytr[idx].ravel()), w.ravel())
        pred[i:i + chunk] = votes.argmax(1)
    return pred


def linear_probe(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, C: float = 1.0) -> np.ndarray:
    """Логістична регресія на заморожених фічах (linear probe). Один клас у мітках — передбачаємо його."""
    from sklearn.linear_model import LogisticRegression

    if len(np.unique(ytr)) < 2:
        return np.full(len(Xte), ytr[0])
    clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
    return clf.fit(Xtr, ytr).predict(Xte)


def macro_recall(y: np.ndarray, p: np.ndarray, n_cls: int) -> float:
    """Середній recall по класах (слайд 46): на VisDrone «car» — більшість, accuracy це ховає."""
    rec = [(p[y == c] == c).mean() for c in range(n_cls) if (y == c).any()]
    return float(np.mean(rec)) if rec else float("nan")


def select_random(n: int, budget: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).choice(n, min(budget, n), replace=False)


def select_kmeans(X: np.ndarray, budget: int, seed: int) -> np.ndarray:
    """Кластерний добір (слайд 30): k-means з budget кластерами, розмічаємо найближчий до центру кроп кожного кластера."""
    budget = min(budget, len(X))
    try:
        import faiss

        km = faiss.Kmeans(X.shape[1], budget, niter=25, seed=seed, spherical=True, verbose=False)
        km.train(np.ascontiguousarray(X, dtype=np.float32))
        C = km.centroids
    except ImportError:
        from sklearn.cluster import KMeans

        C = KMeans(budget, n_init=1, random_state=seed).fit(X).cluster_centers_
    idx = np.unique((X @ l2n(C).T).argmax(0))
    if len(idx) < budget:   # порожні кластери або спільний найближчий — добираємо випадково
        rest = np.setdiff1d(np.arange(len(X)), idx)
        idx = np.concatenate([idx, np.random.default_rng(seed).choice(rest, budget - len(idx), replace=False)])
    return idx


def label_efficiency(embs: Dict[str, np.ndarray], y: np.ndarray, tr: np.ndarray, te: np.ndarray, n_cls: int,
                     budgets: Sequence[int] = (12, 30, 60, 120, 300, 600), seeds: Sequence[int] = (0, 1, 2),
                     methods: Sequence[str] = ("random", "kmeans"), k: int = 20) -> pd.DataFrame:
    """Macro-recall на test для кожного бюджету міток, способу добору, класифікатора (kNN / linear) і моделі."""
    rows = []
    tr_idx, te_idx = np.flatnonzero(tr), np.flatnonzero(te)
    for m, E in embs.items():
        Xtr, Xte, ytr, yte = E[tr_idx], E[te_idx], y[tr_idx], y[te_idx]
        for clf in ("kNN", "linear"):
            p = knn_predict(Xtr, ytr, Xte, n_cls, k) if clf == "kNN" else linear_probe(Xtr, ytr, Xte)
            rows.append(dict(model=m, clf=clf, method="усі мітки", budget=len(tr_idx), seed=0,
                             macro_recall=macro_recall(yte, p, n_cls), classes=len(np.unique(ytr))))
        for b in budgets:
            if b >= len(tr_idx):
                continue
            for meth in methods:
                for s in seeds:
                    sel = select_random(len(Xtr), b, s) if meth == "random" else select_kmeans(Xtr, b, s)
                    for clf in ("kNN", "linear"):
                        p = (knn_predict(Xtr[sel], ytr[sel], Xte, n_cls, k) if clf == "kNN"
                             else linear_probe(Xtr[sel], ytr[sel], Xte))
                        rows.append(dict(model=m, clf=clf, method=meth, budget=b, seed=s,
                                         macro_recall=macro_recall(yte, p, n_cls), classes=len(np.unique(ytr[sel]))))
    return pd.DataFrame(rows)


def plot_label_efficiency(R: pd.DataFrame, n_cls: int):
    import matplotlib.pyplot as plt

    clfs = list(R.clf.unique())
    fig, axes = plt.subplots(1, len(clfs), figsize=(6.5 * len(clfs), 4.3), squeeze=False)
    colors = {m: plt.cm.tab10(i) for i, m in enumerate(R.model.unique())}
    for ax, clf in zip(axes[0], clfs):
        sub = R[R.clf == clf]
        for m in sub.model.unique():
            for meth, ls in (("kmeans", "-"), ("random", "--")):
                g = sub[(sub.model == m) & (sub.method == meth)].groupby("budget").macro_recall.agg(["mean", "std"])
                if g.empty:
                    continue
                ax.plot(g.index, g["mean"], ls, marker="o", ms=4, color=colors[m], label=f"{m} · {meth}")
                ax.fill_between(g.index, g["mean"] - g["std"], g["mean"] + g["std"], color=colors[m], alpha=0.1)
            full = sub[(sub.model == m) & (sub.method == "усі мітки")]
            if len(full):
                ax.axhline(full.macro_recall.iloc[0], color=colors[m], lw=0.8, ls=":")
        ax.axhline(1 / n_cls, color="grey", lw=0.8)
        ax.text(ax.get_xlim()[0] if ax.get_xlim()[0] > 0 else 10, 1 / n_cls + 0.01, "випадково", color="grey", fontsize=8)
        ax.set_xscale("log")
        ax.set_xlabel("розмічених кропів (бюджет)")
        ax.set_ylabel("macro-recall на val + test-dev")
        ax.set_title(f"{clf} · суцільна — кластерний добір, штрих — випадковий, крапки — усі мітки train", fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    plt.show()


SIZE_BINS = (0, 20, 40, 80, 10_000)


def recall_by_size(embs: Dict[str, np.ndarray], y: np.ndarray, short: np.ndarray, tr: np.ndarray, te: np.ndarray,
                   n_cls: int, bins: Sequence[int] = SIZE_BINS, k: int = 20) -> pd.DataFrame:
    """Macro-recall kNN (усі мітки train) окремо за короткою стороною об'єкта в px вихідного кадру."""
    rows = []
    labels = [f"{a}–{b}px" if b < 10_000 else f"≥{a}px" for a, b in zip(bins[:-1], bins[1:])]
    for m, E in embs.items():
        p = knn_predict(E[tr], y[tr], E[te], n_cls, k)
        yt, st = y[te], short[te]
        for (a, b), lbl in zip(zip(bins[:-1], bins[1:]), labels):
            msk = (st >= a) & (st < b)
            rows.append(dict(model=m, size=lbl, n=int(msk.sum()),
                             macro_recall=macro_recall(yt[msk], p[msk], n_cls) if msk.any() else np.nan))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Блок 4 — архів як актив: карта сцен, дублікати, витік
# --------------------------------------------------------------------------------------
def sample_frames(seqs: pd.DataFrame, per_seq: int = 12) -> pd.DataFrame:
    rows = []
    for r in seqs.itertuples():
        for i in np.unique(np.linspace(0, r.n_frames - 1, per_seq).round().astype(int)):
            rows.append(dict(split=r.split, seq=r.seq, vid=r.vid, frame=int(i), path=str(frame_path(r, i))))
    return pd.DataFrame(rows)


def embed_frames(enc: DinoEncoder, frames: pd.DataFrame, cache_dir: str | Path, width: int = 448,
                 batch: int = 32) -> np.ndarray:
    """CLS цілого кадру. Усі кадри VisDrone-VID — 16:9, тож вхід фіксований: width × width·9/16 (кратно патчу)."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    f = cache_dir / f"frames_{enc.cache_key(width, len(frames), frames.path.iloc[0], frames.path.iloc[-1])}.npy"
    if f.exists():
        return np.load(f)
    size = (width, int(width * 9 / 16))
    out = []
    for i in range(0, len(frames), batch):
        ims = []
        for p in frames.path.iloc[i:i + batch]:
            im = Image.open(p)
            im.draft("RGB", (size[0] * 2, size[1] * 2))   # JPEG декодується одразу зменшеним — 4K кадри в рази швидше
            ims.append(im.convert("RGB"))
        out.append(enc.embed(ims, size=size, batch=batch))
    E = np.concatenate(out)
    np.save(f, E)
    return E


def project_2d(X: np.ndarray, seed: int = 0) -> Tuple[np.ndarray, str]:
    try:
        from sklearn.manifold import TSNE

        return TSNE(2, perplexity=min(30, max(5, len(X) // 4)), init="pca", random_state=seed).fit_transform(X), "t-SNE"
    except ImportError:
        Xc = X - X.mean(0)
        return Xc @ np.linalg.svd(Xc, full_matrices=False)[2][:2].T, "PCA (немає scikit-learn)"


def plot_scene_map(embs: Dict[str, np.ndarray], frames: pd.DataFrame, seed: int = 0):
    """Ліворуч — кольором сабсет; праворуч — лише послідовності, вирізані з одного відео (uavNNNNNNN) у train."""
    import matplotlib.pyplot as plt

    multi = frames.groupby("vid").seq.nunique()
    multi = list(multi[multi > 1].index)
    fig, axes = plt.subplots(len(embs), 2, figsize=(13, 5.2 * len(embs)), squeeze=False)
    split_c = {"train": "#4c78a8", "val": "#f58518", "test-dev": "#54a24b"}
    for r, (m, E) in enumerate(embs.items()):
        xy, how = project_2d(E, seed)
        ax = axes[r, 0]
        for sp, c in split_c.items():
            msk = (frames.split == sp).to_numpy()
            ax.scatter(*xy[msk].T, s=7, c=c, label=f"{sp} ({frames[msk].seq.nunique()} посл.)", alpha=0.75)
        for sp in sorted(set(frames.split) - set(split_c)):
            msk = (frames.split == sp).to_numpy()
            ax.scatter(*xy[msk].T, s=7, label=sp, alpha=0.75)
        ax.set_title(f"{m} · {how} кадрів · кольором сабсет", fontsize=10)
        ax.legend(fontsize=8)
        ax = axes[r, 1]
        ax.scatter(*xy.T, s=5, c="#dddddd")
        for i, v in enumerate(multi):
            msk = (frames.vid == v).to_numpy()
            ax.scatter(*xy[msk].T, s=10, color=plt.cm.tab10(i % 10),
                       label=f"{v}: {frames[msk].seq.nunique()} посл.")
        ax.set_title("послідовності з одного вихідного відео", fontsize=10)
        if multi:
            ax.legend(fontsize=7, loc="best")
        for ax in axes[r]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    plt.show()


def cross_video_pairs(E: np.ndarray, frames: pd.DataFrame, top: int = 6) -> pd.DataFrame:
    """Найсхожіші пари кадрів з РІЗНИХ відео: кандидати в «ту саму сцену» або витік між сабсетами."""
    S = E @ E.T
    vid = frames.vid.to_numpy()
    S[vid[:, None] == vid[None, :]] = -1
    seq = frames.seq.to_numpy()
    out, seen = [], set()
    for flat in np.argsort(-S, axis=None):
        i, j = divmod(int(flat), len(E))
        pair = tuple(sorted((seq[i], seq[j])))
        if i >= j or pair in seen:
            continue
        seen.add(pair)
        a, b = frames.iloc[i], frames.iloc[j]
        out.append(dict(cos=float(S[i, j]), a=f"{a.split}/{a.seq}#{a.frame}", b=f"{b.split}/{b.seq}#{b.frame}",
                        cross_split=a.split != b.split, path_a=a.path, path_b=b.path))
        if len(out) >= top:
            break
    return pd.DataFrame(out)


def show_pairs(pairs: pd.DataFrame, title: str = ""):
    import matplotlib.pyplot as plt

    n = len(pairs)
    fig, axes = plt.subplots(n, 2, figsize=(9, 2.6 * n), squeeze=False)
    for r, p in enumerate(pairs.itertuples()):
        for c, (path, lbl) in enumerate(((p.path_a, p.a), (p.path_b, p.b))):
            im = Image.open(path)
            im.draft("RGB", (640, 360))
            axes[r, c].imshow(im.convert("RGB"))
            axes[r, c].set_title(f"{lbl}" + (f" · cos {p.cos:.3f}" if c == 0 else ""), fontsize=8)
            axes[r, c].axis("off")
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    plt.show()


def group_split(groups: np.ndarray, test_frac: float, seed: int) -> np.ndarray:
    """Маска test: цілі групи (кроп / послідовність / відео) потрапляють або в train, або в test."""
    rng = np.random.default_rng(seed)
    uniq = rng.permutation(np.unique(groups))
    sizes = pd.Series(groups).value_counts()
    test, n = [], 0
    for g in uniq:
        if n >= test_frac * len(groups):
            break
        test.append(g)
        n += sizes[g]
    return np.isin(groups, test)


def leakage_experiment(embs: Dict[str, np.ndarray], crops: pd.DataFrame, y: np.ndarray, n_cls: int,
                       test_frac: float = 0.2, seeds: Sequence[int] = range(5), k: int = 20) -> pd.DataFrame:
    """
    Одні й ті самі train-кропи, три способи відкласти 20 % на test:
      кроп випадково -> кропи одного треку й однієї сцени по обидва боки; за послідовністю; за відео uavNNNNNNN.
    """
    tr = (crops.split == "train").to_numpy()
    sub = crops[tr]
    strategies = {"кроп випадково": np.arange(len(sub)), "за послідовністю": sub.seq.to_numpy(),
                  "за відео (uavNNNNNNN)": sub.vid.to_numpy()}
    rows = []
    for m, E in embs.items():
        X, yy = E[tr], y[tr]
        for name, g in strategies.items():
            for s in seeds:
                te = group_split(g, test_frac, s)
                p = knn_predict(X[~te], yy[~te], X[te], n_cls, k)
                rows.append(dict(model=m, split=name, seed=s, macro_recall=macro_recall(yy[te], p, n_cls)))
    return pd.DataFrame(rows)
