"""
Заняття 5 · спільний код трьох демо-ноутбуків.

    lesson05_demo1_dino.ipynb          DINOv2 / DINOv3: фічі без жодної мітки        (LIVE DEMO 1, слайд 32)
    lesson05_demo2_clip_basics.ipynb   CLIP проти SigLIP 2: спільний простір і межі   (LIVE DEMO 2, слайд 40)
    lesson05_demo3_video_search.ipynb  текстовий пошук по аеровідео, FAISS, PQ        (LIVE DEMO 3, слайд 44)

Конвенція курсу: формули, які пояснюються на слайдах (косинус, втрати, softmax / sigmoid, злиття оцінок),
рахуються в клітинках ноутбуків; тут — дані, моделі, кеші, метрики, графіки й віджети.
Усі ембеддинги L2-нормовані, тож скалярний добуток = сирий косинус.
torch / open_clip / transformers / faiss імпортуються ліниво: модуль можна імпортувати
ще до того, як setup_hf() вирішить, чи працювати з Hugging Face офлайн.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import random
import time
import urllib.request
import warnings
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

# ======================================================================================
# Шляхи (від кореня репозиторію — не залежать від того, звідки запущено ноутбук чи tools/)
# ======================================================================================
REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = REPO_ROOT / "artifacts"
VISDRONE_DIR = ARTIFACTS / "visdrone"           # tools/fetch_visdrone.py
OWN_VIDEO_DIR = ARTIFACTS / "aerial_video"      # власне відео для demo3, DATASET = "own"
IMAGES_DIR = ARTIFACTS / "images"               # пара RGB / LWIR із заняття 1
CLIP_DIR = ARTIFACTS / "models" / "clip"        # tools/fetch_clip_models.py
DINO_DIR = ARTIFACTS / "models" / "dino"        # tools/fetch_dino_models.py
DINO_HUB = DINO_DIR / "hf_home" / "hub"         # кеш transformers для DINO

# Панелі одразу малюють перший результат. У nbconvert / nbclient це варто вимкнути (PANELS_AUTORUN=0):
# nbclient імітує віджет Output і після кожного виходу шле весь його стан назад у ядро, і на кількох великих
# картинках відповідь ядра губиться — клітинка чекає до тайм-ауту. У JupyterLab / VS Code такого немає.
AUTORUN = os.environ.get("PANELS_AUTORUN", "1") != "0"

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
VID_EXT = {".mp4", ".mov", ".avi", ".mkv", ".ts", ".m4v"}


# ======================================================================================
# Середовище
# ======================================================================================
def _manifest(models_dir: Path) -> dict:
    f = models_dir / "manifest.json"
    return json.loads(f.read_text()) if f.exists() else {}


def setup_hf(clip: Sequence[str] = (), dino: Sequence[str] = (), size: str = "base") -> str:
    """
    Офлайн-режим Hugging Face, якщо tools/fetch_*_models.py уже записали в маніфести всі потрібні моделі:
    жодних звернень до Hub і попереджень про HF_TOKEN. Діє лише до першого імпорту huggingface_hub,
    тож викликати в першій клітинці (після перезапуску ядра). Повертає "offline" / "online".
    Gated DINOv3, якого немає в маніфесті, не заважає: без доступу його однаково не завантажити,
    і load_dino() пропустить його з підказкою.
    """
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    have_dino = _manifest(DINO_DIR)
    dino = [m for m in dino if m not in GATED or f"{m}-{dino_size(m, size)}" in have_dino]
    need = [(CLIP_DIR, f"{m}-{size}") for m in clip] + [(DINO_DIR, f"{m}-{dino_size(m, size)}") for m in dino]
    if need and all(key in _manifest(d) for d, key in need):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    return "offline" if os.environ.get("HF_HUB_OFFLINE") == "1" else "online"


def pick_device(device: Optional[str] = None) -> str:
    """None / "auto" — cuda, mps або cpu. Якщо попросили cuda, а її немає, — попередження і cpu."""
    import torch

    if device in (None, "auto"):
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        warnings.warn("DEVICE='cuda', але CUDA недоступна — працюємо на CPU")
        return "cpu"
    return device


_POOL = None


def _cpu_pool():
    """Спільний пул потоків для декодування JPEG і PIL-ресайзу (обидва відпускають GIL)."""
    global _POOL
    if _POOL is None:
        from concurrent.futures import ThreadPoolExecutor

        _POOL = ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1), thread_name_prefix="prep")
    return _POOL


def l2n(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# ======================================================================================
# Дані: VisDrone-VID і власне відео
# ======================================================================================
# Категорії VisDrone (стовпець 8 розмітки): 0 ignored, 1 pedestrian, 2 people, 3 bicycle, 4 car, 5 van,
# 6 truck, 7 tricycle, 8 awning-tricycle, 9 bus, 10 motor, 11 others.
CLASSES: Dict[str, List[int]] = {
    "pedestrian": [1], "bicycle": [3], "car": [4], "van": [5],
    "truck": [6], "tricycle": [7], "bus": [9], "motorcycle": [10],
}
VEHICLES = (4, 5, 6, 9)
ANN_COLS = ["frame", "tid", "x", "y", "w", "h", "score", "cat", "trunc", "occ"]


@dataclass
class Source:
    source_id: str
    kind: str  # "video" | "images"
    path: str
    fps: float
    dataset: str = "own"
    ann_path: Optional[str] = None


def _image_files(d: Path) -> List[Path]:
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT)


def discover_visdrone(root: str | Path = VISDRONE_DIR, fps: float = 30.0) -> List[Source]:
    """VisDrone2019-VID-*: <split>/sequences/<seq>/*.jpg + <split>/annotations/<seq>.txt."""
    out = []
    for seq_root in sorted(Path(root).rglob("sequences")):
        if not seq_root.is_dir():
            continue
        ann_root, split = seq_root.parent / "annotations", seq_root.parent.name
        for seq in sorted(p for p in seq_root.iterdir() if p.is_dir()):
            ann = ann_root / f"{seq.name}.txt"
            out.append(Source(f"{split}/{seq.name}", "images", str(seq), fps, "visdrone",
                              str(ann) if ann.exists() else None))
    return out


def discover_generic(root: str | Path, image_seq_fps: float = 30.0, min_images: int = 10) -> List[Source]:
    """Власні дані: відеофайли + теки з послідовностями кадрів (UAVDT, AU-AIR тощо)."""
    root, out = Path(root), []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in VID_EXT:
            cap = cv2.VideoCapture(str(p))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            cap.release()
            out.append(Source(str(p.relative_to(root).with_suffix("")), "video", str(p), float(fps)))
        elif p.is_dir() and len(_image_files(p)) >= min_images:
            out.append(Source(str(p.relative_to(root)), "images", str(p), image_seq_fps))
    return out


def source_seconds(src: Source) -> float:
    if src.kind == "video":
        cap = cv2.VideoCapture(src.path)
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        return float(n) / src.fps if n > 0 else 0.0
    return len(_image_files(Path(src.path))) / src.fps


def video_id(seq: str) -> str:
    """uav0000013_01073_v -> uav0000013: кілька послідовностей VisDrone вирізано з одного відео."""
    return seq.split("_")[0]


def list_sequences(root: str | Path = VISDRONE_DIR) -> pd.DataFrame:
    rows = []
    for s in discover_visdrone(root):
        split_dir, seq = s.source_id.split("/")
        frames = _image_files(Path(s.path))
        if not frames:
            continue
        with Image.open(frames[0]) as im:
            w, h = im.size
        rows.append(dict(split=split_dir.replace("VisDrone2019-VID-", ""), seq=seq, vid=video_id(seq),
                         n_frames=len(frames), width=w, height=h, path=s.path, ann_path=s.ann_path))
    return pd.DataFrame(rows)


def ensure_visdrone(splits: Sequence[str], root: str | Path = VISDRONE_DIR):
    """Докачує відсутні сабсети через tools/fetch_visdrone.py."""
    import subprocess
    import sys

    missing = [sp for sp in splits if not any((Path(root) / f"VisDrone2019-VID-{sp}" / "sequences").glob("*"))]
    if missing:
        subprocess.run([sys.executable, str(REPO_ROOT / "tools" / "fetch_visdrone.py"), "--splits", *missing], check=True)


def frame_path(seq_row, idx: int) -> Path:
    """Кадр за 0-based індексом (файли 0000001.jpg..., у розмітці frame 1-based)."""
    p = seq_row["path"] if isinstance(seq_row, (dict, pd.Series)) else seq_row.path
    return Path(p) / f"{idx + 1:07d}.jpg"


def load_ann(ann_path: str) -> pd.DataFrame:
    return pd.read_csv(ann_path, header=None, usecols=range(10), names=ANN_COLS)


def make_synthetic_visdrone(root: str | Path, n_frames: int = 60, size=(640, 360), seed: int = 0) -> Path:
    """Мініатюрні train / val / test-dev у форматі VisDrone — щоб перевірити конвеєр без даних і ваг."""
    root = Path(root)
    rng = np.random.default_rng(seed)
    W, H = size
    s_id = 0
    for split, n_seq in (("train", 4), ("val", 2), ("test-dev", 2)):
        d = root / f"VisDrone2019-VID-{split}"
        for _ in range(n_seq):
            sd = d / "sequences" / f"uav{s_id:07d}_00000_v"
            s_id += 1
            if sd.exists():
                continue
            sd.mkdir(parents=True)
            (d / "annotations").mkdir(parents=True, exist_ok=True)
            bg = rng.integers(60, 120, size=(H, W, 3), dtype=np.uint8)
            lines = []
            for f in range(n_frames):
                fr = np.roll(bg, shift=f * 3, axis=1).copy()
                for tid, (cat, color, period) in enumerate([(6, (0, 0, 220), 30), (9, (220, 0, 0), 45),
                                                            (4, (0, 200, 0), 20), (1, (200, 200, 0), 25)]):
                    if (f // period) % 2 == 0:
                        x, y = (40 + 5 * f + 70 * cat) % (W - 60), 40 + (37 * cat) % (H - 80)
                        cv2.rectangle(fr, (x, y), (x + 50, y + 34), color, -1)
                        lines.append(f"{f + 1},{tid},{x},{y},50,34,1,{cat},0,0")
                cv2.imwrite(str(sd / f"{f + 1:07d}.jpg"), fr)
            (d / "annotations" / f"{sd.name}.txt").write_text("\n".join(lines) + "\n")
    return root


# --------------------------------------------------------------------------------------
# Кропи об'єктів і фону (demo1 блок 3, demo2 розділ 4)
# --------------------------------------------------------------------------------------
def extract_crops(seqs: pd.DataFrame, classes: Optional[Sequence[str]] = None, frame_step: int = 10,
                  min_px: int = 8, max_occ: int = 1, per_track: int = 2, cap_per_class_seq: int = 40,
                  seed: int = 0) -> pd.DataFrame:
    """
    Кропи з bbox VisDrone без обрізання краєм кадру. Беремо кожен frame_step-й кадр (сусідні майже однакові),
    з треку — не більше per_track кропів, з пари (послідовність, клас) — не більше cap_per_class_seq,
    щоб одна довга послідовність не заполонила клас.
    """
    classes = list(classes or CLASSES)
    cat2cls = {c: k for k in classes for c in CLASSES[k]}
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
        a = pd.concat([g.sort_values("frame").iloc[np.unique(np.linspace(0, len(g) - 1, min(per_track, len(g)))
                                                              .round().astype(int))]
                       for _, g in a.groupby("tid")])
        for _, g in a.groupby("cls"):
            if len(g) > cap_per_class_seq:
                g = g.iloc[rng.choice(len(g), cap_per_class_seq, replace=False)]
            for b in g.itertuples():
                rows.append(dict(split=r.split, seq=r.seq, vid=r.vid, frame=int(b.frame) - 1, tid=int(b.tid),
                                 cls=b.cls, x=int(b.x), y=int(b.y), w=int(b.w), h=int(b.h),
                                 short=int(min(b.w, b.h)), path=str(frame_path(r, int(b.frame) - 1))))
    df = pd.DataFrame(rows)
    return df.sort_values(["split", "seq", "frame", "tid"], kind="stable").reset_index(drop=True) if len(df) else df


def background_crops(seqs: pd.DataFrame, n: int, pad: float = 0.25, side=(64, 160), seed: int = 0) -> pd.DataFrame:
    """
    Кропи «фону»: випадкові вікна, які разом із контекстом pad не перетинаються з жодною розміченою рамкою
    (включно з ignored regions). Це «порожні» кадри, з яких на 99 % складається архів (слайд 25).
    """
    rng = random.Random(seed)
    cand = [(r, f) for r in seqs.itertuples() if r.ann_path for f in rng.sample(range(r.n_frames), min(3, r.n_frames))]
    rng.shuffle(cand)
    rows, ann_cache = [], {}
    for r, f in cand:
        if len(rows) >= n:
            break
        a = ann_cache.setdefault(r.ann_path, load_ann(r.ann_path))
        b = a.loc[a.frame == f + 1, ["x", "y", "w", "h"]].to_numpy()
        for _ in range(30):
            s = rng.randint(*side)
            ctx = int(s * pad) + 8
            x0, y0 = rng.randint(ctx, r.width - s - ctx), rng.randint(ctx, r.height - s - ctx)
            if not len(b) or not np.any((x0 - ctx < b[:, 0] + b[:, 2]) & (x0 + s + ctx > b[:, 0])
                                       & (y0 - ctx < b[:, 1] + b[:, 3]) & (y0 + s + ctx > b[:, 1])):
                rows.append(dict(split=r.split, seq=r.seq, vid=r.vid, frame=f, tid=-1, cls="background",
                                 x=x0, y=y0, w=s, h=s, short=s, path=str(frame_path(r, f))))
                break
    return pd.DataFrame(rows)


def crop_image(im: Image.Image, x: int, y: int, w: int, h: int, pad: float = 0.25, size: int = 224) -> Image.Image:
    """Квадрат навколо bbox з контекстом pad з кожного боку; за межами кадру — чорне поле."""
    cx, cy, side = x + w / 2, y + h / 2, max(w, h) * (1 + 2 * pad)
    box = tuple(int(round(v)) for v in (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2))
    return im.crop(box).resize((size, size), Image.BICUBIC)


def materialize_crops(crops: pd.DataFrame, cache_dir: str | Path, size: int = 224, pad: float = 0.25) -> pd.DataFrame:
    """Вирізає кропи один раз у JPEG (кадр 4K декодується довго, а моделей кілька). Додає колонку crop_path."""
    key = hashlib.sha1(pd.util.hash_pandas_object(crops[["path", "x", "y", "w", "h"]], index=False).values.tobytes()
                       + f"{size}-{pad}".encode()).hexdigest()[:10]
    d = Path(cache_dir) / f"crops_{key}"
    d.mkdir(parents=True, exist_ok=True)
    crops = crops.assign(crop_path=[str(d / f"{i:06d}.jpg") for i in range(len(crops))])
    todo = crops[[not Path(p).exists() for p in crops.crop_path]]
    for path, g in todo.groupby("path"):
        im = Image.open(path).convert("RGB")
        for b in g.itertuples():
            crop_image(im, b.x, b.y, b.w, b.h, pad, size).save(b.crop_path, quality=92)
    return crops


def thumb(im: Image.Image | str, max_side: int = 480) -> Image.Image:
    """Мініатюра для показу. Повнорозмірні кадри у великих фігурах роздувають вихід ноутбука до мегабайтів,
    а всередині віджета Output це ще й гальмує фронтенд (nbconvert чекає до тайм-ауту клітинки)."""
    im = Image.open(im) if isinstance(im, (str, Path)) else im.copy()
    im.draft("RGB", (max_side, max_side))
    im = im.convert("RGB")
    im.thumbnail((max_side, max_side))
    return im


def load_images(paths: Iterable[str]) -> List[Image.Image]:
    return [Image.open(p).convert("RGB") for p in paths]


def sample_frames(seqs: pd.DataFrame, per_seq: int = 6) -> pd.DataFrame:
    """
    Цілі кадри (рівномірно per_seq з кожної послідовності) + з розмітки: скільки на кадрі машин
    (≥ 20 px), транспорту й людей — ground truth для запитів «з машинами» / «без машин».
    """
    rows = []
    for r in seqs.itertuples():
        a = load_ann(r.ann_path) if r.ann_path else pd.DataFrame(columns=ANN_COLS)
        a = a[a.score == 1]
        for i in np.unique(np.linspace(0, r.n_frames - 1, per_seq).round().astype(int)):
            m = a[a.frame == i + 1]
            rows.append(dict(split=r.split, seq=r.seq, vid=r.vid, frame=int(i), path=str(frame_path(r, i)),
                             cars=int(((m.cat == 4) & (np.minimum(m.w, m.h) >= 20)).sum()),
                             vehicles=int(m.cat.isin(VEHICLES).sum()), people=int(m.cat.isin([1, 2]).sum())))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Тестові зображення для батареї відмов (demo2)
# --------------------------------------------------------------------------------------
def _font(px: int):
    for name in ("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "arialbd.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, px)
        except OSError:
            continue
    return ImageFont.load_default()


def stamp_text(im: Image.Image, text: str, frac: float = 0.22) -> Image.Image:
    """Типографічна атака: біла табличка з написом унизу кадру (слайд 37)."""
    im = im.copy().convert("RGB")
    if not text:
        return im
    d, h = ImageDraw.Draw(im), int(im.height * frac)
    d.rectangle((0, im.height - h, im.width, im.height), fill="white")
    d.text((im.width // 2, im.height - h // 2), text, fill="black", font=_font(int(h * 0.6)), anchor="mm")
    return im


def paste_copies(obj: Image.Image, n: int, size: int = 336, fill=(120, 120, 120), seed: int = 0) -> Image.Image:
    """n копій одного об'єкта на однотонному «асфальті» без перекриття — тест «рахунку» (слайд 37)."""
    rng = random.Random(seed)
    canvas = Image.new("RGB", (size, size), fill)
    s = size // 4
    o = obj.convert("RGB").resize((s, int(s * obj.height / obj.width)))
    for x, y in rng.sample([(c * s + s // 8, r * s + s // 8) for r in range(4) for c in range(4)], n):
        canvas.paste(o, (x, y))
    return canvas


def transform(im: Image.Image, text: str = "", rotate: int = 0, invert: bool = False) -> Image.Image:
    """Модифікації кадру з батареї відмов: напис, поворот (слайд 38), інверсія полярності ІЧ (слайд 39)."""
    im = im.convert("RGB")
    if invert:
        im = ImageOps.invert(im)
    if rotate:
        im = im.rotate(rotate, expand=True)
    return stamp_text(im, text)


# ======================================================================================
# Моделі DINO (transformers): DINOv2, DINOv2 + регістри, DINOv3
# ======================================================================================
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


def dino_size(name: str, size: str) -> str:
    """Розмір, якщо він є для моделі, інакше перший доступний (dinov3sat — лише large)."""
    return size if size in DINO_ZOO[name] else next(iter(DINO_ZOO[name]))


def _tiny_dino_config(name: str):
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
    Ваги беруться з DINO_HUB (кеш tools/fetch_dino_models.py) незалежно від HF_HOME процесу.
    """

    def __init__(self, name: str, size: str = "base", device: Optional[str] = None, random_weights: bool = False):
        import torch
        import transformers as T

        self.name, self.size = name, dino_size(name, size)
        self.repo = DINO_ZOO[name][self.size]
        self.device = pick_device(device)
        self.random_weights = random_weights
        t0 = time.time()
        if random_weights:
            self.model = T.AutoModel.from_config(_tiny_dino_config(name))
            mean, std = IMAGENET_MEAN, IMAGENET_STD
        else:
            # Прогрес-бар «Loading weights» у Jupyter — тимчасовий ipywidget: після закриття VS Code
            # показує на його місці «Error displaying widget: model not found». Вимикаємо на час завантаження.
            from transformers.utils import logging as hf_logging

            bar_was_on = hf_logging.is_progress_bar_enabled()
            hf_logging.disable_progress_bar()
            try:
                self.model = T.AutoModel.from_pretrained(self.repo, cache_dir=str(DINO_HUB))
                # Нормалізація — з препроцесора моделі: у dinov3sat вона своя, не ImageNet.
                proc = T.AutoImageProcessor.from_pretrained(self.repo, cache_dir=str(DINO_HUB))
            finally:
                if bar_was_on:
                    hf_logging.enable_progress_bar()
            mean, std = tuple(proc.image_mean), tuple(proc.image_std)
        self.load_s = time.time() - t0
        self.model.eval().to(self.device)
        cfg = self.model.config
        self.patch = int(cfg.patch_size if isinstance(cfg.patch_size, int) else cfg.patch_size[0])
        self.n_reg = int(getattr(cfg, "num_register_tokens", 0) or 0)
        self.prefix = 1 + self.n_reg
        self.dim = int(cfg.hidden_size)
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self._mean = torch.tensor(mean).view(1, 3, 1, 1).to(self.device)
        self._std = torch.tensor(std).view(1, 3, 1, 1).to(self.device)
        self.img_per_s: Optional[float] = None

    def grid_size(self, w: int, h: int, max_side: int) -> Tuple[int, int]:
        """Вхід (W, H), кратний патчу, з довшою стороною ≈ max_side (слайд 17: 1280×720 -> 1274×714 для P=14)."""
        s, p = max_side / max(w, h), self.patch
        return max(p, int(round(w * s / p)) * p), max(p, int(round(h * s / p)) * p)

    def _tensor(self, ims: Sequence[Image.Image], W: int, H: int):
        import torch

        arr = np.stack(list(_cpu_pool().map(lambda im: np.asarray(im.convert("RGB").resize((W, H), Image.BICUBIC)), ims)))
        x = torch.from_numpy(arr).to(self.device).permute(0, 3, 1, 2).float().div_(255)
        return (x - self._mean) / self._std

    def _autocast(self):
        import contextlib

        import torch

        return torch.autocast("cuda", dtype=torch.float16) if self.device.startswith("cuda") else contextlib.nullcontext()

    def _forward(self, x, attentions: bool = False):
        import torch

        if not attentions:
            with torch.inference_mode(), self._autocast():
                return self.model(pixel_values=x)
        # sdpa не віддає ваги уваги: на один прохід перемикаємось на eager. І без autocast: у DINOv3 матриця
        # уваги в fp16 переповнюється й дає NaN, а прохід один на кадр — fp32 тут дешевий.
        with torch.inference_mode():
            self.model.set_attn_implementation("eager")
            try:
                return self.model(pixel_values=x, output_attentions=True)
            finally:
                self.model.set_attn_implementation("sdpa")

    def features(self, im: Image.Image, max_side: int = 896) -> dict:
        """CLS (D,) і патчі (h, w, D) одного кадру; float32 на CPU."""
        W, H = self.grid_size(*im.size, max_side)
        t = self._forward(self._tensor([im], W, H)).last_hidden_state[0].float().cpu().numpy()
        gh, gw = H // self.patch, W // self.patch
        return dict(cls=t[0], patches=t[self.prefix:].reshape(gh, gw, -1), input=(W, H), grid=(gh, gw))

    def cls_attention(self, im: Image.Image, max_side: int = 672) -> np.ndarray:
        """Увага [CLS] -> патчі в останньому шарі, середнє по головах: (h, w).
        Eager-увага тримає матриці N×N усіх шарів (O(N²), слайд 19), тому кадр менший."""
        W, H = self.grid_size(*im.size, max_side)
        out = self._forward(self._tensor([im], W, H), attentions=True)
        a = out.attentions[-1][0, :, 0, self.prefix:].float().mean(0).cpu().numpy()
        return a.reshape(H // self.patch, W // self.patch)

    def embed(self, ims: Sequence[Image.Image], size: Tuple[int, int] = (224, 224), batch: int = 64) -> np.ndarray:
        """CLS-ембеддинги (N, D), L2-нормовані. size=(W, H) округлюється до кратного патчу."""
        p = self.patch
        W, H = max(p, round(size[0] / p) * p), max(p, round(size[1] / p) * p)
        out, t0 = [], time.time()
        for i in range(0, len(ims), batch):
            out.append(self._forward(self._tensor(ims[i:i + batch], W, H)).last_hidden_state[:, 0].float().cpu().numpy())
        if len(ims):
            self.img_per_s = len(ims) / max(time.time() - t0, 1e-6)
        return l2n(np.concatenate(out) if out else np.zeros((0, self.dim), np.float32))

    def cache_key(self, *parts) -> str:
        h = hashlib.sha1(json.dumps([self.repo, self.random_weights, *map(str, parts)]).encode()).hexdigest()[:10]
        return f"{self.name}-{self.size}_{h}"

    def __repr__(self) -> str:
        return (f"DinoEncoder({self.repo}{' RANDOM' if self.random_weights else ''}, patch={self.patch}, "
                f"registers={self.n_reg}, dim={self.dim}, {self.n_params / 1e6:.0f}M, {self.device})")


def load_dino(models: Sequence[str], size: str = "base", device: Optional[str] = None,
              random_weights: bool = False) -> Dict[str, DinoEncoder]:
    """Вантажить моделі; gated DINOv3 без доступу пропускає з інструкцією, а не падає."""
    encs = {}
    for name in models:
        try:
            encs[name] = DinoEncoder(name, size, device=device, random_weights=random_weights)
            print(" ", encs[name])
        except Exception as e:  # noqa: BLE001
            repo = DINO_ZOO[name][dino_size(name, size)]
            print(f"  ✗ {name}: {type(e).__name__}: {str(e).splitlines()[0][:140]}")
            if name in GATED:
                print(f"    gated: прийміть ліцензію на https://huggingface.co/{repo}, `hf auth login`,\n"
                      f"    потім python tools/fetch_dino_models.py --models {name} --size {dino_size(name, size)}. Поки що — без {name}.")
            else:
                print(f"    python tools/fetch_dino_models.py --models {name} --size {dino_size(name, size)}")
    return encs


def embed_crops(enc, crops: pd.DataFrame, cache_dir: str | Path, batch: int = 128) -> np.ndarray:
    """Ембеддинги кропів (crop_path) з кешем на диску; enc — DinoEncoder або ClipEncoder."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = enc.cache_key(len(crops), crops.crop_path.iloc[0], crops.crop_path.iloc[-1])
    f = cache_dir / f"crops_{key}.npy"
    if f.exists():
        return np.load(f)
    E = np.concatenate([enc.embed(load_images(crops.crop_path.iloc[i:i + batch]))
                        for i in range(0, len(crops), batch)])
    np.save(f, E)
    return E


def embed_frames(enc: DinoEncoder, frames: pd.DataFrame, cache_dir: str | Path, width: int = 448,
                 batch: int = 32) -> np.ndarray:
    """CLS цілого кадру. Кадри VisDrone-VID 16:9, тож вхід фіксований: width × width·9/16 (кратно патчу)."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    f = cache_dir / f"frames_{enc.cache_key(width, len(frames), frames.path.iloc[0], frames.path.iloc[-1])}.npy"
    if f.exists():
        return np.load(f)
    size = (width, int(width * 9 / 16))

    def load(p):
        im = Image.open(p)
        im.draft("RGB", (size[0] * 2, size[1] * 2))   # JPEG декодується одразу зменшеним — 4K у рази швидше
        return im.convert("RGB")

    E = np.concatenate([enc.embed(list(_cpu_pool().map(load, frames.path.iloc[i:i + batch])), size=size, batch=batch)
                        for i in range(0, len(frames), batch)])
    np.save(f, E)
    return E


# ======================================================================================
# Моделі CLIP-сімейства (open_clip): CLIP, SigLIP 2, RemoteCLIP, SkyCLIP
# ======================================================================================
SKY_BASE_URL = "https://opendatasharing.s3.us-west-2.amazonaws.com/SkyScript/ckpt/"
CLIP_ZOO: Dict[str, Dict[str, dict]] = {
    # OpenAI CLIP (Radford et al., 2021): еталон «вебової» моделі з softmax-втратою.
    # Ваги навчено з QuickGELU; open_clip 3.x без force_quick_gelu ставить звичайний GELU лише з UserWarning.
    "clip": {
        "base": dict(arch="ViT-B-16", pretrained="openai", quick_gelu=True),
        "large": dict(arch="ViT-L-14", pretrained="openai", quick_gelu=True),
    },
    # SigLIP 2 (Google, 2025): вебові дані, сигмоїдна втрата. Ваги з HF через open_clip.
    "siglip2": {
        "base": dict(arch="ViT-B-16-SigLIP2", pretrained="webli"),
        "large": dict(arch="ViT-L-16-SigLIP2-256", pretrained="webli"),
    },
    # RemoteCLIP (Liu et al., TGRS 2024): ваги у форматі OpenCLIP на HF, завантаження — як у README авторів.
    "remoteclip": {
        "base": dict(arch="ViT-B-32", hf_repo="chendelong/RemoteCLIP", hf_file="RemoteCLIP-ViT-B-32.pt"),
        "large": dict(arch="ViT-L-14", hf_repo="chendelong/RemoteCLIP", hf_file="RemoteCLIP-ViT-L-14.pt"),
    },
    # SkyCLIP (SkyScript, AAAI 2024): архіви з S3 авторів. ViT-L-14 ініціалізовано з OpenAI CLIP -> quick_gelu.
    "skyclip": {
        "base": dict(arch="ViT-B-32", url=SKY_BASE_URL + "SkyCLIP_ViT_B32_top50pct.zip"),
        "large": dict(arch="ViT-L-14", url=SKY_BASE_URL + "SkyCLIP_ViT_L14_top30pct.zip", quick_gelu=True),
    },
}

# Шаблони промптів (слайд 38). "{}" — опис цілі з артиклем: "a truck".
PLAIN_TEMPLATES = ["{}"]
PHOTO_TEMPLATES = ["a photo of {}."]
AERIAL_TEMPLATES = [
    "an aerial photo of {}",
    "a drone image of {}",
    "an overhead view of {}",
    "a photo of {} taken from a drone",
    "a high-altitude aerial view of {}",
    "a top-down aerial image showing {}",
]


def _download(url: str, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    print(f"[download] {url}\n           -> {out}")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total, done, last = int(r.headers.get("Content-Length", 0)), 0, -1
        while buf := r.read(1 << 20):
            f.write(buf)
            done += len(buf)
            if total and (pct := int(100 * done / total)) // 10 != last // 10:
                print(f"           {pct}% ({done / 1e6:.0f} / {total / 1e6:.0f} MB)")
                last = pct
    tmp.rename(out)
    return out


def resolve_checkpoint(name: str, size: str, models_dir: str | Path = CLIP_DIR, download: bool = True) -> Path:
    """Локальний шлях до чекпойнта RemoteCLIP / SkyCLIP (за потреби завантажує)."""
    spec, models_dir = CLIP_ZOO[name][size], Path(models_dir)
    if "hf_repo" in spec:
        from huggingface_hub import hf_hub_download

        p = Path(hf_hub_download(spec["hf_repo"], spec["hf_file"], cache_dir=str(models_dir / "hf"),
                                 local_files_only=not download))
        st = p.with_suffix(".safetensors")   # копія з fetch_clip_models.py --to-safetensors
        return st if st.exists() else p
    if "url" in spec:
        zip_name = spec["url"].rsplit("/", 1)[-1]
        zpath = models_dir / "skyclip" / zip_name
        xdir = zpath.with_suffix("")
        if not xdir.exists():
            if not zpath.exists():
                if not download:
                    raise FileNotFoundError(f"{zpath} немає. Запустіть python tools/fetch_clip_models.py")
                _download(spec["url"], zpath)
            with zipfile.ZipFile(zpath) as z:
                z.extractall(xdir)
        # Вміст архіву — вихід тренування OpenCLIP. Беремо *.safetensors, інакше *.pt.
        cands = sorted(xdir.rglob("*.safetensors")) or sorted(xdir.rglob("*.pt"))
        if not cands:
            raise FileNotFoundError(f"У {xdir} не знайдено *.pt / *.safetensors")
        latest = [c for c in cands if "latest" in c.name]
        return latest[0] if latest else cands[-1]
    raise ValueError(f"{name}/{size}: чекпойнт завантажується через open_clip (pretrained=...)")


def _numpy_safe_globals() -> list:
    """
    Типи numpy з чекпойнтів тренування OpenCLIP (стан GradScaler у SkyCLIP): скаляр і dtype — дані, а не код,
    тож дозволяємо їх у weights_only. Pickle посилається на numpy.core.multiarray.scalar, а в numpy 2
    він живе в numpy._core — реєструємо під старим ім'ям.
    """
    try:
        from numpy._core.multiarray import scalar
    except ImportError:  # numpy 1.x
        from numpy.core.multiarray import scalar
    dtypes = [type(np.dtype(c)) for c in "?bBhHiIlLqQefdgFD"]
    return [(scalar, "numpy.core.multiarray.scalar"), scalar, np.dtype, *dict.fromkeys(dtypes)]


def load_state_dict_safely(path: str | Path, trust_pickle: bool = False) -> dict:
    """
    .pt — це pickle: з weights_only=False він може виконати довільний код. Тому за замовчуванням вантажимо
    лише тензори (+ numpy-скаляри); якщо файл містить щось інше — падаємо з поясненням (air-gapped контур).
    """
    import torch

    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        sd = load_file(str(path))
    else:
        try:
            with torch.serialization.safe_globals(_numpy_safe_globals()):
                ckpt = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as e:  # noqa: BLE001
            if not trust_pickle:
                raise RuntimeError(
                    f"{path.name} не вантажиться з weights_only=True ({type(e).__name__}). Якщо джерело перевірене — "
                    "передайте trust_pickle=True і збережіть ваги в safetensors "
                    "(python tools/fetch_clip_models.py --to-safetensors).") from e
            warnings.warn(f"{path.name}: завантаження з weights_only=False (pickle).")
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    if next(iter(sd)).startswith("module."):
        sd = {k[len("module."):]: v for k, v in sd.items()}
    return sd


def save_safetensors(state_dict: dict, out: str | Path) -> Path:
    from safetensors.torch import save_file

    save_file({k: v.contiguous() for k, v in state_dict.items()}, str(out))
    return Path(out)


def _get_tokenizer(open_clip, arch: str, cache_dir: Path):
    """
    Токенайзер open_clip. Офлайн get_tokenizer для SigLIP 2 падає (transformers шукає config.json, якого
    в репозиторії токенайзера немає), тож будуємо той самий HFTokenizer з локальної копії в кеші.
    """
    from huggingface_hub import constants

    cfg = (open_clip.get_model_config(arch) or {}).get("text_cfg", {})
    name = cfg.get("hf_tokenizer_name")
    if not (name and constants.HF_HUB_OFFLINE):
        return open_clip.get_tokenizer(arch, cache_dir=str(cache_dir))
    from huggingface_hub import snapshot_download
    from open_clip.tokenizer import DEFAULT_CONTEXT_LENGTH, HFTokenizer

    local = snapshot_download(name, cache_dir=str(cache_dir), local_files_only=True)
    return HFTokenizer(local, context_length=cfg.get("context_length", DEFAULT_CONTEXT_LENGTH),
                       tokenizer_mode=cfg.get("tokenizer_mode"), **cfg.get("tokenizer_kwargs", {}))


def _create_bare_model(open_clip, arch: str, **kw):
    """Архітектура без ваг. Лог «Model initialized randomly» для RemoteCLIP / SkyCLIP — хибна тривога, глушимо."""
    import logging

    class _Drop(logging.Filter):
        def filter(self, rec):
            return "initialized randomly" not in rec.getMessage()

    f, root = _Drop(), logging.getLogger()
    root.addFilter(f)
    try:
        return open_clip.create_model(arch, **kw)
    finally:
        root.removeFilter(f)


class ClipEncoder:
    """
    Єдиний інтерфейс до CLIP / SigLIP 2 / RemoteCLIP / SkyCLIP через open_clip.

    Препроцесинг розрізано на дві частини. PIL-кроки (ресайз, letterbox) — на CPU у пулі потоків.
    ToTensor + Normalize — одним батчем на пристрої моделі: по кропу це крихітні тензорні операції,
    які torch розпаралелює на всі ядра; потоки OpenMP більше чекають, ніж рахують, і GPU простоює.
    """

    def __init__(self, name: str, size: str = "base", device: Optional[str] = None, resize_mode: str = "longest",
                 random_weights: bool = False, trust_pickle: bool = False, models_dir: str | Path = CLIP_DIR,
                 download: bool = True):
        import open_clip
        import torch
        from torchvision.transforms import Compose

        self.name, self.size = name, size
        self.key = f"{name}-{size}" + ("-RANDOM" if random_weights else "")
        self.device = pick_device(device)
        self.resize_mode = resize_mode
        models_dir = Path(models_dir)
        spec = dict(CLIP_ZOO[name][size])
        qg = spec.get("quick_gelu", False)

        if random_weights:      # лише smoke-тест без мережі: архітектура з вбудованим токенайзером
            self.arch = "ViT-B-32"
            model = _create_bare_model(open_clip, self.arch)
        elif "pretrained" in spec:
            self.arch = spec["arch"]
            model = open_clip.create_model(self.arch, pretrained=spec["pretrained"], force_quick_gelu=qg,
                                           cache_dir=str(models_dir / "open_clip"), require_pretrained=True)
        else:
            self.arch = spec["arch"]
            ckpt = resolve_checkpoint(name, size, models_dir, download=download)
            model = _create_bare_model(open_clip, self.arch, force_quick_gelu=qg)
            res = model.load_state_dict(load_state_dict_safely(ckpt, trust_pickle), strict=False)
            if res.missing_keys:
                raise RuntimeError(f"{self.key}: бракує ключів у чекпойнті: {res.missing_keys[:5]} ...")
            if res.unexpected_keys:
                warnings.warn(f"{self.key}: зайві ключі проігноровано: {res.unexpected_keys[:5]}")
            self.checkpoint = str(ckpt)

        self.tokenizer = _get_tokenizer(open_clip, self.arch, models_dir / "open_clip")
        pp = open_clip.get_model_preprocess_cfg(model)
        # resize_mode="longest" = letterbox: увесь кадр у полі зору моделі, без center-crop
        # (center-crop 16:9 -> 1:1 викидає 44 % ширини, слайд 22).
        self.preprocess = open_clip.image_transform(
            pp["size"], is_train=False, mean=pp["mean"], std=pp["std"], resize_mode=resize_mode,
            interpolation=pp.get("interpolation"), fill_color=pp.get("fill_color", 0))
        tf = list(self.preprocess.transforms)
        self._pil_tf = Compose(tf[:next(i for i, t in enumerate(tf) if "ToTensor" in type(t).__name__)])
        self.input_size, self.mean = pp["size"], tuple(pp["mean"])
        self._torch = torch
        self.model = model.to(self.device).eval()
        self._mean = torch.tensor(pp["mean"], device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(pp["std"], device=self.device).view(1, 3, 1, 1)
        self.n_params = sum(p.numel() for p in model.parameters())
        self.dim = int(self.encode_texts(["test"]).shape[1])
        self.img_per_s: Optional[float] = None
        self.cached = False

    # ---------------------------------------------------------------- параметри голови
    @property
    def logit_scale(self) -> float:
        """1/τ — температура, з якою модель навчалась (слайд 35)."""
        return float(self.model.logit_scale.detach().exp())

    @property
    def logit_bias(self) -> Optional[float]:
        """Зсув сигмоїди — є лише в SigLIP."""
        b = getattr(self.model, "logit_bias", None)
        return None if b is None else float(b.detach())

    # ---------------------------------------------------------------- кодування
    def _autocast(self):
        import contextlib

        torch = self._torch
        return torch.autocast(device_type="cuda", dtype=torch.float16) if self.device == "cuda" else contextlib.nullcontext()

    def prepare_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        """CPU-частина: PIL-ресайз / letterbox у пулі потоків -> батч uint8 (N, H, W, 3)."""
        def one(im):
            return np.asarray(self._pil_tf(im.convert("RGB")), dtype=np.uint8)

        return np.stack([one(im) for im in images] if len(images) < 8 else list(_cpu_pool().map(one, images)))

    def encode_prepared(self, batch: np.ndarray) -> np.ndarray:
        """GPU-частина: uint8 -> float, нормалізація і image tower одним батчем."""
        torch = self._torch
        x = torch.from_numpy(batch).to(self.device, non_blocking=True).permute(0, 3, 1, 2).float().div_(255)
        with torch.no_grad(), self._autocast():
            f = self.model.encode_image((x - self._mean) / self._std, normalize=True)
        return f.float().cpu().numpy()

    def encode_images(self, images: Sequence[Image.Image], batch: int = 128) -> np.ndarray:
        return np.concatenate([self.encode_prepared(self.prepare_images(images[i:i + batch]))
                               for i in range(0, len(images), batch)])

    embed = encode_images   # той самий інтерфейс, що й у DinoEncoder (embed_crops)

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        torch = self._torch
        t = self.tokenizer(list(texts)).to(self.device)
        with torch.no_grad(), self._autocast():
            f = self.model.encode_text(t, normalize=True)
        return f.float().cpu().numpy()

    def text_query(self, text: str, templates: Sequence[str] = AERIAL_TEMPLATES) -> np.ndarray:
        """Ансамбль промптів: усереднюємо нормовані вектори і знову нормуємо (слайд 38)."""
        return l2n(self.encode_texts([tpl.format(text) for tpl in templates]).mean(0)).astype("float32")

    def image_query(self, image: Image.Image) -> np.ndarray:
        return self.encode_images([image])[0].astype("float32")

    def cache_key(self, *parts) -> str:
        h = hashlib.sha1(json.dumps([self.key, self.resize_mode, *map(str, parts)]).encode()).hexdigest()[:10]
        return f"{self.key}_{h}"

    def __repr__(self) -> str:
        return f"ClipEncoder({self.key}, arch={self.arch}, {self.n_params / 1e6:.0f}M, dim={self.dim}, {self.device})"


# ======================================================================================
# Класифікація на заморожених ознаках, метрики й витік (demo1, demo2)
# ======================================================================================
def macro_recall(y: np.ndarray, p: np.ndarray) -> float:
    """Середній recall по класах, присутніх у y (слайд 46): accuracy ховає провал на рідкісних класах."""
    rec = [(p[y == c] == c).mean() for c in np.unique(y)]
    return float(np.mean(rec)) if rec else float("nan")


def bootstrap_diff(y: np.ndarray, pa: np.ndarray, pb: np.ndarray, n: int = 2000, seed: int = 0) -> np.ndarray:
    """2.5 / 50 / 97.5 перцентилі різниці macro-recall (pa − pb) при ресемплінгу вибірки."""
    r = np.random.default_rng(seed)
    d = [macro_recall(y[i], pa[i]) - macro_recall(y[i], pb[i]) for i in (r.integers(0, len(y), len(y)) for _ in range(n))]
    return np.percentile(d, [2.5, 50, 97.5])


def knn_predict(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, k: int = 20, T: float = 0.07,
                chunk: int = 4096) -> np.ndarray:
    """Зважений kNN по косинусу, як в оцінці DINO: голос сусіда = exp(cos / T). Мітки — цілі 0..C−1."""
    k, n_cls = max(1, min(k, len(Xtr))), int(ytr.max()) + 1
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
    """Логістична регресія на заморожених ознаках. Один клас у мітках — передбачаємо його."""
    from sklearn.linear_model import LogisticRegression

    if len(np.unique(ytr)) < 2:
        return np.full(len(Xte), ytr[0])
    return LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(Xtr, ytr).predict(Xte)


def select_labels(X: np.ndarray, y: np.ndarray, budget: int, method: str, seed: int) -> np.ndarray:
    """
    Які кропи «розмітить експерт»:
      random    — budget випадкових;
      per_class — budget на кожен клас (мітки вже відомі; так рахують few-shot);
      kmeans    — k-means з budget кластерами, кроп, найближчий до центру кожного (слайд 29). Мітки не потрібні.
    """
    rng = np.random.default_rng(seed)
    if method == "random":
        return rng.choice(len(X), min(budget, len(X)), replace=False)
    if method == "per_class":
        return np.concatenate([rng.choice(np.flatnonzero(y == c), min(budget, (y == c).sum()), replace=False)
                               for c in np.unique(y)])
    budget = min(budget, len(X))
    try:
        import faiss

        # Кластерів навмисно багато (budget = кількість міток): FAISS за замовчуванням хоче ≥ 39 точок
        # на центроїд і інакше лише попереджає — результат той самий.
        km = faiss.Kmeans(X.shape[1], budget, niter=25, seed=seed, spherical=True, verbose=False,
                          min_points_per_centroid=1)
        km.train(np.ascontiguousarray(X, dtype=np.float32))
        C = km.centroids
    except ImportError:
        from sklearn.cluster import KMeans

        C = KMeans(budget, n_init=1, random_state=seed).fit(X).cluster_centers_
    idx = np.unique((X @ l2n(C).T).argmax(0))
    if len(idx) < budget:   # порожні кластери або спільний найближчий — добираємо випадково
        idx = np.concatenate([idx, rng.choice(np.setdiff1d(np.arange(len(X)), idx), budget - len(idx), replace=False)])
    return idx


def label_efficiency(embs: Dict[str, np.ndarray], y: np.ndarray, tr: np.ndarray, te: np.ndarray,
                     budgets: Sequence[int], methods: Sequence[str] = ("random", "kmeans"),
                     seeds: Sequence[int] = (0, 1, 2), k: int = 20) -> pd.DataFrame:
    """Macro-recall на test для кожного бюджету міток, способу добору, класифікатора (kNN / linear) і моделі."""
    rows = []
    tr_idx, te_idx = np.flatnonzero(tr), np.flatnonzero(te)
    clfs = {"kNN": lambda a, b, c: knn_predict(a, b, c, k), "linear probe": linear_probe}
    for m, E in embs.items():
        Xtr, Xte, ytr, yte = E[tr_idx], E[te_idx], y[tr_idx], y[te_idx]
        plans = [("усі мітки", len(tr_idx), 0, np.arange(len(tr_idx)))]
        plans += [(meth, b, s, select_labels(Xtr, ytr, b, meth, s))
                  for b in budgets for meth in methods for s in seeds if meth == "per_class" or b < len(tr_idx)]
        for meth, b, s, sel in plans:
            for clf, fn in clfs.items():
                rows.append(dict(model=m, clf=clf, method=meth, budget=b, seed=s,
                                 macro_recall=macro_recall(yte, fn(Xtr[sel], ytr[sel], Xte)),
                                 classes=len(np.unique(ytr[sel]))))
    return pd.DataFrame(rows)


def recall_by_size(embs: Dict[str, np.ndarray], y: np.ndarray, short: np.ndarray, tr: np.ndarray, te: np.ndarray,
                   bins: Sequence[int] = (0, 20, 40, 80, 10_000), k: int = 20) -> pd.DataFrame:
    """Macro-recall kNN (усі мітки train) окремо за короткою стороною об'єкта в px вихідного кадру."""
    rows = []
    for m, E in embs.items():
        p, yt, st = knn_predict(E[tr], y[tr], E[te], k), y[te], short[te]
        for a, b in zip(bins[:-1], bins[1:]):
            msk = (st >= a) & (st < b)
            rows.append(dict(model=m, size=f"{a}–{b}px" if b < 10_000 else f"≥{a}px", n=int(msk.sum()),
                             macro_recall=macro_recall(yt[msk], p[msk]) if msk.any() else np.nan))
    return pd.DataFrame(rows)


def group_split(groups: np.ndarray, test_frac: float, seed: int) -> np.ndarray:
    """Маска test: цілі групи (кроп / послідовність / відео) потрапляють або в train, або в test."""
    rng = np.random.default_rng(seed)
    sizes = pd.Series(groups).value_counts()
    test, n = [], 0
    for g in rng.permutation(np.unique(groups)):
        if n >= test_frac * len(groups):
            break
        test.append(g)
        n += sizes[g]
    return np.isin(groups, test)


def leakage_experiment(embs: Dict[str, np.ndarray], crops: pd.DataFrame, y: np.ndarray,
                       test_frac: float = 0.2, seeds: Sequence[int] = range(5), k: int = 20) -> pd.DataFrame:
    """Одні й ті самі train-кропи, три способи відкласти 20 % на test: кроп випадково / послідовність / відео."""
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
                rows.append(dict(model=m, split=name, seed=s,
                                 macro_recall=macro_recall(yy[te], knn_predict(X[~te], yy[~te], X[te], k))))
    return pd.DataFrame(rows)


def cross_video_pairs(E: np.ndarray, frames: pd.DataFrame, top: int = 6) -> pd.DataFrame:
    """Найсхожіші пари кадрів з РІЗНИХ відео: кандидати в «ту саму локацію», тобто у витік між сабсетами."""
    S = E @ E.T
    vid, seq = frames.vid.to_numpy(), frames.seq.to_numpy()
    S[vid[:, None] == vid[None, :]] = -1
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


# ======================================================================================
# Конвеєр пошуку по відео (demo3): ключові кадри -> тайли -> ембеддинги -> FAISS -> метрики -> PQ
# ======================================================================================
def _iter_frames(src: Source, step: int) -> Iterable[Tuple[int, np.ndarray, Optional[Path]]]:
    """(індекс кадру, BGR, файл-оригінал або None для відео)."""
    if src.kind == "video":
        cap, i = cv2.VideoCapture(src.path), 0
        while True:
            if i % step == 0:
                ok, fr = cap.read()
                if not ok:
                    break
                yield i, fr, None
            elif not cap.grab():
                break
            i += 1
        cap.release()
    else:
        files = _image_files(Path(src.path))
        for i in range(0, len(files), step):
            fr = cv2.imread(str(files[i]))
            if fr is not None:
                yield i, fr, files[i]


def _n_sampled(src: Source, step: int) -> int:
    if src.kind == "video":
        cap = cv2.VideoCapture(src.path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    else:
        n = len(_image_files(Path(src.path)))
    return -(-max(n, 0) // step)


def extract_keyframes(sources: Sequence[Source], out_dir: str | Path, sample_fps: float = 1.0, dedup_thr: float = 2.0,
                      max_side: int = 1920, jpeg_quality: int = 90, force: bool = False) -> pd.DataFrame:
    """
    Ключові кадри: прорідження до sample_fps + дедуплікація по мініатюрі 32×18 (середня абсолютна різниця).
    Дедуплікація не залежить від моделі, тож усі моделі індексують однаковий набір кадрів.
    Для статичної камери малий рухомий об'єкт мініатюру майже не змінює — там dedup_thr=0
    або дедуплікація по треку детектора.

    У out_dir/frames лягають кадри з відео (довільний доступ до H.264 повільний) і кадри послідовностей,
    які довелося зменшити до max_side; інакше keyframes.csv посилається на оригінал.
    Кеш: ті самі параметри й джерела -> збережений результат; статистика — у df.attrs["stats"].
    """
    from tqdm.auto import tqdm

    out_dir = Path(out_dir)
    meta_csv, params_json, stats_json = out_dir / "keyframes.csv", out_dir / "keyframes_params.json", out_dir / "keyframes_stats.json"
    params = dict(sample_fps=sample_fps, dedup_thr=dedup_thr, max_side=max_side, layout="link-originals",
                  sources=[asdict(s) for s in sources])
    if not force and all(f.exists() for f in (meta_csv, params_json, stats_json)):
        if json.loads(params_json.read_text()) == json.loads(json.dumps(params)):
            df = pd.read_csv(meta_csv)
            if all(Path(p).exists() for p in df["path"]):
                df.attrs["stats"] = dict(json.loads(stats_json.read_text()), cached=True)
                return df

    params_json.unlink(missing_ok=True)   # якщо прогін обірветься, напівготовий кеш не підхопиться
    frames_dir = (out_dir / "frames").resolve()
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old in frames_dir.glob("*.jpg"):
        old.unlink()

    steps = [max(1, int(round(s.fps / sample_fps))) for s in sources]
    rows, per_source, kf_id, n_copied = [], [], 0, 0
    with tqdm(total=sum(_n_sampled(s, st) for s, st in zip(sources, steps)), desc="ключові кадри", unit="кадр") as bar:
        for s, step in zip(sources, steps):
            bar.set_postfix_str(s.source_id.rsplit("/", 1)[-1], refresh=False)
            last_thumb, n_seen, n_kept = None, 0, 0
            for idx, fr, orig in _iter_frames(s, step):
                n_seen += 1
                bar.update()
                thumb = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), (32, 18), interpolation=cv2.INTER_AREA)
                thumb = thumb.astype(np.float32)
                if last_thumb is not None and np.abs(thumb - last_thumb).mean() < dedup_thr:
                    continue
                last_thumb = thumb
                h0, w0 = fr.shape[:2]
                sc = min(1.0, max_side / max(h0, w0))
                if sc < 1.0:
                    fr = cv2.resize(fr, (round(w0 * sc), round(h0 * sc)), interpolation=cv2.INTER_AREA)
                if orig is not None and sc == 1.0:
                    fp = orig.resolve()
                else:
                    fp = frames_dir / f"{kf_id:07d}.jpg"
                    cv2.imwrite(str(fp), fr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                    n_copied += 1
                rows.append(dict(kf_id=kf_id, source_id=s.source_id, dataset=s.dataset, frame_idx=idx,
                                 t_sec=idx / s.fps, path=str(fp), width=fr.shape[1], height=fr.shape[0],
                                 orig_w=w0, orig_h=h0, scale=sc))
                kf_id += 1
                n_kept += 1
            per_source.append(dict(source_id=s.source_id, seen=n_seen, kept=n_kept))
        n_done = bar.n            # FRAME_COUNT відео приблизний: недобраний бар став би червоним
        bar.reset(total=n_done)
        bar.update(n_done)

    df = pd.DataFrame(rows)
    stats = dict(n_seen=sum(r["seen"] for r in per_source), n_kept=kf_id, n_copied=n_copied,
                 frames_dir=str(frames_dir), per_source=per_source)
    df.to_csv(meta_csv, index=False)
    stats_json.write_text(json.dumps(stats, ensure_ascii=False, indent=1))
    params_json.write_text(json.dumps(params))   # останнім: без нього кеш не вважається готовим
    df.attrs["stats"] = dict(stats, cached=False)
    return df


def make_tiles(kf: pd.DataFrame, grid: Optional[Tuple[int, int]] = (2, 3), overlap: float = 0.15) -> pd.DataFrame:
    """Тайл 0 — увесь кадр; далі сітка rows × cols з перекриттям (SAHI-подібно, слайд 23)."""
    rows = []
    for r in kf.itertuples(index=False):
        rows.append((r.kf_id, 0, 0, 0, r.width, r.height))
        if grid:
            gr, gc = grid
            tw, th = r.width / (gc - (gc - 1) * overlap), r.height / (gr - (gr - 1) * overlap)
            for t, (i, j) in enumerate(((i, j) for i in range(gr) for j in range(gc)), start=1):
                x0, y0 = j * tw * (1 - overlap), i * th * (1 - overlap)
                rows.append((r.kf_id, t, int(round(x0)), int(round(y0)),
                             int(round(min(r.width, x0 + tw))), int(round(min(r.height, y0 + th)))))
    return pd.DataFrame(rows, columns=["kf_id", "tile_idx", "x0", "y0", "x1", "y1"])


def embed_tiles(enc: ClipEncoder, kf: pd.DataFrame, tiles: pd.DataFrame, cache_dir: str | Path,
                batch_size: int = 64, force: bool = False, save_every: int = 10) -> np.ndarray:
    """
    Ембеддинги всіх тайлів (порядок рядків = порядок tiles) з кешем на диску.
    Ключ кешу — хеш тайлів, кадрів, моделі й препроцесингу. Рахуємо прямо у memmap і кожні save_every
    батчів фіксуємо прогрес: обірваний прогін (перезапуск ядра, Ctrl+C) продовжиться з місця зупинки.
    Конвеєр на два батчі: поки GPU рахує батч b, CPU уже декодує й ресайзить b + 1.
    """
    from concurrent.futures import ThreadPoolExecutor

    from tqdm.auto import tqdm

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    sig = hashlib.md5()
    sig.update(pd.util.hash_pandas_object(tiles, index=False).values.tobytes())
    sig.update(pd.util.hash_pandas_object(kf[["kf_id", "path"]], index=False).values.tobytes())
    sig.update(f"{enc.key}|{enc.resize_mode}|{getattr(enc, 'checkpoint', '')}".encode())
    f = cache_dir / f"emb_{enc.key}_{sig.hexdigest()[:10]}.npy"
    meta_f, part_f, prog_f = f.with_suffix(".json"), f.with_suffix(".partial.npy"), f.with_suffix(".partial.json")
    n = len(tiles)
    if f.exists() and not force:
        enc.cached = True
        if meta_f.exists():
            enc.img_per_s = json.loads(meta_f.read_text()).get("img_per_s")
        return np.load(f)
    enc.cached = False

    done, out = 0, None
    if part_f.exists() and prog_f.exists() and not force:
        out = np.lib.format.open_memmap(part_f, mode="r+")
        if out.shape == (n, enc.dim):
            done = int(json.loads(prog_f.read_text())["done"])
        else:
            del out
            out = None
    if out is None:
        out = np.lib.format.open_memmap(part_f, mode="w+", dtype=np.float32, shape=(n, enc.dim))

    paths = kf.set_index("kf_id")["path"].to_dict()
    boxes = tiles[["kf_id", "x0", "y0", "x1", "y1"]].to_numpy()

    def load_batch(i):
        bb = boxes[i:i + batch_size]
        ids = list(dict.fromkeys(bb[:, 0].tolist()))
        frames = dict(zip(ids, _cpu_pool().map(lambda k: Image.open(paths[k]).convert("RGB"), ids)))
        return enc.prepare_images([frames[k].crop((x0, y0, x1, y1)) for k, x0, y0, x1, y1 in bb])

    starts = list(range(done, n, batch_size))
    t0, n_new = time.time(), 0
    with tqdm(total=n, initial=done, desc=enc.key, unit="тайл", postfix="продовжуємо з кешу" if done else None) as bar, \
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="prefetch") as prefetch:
        nxt = prefetch.submit(load_batch, starts[0]) if starts else None
        for b, i in enumerate(starts):
            batch = nxt.result()
            if b + 1 < len(starts):
                nxt = prefetch.submit(load_batch, starts[b + 1])
            out[i:i + len(batch)] = enc.encode_prepared(batch)
            n_new += len(batch)
            bar.update(len(batch))
            if (b + 1) % save_every == 0:
                out.flush()
                prog_f.write_text(json.dumps(dict(done=i + len(batch))))
    out.flush()
    del out  # закрити memmap, інакше Windows не дасть перейменувати файл
    dt = time.time() - t0
    enc.img_per_s = n_new / max(dt, 1e-6) if n_new else None
    part_f.replace(f)
    prog_f.unlink(missing_ok=True)
    meta_f.write_text(json.dumps(dict(img_per_s=enc.img_per_s, n=n, seconds=dt, resumed_from=done)))
    return np.load(f)


def build_flat_index(emb: np.ndarray):
    """Точний пошук за скалярним добутком (= cos для нормованих векторів)."""
    import faiss

    idx = faiss.IndexIDMap2(faiss.IndexFlatIP(emb.shape[1]))
    idx.add_with_ids(np.ascontiguousarray(emb, dtype="float32"), np.arange(len(emb), dtype="int64"))
    return idx


def all_tile_scores(index, q: np.ndarray) -> np.ndarray:
    """Сирий cos для кожного тайла (у порядку рядків tiles)."""
    D, I = index.search(np.ascontiguousarray(q[None], dtype="float32"), index.ntotal)
    s = np.full(index.ntotal, -np.inf, dtype=np.float32)
    s[I[0]] = D[0]
    return s


def frame_scores(tile_s: np.ndarray, tiles: pd.DataFrame, n_frames: int, use_tiles: bool = True) -> np.ndarray:
    """Оцінка кадру = max по його тайлах (або лише тайл 0 = увесь кадр)."""
    mask = np.ones(len(tiles), bool) if use_tiles else (tiles["tile_idx"].to_numpy() == 0)
    out = np.full(n_frames, -np.inf, dtype=np.float32)
    np.maximum.at(out, tiles["kf_id"].to_numpy()[mask], tile_s[mask])
    return out


def top_frames(tile_s: np.ndarray, tiles: pd.DataFrame, kf: pd.DataFrame, k: int = 8, use_tiles: bool = True,
               exclude: Iterable[int] = ()) -> pd.DataFrame:
    """Топ-k кадрів за оцінками тайлів: для кожного кадру — тайл з найбільшою оцінкою (його рамку й показуємо)."""
    t = tiles.assign(score=tile_s)
    if not use_tiles:
        t = t[t.tile_idx == 0]
    t = t[~t.kf_id.isin(set(exclude))].sort_values("score", ascending=False).drop_duplicates("kf_id").head(k)
    res = t.merge(kf[["kf_id", "source_id", "t_sec", "path"]], on="kf_id", how="left")
    res.insert(0, "rank", np.arange(1, len(res) + 1))
    return res


# --------------------------------------------------------------------------------------
# Ground truth і метрики (слайд 46)
# --------------------------------------------------------------------------------------
VISDRONE_QUERIES: Dict[str, dict] = {
    "truck": dict(cats=[6], text="a truck"),
    "bus": dict(cats=[9], text="a bus"),
    "van": dict(cats=[5], text="a van"),
    "tricycle": dict(cats=[7, 8], text="a three-wheeled cargo tricycle"),
    "bicycle": dict(cats=[3], text="a person riding a bicycle"),
    "motorcycle": dict(cats=[10], text="a motorcycle"),
    "person": dict(cats=[1, 2], text="people walking"),
    "car": dict(cats=[4], text="a car"),
}


def _group_events(kf_sub: pd.DataFrame, gap_s: float) -> List[List[int]]:
    """Сусідні релевантні кадри одного джерела -> одна «подія» (одна ціль для оператора)."""
    events = []
    for _, g in kf_sub.sort_values(["source_id", "t_sec"]).groupby("source_id"):
        cur, last_t = [], None
        for r in g.itertuples(index=False):
            if last_t is not None and r.t_sec - last_t > gap_s:
                events.append(cur)
                cur = []
            cur.append(int(r.kf_id))
            last_t = r.t_sec
        if cur:
            events.append(cur)
    return events


def load_visdrone_gt(kf: pd.DataFrame, sources: Sequence[Source], queries: Dict[str, dict] = VISDRONE_QUERIES,
                     min_px: int = 24, max_occlusion: int = 1, gap_s: float = 3.0) -> Dict[str, dict]:
    """Авто-GT: кадр релевантний запиту, якщо на ньому є об'єкт потрібної категорії з короткою стороною ≥ min_px."""
    frame_cats: Dict[Tuple[str, int], set] = {}
    for s in sources:
        if not s.ann_path:
            continue
        a = load_ann(s.ann_path)
        a = a[(a.score == 1) & (a.cat > 0) & (np.minimum(a.w, a.h) >= min_px) & (a.occ <= max_occlusion)]
        for fr, g in a.groupby("frame"):
            frame_cats[(s.source_id, int(fr) - 1)] = set(g.cat.astype(int))
    gt = {}
    for qid, q in queries.items():
        sub = kf[[bool(frame_cats.get((r.source_id, int(r.frame_idx)), set()) & set(q["cats"]))
                  for r in kf.itertuples(index=False)]]
        gt[qid] = dict(text=q["text"], relevant=set(sub.kf_id.astype(int)), events=_group_events(sub, gap_s))
    return gt


def load_csv_gt(kf: pd.DataFrame, csv_path: str | Path, tol_s: float = 1.0) -> Dict[str, dict]:
    """Власний GT: CSV query_id,text,source_id,start_s,end_s — рядок на інтервал, де ціль видно ("*" — будь-яке джерело)."""
    gt: Dict[str, dict] = {}
    for qid, rows in pd.read_csv(csv_path).groupby("query_id"):
        events, rel = [], set()
        for r in rows.itertuples(index=False):
            m = (kf.t_sec >= r.start_s - tol_s) & (kf.t_sec <= r.end_s + tol_s)
            if str(r.source_id) != "*":
                m &= kf.source_id.astype(str) == str(r.source_id)
            ids = kf.loc[m, "kf_id"].astype(int).tolist()
            if ids:
                events.append(ids)
                rel |= set(ids)
            else:
                warnings.warn(f"{qid}: інтервал {r.source_id} {r.start_s}-{r.end_s} c не містить ключових кадрів")
        gt[str(qid)] = dict(text=str(rows.text.iloc[0]), relevant=rel, events=events)
    return gt


def gt_summary(gt: Dict[str, dict], n_frames: int) -> pd.DataFrame:
    return pd.DataFrame([dict(query=q, text=v["text"], relevant_frames=len(v["relevant"]), events=len(v["events"]),
                              prevalence=len(v["relevant"]) / max(n_frames, 1)) for q, v in gt.items()]
                        ).sort_values("prevalence")


def rank_metrics(fscore: np.ndarray, g: dict, ks: Sequence[int] = (10, 50)) -> dict:
    """
    EventR@k — частка подій з хоча б одним кадром у топ-k («чи потрапила ціль у те, що побачить оператор»);
    P@k — частка релевантних кадрів у топ-k («скільки сміття вгорі»); AP — average precision по кадрах.
    """
    n = len(fscore)
    order = np.argsort(-fscore, kind="stable")
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n)
    rel = np.zeros(n, bool)
    rel[list(g["relevant"])] = True
    out = dict(n_rel=int(rel.sum()), n_events=len(g["events"]))
    if not rel.any():
        return out
    ev_best = np.array([rank[e].min() for e in g["events"]])
    for k in ks:
        out[f"EventR@{k}"] = float((ev_best < k).mean())
        out[f"P@{k}"] = float(rel[order[:k]].mean())
    hits = rel[order]
    out["AP"] = float((np.cumsum(hits) / np.arange(1, n + 1) * hits).sum() / hits.sum())
    out["prevalence"] = float(rel.mean())
    return out


# --------------------------------------------------------------------------------------
# PQ: що втрачається, коли по каналу йдуть коди замість векторів (слайди 50–51)
# --------------------------------------------------------------------------------------
def build_pq(emb: np.ndarray, M: int = 64, nbits: int = 8, seed: int = 0, train: Optional[np.ndarray] = None):
    """
    IndexPQ (L2; для нормованих векторів порядок той самий, що й за cos). train — на чому вчити кодову книгу
    (у реальній системі великий архів на землі). Замало векторів для 2^nbits центроїдів — зменшуємо nbits;
    FAISS радить ≥ 39 на центроїд — попереджаємо один раз замість M однакових попереджень FAISS.
    """
    import faiss

    x = np.ascontiguousarray(emb, dtype="float32")
    xt = x if train is None else np.ascontiguousarray(train, dtype="float32")
    n, d = xt.shape
    assert d % M == 0, f"d={d} не ділиться на M={M}"
    nb = nbits
    while nb > 1 and n < (1 << nb):
        nb -= 1
    if nb != nbits:
        warnings.warn(f"PQ: {n} векторів для навчання менше, ніж {1 << nbits} центроїдів; використано nbits={nb}.")
    elif n < 39 * (1 << nb):
        warnings.warn(f"PQ: {n / (1 << nb):.0f} векторів на центроїд (FAISS радить ≥ 39): кодова книга грубша, "
                      "ніж була б на великому архіві, втрати PQ тут — оцінка зверху.")
    pq = faiss.IndexPQ(d, M, nb)
    pq.pq.cp.seed = seed
    pq.pq.cp.min_points_per_centroid = 1
    pq.train(xt)
    pq.add(x)
    return pq, dict(M=M, nbits=nb, code_bytes=M * nb / 8, fp16_bytes=d * 2, n_train=n)


def pq_tile_scores(pq, q: np.ndarray, emb: Optional[np.ndarray] = None, rerank_top: int = 0) -> np.ndarray:
    """
    Оцінки за PQ-кодами: cos ≈ 1 − d²/2 (FAISS L2 повертає квадрат відстані). rerank_top > 0 — топ-R кандидатів
    переранжовуємо точним cos по повних векторах («дозапит повних векторів для топу видачі», слайд 51).
    """
    D, I = pq.search(np.ascontiguousarray(q[None], dtype="float32"), pq.ntotal)
    s = np.empty(pq.ntotal, dtype=np.float32)
    s[I[0]] = 1.0 - D[0] / 2.0
    if rerank_top and emb is not None:
        top = I[0][:rerank_top]
        s[top] = emb[top] @ q + 10.0  # кандидати гарантовано вище за решту, порядок — за точним cos
    return s


# ======================================================================================
# Графіки
# ======================================================================================
def show_jpeg(fig=None, quality: int = 82):
    """Показ фігури з фотографіями як JPEG: PNG таких фігур важить мегабайти, а всередині віджета Output
    це ще й гальмує фронтенд. Графіки з лініями лишаємо звичайним plt.show()."""
    from IPython.display import Image as IPImage, display

    fig = fig or plt.gcf()
    buf = io.BytesIO()
    fig.savefig(buf, format="jpeg", dpi=90, bbox_inches="tight", pil_kwargs=dict(quality=quality))
    plt.close(fig)
    display(IPImage(buf.getvalue(), format="jpeg"))


def show_grid(images: Sequence[Image.Image], titles: Optional[Sequence[str]] = None, ncols: int = 6,
              size: float = 2.2, suptitle: str = "", colors: Optional[Sequence[str]] = None):
    nrows = int(np.ceil(len(images) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(size * ncols, size * nrows + (0.4 if suptitle else 0)), squeeze=False,
                             layout="constrained")
    for ax in axes.ravel():
        ax.axis("off")
    for i, im in enumerate(images):
        ax = axes.ravel()[i]
        ax.imshow(thumb(im, 320))
        if titles is not None:
            ax.set_title(titles[i], fontsize=8, color=colors[i] if colors else "black")
    if suptitle:
        fig.suptitle(suptitle)
    show_jpeg(fig)


def show_matrix(mats: Dict[str, np.ndarray], row_labels: Sequence, col_labels: Sequence[str], fmt: str = ".3f",
                cmap: str = "viridis", vmin=None, vmax=None, titles: Optional[Dict[str, str]] = None):
    """Матриці для кількох моделей поруч. row_labels — підписи або мініатюри (PIL) рядків."""
    thumbs = row_labels and isinstance(row_labels[0], Image.Image)
    n = len(mats) + bool(thumbs)
    fig, axes = plt.subplots(1, n, figsize=(0.85 * len(col_labels) * len(mats) + 1.6 * bool(thumbs) + 1.5,
                                            0.55 * len(row_labels) + 1.6),
                             gridspec_kw=dict(width_ratios=[0.3] * bool(thumbs) + [1] * len(mats)), squeeze=False,
                             layout="constrained")
    axes = axes[0]
    if thumbs:
        axes[0].imshow(np.concatenate([np.asarray(im.convert("RGB").resize((64, 64))) for im in row_labels]))
        axes[0].axis("off")
    for ax, (name, m) in zip(axes[bool(thumbs):], mats.items()):
        lo, hi = (m.min() if vmin is None else vmin), (m.max() if vmax is None else vmax)
        ax.imshow(m, cmap=cmap, vmin=lo, vmax=hi, aspect="auto")
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(j, i, format(m[i, j], fmt), ha="center", va="center", fontsize=7,
                        color="white" if m[i, j] < (lo + hi) / 2 else "black")
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=35, ha="right", fontsize=8)
        ax.set_yticks([] if thumbs else range(len(row_labels)))
        if not thumbs:
            ax.set_yticklabels(row_labels, fontsize=8)
        ax.set_title((titles or {}).get(name, name), fontsize=10)
    plt.show()


def confusion(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> np.ndarray:
    idx = {l: i for i, l in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)), int)
    for t, p in zip(y_true, y_pred):
        m[idx[t], idx[p]] += 1
    return m


def plot_confusions(cms: Dict[str, np.ndarray], labels: Sequence[str]):
    fig, axes = plt.subplots(1, len(cms), figsize=(5.2 * len(cms), 4.6), squeeze=False, layout="constrained")
    for ax, (name, m) in zip(axes[0], cms.items()):
        r = m / np.maximum(m.sum(1, keepdims=True), 1)
        ax.imshow(r, cmap="Blues", vmin=0, vmax=1)
        for i, j in zip(*np.nonzero(m)):
            ax.text(j, i, m[i, j], ha="center", va="center", fontsize=7, color="white" if r[i, j] > 0.5 else "black")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("передбачено")
        ax.set_ylabel("істина")
        ax.set_title(name)
    plt.show()


def plot_hist_pair(panels: Sequence[dict], xlabel: str, title: str = ""):
    """Гістограми (щільність) двох груп для кількох моделей: dict(title, pos, neg, pos_label, neg_label)."""
    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 3.4), squeeze=False, layout="constrained")
    for ax, p in zip(axes[0], panels):
        bins = np.linspace(min(p["pos"].min(), p["neg"].min()), max(p["pos"].max(), p["neg"].max()), 40)
        ax.hist(p["pos"], bins, alpha=0.6, color="tab:orange", density=True,
                label=f'{p.get("pos_label", "ціль")} (n={len(p["pos"])})')
        ax.hist(p["neg"], bins, alpha=0.6, color="tab:gray", density=True,
                label=f'{p.get("neg_label", "решта")} (n={len(p["neg"])})')
        ax.set_title(p["title"], fontsize=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("щільність")
        ax.legend(fontsize=8)
    if title:
        fig.suptitle(title)
    plt.show()


def plot_label_efficiency(R: pd.DataFrame, chance: Optional[float] = None, zero_shot: Optional[Dict[str, float]] = None,
                          xlabel: str = "розмічених кропів (бюджет)"):
    """
    Macro-recall від бюджету міток: панель на класифікатор, лінія на (модель, спосіб добору),
    крапки — «усі мітки», штрих-пунктир — zero-shot без жодної мітки (demo2).
    """
    clfs = list(R.clf.unique())
    models = list(dict.fromkeys(list(R.model.unique()) + list(zero_shot or {})))
    colors = {m: plt.cm.tab10(i) for i, m in enumerate(models)}
    styles = {"kmeans": "-", "random": "--", "per_class": "-"}
    fig, axes = plt.subplots(1, len(clfs), figsize=(6.5 * len(clfs), 4.2), squeeze=False, sharey=True,
                             layout="constrained")
    for ax, clf in zip(axes[0], clfs):
        sub = R[R.clf == clf]
        for m in sub.model.unique():
            for meth in [x for x in sub.method.unique() if x != "усі мітки"]:
                g = sub[(sub.model == m) & (sub.method == meth)].groupby("budget").macro_recall.agg(["mean", "std"])
                ax.plot(g.index, g["mean"], styles.get(meth, "-"), marker="o", ms=4, color=colors[m], label=f"{m} · {meth}")
                ax.fill_between(g.index, g["mean"] - g["std"], g["mean"] + g["std"], color=colors[m], alpha=0.1)
            full = sub[(sub.model == m) & (sub.method == "усі мітки")]
            if len(full):
                ax.axhline(full.macro_recall.iloc[0], color=colors[m], lw=0.8, ls=":")
        for m, v in (zero_shot or {}).items():
            ax.axhline(v, color=colors[m], lw=1.4, ls="-.", label=f"{m} · zero-shot, 0 міток")
        if chance:
            ax.axhline(chance, color="grey", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_title(f"{clf} · крапки — усі мітки train", fontsize=9)
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("macro-recall на test (інші відео)")
    axes[0, -1].legend(fontsize=7, ncol=2)
    plt.show()


def show_pairs(pairs: pd.DataFrame, title: str = ""):
    fig, axes = plt.subplots(len(pairs), 2, figsize=(9, 2.6 * len(pairs)), squeeze=False, layout="constrained")
    for r, p in enumerate(pairs.itertuples()):
        for c, (path, lbl) in enumerate(((p.path_a, p.a), (p.path_b, p.b))):
            axes[r, c].imshow(thumb(path, 640))
            axes[r, c].set_title(lbl + (f" · cos {p.cos:.3f}" if c == 0 else ""), fontsize=8)
            axes[r, c].axis("off")
    if title:
        fig.suptitle(title, fontsize=10)
    show_jpeg(fig)


def show_tiles(kf_row, tiles: pd.DataFrame, title: str = ""):
    import matplotlib.patches as patches

    fig, ax = plt.subplots(figsize=(8, 4.6))
    full = Image.open(kf_row["path"])
    ax.imshow(thumb(full, 960), extent=(0, full.width, full.height, 0))
    for r in tiles[(tiles.kf_id == kf_row["kf_id"]) & (tiles.tile_idx > 0)].itertuples(index=False):
        ax.add_patch(patches.Rectangle((r.x0, r.y0), r.x1 - r.x0, r.y1 - r.y0, fill=False, lw=1.5,
                                       ec=plt.cm.tab10(r.tile_idx % 10)))
        ax.text(r.x0 + 8, r.y0 + 28, str(r.tile_idx), color="w", fontsize=11, weight="bold")
    ax.set_title(title)
    ax.axis("off")
    show_jpeg(fig)


def show_results(res: pd.DataFrame, title: str = "", ncols: int = 4, relevant: Optional[set] = None, thumb_w: float = 3.6):
    """Видача: мініатюри з cos, таймкодом і рамкою тайла, що дав максимум; ✓ / ✗ — за GT."""
    import matplotlib.patches as patches

    if not len(res):
        print("порожня видача")
        return
    nrows = int(np.ceil(len(res) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(thumb_w * ncols, thumb_w * 0.75 * nrows + 0.3), squeeze=False, layout="constrained")
    for ax in axes.ravel():
        ax.axis("off")
    for ax, r in zip(axes.ravel(), res.itertuples(index=False)):
        full = Image.open(r.path)
        W0, H0 = full.size
        ax.imshow(thumb(full), extent=(0, W0, H0, 0))
        ok = relevant is None or r.kf_id in relevant
        if r.tile_idx != 0:
            ax.add_patch(patches.Rectangle((r.x0, r.y0), r.x1 - r.x0, r.y1 - r.y0, fill=False, lw=2, ec="#f1c40f"))
        mark = "" if relevant is None else (" ✓" if ok else " ✗")
        ax.set_title(f"#{r.rank} {r.score:.3f}{mark}\n{str(r.source_id).rsplit('/', 1)[-1]} · t={r.t_sec:.1f}s",
                     fontsize=8, color="#1e8449" if ok else "#c0392b")
    fig.suptitle(title, fontsize=10)
    show_jpeg(fig)


def plot_metrics(R: pd.DataFrame, metrics: Sequence[str] = ("EventR@50", "P@10"), baselines: Optional[Dict[str, float]] = None):
    """Метрика, усереднена по запитах, для кожної моделі × (prompt, input); пунктир — випадкове ранжування."""
    fig, axes = plt.subplots(1, len(metrics), figsize=(6.5 * len(metrics), 4), squeeze=False, layout="constrained")
    for ax, metric in zip(axes[0], metrics):
        R.groupby(["model", "prompt", "input"])[metric].mean().unstack(["prompt", "input"]).plot.bar(ax=ax, rot=0, width=0.8)
        ax.set_title(f"{metric} (середнє по {R['query'].nunique()} запитах)")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8, title="prompt / input")
        if metric in (baselines or {}):
            ax.axhline(baselines[metric], ls="--", c="k", lw=1)
            ax.text(-0.4, baselines[metric] + 0.02, "випадкове ранжування", fontsize=8)
    plt.show()


# ======================================================================================
# demo1 · що бачить DINO: PCA, норми токенів, увага [CLS]; подібність патчів за кліком
# ======================================================================================
def pca_rgb(patches: np.ndarray, q: float = 1.0) -> np.ndarray:
    """Перші 3 головні компоненти патч-фіч як RGB (h, w, 3) у [0, 1]. Знак і порядок компонент довільні."""
    gh, gw, d = patches.shape
    X = patches.reshape(-1, d).astype(np.float32)
    X -= X.mean(0)
    P = X @ np.linalg.svd(X, full_matrices=False)[2][:3].T
    lo, hi = np.percentile(P, q, axis=0), np.percentile(P, 100 - q, axis=0)
    return np.clip((P - lo) / np.maximum(hi - lo, 1e-6), 0, 1).reshape(gh, gw, 3)


def high_norm_tokens(patches: np.ndarray, k: float = 3.0) -> Tuple[np.ndarray, int]:
    """Норми патч-токенів і кількість «викидів» (> медіана + k·MAD) — артефакти, які прибирають регістри."""
    n = np.linalg.norm(patches, axis=-1)
    med = np.median(n)
    return n, int((n > med + k * 1.4826 * (np.median(np.abs(n - med)) + 1e-6)).sum())


def show_feature_maps(im: Image.Image, encs: Dict[str, DinoEncoder], max_side: int = 896, attn_max_side: int = 672,
                      title: str = ""):
    """Рядок на модель: кадр | PCA патчів | норми токенів | увага [CLS]."""
    fig, axes = plt.subplots(len(encs), 4, figsize=(17, 2.75 * len(encs) + 0.6), squeeze=False, layout="constrained")
    ext, small = (0, im.width, im.height, 0), thumb(im, 640)
    for r, (name, enc) in enumerate(encs.items()):
        f = enc.features(im, max_side)
        gh, gw = f["grid"]
        norms, n_out = high_norm_tokens(f["patches"])
        att = enc.cls_attention(im, attn_max_side)
        axes[r, 0].imshow(small, extent=ext)
        axes[r, 0].set_title(f"{name} · вхід {f['input'][0]}×{f['input'][1]} · patch {enc.patch} → {gw}×{gh} = {gh * gw} ток.",
                             fontsize=9)
        axes[r, 1].imshow(pca_rgb(f["patches"]), extent=ext, interpolation="nearest")
        axes[r, 1].set_title("PCA патч-фіч (3 компоненти = RGB)", fontsize=9)
        axes[r, 2].imshow(norms, extent=ext, cmap="magma", interpolation="nearest")
        axes[r, 2].set_title(f"‖токен‖ · високонормних: {n_out}", fontsize=9)
        axes[r, 3].imshow(small, extent=ext, alpha=0.35)
        axes[r, 3].imshow(att, extent=ext, cmap="viridis", alpha=0.75, interpolation="nearest", vmax=np.percentile(att, 99.5))
        axes[r, 3].set_title(f"увага [CLS], останній шар (≤{attn_max_side}px)", fontsize=9)
        for ax in axes[r]:
            ax.set_xlim(0, im.width)
            ax.set_ylim(im.height, 0)
            ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=11)
    show_jpeg(fig)


def _widgets():
    try:
        import ipywidgets as W
        from IPython.display import display
        return W, display
    except ImportError:
        print("ipywidgets не встановлено: pip install ipywidgets")
        return None, None


def feature_maps_panel(seqs: pd.DataFrame, encs: Dict[str, DinoEncoder], max_side: int = 896, attn_max_side: int = 672,
                       start_seq: Optional[str] = None):
    """Блок 1 demo1: послідовність, позиція кадру, роздільність входу."""
    W, display = _widgets()
    if W is None:
        return
    opts = [(f"{r.split}/{r.seq} ({r.width}×{r.height})", i) for i, r in seqs.iterrows()]
    start = next((i for i, r in seqs.iterrows() if r.seq == start_seq), opts[0][1])
    seq_dd = W.Dropdown(options=opts, value=start, description="послідовність", layout=W.Layout(width="60%"))
    pos_sl = W.FloatSlider(value=0.5, min=0, max=1, step=0.05, description="позиція", continuous_update=False)
    side_sl = W.IntSlider(value=max_side, min=224, max=1344, step=112, description="max side", continuous_update=False)
    out = W.Output()

    def go(_=None):
        r = seqs.loc[seq_dd.value]
        idx = int(round(pos_sl.value * (r.n_frames - 1)))
        with out:
            out.clear_output(wait=True)
            show_feature_maps(Image.open(frame_path(r, idx)), encs, side_sl.value, attn_max_side,
                              f"{r.split}/{r.seq} · кадр {idx + 1}/{r.n_frames}")

    for w in (seq_dd, pos_sl, side_sl):
        w.observe(go, names="value")
    display(W.VBox([W.HBox([seq_dd, pos_sl, side_sl]),
                    W.HTML("<i>max side кратний 112 = 7·16 = 8·14: однакова довша сторона для patch 14 і 16.</i>"), out]))
    if AUTORUN:
        go()


def largest_box(seq_row, idx: int, cats: Sequence[int] = (4,)) -> Optional[Tuple[float, float, float, float]]:
    """Найбільший bbox потрібних категорій на кадрі — стартова точка панелі подібності."""
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
    """По одному кадру з n послідовностей різних відео."""
    cand = seqs[seqs.split.isin(splits) & ~seqs.vid.isin(list(exclude_vids))].drop_duplicates("vid")
    cand = cand.sample(frac=1.0, random_state=seed).head(n)
    return [(f"{r.split}/{r.seq}", frame_path(r, int(pos * (r.n_frames - 1)))) for r in cand.itertuples()]


def _overlay(im: Image.Image, sim: np.ndarray, vmin: float, vmax: float, cmap: str = "inferno") -> np.ndarray:
    import matplotlib

    s = cv2.resize(np.clip((sim - vmin) / max(vmax - vmin, 1e-6), 0, 1).astype(np.float32), im.size,
                   interpolation=cv2.INTER_NEAREST)
    return 0.45 * np.asarray(im.convert("RGB"), dtype=np.float32) / 255 + 0.55 * matplotlib.colormaps[cmap](s)[..., :3]


def _jpeg(im: Image.Image, quality: int = 85) -> bytes:
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "JPEG", quality=quality)
    return buf.getvalue()


class PatchSimilarityPanel:
    """
    Клік по кадру-запиту -> косинус фічі цього патча (або окілу 3×3) до всіх патчів запиту й галереї.
    Рядок на модель; сітки патчів у моделей різні (14 vs 16), клік перераховується в кожну окремо.
    """
    PREVIEW_W = 760

    def __init__(self, encs: Dict[str, DinoEncoder], queries: List[Tuple[str, Path]], gallery: List[Tuple[str, Path]],
                 max_side: int = 896, start_xy: Optional[Tuple[float, float]] = None):
        import ipywidgets as W

        self.encs, self.queries, self.gallery, self.max_side = encs, queries, gallery, max_side
        self.g_ims = load_images(p for _, p in gallery)
        self.g_feats = {m: [l2n(e.features(im, max_side)["patches"]) for im in self.g_ims] for m, e in encs.items()}
        self.xy = start_xy
        self.q_dd = W.Dropdown(options=[(lbl, i) for i, (lbl, _) in enumerate(queries)], description="запит",
                               layout=W.Layout(width="55%"))
        self.rad = W.ToggleButtons(options=[("1 патч", 0), ("окіл 3×3", 1)], value=0)
        self.preview = W.Image(format="jpeg", layout=W.Layout(width=f"{self.PREVIEW_W}px"))
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
        self.ui = W.VBox([W.HBox([self.q_dd, self.rad]), W.HTML(f"<i>{hint}</i>"), self.preview, self.out])
        self._load(0, keep_xy=start_xy is not None, render=AUTORUN)

    def _load(self, i: int, keep_xy: bool = False, render: bool = True):
        self.q_im = Image.open(self.queries[i][1]).convert("RGB")
        self.q_feats = {m: l2n(e.features(self.q_im, self.max_side)["patches"]) for m, e in self.encs.items()}
        if not keep_xy:
            self.xy = (self.q_im.width / 2, self.q_im.height / 2)
        if render:
            self._render()

    def _on_click(self, ev):
        rw, rh = ev.get("boundingRectWidth"), ev.get("boundingRectHeight")
        if rw and rh:
            self.set_point(ev.get("relativeX", 0) / rw * self.q_im.width, ev.get("relativeY", 0) / rh * self.q_im.height)

    def set_point(self, x: float, y: float):
        self.xy = (float(x), float(y))
        self._render()

    def _query_vec(self, F: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        gh, gw, _ = F.shape
        x, y = self.xy
        r, c = min(int(y / self.q_im.height * gh), gh - 1), min(int(x / self.q_im.width * gw), gw - 1)
        k = self.rad.value
        return l2n(F[max(r - k, 0):r + k + 1, max(c - k, 0):c + k + 1].reshape(-1, F.shape[-1]).mean(0)), (r, c)

    def _render(self):
        pv = self.q_im.copy()
        x, y = self.xy
        ImageDraw.Draw(pv).ellipse([x - 9, y - 9, x + 9, y + 9], outline=(0, 255, 255), width=3)
        self.preview.value = _jpeg(pv.resize((self.PREVIEW_W, round(pv.height * self.PREVIEW_W / pv.width))))
        n_g = len(self.g_ims)
        with self.out:
            self.out.clear_output(wait=True)
            fig, axes = plt.subplots(len(self.encs), n_g + 1, figsize=(2.9 * (n_g + 1), 2.0 * len(self.encs) + 0.5),
                                     squeeze=False, layout="constrained")
            for r, m in enumerate(self.encs):
                q, (pr, pc) = self._query_vec(self.q_feats[m])
                s_q, s_g = self.q_feats[m] @ q, [F @ q for F in self.g_feats[m]]
                # Шкала — сирі cos від медіани до максимуму: окремо для запиту (сам об'єкт дає ~1)
                # і спільна для галереї рядка, щоб кадри галереї порівнювались чесно.
                s_q_wo = s_q.copy()
                s_q_wo[pr, pc] = np.nan
                pool = np.concatenate([s.ravel() for s in s_g])
                vmin, vmax = np.percentile(pool, 50), pool.max()
                axes[r, 0].imshow(_overlay(thumb(self.q_im), s_q, np.nanpercentile(s_q_wo, 50), np.nanmax(s_q_wo)))
                axes[r, 0].set_title(f"{m} · запит · галерея: cos {vmin:.2f}…{vmax:.2f}", fontsize=8)
                for j, (im, s) in enumerate(zip(self.g_ims, s_g)):
                    axes[r, j + 1].imshow(_overlay(thumb(im), s, vmin, vmax))
                    axes[r, j + 1].set_title(f"{self.gallery[j][0].split('/')[-1][:18]} · max {s.max():.2f}", fontsize=8)
                for ax in axes[r]:
                    ax.axis("off")
            show_jpeg(fig)


def patch_similarity_panel(encs, queries, gallery, max_side: int = 896, start_xy=None):
    W, display = _widgets()
    if W is None:
        return None
    p = PatchSimilarityPanel(encs, queries, gallery, max_side, start_xy)
    display(p.ui)
    return p


# ======================================================================================
# demo2 · пісочниця zero-shot
# ======================================================================================
def zero_shot_playground(images: Dict[str, Image.Image], score_fn: Callable, models: Sequence[str],
                         prompts: Sequence[str], image: Optional[str] = None, text: str = "", rotate: int = 0,
                         invert: bool = False):
    """
    Пісочниця zero-shot (слайди 35–39). Що рахувати, вирішує ноутбук:
        score_fn(model, image: PIL, prompts: list[str]) -> dict(cos=array, prob=array, prob_name=str)
    Віджет змінює кадр (напис — слайд 37, поворот — слайд 38, інверсія ІЧ — слайд 39) і список промптів.
    """
    W, display = _widgets()
    if W is None:
        return
    L = W.Layout
    img_dd = W.Dropdown(options=list(images), value=image or next(iter(images)), description="кадр", layout=L(width="45%"))
    txt = W.Text(value=text, description="напис", placeholder="напр. AMBULANCE", continuous_update=False)
    rot = W.ToggleButtons(options=[("0°", 0), ("90°", 90), ("180°", 180), ("270°", 270)], value=rotate)
    rot.style.button_width = "48px"
    inv = W.Checkbox(value=invert, description="інвертувати (white-hot ↔ black-hot)", indent=False)
    area = W.Textarea(value="\n".join(prompts), description="промпти", layout=L(width="60%", height="150px"),
                      continuous_update=False)
    out = W.Output()

    def go(_=None):
        im = transform(images[img_dd.value], txt.value.strip(), rot.value, inv.value)
        ps = [p.strip() for p in area.value.splitlines() if p.strip()]
        with out:
            out.clear_output(wait=True)
            if not ps:
                print("додайте хоча б один промпт")
                return
            fig = plt.figure(figsize=(5.5 + 4.2 * len(models), max(3.2, 0.32 * len(ps) + 1.4)), layout="constrained")
            gs = fig.add_gridspec(1, 1 + 2 * len(models), width_ratios=[2.4] + [1, 1] * len(models))
            ax = fig.add_subplot(gs[0])
            ax.imshow(thumb(im, 400))
            ax.axis("off")
            for k, m in enumerate(models):
                res = score_fn(m, im, ps)
                best = int(np.argmax(res["cos"]))
                for j, (key, name) in enumerate((("cos", "сирий cos"), ("prob", res["prob_name"]))):
                    a = fig.add_subplot(gs[1 + 2 * k + j])
                    v = res[key]
                    a.barh(range(len(ps)), v, color=["tab:orange" if i == best else "tab:gray" for i in range(len(ps))])
                    a.set_yticks(range(len(ps)))
                    a.set_yticklabels(ps if k == 0 and j == 0 else [], fontsize=8)
                    a.invert_yaxis()
                    a.set_title(f"{m}: {name}", fontsize=9)
                    for i, x in enumerate(v):
                        a.text(x, i, f" {x:.3f}" if key == "cos" else f" {x:.2f}", va="center", fontsize=7)
                    a.set_xlim(min(0, v.min()), v.max() * 1.35 + 1e-6)
                    a.tick_params(axis="x", labelsize=7)
            show_jpeg(fig)

    for w in (img_dd, txt, rot, inv, area):
        w.observe(go, names="value")
    display(W.VBox([W.HBox([img_dd, txt]), W.HBox([rot, inv]), area, out]))
    if AUTORUN:
        go()


# ======================================================================================
# demo3 · панелі пошуку і порогів
# ======================================================================================
def search_box(on_query: Callable[..., None], models: Sequence[str], gt_texts: Optional[Dict[str, str]] = None,
               value: str = "a truck parked next to a building", gt_query: Optional[str] = None):
    """
    Текстовий пошук (ipywidgets). Що робити із запитом, вирішує ноутбук:
        on_query(text, models=[...], k=int, use_tiles=bool, aerial=bool, gt_query=str|None)
    gt_texts — {query_id: text}: вибір запиту з GT підставляє його текст і вмикає позначки ✓ / ✗.
    """
    W, display = _widgets()
    if W is None:
        return
    gt_texts = gt_texts or {}
    box = W.Text(value=gt_texts.get(gt_query, value), placeholder="опис цілі англійською", layout=W.Layout(width="55%"))
    btn = W.Button(description="Шукати", button_style="primary", icon="search")
    gt_dd = W.Dropdown(options=[("— без GT —", None)] + [(f"{q}: {t}", q) for q, t in gt_texts.items()],
                       value=gt_query, description="GT", layout=W.Layout(width="45%"))
    model_cb = [W.Checkbox(value=True, description=m, indent=False, layout=W.Layout(width="auto")) for m in models]
    k_sl = W.IntSlider(value=8, min=4, max=24, step=4, description="top-k", continuous_update=False)
    tiles_tb = W.ToggleButtons(options=[("тайли", True), ("увесь кадр", False)], value=True)
    prompt_tb = W.ToggleButtons(options=[("aerial", True), ("plain", False)], value=True,
                                tooltips=["ансамбль промптів з ракурсом", "лише сам опис"])
    for tb in (tiles_tb, prompt_tb):
        tb.style.button_width = "90px"
    out = W.Output()

    def _pick_gt(ch):
        if ch["new"] is not None:
            box.value = gt_texts[ch["new"]]

    def _go(_=None):
        sel = [m for m, cb in zip(models, model_cb) if cb.value]
        with out:
            out.clear_output(wait=True)
            if not box.value.strip() or not sel:
                print("введіть запит і виберіть хоча б одну модель")
                return
            # GT-позначки мають сенс, лише поки текст не змінено вручну
            gq = gt_dd.value if gt_dd.value is not None and box.value == gt_texts[gt_dd.value] else None
            on_query(box.value.strip(), models=sel, k=k_sl.value, use_tiles=tiles_tb.value, aerial=prompt_tb.value,
                     gt_query=gq)

    gt_dd.observe(_pick_gt, names="value")
    btn.on_click(_go)
    box.on_submit(_go)
    display(W.VBox([W.HBox([box, btn]), gt_dd, W.HBox([W.Label("моделі:"), *model_cb]),
                    W.HBox([k_sl, tiles_tb, prompt_tb]), out]))
    if AUTORUN:
        _go()


def zones_panel(get_scores: Callable[[str, str], Tuple[np.ndarray, np.ndarray]], models: Sequence[str],
                queries: Dict[str, str], sec_per_frame: float = 2.0):
    """
    Три зони впевненості (слайд 49). get_scores(model, query) -> (оцінки кадрів, маска «з ціллю») рахує ноутбук.
    Повзунки T_low / T_high — у сирому cos; таблиця: скільки кадрів у кожній зоні, скільки цілей автоматично
    відкинуто і скільки часу оператора коштує зона 2 + 3 (sec_per_frame секунд на кадр).
    """
    W, display = _widgets()
    if W is None:
        return
    m_dd = W.Dropdown(options=list(models), description="модель")
    q_dd = W.Dropdown(options=[(f"{q}: {t}", q) for q, t in queries.items()], description="запит", layout=W.Layout(width="45%"))
    lo = W.FloatSlider(description="T_low", step=0.001, readout_format=".3f", continuous_update=False, layout=W.Layout(width="48%"))
    hi = W.FloatSlider(description="T_high", step=0.001, readout_format=".3f", continuous_update=False, layout=W.Layout(width="48%"))
    out = W.Output()
    state = {}

    def load(_=None):
        fs, rel = get_scores(m_dd.value, q_dd.value)
        state.update(fs=fs, rel=rel, loading=True)     # зміна меж повзунків не має перемальовувати тричі
        a, b = float(fs.min()), float(fs.max())
        with lo.hold_trait_notifications(), hi.hold_trait_notifications():
            lo.min = hi.min = -1.0
            lo.max = hi.max = 1.0
            lo.min, lo.max, hi.min, hi.max = a, b, a, b
            # старт: T_low зберігає 90 % кадрів з ціллю, T_high пропускає 1 % порожніх
            lo.value = float(np.quantile(fs[rel], 0.10)) if rel.any() else a
            hi.value = max(lo.value, float(np.quantile(fs[~rel], 0.99)))
        state["loading"] = False
        draw()

    def draw(_=None):
        if "fs" not in state:
            return load()
        if state.get("loading"):
            return
        fs, rel = state["fs"], state["rel"]
        t_lo, t_hi = lo.value, max(lo.value, hi.value)
        z = np.digitize(fs, [t_lo, t_hi])          # 0 — пропуск, 1 — оператор, 2 — пріоритет
        with out:
            out.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(9, 3.2), layout="constrained")
            bins = np.linspace(fs.min(), fs.max(), 60)
            ax.hist(fs[~rel], bins, alpha=0.6, color="tab:gray", log=True, label="порожні кадри")
            ax.hist(fs[rel], bins, alpha=0.8, color="tab:orange", log=True, label="кадри з ціллю")
            for x0, x1, c in ((fs.min(), t_lo, "#dddddd"), (t_lo, t_hi, "#9ecae1"), (t_hi, fs.max(), "#a1d99b")):
                ax.axvspan(x0, x1, color=c, alpha=0.35, lw=0)
            ax.set_xlabel("сирий cos із запитом (не softmax)")
            ax.legend(fontsize=8)
            plt.show()
            rows = []
            for k, name in enumerate(["1 · автоматичний пропуск", "2 · увага оператора", "3 · пріоритетне оповіщення"]):
                msk = z == k
                rows.append({"зона": name, "кадрів": int(msk.sum()), "з ціллю": int((msk & rel).sum()),
                             "частка цілей": f"{(msk & rel).sum() / max(rel.sum(), 1):.0%}",
                             "точність зони": f"{(msk & rel).sum() / max(msk.sum(), 1):.0%}"})
            from IPython.display import display as _d

            _d(pd.DataFrame(rows).set_index("зона"))
            print(f"Оператор переглядає {int((z > 0).sum())} кадрів ≈ {(z > 0).sum() * sec_per_frame / 60:.0f} хв "
                  f"(по {sec_per_frame:.0f} с на кадр); автоматично відкинуто {(z == 0)[rel].mean():.0%} кадрів з ціллю.")

    m_dd.observe(load, names="value")
    q_dd.observe(load, names="value")
    lo.observe(draw, names="value")
    hi.observe(draw, names="value")
    display(W.VBox([W.HBox([m_dd, q_dd]), W.HBox([lo, hi]), out]))
    if AUTORUN:
        load()


def image_search_panel(on_query: Callable[..., None], kf: pd.DataFrame, models: Sequence[str],
                       gt_texts: Optional[Dict[str, str]] = None, start_kf: Optional[int] = None,
                       start_box: Optional[Tuple[int, int, int, int]] = None, text: str = "", gt_query: Optional[str] = None):
    """
    Пошук за зразком (ipywidgets): кадр з архіву або своє фото, регіон рамкою мишею (ipyevents) або слайдерами,
    необов'язковий текст-модифікатор з вагою λ (слайд 42). Що робити з регіоном, вирішує ноутбук:
        on_query(crop: PIL, text=str, lam=float, models=[...], k=int, use_tiles=bool, src_kf=int|None, gt_query=str|None)
    """
    W, display = _widgets()
    if W is None:
        return None
    p = _ImageQueryPanel(W, on_query, kf, models, gt_texts or {}, start_kf, start_box, text, gt_query)
    display(p.ui)
    if AUTORUN:
        p._go()
    return p


class _ImageQueryPanel:
    PREVIEW_W = 640     # ширина прев'ю в пікселях екрана
    MIN_BOX = 16        # мінімальна сторона рамки в пікселях кадру
    DRAG_MIN = 4        # менший рух миші — це клік, а не протягування

    def __init__(self, W, on_query, kf, models, gt_texts, start_kf, start_box, text, gt_query):
        self.on_query, self.kf, self.models = on_query, kf.reset_index(drop=True), models
        self.photo: Optional[Image.Image] = None
        self.src_kf: Optional[int] = None
        self._drag: Optional[Tuple[float, float]] = None
        L = W.Layout
        self.frame_sl = W.IntSlider(min=0, max=max(len(self.kf) - 1, 0), description="кадр", continuous_update=False,
                                    layout=L(width="55%"))
        self.frame_lbl = W.Label()
        self.upload = W.FileUpload(accept="image/*", multiple=False, description="Обрати фото")
        self.preview = W.Image(format="jpeg", layout=L(width=f"{self.PREVIEW_W}px"))
        self.crop_img = W.Image(format="jpeg", layout=L(max_width="200px", max_height="200px"))
        self.crop_lbl = W.Label()
        self.x_rng = W.IntRangeSlider(description="x", continuous_update=False, layout=L(width=f"{self.PREVIEW_W}px"))
        self.y_rng = W.IntRangeSlider(description="y", continuous_update=False, layout=L(width=f"{self.PREVIEW_W}px"))
        self.full_btn = W.Button(description="Увесь кадр", icon="expand")
        self.go_btn = W.Button(description="Шукати схожі", button_style="primary", icon="search")
        self.text = W.Text(value=text, description="+ текст", placeholder="напр. many cars (необов'язково)",
                           layout=L(width="45%"))
        self.lam = W.FloatSlider(value=1.0, min=0, max=3, step=0.25, description="λ тексту", continuous_update=False)
        self.model_cb = [W.Checkbox(value=True, description=m, indent=False, layout=L(width="auto")) for m in models]
        self.k_sl = W.IntSlider(value=8, min=4, max=24, step=4, description="top-k", continuous_update=False)
        self.tiles_tb = W.ToggleButtons(options=[("тайли", True), ("увесь кадр", False)], value=True)
        self.tiles_tb.style.button_width = "90px"
        self.gt_dd = W.Dropdown(options=[("— без GT —", None)] + [(f"{q}: {t}", q) for q, t in gt_texts.items()],
                                value=gt_query, description="GT", layout=L(width="45%"))
        self.out = W.Output()
        try:
            from ipyevents import Event

            self.mouse = Event(source=self.preview, prevent_default_action=True,
                               watched_events=["mousedown", "mousemove", "mouseup", "mouseleave"],
                               throttle_or_debounce="throttle", wait=40)
            self.mouse.on_dom_event(self._on_mouse)
            hint = "Протягніть мишею по кадру, щоб виділити регіон; слайдери x / y — для точного налаштування."
        except ImportError:
            hint = "Миша недоступна (немає ipyevents): регіон — слайдерами x / y."
        self.frame_sl.observe(lambda ch: self._load_kf(ch["new"]), names="value")
        self.upload.observe(self._on_upload, names="value")
        for s in (self.x_rng, self.y_rng):
            s.observe(lambda _: self._redraw(), names="value")
        self.full_btn.on_click(lambda _: self._set_box(0, 0, *self.photo.size))
        self.go_btn.on_click(self._go)
        self.ui = W.VBox([
            W.HBox([self.frame_sl, self.frame_lbl]), W.HBox([self.upload, self.full_btn]), W.HTML(f"<i>{hint}</i>"),
            W.HBox([self.preview, W.VBox([W.Label("регіон-запит:"), self.crop_img, self.crop_lbl])]),
            self.x_rng, self.y_rng, W.HBox([self.text, self.lam]),
            W.HBox([W.Label("моделі:"), *self.model_cb]), W.HBox([self.k_sl, self.tiles_tb]),
            W.HBox([self.gt_dd, self.go_btn]), self.out])
        row = 0
        if start_kf is not None and (self.kf.kf_id == start_kf).any():
            row = int(np.flatnonzero(self.kf.kf_id.to_numpy() == start_kf)[0])
        self.frame_sl.value = row
        self._load_kf(row)
        if start_box is not None:
            self._set_box(*start_box)

    def _set_photo(self, im: Image.Image, src_kf: Optional[int], label: str):
        self.photo, self.src_kf = im.convert("RGB"), src_kf
        w, h = self.photo.size
        self.frame_lbl.value = label
        with self.x_rng.hold_trait_notifications(), self.y_rng.hold_trait_notifications():
            self.x_rng.max, self.y_rng.max = w, h
            self.x_rng.value, self.y_rng.value = (0, w), (0, h)
        self._redraw()

    def _load_kf(self, row: int):
        if len(self.kf):
            r = self.kf.iloc[row]
            self._set_photo(Image.open(r.path), int(r.kf_id), f"{r.source_id} · t={r.t_sec:.1f}s · kf {r.kf_id}")

    def _on_upload(self, ch):
        if not ch["new"]:
            return
        f = ch["new"][0]
        try:
            im = Image.open(io.BytesIO(bytes(f["content"])))
        except Exception as e:  # noqa: BLE001
            self.frame_lbl.value = f"не вдалося відкрити {f['name']}: {type(e).__name__} (потрібен JPG або PNG)"
            return
        self._set_photo(im, None, f"своє фото: {f['name']} ({im.width}×{im.height})")

    def _box(self) -> Tuple[int, int, int, int]:
        (x0, x1), (y0, y1) = self.x_rng.value, self.y_rng.value
        return x0, y0, x1, y1

    def _set_box(self, x0, y0, x1, y1):
        w, h = self.photo.size
        x0, x1 = sorted((int(round(x0)), int(round(x1))))
        y0, y1 = sorted((int(round(y0)), int(round(y1))))
        if x1 - x0 < self.MIN_BOX:
            x0, x1 = max(0, min(x0, w - self.MIN_BOX)), min(w, max(x0, 0) + self.MIN_BOX)
        if y1 - y0 < self.MIN_BOX:
            y0, y1 = max(0, min(y0, h - self.MIN_BOX)), min(h, max(y0, 0) + self.MIN_BOX)
        with self.x_rng.hold_trait_notifications(), self.y_rng.hold_trait_notifications():
            self.x_rng.value, self.y_rng.value = (x0, x1), (y0, y1)
        self._redraw()

    def crop(self) -> Image.Image:
        return self.photo.crop(self._box())

    def _redraw(self):
        if self.photo is None:
            return
        w, h = self.photo.size
        sc = self.PREVIEW_W / w
        pv = self.photo.resize((self.PREVIEW_W, max(1, round(h * sc))), Image.BILINEAR)
        x0, y0, x1, y1 = self._box()
        if (x0, y0, x1, y1) != (0, 0, w, h):
            # затемнюємо все поза регіоном, щоб рамку було видно на будь-якому фоні
            mask = Image.new("L", pv.size, 110)
            ImageDraw.Draw(mask).rectangle([x0 * sc, y0 * sc, x1 * sc, y1 * sc], fill=0)
            pv = Image.composite(Image.new("RGB", pv.size), pv, mask)
            ImageDraw.Draw(pv).rectangle([x0 * sc, y0 * sc, x1 * sc, y1 * sc], outline=(241, 196, 15), width=2)
        self.preview.value = _jpeg(pv)
        cr = self.crop()
        cr.thumbnail((200, 200))
        self.crop_img.value = _jpeg(cr, 90)
        self.crop_lbl.value = f"{x1 - x0}×{y1 - y0} px"

    def _event_xy(self, ev) -> Optional[Tuple[float, float]]:
        """Координати події в пікселях кадру; розмір рамки — з події, бо браузер міг відмасштабувати прев'ю."""
        rw, rh = ev.get("boundingRectWidth"), ev.get("boundingRectHeight")
        if not rw or not rh or self.photo is None:
            return None
        w, h = self.photo.size
        return min(max(ev.get("relativeX", 0) / rw * w, 0), w), min(max(ev.get("relativeY", 0) / rh * h, 0), h)

    def _on_mouse(self, ev):
        pt = self._event_xy(ev)
        if pt is None:
            return
        kind = ev.get("type")
        if kind == "mousedown":
            self._drag = pt
        elif self._drag is not None and kind in ("mousemove", "mouseup", "mouseleave"):
            (sx, sy), (x, y) = self._drag, pt
            if abs(x - sx) >= self.DRAG_MIN or abs(y - sy) >= self.DRAG_MIN:
                self._set_box(sx, sy, x, y)
            if kind != "mousemove":
                self._drag = None

    def _go(self, _=None):
        sel = [m for m, cb in zip(self.models, self.model_cb) if cb.value]
        with self.out:
            self.out.clear_output(wait=True)
            if self.photo is None or not sel:
                print("виберіть зображення і хоча б одну модель")
                return
            self.on_query(self.crop(), text=self.text.value.strip(), lam=self.lam.value, models=sel, k=self.k_sl.value,
                          use_tiles=self.tiles_tb.value, src_kf=self.src_kf, gt_query=self.gt_dd.value)
