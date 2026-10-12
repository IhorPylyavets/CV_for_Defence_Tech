"""
Заняття 6 · спільний код трьох демо-ноутбуків.

    lesson06_demo1_sam.ipynb                  SAM / SAM 2.1: промпти, неоднозначність, де зникає час   (LIVE DEMO 1, слайд 25)
    lesson06_demo2_ov_detectors.ipynb         OV-детектори: S = O·Pᵀ, промпт, точність / час, тайлінг (LIVE DEMO 2, слайд 43)
    lesson06_demo3_grounded_sam2_video.ipynb  Grounded-SAM 2 на відео: пам'ять, оклюзії, ціна, ІЧ   (LIVE DEMO 3, слайд 55)

Конвенція курсу: формули, які пояснюються на слайдах (кількість токенів, S = O·Pᵀ, T(k) = T_enc + k·T_dec,
формула тайлів, T_frame, байти на об'єкт), рахуються в клітинках ноутбуків; тут — дані, моделі, кеші, метрики,
графіки й віджети.

Ваги лежать в artifacts/models/{sam,ov,ultralytics}/ (tools/fetch_sam_models.py, tools/fetch_ov_models.py).
torch / transformers / sam2 / ultralytics імпортуються ліниво: модуль можна імпортувати ще до того,
як setup_hf() вирішить, чи працювати з Hugging Face офлайн.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import os
import re
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

# ======================================================================================
# Шляхи (від кореня репозиторію — не залежать від того, звідки запущено ноутбук чи tools/)
# ======================================================================================
REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = REPO_ROOT / "artifacts"
VISDRONE_DIR = ARTIFACTS / "visdrone"
DET_VAL_DIR = VISDRONE_DIR / "VisDrone2019-DET-val"     # tools/fetch_visdrone.py --dataset visdrone-det
VID_VAL_DIR = VISDRONE_DIR / "VisDrone2019-VID-val"     # tools/fetch_visdrone.py --splits val
IMAGES_DIR = ARTIFACTS / "images"                       # пара RGB / LWIR із заняття 1
SAM_DIR = ARTIFACTS / "models" / "sam"                  # tools/fetch_sam_models.py
OV_DIR = ARTIFACTS / "models" / "ov"                    # tools/fetch_ov_models.py
ULTRA_DIR = ARTIFACTS / "models" / "ultralytics"        # YOLOE, YOLO-World, YOLO11, MobileSAM, FastSAM + текстові енкодери


def hub(models_dir: Path) -> Path:
    """Кеш Hugging Face всередині artifacts: тека самодостатня, її можна перенести в закритий контур."""
    return models_dir / "hf_home" / "hub"


# Віджети одразу малюють перший результат. У nbconvert / nbclient це варто вимкнути (PANELS_AUTORUN=0) — як у заняття 5.
AUTORUN = os.environ.get("PANELS_AUTORUN", "1") != "0"

# ======================================================================================
# Реєстр моделей. Ключ — те, що пишемо в ноутбуці й у --models скриптів завантаження.
# ======================================================================================
# kind: "sam2" — офіційний пакет sam2 (API зі слайдів 22 і 53); "hf-*" — transformers; "ultra" — ultralytics.
SAM_ZOO: Dict[str, dict] = {
    "sam-vit-base":   dict(kind="hf-sam", repo="facebook/sam-vit-base", label="SAM ViT-B"),
    "sam-vit-huge":   dict(kind="hf-sam", repo="facebook/sam-vit-huge", label="SAM ViT-H"),
    "sam-hq-vit-base": dict(kind="hf-samhq", repo="syscv-community/sam-hq-vit-base", label="HQ-SAM ViT-B"),
    "sam2.1-tiny":    dict(kind="sam2", repo="facebook/sam2.1-hiera-tiny", label="SAM 2.1 T"),
    "sam2.1-small":   dict(kind="sam2", repo="facebook/sam2.1-hiera-small", label="SAM 2.1 S"),
    "sam2.1-base-plus": dict(kind="sam2", repo="facebook/sam2.1-hiera-base-plus", label="SAM 2.1 B+"),
    "sam2.1-large":   dict(kind="sam2", repo="facebook/sam2.1-hiera-large", label="SAM 2.1 L"),
    "mobile-sam":     dict(kind="ultra", file="mobile_sam.pt", label="MobileSAM"),
    "fastsam":        dict(kind="ultra", file="FastSAM-s.pt", label="FastSAM-s"),
    # gated: SAM License (не Apache) — прийміть умови на https://huggingface.co/facebook/sam3 і `hf auth login`
    "sam3":           dict(kind="hf-sam3", repo="facebook/sam3", label="SAM 3", gated=True),
}
SAM_DEFAULT = ["sam-vit-base", "sam-vit-huge", "sam-hq-vit-base", "sam2.1-tiny", "sam2.1-small",
               "sam2.1-base-plus", "sam2.1-large", "mobile-sam", "fastsam"]

OV_ZOO: Dict[str, dict] = {
    "gdino-tiny":    dict(kind="hf-gdino", repo="IDEA-Research/grounding-dino-tiny", label="Grounding DINO-T"),
    "gdino-base":    dict(kind="hf-gdino", repo="IDEA-Research/grounding-dino-base", label="Grounding DINO-B"),
    "owlv2":         dict(kind="hf-owlv2", repo="google/owlv2-base-patch16-ensemble", label="OWLv2-B/16"),
    "florence2":     dict(kind="hf-florence2", repo="florence-community/Florence-2-large", label="Florence-2-L"),
    "qwen2.5-vl":    dict(kind="hf-qwen", repo="Qwen/Qwen2.5-VL-3B-Instruct", label="Qwen2.5-VL-3B"),
    # ultralytics: ваги + текстовий енкодер (YOLOE — MobileCLIP, YOLO-World — CLIP ViT-B/32) в ULTRA_DIR
    "yoloe":         dict(kind="ultra-yoloe", file="yoloe-11l-seg.pt", label="YOLOE-11L"),
    "yoloworld":     dict(kind="ultra-world", file="yolov8x-worldv2.pt", label="YOLO-World-X"),
    "yolo11":        dict(kind="ultra-yolo", file="yolo11m.pt", label="YOLO11m (COCO)"),
}
OV_DEFAULT = ["gdino-tiny", "gdino-base", "owlv2", "florence2", "yoloe", "yoloworld", "yolo11"]

# Роздільність, до якої модель зводить кадр (для розрахунку «пікселів на цілі», слайд 45)
INPUT_RES = {
    "SAM / SAM 2 (довша сторона → 1024)": ("long", 1024),
    "Grounding DINO (довша ≤ 1333, коротша 800)": ("gdino", (800, 1333)),
    "OWLv2 (квадрат 960²)": ("long", 960),
    "YOLO, imgsz = 1280": ("long", 1280),
    "CLIP / ViT-B (224²)": ("long", 224),
}


# ======================================================================================
# Середовище
# ======================================================================================
def _manifest(models_dir: Path) -> dict:
    f = models_dir / "manifest.json"
    return json.loads(f.read_text()) if f.exists() else {}


def have(name: str) -> bool:
    """Чи лежать ваги моделі в artifacts (за маніфестом скрипта завантаження)."""
    d = SAM_DIR if name in SAM_ZOO else OV_DIR
    return name in _manifest(d)


def setup_hf(sam: Sequence[str] = (), ov: Sequence[str] = ()) -> str:
    """
    Офлайн-режим Hugging Face, якщо tools/fetch_*_models.py уже записали в маніфести всі потрібні моделі:
    жодних звернень до Hub. Діє лише до першого імпорту huggingface_hub, тож викликати в першій клітинці
    (після перезапуску ядра). Повертає "offline" / "online". Gated SAM 3, якого немає в маніфесті, не заважає:
    ноутбук просто пропустить розділ із ним.
    """
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    quiet()
    need = [(SAM_DIR, m) for m in sam if not SAM_ZOO[m].get("gated") or have(m)] + [(OV_DIR, m) for m in ov]
    if need and all(m in _manifest(d) for d, m in need):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    return "offline" if os.environ.get("HF_HUB_OFFLINE") == "1" else "online"


def quiet():
    """Прибирає шум, що не стосується демо: torch.jit.script у sam2 і відсутнє CUDA-розширення sam2._C
    (без нього пропускається лише заповнення дрібних дірок у масках — див. README)."""
    warnings.filterwarnings("ignore", message=r".*torch\.jit\.(script|load).*")
    warnings.filterwarnings("ignore", message=r"cannot import name '_C'")
    warnings.filterwarnings("ignore", message=r".*memory_attention_rope_theta.*")


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


def gpu_name(device: str) -> str:
    import torch

    if device.startswith("cuda"):
        return torch.cuda.get_device_name(0)
    return "Apple MPS" if device == "mps" else "CPU"


def free_memory():
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# ======================================================================================
# Таймінг. На GPU без synchronize() міряється час запуску ядер, а не їх виконання.
# ======================================================================================
def sync():
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


@contextlib.contextmanager
def timed(store: Optional[list] = None):
    sync()
    t0 = time.perf_counter()
    yield
    sync()
    if store is not None:
        store.append((time.perf_counter() - t0) * 1000)


def bench(fn: Callable, n_warmup: int = 3, n_iter: int = 10) -> dict:
    """Медіана / p90 / максимум у мс. На борту важить найгірший випадок, а не середнє (слайд 59)."""
    for _ in range(n_warmup):
        fn()
    ts: list = []
    for _ in range(n_iter):
        with timed(ts):
            fn()
    a = np.array(ts)
    return {"median_ms": float(np.median(a)), "p90_ms": float(np.percentile(a, 90)), "max_ms": float(a.max())}


def autocast(device: str):
    """bf16 autocast на GPU (рекомендація SAM 2), no-op на CPU / MPS."""
    import torch

    if device.startswith("cuda"):
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def n_params(module) -> int:
    return sum(p.numel() for p in module.parameters())


# ======================================================================================
# Завантаження моделей
# ======================================================================================
def _need(name: str, models_dir: Path, script: str):
    if name not in _manifest(models_dir) and os.environ.get("HF_HUB_OFFLINE") == "1":
        raise FileNotFoundError(f"{name}: ваг немає в {models_dir}. Запустіть `python tools/{script} --models {name}`")


def sam2_checkpoint(name: str, download: bool = True) -> Tuple[str, Path]:
    """(конфіг hydra, шлях до .pt) для SAM 2.1. Чекпойнт — у кеші HF всередині artifacts/models/sam."""
    from huggingface_hub import hf_hub_download
    from sam2.build_sam import HF_MODEL_ID_TO_FILENAMES

    repo = SAM_ZOO[name]["repo"]
    cfg, fname = HF_MODEL_ID_TO_FILENAMES[repo]
    path = hf_hub_download(repo, fname, cache_dir=str(hub(SAM_DIR)), local_files_only=not download)
    return cfg, Path(path)


def load_sam2_image(name: str = "sam2.1-large", device: str = "cuda"):
    """SAM2ImagePredictor (офіційний пакет sam2): set_image() — енкодер, predict() — prompt encoder + декодер."""
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    _need(name, SAM_DIR, "fetch_sam_models.py")
    cfg, ckpt = sam2_checkpoint(name, download=os.environ.get("HF_HUB_OFFLINE") != "1")
    return SAM2ImagePredictor(build_sam2(cfg, str(ckpt), device=device))


def load_sam2_video(name: str = "sam2.1-small", device: str = "cuda"):
    """SAM2VideoPredictor: пам'ять, голова оклюзії, поширення по відео (слайди 28–29)."""
    from sam2.build_sam import build_sam2_video_predictor

    _need(name, SAM_DIR, "fetch_sam_models.py")
    cfg, ckpt = sam2_checkpoint(name, download=os.environ.get("HF_HUB_OFFLINE") != "1")
    return build_sam2_video_predictor(cfg, str(ckpt), device=device)


def load_hf(name: str, device: str = "cpu", dtype=None):
    """(processor, model) для моделей transformers із SAM_ZOO / OV_ZOO."""
    import transformers as T

    T.logging.set_verbosity_error()          # без попереджень про застарілі поля конфігів
    zoo, d = (SAM_ZOO, SAM_DIR) if name in SAM_ZOO else (OV_ZOO, OV_DIR)
    spec = zoo[name]
    _need(name, d, "fetch_sam_models.py" if d == SAM_DIR else "fetch_ov_models.py")
    kind, repo, cache = spec["kind"], spec["repo"], str(hub(d))
    classes = {
        "hf-sam": (T.SamProcessor, T.SamModel),
        "hf-samhq": (T.SamHQProcessor, T.SamHQModel),
        "hf-sam3": (T.Sam3Processor, T.Sam3Model),
        "hf-gdino": (T.AutoProcessor, T.AutoModelForZeroShotObjectDetection),
        "hf-owlv2": (T.Owlv2Processor, T.Owlv2ForObjectDetection),
        "hf-florence2": (T.AutoProcessor, T.Florence2ForConditionalGeneration),
        "hf-qwen": (T.AutoProcessor, T.Qwen2_5_VLForConditionalGeneration),
    }
    P, M = classes[kind]
    proc = P.from_pretrained(repo, cache_dir=cache)
    kw = {"dtype": dtype} if dtype is not None else {}
    model = M.from_pretrained(repo, cache_dir=cache, **kw).to(device).eval()
    return proc, model


@contextlib.contextmanager
def ultra_home():
    """
    Ultralytics кладе ваги й текстові енкодери в поточну теку (mobileclip_blt.ts) і в settings.weights_dir (CLIP ViT-B/32).
    На час завантаження переходимо в artifacts/models/ultralytics і підміняємо теку для CLIP — без зміни глобальних налаштувань.
    """
    import ultralytics.nn.text_model as tm

    ULTRA_DIR.mkdir(parents=True, exist_ok=True)
    cwd, wd = os.getcwd(), tm.WEIGHTS_DIR
    os.chdir(ULTRA_DIR)
    tm.WEIGHTS_DIR = ULTRA_DIR
    try:
        yield ULTRA_DIR
    finally:
        os.chdir(cwd)
        tm.WEIGHTS_DIR = wd


def load_ultra(name: str, classes: Optional[Sequence[str]] = None):
    """YOLOE / YOLO-World / YOLO11 / MobileSAM / FastSAM. Для OV-моделей classes «запікаються» одразу (слайд 41)."""
    from ultralytics import SAM, YOLO, YOLOE, FastSAM, YOLOWorld

    spec = SAM_ZOO.get(name) or OV_ZOO[name]
    with ultra_home():
        f = spec["file"]
        if name == "mobile-sam":
            return SAM(f)
        if name == "fastsam":
            return FastSAM(f)
        if spec["kind"] == "ultra-yoloe":
            m = YOLOE(f)
            if classes:
                m.set_classes(list(classes), m.get_text_pe(list(classes)))
            return m
        if spec["kind"] == "ultra-world":
            m = YOLOWorld(f)
            if classes:
                m.set_classes(list(classes))
            return m
        return YOLO(f)


def set_ultra_classes(model, classes: Sequence[str]):
    """Нова «таблиця класів» для YOLOE / YOLO-World: текстовий енкодер працює лише тут, на землі (слайд 41)."""
    from ultralytics import YOLOE

    with ultra_home():
        if isinstance(model, YOLOE):
            model.set_classes(list(classes), model.get_text_pe(list(classes)))
        else:
            model.set_classes(list(classes))


# ======================================================================================
# Завантаження ваг (викликається з tools/fetch_sam_models.py і tools/fetch_ov_models.py)
# ======================================================================================
_HF_PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"]


def download_model(name: str) -> List[Path]:
    """Тягне ваги однієї моделі в artifacts/models/... і повертає список файлів (для маніфесту з SHA-256)."""
    spec = SAM_ZOO.get(name) or OV_ZOO[name]
    models_dir = SAM_DIR if name in SAM_ZOO else OV_DIR
    models_dir.mkdir(parents=True, exist_ok=True)
    kind = spec["kind"]
    if kind == "sam2":
        _, ckpt = sam2_checkpoint(name, download=True)
        return [ckpt]
    if kind.startswith("hf-"):
        from huggingface_hub import snapshot_download

        local = Path(snapshot_download(spec["repo"], allow_patterns=_HF_PATTERNS, cache_dir=str(hub(models_dir))))
        return sorted(p for p in local.rglob("*") if p.is_file())
    # ultralytics: самі ваги + текстовий енкодер, який підтягується при першому set_classes
    before = {p for p in ULTRA_DIR.rglob("*") if p.is_file()} if ULTRA_DIR.exists() else set()
    m = load_ultra(name)
    if kind == "ultra-yoloe":
        set_ultra_classes(m, ["car"])
    elif kind == "ultra-world":
        set_ultra_classes(m, ["car"])
    after = {p for p in ULTRA_DIR.rglob("*") if p.is_file()}
    files = {ULTRA_DIR / spec["file"]} | (after - before)
    if kind == "ultra-yoloe":
        files |= set(ULTRA_DIR.glob("mobileclip*.ts"))
    if kind == "ultra-world":
        files |= set((ULTRA_DIR / "clip").glob("*.pt"))
    return sorted(f for f in files if f.exists())


def fetch_models(names: Sequence[str], models_dir: Path) -> List[str]:
    """Завантажує моделі, пише manifest.json (шляхи й SHA-256). Повертає список тих, що не вдалося."""
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    from huggingface_hub.errors import GatedRepoError

    manifest_f = models_dir / "manifest.json"
    models_dir.mkdir(parents=True, exist_ok=True)
    manifest = _manifest(models_dir)
    failed = []
    for name in names:
        spec = SAM_ZOO.get(name) or OV_ZOO[name]
        print(f"\n=== {name}: {spec.get('repo') or spec.get('file')}")
        try:
            files = download_model(name)
        except GatedRepoError:
            print(f"  ✗ gated: прийміть ліцензію на https://huggingface.co/{spec['repo']} і залогіньтесь (`hf auth login`)")
            failed.append(name)
            continue
        except Exception as e:  # noqa: BLE001 — мережа, квоти, відсутній пакет: решту моделей однаково тягнемо
            print(f"  ✗ {type(e).__name__}: {' '.join(str(e).split())[:300]}")
            failed.append(name)
            continue
        size = sum(f.stat().st_size for f in files) / 2**20
        print(f"  OK: {len(files)} файлів, {size:.0f} МБ")
        manifest[name] = dict(spec=spec, files=[dict(path=Path(os.path.relpath(f.resolve(), REPO_ROOT)).as_posix(),
                                                     sha256=_sha256(f.resolve())) for f in files])
        manifest_f.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\nManifest: {manifest_f}")
    return failed


# ======================================================================================
# Дані: VisDrone2019-DET-val (кадри) і VisDrone2019-VID-val (відео з треками й оклюзіями)
# ======================================================================================
# Категорії VisDrone: 0 ignored, 1 pedestrian, 2 people, 3 bicycle, 4 car, 5 van, 6 truck, 7 tricycle,
# 8 awning-tricycle, 9 bus, 10 motor, 11 others.
# Одна «наша» категорія = кілька категорій VisDrone (van — це теж car для OV-запиту «car»).
EVAL_GROUPS = {"person": [1, 2], "car": [4, 5], "truck": [6], "bus": [9]}
VEHICLES = (4, 5, 6, 9)
VID_COLS = ["frame", "tid", "x", "y", "w", "h", "score", "cat", "trunc", "occ"]


@dataclass
class GT:
    boxes: np.ndarray      # (N, 4) xyxy
    cats: np.ndarray       # (N,) категорія VisDrone
    ignore: np.ndarray     # (M, 4) xyxy — ignored regions (cat 0): детекції там не штрафуються


def ensure_det_val():
    if not (DET_VAL_DIR / "images").is_dir():
        raise FileNotFoundError(f"Немає {DET_VAL_DIR}. Запустіть `python tools/fetch_visdrone.py --dataset visdrone-det`")


def ensure_vid_val():
    if not (VID_VAL_DIR / "sequences").is_dir():
        raise FileNotFoundError(f"Немає {VID_VAL_DIR}. Запустіть `python tools/fetch_visdrone.py --splits val`")


def load_det_ann(txt: Path) -> GT:
    """Рядок DET: x,y,w,h,score,category,truncation,occlusion. score = 0 — ignored region."""
    rows = [[int(float(t)) for t in ln.strip().strip(",").split(",")[:8]]
            for ln in Path(txt).read_text().strip().splitlines() if ln.strip()]
    a = np.array(rows, dtype=np.float32).reshape(-1, 8)
    xyxy = np.stack([a[:, 0], a[:, 1], a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]], 1)
    cat = a[:, 5].astype(int)
    ign = (cat == 0) | (a[:, 4] == 0)
    keep = ~ign & (cat != 11) & (a[:, 2] > 0) & (a[:, 3] > 0)
    return GT(xyxy[keep], cat[keep], xyxy[ign])


def det_items(n: Optional[int] = None, seed: int = 0, offset: int = 0) -> List[Tuple[Path, GT]]:
    """[(шлях кадру, GT)] у детермінованому порядку; n — випадкова підвибірка, offset — зсув (для dev / test)."""
    ensure_det_val()
    imgs = sorted((DET_VAL_DIR / "images").glob("*.jpg"))
    if n is not None and n < len(imgs):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(imgs))[offset:offset + n]
        imgs = sorted(imgs[i] for i in idx)
    return [(p, load_det_ann(DET_VAL_DIR / "annotations" / (p.stem + ".txt"))) for p in imgs]


def imread_rgb(p) -> np.ndarray:
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(p)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def image_size(p) -> Tuple[int, int]:
    """(W, H) без декодування пікселів."""
    with Image.open(p) as im:
        return im.size


def vid_sequences() -> pd.DataFrame:
    ensure_vid_val()
    rows = []
    for s in sorted((VID_VAL_DIR / "sequences").iterdir()):
        fs = sorted(s.glob("*.jpg"))
        w, h = image_size(fs[0])
        rows.append(dict(seq=s.name, frames=len(fs), W=w, H=h))
    return pd.DataFrame(rows)


def vid_annotations(seq: str) -> pd.DataFrame:
    """frame (з 1), tid, x, y, w, h, score, cat, trunc, occ (0 — видно, 1 — частково, 2 — сильно перекрито)."""
    a = pd.read_csv(VID_VAL_DIR / "annotations" / f"{seq}.txt", header=None, names=VID_COLS)
    return a[(a.cat > 0) & (a.cat < 11) & (a.w > 0) & (a.h > 0)].reset_index(drop=True)


@dataclass
class Clip:
    seq: str
    dir: Path                 # теку з кадрами 00000.jpg, 00001.jpg, … — формат SAM2VideoPredictor
    frame_ids: List[int]      # номер кадру VisDrone для кожного кадру кліпу
    scale: float              # кліп / оригінал
    size: Tuple[int, int]     # (W, H) кліпу

    @property
    def frames(self) -> List[Path]:
        return [self.dir / f"{i:05d}.jpg" for i in range(len(self.frame_ids))]

    def read(self, i: int) -> np.ndarray:
        return imread_rgb(self.frames[i])


def make_clip(seq: str, start: int, n: int, stride: int, out_root: Path, max_side: int = 1920) -> Clip:
    """
    Вирізає шматок послідовності VisDrone-VID у теку 00000.jpg… (SAM 2 читає саме так) і зменшує до max_side.
    SAM 2 однаково стисне кадр до 1024, а повнороздільні маски 4K на сотню кадрів не влізуть у RAM.
    """
    ensure_vid_val()
    src = sorted((VID_VAL_DIR / "sequences" / seq).glob("*.jpg"))
    ids = [int(p.stem) for p in src][start - 1:start - 1 + n * stride:stride]
    W, H = image_size(src[0])
    s = min(1.0, max_side / max(W, H))
    size = (round(W * s), round(H * s))
    d = Path(out_root) / f"{seq}_{start}_{n}x{stride}_{size[0]}"
    if len(list(d.glob("*.jpg"))) != len(ids):
        d.mkdir(parents=True, exist_ok=True)
        for k, fid in enumerate(ids):
            im = cv2.imread(str(src[fid - 1]))
            if s < 1:
                im = cv2.resize(im, size, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(d / f"{k:05d}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return Clip(seq, d, ids, s, size)


def clip_gt(clip: Clip, ann: pd.DataFrame, cats: Sequence[int] = VEHICLES) -> pd.DataFrame:
    """GT кліпу в його координатах: k (кадр кліпу), tid, box xyxy, occ."""
    a = ann[ann.cat.isin(cats) & ann.frame.isin(clip.frame_ids)].copy()
    k_of = {f: k for k, f in enumerate(clip.frame_ids)}
    a["k"] = a.frame.map(k_of)
    s = clip.scale
    a["x0"], a["y0"], a["x1"], a["y1"] = a.x * s, a.y * s, (a.x + a.w) * s, (a.y + a.h) * s
    return a[["k", "tid", "cat", "x0", "y0", "x1", "y1", "occ"]].reset_index(drop=True)


# ======================================================================================
# Геометрія, тайлінг, NMS
# ======================================================================================
def box_iou(a, b) -> np.ndarray:
    a = np.asarray(a, np.float32).reshape(-1, 4)
    b = np.asarray(b, np.float32).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-9)


def box_ios(a, b) -> np.ndarray:
    """Перетин / площа МЕНШОГО боксу: 1.0, якщо один бокс цілком усередині іншого (половина машини на межі тайла)."""
    a = np.asarray(a, np.float32).reshape(-1, 4)
    b = np.asarray(b, np.float32).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])  # noqa: E731
    return inter / (np.minimum(area(a)[:, None], area(b)[None]) + 1e-9)


def mask_to_box(m: np.ndarray) -> Optional[np.ndarray]:
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], np.float32)


def tile_origins(W: int, H: int, t: int = 640, o: float = 0.2) -> List[Tuple[int, int]]:
    """Лівий верхній кут кожного тайла; кількість — за формулою слайду 46, тайли рівномірно до краю кадру."""
    step = t * (1 - o)
    nx = 1 if W <= t else math.ceil((W - t) / step) + 1
    ny = 1 if H <= t else math.ceil((H - t) / step) + 1
    xs = np.linspace(0, max(0, W - t), nx).round().astype(int)
    ys = np.linspace(0, max(0, H - t), ny).round().astype(int)
    return [(int(x), int(y)) for y in ys for x in xs]


@dataclass
class Dets:
    boxes: np.ndarray                      # (N, 4) xyxy у пікселях кадру
    scores: np.ndarray                     # (N,)
    labels: np.ndarray                     # (N,) індекс класу; -1 — фраза не зіставилась із жодним класом
    names: Optional[List[str]] = None      # фраза / назва класу, як її повернула модель

    def __len__(self):
        return len(self.boxes)

    def take(self, idx) -> "Dets":
        idx = np.asarray(idx)
        return Dets(self.boxes[idx], self.scores[idx], self.labels[idx],
                    None if self.names is None else [self.names[i] for i in np.atleast_1d(idx)])

    @staticmethod
    def empty() -> "Dets":
        return Dets(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, int), [])


def nms(d: Dets, iou: float = 0.5, class_aware: bool = True) -> Dets:
    import torch
    import torchvision

    if len(d) == 0:
        return d
    b, s = torch.as_tensor(d.boxes, dtype=torch.float32), torch.as_tensor(d.scores, dtype=torch.float32)
    l = torch.as_tensor(d.labels if class_aware else np.zeros(len(d), int), dtype=torch.int64)
    keep = torchvision.ops.batched_nms(b, s, l, iou).numpy()
    return d.take(keep)


def merge_contained(d: Dets, ios: float = 0.85) -> Dets:
    """
    Склейка тайлів, як GREEDYNMM у SAHI: бокс того самого класу, що майже весь лежить у більшому (IoS ≥ ios), —
    це шматок об'єкта, обрізаний межею тайла. Лишаємо більший бокс із максимальним score пари.
    """
    if len(d) < 2:
        return d
    area = (d.boxes[:, 2] - d.boxes[:, 0]) * (d.boxes[:, 3] - d.boxes[:, 1])
    scores, keep = d.scores.copy(), []
    for i in np.argsort(-area):
        if keep:
            k = np.asarray(keep)
            hit = np.flatnonzero((box_ios(d.boxes[i], d.boxes[k])[0] >= ios) & (d.labels[k] == d.labels[i]))
            if len(hit):
                scores[k[hit[0]]] = max(scores[k[hit[0]]], scores[i])
                continue
        keep.append(i)
    out = d.take(np.array(keep, int))
    out.scores = scores[np.array(keep, int)]
    return out


def detect_tiled(detect_fn: Callable[[np.ndarray], Dets], img: np.ndarray, t: int = 640, o: float = 0.2,
                 with_full: bool = True, iou: float = 0.5, ios: Optional[float] = 0.85) -> Tuple[Dets, int]:
    """SAHI-подібний прохід: тайли з перекриттям + (опційно) повний кадр; склейка — class-aware NMS і злиття
    вкладених боксів (ios=None — лише NMS). Повертає (Dets, проходів)."""
    H, W = img.shape[:2]
    parts, passes = [], 0
    for x, y in tile_origins(W, H, t, o):
        d = detect_fn(img[y:y + t, x:x + t])
        passes += 1
        if len(d):
            parts.append(Dets(d.boxes + np.array([x, y, x, y], np.float32), d.scores, d.labels, d.names))
    if with_full:
        d = detect_fn(img)
        passes += 1
        if len(d):
            parts.append(d)
    if not parts:
        return Dets.empty(), passes
    cat = Dets(np.concatenate([p.boxes for p in parts]), np.concatenate([p.scores for p in parts]),
               np.concatenate([p.labels for p in parts]), sum([list(p.names or [""] * len(p)) for p in parts], []))
    merged = nms(cat, iou)
    return (merge_contained(merged, ios) if ios else merged), passes


# ======================================================================================
# Обгортки OV-детекторів: однаковий вихід Dets
# ======================================================================================
def phrase_to_class(phrase: str, classes: Sequence[str]) -> int:
    """GDINO повертає фрагмент промпта (іноді склеєний із сусіднього класу). Перший клас, що входить у фразу."""
    p = phrase.lower()
    for i, c in enumerate(classes):
        if c.lower() in p:
            return i
    return -1


class GDino:
    """Grounding DINO / MM-Grounding DINO через transformers. Будь-яка зміна тексту — повний прохід (слайди 37–38)."""

    def __init__(self, name: str = "gdino-base", device: str = "cuda"):
        self.name, self.device = name, device
        self.proc, self.model = load_hf(name, device)

    def raw(self, img: np.ndarray, prompt: str):
        import torch

        inp = self.proc(images=Image.fromarray(img), text=prompt, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            return inp, self.model(**inp)

    def __call__(self, img: np.ndarray, prompt: str, classes: Optional[Sequence[str]] = None,
                 box_thr: float = 0.35, text_thr: float = 0.25) -> Dets:
        inp, out = self.raw(img, prompt)
        r = self.proc.post_process_grounded_object_detection(
            out, inp["input_ids"], threshold=box_thr, text_threshold=text_thr, target_sizes=[img.shape[:2]])[0]
        boxes = r["boxes"].float().cpu().numpy().reshape(-1, 4)
        # text_labels — рядки (transformers ≥ 4.51); при нулі боксів буває список не тієї довжини
        names = [str(t) for t in (r["text_labels"] if "text_labels" in r else r["labels"])][:len(boxes)]
        classes = classes or [c.strip() for c in prompt.split(".") if c.strip()]
        return Dets(boxes, r["scores"].float().cpu().numpy()[:len(boxes)],
                    np.array([phrase_to_class(t, classes) for t in names], int).reshape(-1), names)


class Owl:
    """
    OWLv2 через transformers. Ознаки кадру рахуються один раз і кешуються, текст «зустрічається» з ними лише
    в кінці — тому новий запит майже нічого не коштує (гілка «aligned», слайд 35).
    ПАСТКА (слайд 36): кадр доповнюється до квадрата справа / знизу, тож координати — у квадраті S × S, S = max(H, W).
    """

    def __init__(self, name: str = "owlv2", device: str = "cuda"):
        self.name, self.device = name, device
        self.proc, self.model = load_hf(name, device)
        self._key, self._feats = None, None

    def image_features(self, img: np.ndarray, cache: bool = True):
        """(feature_map (1, h, w, d), image_feats (1, h·w, d)). Кеш — за вмістом кадру (id масиву перевикористовується
        для тимчасових тайлів, і кеш за id віддавав би ознаки чужого тайла)."""
        import torch

        key = (img.shape, hashlib.blake2b(np.ascontiguousarray(img[::7, ::7]).tobytes(), digest_size=16).hexdigest())
        if not cache or key != self._key:
            px = self.proc(images=Image.fromarray(img), return_tensors="pt")["pixel_values"].to(self.device)
            with torch.inference_mode():
                fmap = self.model.image_embedder(pixel_values=px)[0]
            b, h, w, d = fmap.shape
            self._key, self._feats = key, (fmap, fmap.reshape(b, h * w, d))
        return self._feats

    def text_embeds(self, queries: Sequence[str]):
        import torch

        tok = self.proc(text=[list(queries)], return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = self.model.owlv2.get_text_features(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"])
        return getattr(out, "pooler_output", out)          # transformers 5 повертає ModelOutput, 4.x — тензор

    def patch_embeds(self, img: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """O: L2-нормовані ембеддинги регіонів (по одному на патч-токен), і сітка (h, w)."""
        import torch

        fmap, feats = self.image_features(img)
        with torch.inference_mode():
            emb = self.model.class_head.dense0(feats)[0]
        emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return emb.float().cpu().numpy(), tuple(fmap.shape[1:3])

    def query_matrix(self, queries: Sequence[str]) -> np.ndarray:
        """P: L2-нормовані ембеддинги текстових запитів."""
        t = self.text_embeds(queries)
        t = t / t.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return t.float().cpu().numpy()

    def calibrate(self, S: np.ndarray, img: np.ndarray) -> np.ndarray:
        """Логіти моделі з «сирого» S = O·Pᵀ: на кожен патч свій зсув і масштаб (logit_shift, logit_scale)."""
        import torch

        _, feats = self.image_features(img)
        h = self.model.class_head
        with torch.inference_mode():
            shift = h.logit_shift(feats)[0].float().cpu().numpy()
            scale = (h.elu(h.logit_scale(feats)) + 1)[0].float().cpu().numpy()
        return (S + shift) * scale

    def boxes(self, img: np.ndarray, square: bool = True) -> np.ndarray:
        """Бокс на кожен патч-токен у пікселях. square=False — та сама пастка, що й target_size=(H, W)."""
        import torch

        fmap, feats = self.image_features(img)
        with torch.inference_mode():
            b = self.model.box_predictor(feats, fmap)[0].float().cpu().numpy()   # cx, cy, w, h у [0, 1] квадрата
        H, W = img.shape[:2]
        sx, sy = (max(H, W), max(H, W)) if square else (W, H)
        return np.stack([(b[:, 0] - b[:, 2] / 2) * sx, (b[:, 1] - b[:, 3] / 2) * sy,
                         (b[:, 0] + b[:, 2] / 2) * sx, (b[:, 1] + b[:, 3] / 2) * sy], 1)

    def __call__(self, img: np.ndarray, queries: Sequence[str], thr: float = 0.1, nms_iou: float = 0.3,
                 square: bool = True) -> Dets:
        O, _ = self.patch_embeds(img)
        logits = self.calibrate(O @ self.query_matrix(queries).T, img)
        prob = 1 / (1 + np.exp(-logits))
        lab, sc = prob.argmax(1), prob.max(1)
        keep = sc >= thr
        b = self.boxes(img, square)[keep]
        H, W = img.shape[:2]
        b[:, [0, 2]] = b[:, [0, 2]].clip(0, W)
        b[:, [1, 3]] = b[:, [1, 3]].clip(0, H)
        d = Dets(b, sc[keep], lab[keep], [queries[i] for i in lab[keep]])
        return nms(d, nms_iou)

    def exemplar_embeds(self, img: np.ndarray, boxes: np.ndarray, iou: float = 0.4) -> np.ndarray:
        """
        Запит зразком (слайд 36): ембеддинг регіону-зразка замість тексту. Беремо патч-токени, чиї бокси лягають на
        бокс зразка (IoU ≥ iou), і усереднюємо їх ембеддинги. Рядки результату — нові рядки матриці P.
        """
        O, _ = self.patch_embeds(img)
        B = self.boxes(img)
        out = []
        for b in np.asarray(boxes, np.float32).reshape(-1, 4):
            m = box_iou(b, B)[0]
            idx = np.flatnonzero(m >= iou) if (m >= iou).any() else [int(m.argmax())]
            v = O[idx].mean(0)
            out.append(v / np.linalg.norm(v))
        return np.stack(out)

    def search(self, img: np.ndarray, Q: np.ndarray, thr: float = 0.75, nms_iou: float = 0.3) -> Dets:
        """Пошук за ембеддингами-зразками: S = O·Qᵀ, максимум по зразках, поріг на сирий косинус.
        Калібрування shift / scale навчене під текстові запити, тому для зразків порогуємо cos напряму."""
        O, _ = self.patch_embeds(img)
        S = (O @ np.asarray(Q).T).max(1)
        k = S >= thr
        H, W = img.shape[:2]
        b = self.boxes(img)[k]
        b[:, [0, 2]] = b[:, [0, 2]].clip(0, W)
        b[:, [1, 3]] = b[:, [1, 3]].clip(0, H)
        return nms(Dets(b, S[k], np.zeros(int(k.sum()), int), ["зразок"] * int(k.sum())), nms_iou)


def ultra_detect(model, img: np.ndarray, conf: float = 0.25, imgsz: int = 1280, **kw) -> Dets:
    """YOLOE / YOLO-World / YOLO11. ПАСТКА: ultralytics чекає numpy у BGR (як cv2); RGB мовчки погіршить якість."""
    r = model.predict(np.ascontiguousarray(img[..., ::-1]), conf=conf, imgsz=imgsz, verbose=False, **kw)[0]
    lab = r.boxes.cls.cpu().numpy().astype(int)
    return Dets(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), lab, [r.names[int(i)] for i in lab])


def ultra_visual_prompt(model, img: np.ndarray, ref_img: np.ndarray, ref_boxes: np.ndarray,
                        conf: float = 0.1, imgsz: int = 1280) -> Dets:
    """YOLOE visual prompt: бокс-зразок на референсному кадрі → пошук на цільовому. Текстовий енкодер не потрібен."""
    from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

    vp = dict(bboxes=np.asarray(ref_boxes, np.float32).reshape(-1, 4), cls=np.zeros(len(ref_boxes), int))
    r = model.predict(np.ascontiguousarray(img[..., ::-1]), refer_image=np.ascontiguousarray(ref_img[..., ::-1]),
                      visual_prompts=vp, predictor=YOLOEVPSegPredictor, conf=conf, imgsz=imgsz, verbose=False)[0]
    return Dets(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), np.zeros(len(r.boxes), int),
                ["зразок"] * len(r.boxes))


# ======================================================================================
# VLM-grounding (слайд 42): бокси як текст
# ======================================================================================
def florence_generate(proc, model, img: np.ndarray, task: str = "<CAPTION_TO_PHRASE_GROUNDING>", text: str = "",
                      device: str = "cuda") -> str:
    """Сирий вихід Florence-2 з токенами координат <loc_0>…<loc_999>."""
    import torch

    inp = proc(text=task + text, images=Image.fromarray(img), return_tensors="pt").to(device, model.dtype)
    with torch.inference_mode():
        ids = model.generate(input_ids=inp["input_ids"], pixel_values=inp["pixel_values"], max_new_tokens=1024,
                             num_beams=3, do_sample=False)
    return proc.batch_decode(ids, skip_special_tokens=False)[0]


def parse_loc_tokens(raw: str) -> Tuple[List[str], np.ndarray]:
    """'car<loc_12><loc_40><loc_80><loc_99>…' → (фрази, бокси у бінах 0…999). Бінів 1000 на кожну вісь кадру."""
    names, boxes, cur = [], [], ""
    for m in re.finditer(r"([^<]+)|<loc_(\d+)>", raw.replace("</s>", "").replace("<s>", "").replace("<pad>", "")):
        if m.group(1) is not None:
            cur = m.group(1).strip() or cur
        else:
            boxes.append(int(m.group(2)))
            if len(boxes) % 4 == 0:
                names.append(cur)
    b = np.array(boxes[:len(boxes) // 4 * 4], np.float32).reshape(-1, 4)
    return names, b


def qwen_ground(proc, model, img: np.ndarray, prompt: str, device: str = "cuda", max_new_tokens: int = 1024,
                max_pixels: Optional[int] = 1280 * 28 * 28):
    """
    Qwen2.5-VL: детекція як генерація JSON. Повертає (сирий текст, бокси xyxy у пікселях ВХОДУ моделі, (W', H') входу).
    ПАСТКА (слайд 42): Qwen2.5-VL рахує координати в кадрі після власного ресайзу (кратно 28 px), а не в оригіналі.
    max_pixels = 1280·28² — типове обмеження з model card заради швидкості й пам'яті; None — майже без ресайзу.
    """
    import torch

    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    kw = {"max_pixels": max_pixels} if max_pixels else {}
    inp = proc(text=[text], images=[Image.fromarray(img)], return_tensors="pt", **kw).to(device)
    with torch.inference_mode():
        ids = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
    raw = proc.batch_decode(ids[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    t, gh, gw = inp["image_grid_thw"][0].tolist()
    patch = proc.image_processor.patch_size
    in_wh = (gw * patch, gh * patch)
    boxes = []
    for m in re.finditer(r'"bbox_2d"\s*:\s*\[\s*([\d.\-]+)\s*,\s*([\d.\-]+)\s*,\s*([\d.\-]+)\s*,\s*([\d.\-]+)\s*\]', raw):
        boxes.append([float(v) for v in m.groups()])
    return raw, np.array(boxes, np.float32).reshape(-1, 4), in_wh


# ======================================================================================
# Метрики
# ======================================================================================
def coco_eval(items, preds: Dict[str, Dets], class_names: Sequence[str], groups: Dict[str, List[int]] = EVAL_GROUPS) -> dict:
    """AP, AP50, AP_small (COCO, < 32² px) з урахуванням ignored regions VisDrone. preds: {stem: Dets}."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    cat_of = {vid: ci for ci, name in enumerate(class_names) for vid in groups[name]}
    images, anns, dets, aid = [], [], [], 1
    for img_id, (p, gt) in enumerate(items, 1):
        w, h = image_size(p)
        images.append({"id": img_id, "width": w, "height": h})
        for b, c in zip(gt.boxes, gt.cats):
            if int(c) in cat_of:
                x0, y0, x1, y1 = map(float, b)
                anns.append({"id": aid, "image_id": img_id, "category_id": cat_of[int(c)] + 1, "iscrowd": 0,
                             "bbox": [x0, y0, x1 - x0, y1 - y0], "area": (x1 - x0) * (y1 - y0)})
                aid += 1
        for b in gt.ignore:          # ignored regions → crowd для кожного класу
            x0, y0, x1, y1 = map(float, b)
            for ci in range(len(class_names)):
                anns.append({"id": aid, "image_id": img_id, "category_id": ci + 1, "iscrowd": 1,
                             "bbox": [x0, y0, x1 - x0, y1 - y0], "area": (x1 - x0) * (y1 - y0)})
                aid += 1
        d = preds.get(p.stem, Dets.empty())
        for b, s, l in zip(d.boxes, d.scores, d.labels):
            if l >= 0:
                x0, y0, x1, y1 = map(float, b)
                dets.append({"image_id": img_id, "category_id": int(l) + 1, "bbox": [x0, y0, x1 - x0, y1 - y0],
                             "score": float(s)})
    if not dets:
        return {"AP": 0.0, "AP50": 0.0, "AP_small": 0.0, "n_det": 0}
    with contextlib.redirect_stdout(io.StringIO()):
        gt_coco = COCO()
        gt_coco.dataset = {"images": images, "annotations": anns,
                           "categories": [{"id": i + 1, "name": n} for i, n in enumerate(class_names)]}
        gt_coco.createIndex()
        E = COCOeval(gt_coco, gt_coco.loadRes(dets), "bbox")
        E.params.maxDets = [1, 100, 500]        # на VisDrone десятки–сотні об'єктів на кадр
        E.evaluate()
        E.accumulate()
        E.summarize()
    s = E.stats
    return {"AP": float(s[0]), "AP50": float(s[1]), "AP_small": float(s[3]), "n_det": len(dets)}


def match_pr(det_boxes: np.ndarray, gt_boxes: np.ndarray, ignore: Optional[np.ndarray] = None, iou: float = 0.5) -> dict:
    """Precision / recall у робочій точці (слайд 50): жадібне зіставлення за IoU; детекції в ignored regions не рахуються."""
    det_boxes = np.asarray(det_boxes, np.float32).reshape(-1, 4)
    if ignore is not None and len(ignore) and len(det_boxes):
        ctr = np.stack([(det_boxes[:, 0] + det_boxes[:, 2]) / 2, (det_boxes[:, 1] + det_boxes[:, 3]) / 2], 1)
        inside = ((ctr[:, None, 0] >= ignore[None, :, 0]) & (ctr[:, None, 0] <= ignore[None, :, 2]) &
                  (ctr[:, None, 1] >= ignore[None, :, 1]) & (ctr[:, None, 1] <= ignore[None, :, 3])).any(1)
        det_boxes = det_boxes[~inside]
    M = box_iou(det_boxes, gt_boxes)
    tp, used = 0, set()
    for i in np.argsort(-M.max(1)) if M.size else []:
        j = int(np.argmax(np.where(np.isin(np.arange(M.shape[1]), list(used)), -1, M[i])))
        if M[i, j] >= iou:
            tp += 1
            used.add(j)
    return {"tp": tp, "fp": len(det_boxes) - tp, "fn": len(gt_boxes) - tp}


def pr_table(rows: Sequence[dict]) -> Tuple[float, float]:
    tp = sum(r["tp"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows)
    return tp / max(tp + fp, 1), tp / max(tp + fn, 1)


# ======================================================================================
# Візуалізація
# ======================================================================================
PALETTE = np.array([[0, 200, 170], [255, 160, 40], [90, 160, 255], [230, 80, 120], [160, 220, 60],
                    [200, 120, 255], [255, 220, 0], [0, 180, 255], [255, 110, 70], [120, 255, 200]], np.uint8)


def overlay_masks(img: np.ndarray, masks, alpha: float = 0.5, ids=None, contour: bool = True) -> np.ndarray:
    out = img.copy()
    for k, m in enumerate(masks):
        m = np.asarray(m).astype(bool)
        if m.ndim == 3:
            m = m[0]
        if m.shape != out.shape[:2]:
            m = cv2.resize(m.astype(np.uint8), out.shape[1::-1], interpolation=cv2.INTER_NEAREST).astype(bool)
        c = PALETTE[(ids[k] if ids is not None else k) % len(PALETTE)]
        out[m] = (out[m] * (1 - alpha) + c * alpha).astype(np.uint8)
        if contour:
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, cs, -1, tuple(int(v) for v in c), max(1, out.shape[1] // 900))
    return out


def draw_boxes(img: np.ndarray, boxes, labels=None, color=(0, 220, 170), thick: Optional[int] = None,
               font: Optional[float] = None) -> np.ndarray:
    out = img.copy()
    thick = thick or max(1, img.shape[1] // 800)
    font = font or max(0.4, img.shape[1] / 2400)
    for i, b in enumerate(np.asarray(boxes, np.float32).reshape(-1, 4)):
        x0, y0, x1, y1 = [int(round(v)) for v in b]
        cv2.rectangle(out, (x0, y0), (x1, y1), color, thick)
        if labels is not None and str(labels[i]):
            cv2.putText(out, str(labels[i]), (x0, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, font, color,
                        max(1, thick // 2 + 1), cv2.LINE_AA)
    return out


def crop_around(img: np.ndarray, box, pad: int = 60) -> Tuple[np.ndarray, Tuple[int, int]]:
    H, W = img.shape[:2]
    x0, y0 = int(max(0, box[0] - pad)), int(max(0, box[1] - pad))
    x1, y1 = int(min(W, box[2] + pad)), int(min(H, box[3] + pad))
    return img[y0:y1, x0:x1], (x0, y0)


def show_jpeg(fig=None, quality: int = 85, save: Optional[Path] = None):
    """Фігура з фотографіями — як JPEG (PNG таких фігур важить мегабайти). save — ще й копія у файл для слайдів."""
    from IPython.display import Image as IPImage, display

    fig = fig or plt.gcf()
    if save is not None:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=150, bbox_inches="tight")
    buf = io.BytesIO()
    fig.savefig(buf, format="jpeg", dpi=90, bbox_inches="tight", pil_kwargs=dict(quality=quality))
    plt.close(fig)
    display(IPImage(buf.getvalue(), format="jpeg"))


def show_row(images: Sequence[np.ndarray], titles: Optional[Sequence[str]] = None, size: float = 4.5,
             suptitle: str = "", save: Optional[Path] = None, ncols: Optional[int] = None):
    n = len(images)
    ncols = ncols or n
    nrows = math.ceil(n / ncols)
    ar = np.mean([im.shape[0] / im.shape[1] for im in images])
    fig, axes = plt.subplots(nrows, ncols, figsize=(size * ncols, size * ar * nrows + (0.5 if suptitle else 0.2)),
                             squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for k, im in enumerate(images):
        ax = axes.ravel()[k]
        ax.imshow(im)
        if titles is not None:
            ax.set_title(titles[k], fontsize=9)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    show_jpeg(fig, save=save)


def save_fig(fig, path: Path):
    """Графік (лінії / стовпці) у файл для слайдів і на екран."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()


def _widgets():
    try:
        import ipywidgets as W
        from IPython.display import display
        return W, display
    except ImportError:
        print("ipywidgets не встановлено: pip install ipywidgets")
        return None, None


def _jpeg(img: np.ndarray, quality: int = 85) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "JPEG", quality=quality)
    return buf.getvalue()


# ======================================================================================
# demo1 · клік мишею → маски SAM (інтерактивний режим, слайд 21)
# ======================================================================================
class ClickSamPanel:
    """
    Клік по кадру — точка об'єкта, Shift+клік (або перемикач) — точка фону. Енкодер уже порахований (set_image),
    тож кожен клік — лише декодер: час видно в заголовку. predict_fn(points (N, 2), labels (N,), multimask) ->
    (masks (K, H, W), predicted_iou (K,), мс) приходить із ноутбука.
    """
    PREVIEW_W = 820

    def __init__(self, img: np.ndarray, predict_fn: Callable, view_box=None):
        import ipywidgets as W

        self.img, self.fn = img, predict_fn
        H, W_ = img.shape[:2]
        self.view = np.array(view_box if view_box is not None else [0, 0, W_, H], int)   # що показуємо (зум)
        self.points, self.labels = [], []
        self.mode = W.ToggleButtons(options=[("об'єкт (1)", 1), ("фон (0)", 0)], value=1)
        self.multi = W.Checkbox(value=True, description="multimask_output (3 маски)")
        self.reset = W.Button(description="скинути точки")
        self.preview = W.Image(format="jpeg", layout=W.Layout(width=f"{self.PREVIEW_W}px"))
        self.out = W.Output()
        try:
            from ipyevents import Event

            self.ev = Event(source=self.preview, watched_events=["click"], prevent_default_action=True)
            self.ev.on_dom_event(self._on_click)
            hint = "Клік — точка об'єкта, Shift+клік — точка фону."
        except ImportError:
            hint = "ipyevents не встановлено: точки задавайте через panel.add_point(x, y, label) у пікселях кадру."
        self.reset.on_click(lambda _: self._reset())
        self.multi.observe(lambda _: self._render(), names="value")
        self.ui = W.VBox([W.HBox([self.mode, self.multi, self.reset]), W.HTML(f"<i>{hint}</i>"), self.preview, self.out])
        self._render()

    def _reset(self):
        self.points, self.labels = [], []
        self._render()

    def _on_click(self, ev):
        rw, rh = ev.get("boundingRectWidth"), ev.get("boundingRectHeight")
        if not (rw and rh):
            return
        x0, y0, x1, y1 = self.view
        x = x0 + ev.get("relativeX", 0) / rw * (x1 - x0)
        y = y0 + ev.get("relativeY", 0) / rh * (y1 - y0)
        self.add_point(x, y, 0 if ev.get("shiftKey") else self.mode.value)

    def add_point(self, x: float, y: float, label: int = 1):
        self.points.append((float(x), float(y)))
        self.labels.append(int(label))
        self._render()

    def _render(self):
        x0, y0, x1, y1 = self.view
        view = self.img[y0:y1, x0:x1].copy()
        masks, iou, ms = None, None, None
        if self.points:
            masks, iou, ms = self.fn(np.array(self.points), np.array(self.labels), self.multi.value)
            best = int(np.argmax(iou))
            view = overlay_masks(view, [masks[best][y0:y1, x0:x1]], alpha=.5)
        r = max(3, (x1 - x0) // 150)
        for (x, y), l in zip(self.points, self.labels):
            cv2.circle(view, (int(x - x0), int(y - y0)), r, (60, 230, 60) if l else (240, 60, 60), -1)
        pw = self.PREVIEW_W
        self.preview.value = _jpeg(cv2.resize(view, (pw, round(view.shape[0] * pw / view.shape[1])),
                                              interpolation=cv2.INTER_AREA if view.shape[1] > pw else cv2.INTER_NEAREST))
        with self.out:
            self.out.clear_output(wait=True)
            if masks is None:
                print("Клікніть по об'єкту.")
                return
            order = np.argsort([m.sum() for m in masks])
            fig, axes = plt.subplots(1, len(masks), figsize=(3.6 * len(masks), 3.2), squeeze=False)
            for ax, k in zip(axes[0], order):
                ax.imshow(overlay_masks(self.img[y0:y1, x0:x1], [masks[k][y0:y1, x0:x1]], alpha=.55))
                ax.set_title(f"predicted_iou {iou[k]:.2f} · {int(masks[k].sum())} px", fontsize=9)
                ax.axis("off")
            fig.suptitle(f"{len(self.points)} точок · декодер {ms:.1f} мс (енкодер уже в кеші)", fontsize=10)
            fig.tight_layout()
            show_jpeg(fig)


def click_sam_panel(img: np.ndarray, predict_fn: Callable, view_box=None, start_point=None):
    W, display = _widgets()
    if W is None:
        return None
    p = ClickSamPanel(img, predict_fn, view_box)
    if start_point is not None and AUTORUN:
        p.add_point(*start_point)
    display(p.ui)
    return p


# ======================================================================================
# demo3 · SAM 2 на відео
# ======================================================================================
@dataclass
class VideoRun:
    boxes: Dict[int, np.ndarray]                 # obj -> (T, 4), NaN, якщо маски на кадрі немає
    areas: Dict[int, np.ndarray]                 # obj -> (T,) площа маски, px кліпу
    small: Dict[int, Dict[int, np.ndarray]]      # кадр -> obj -> маска, зменшена в keep_scale (для картинок)
    ms: List[float]                              # мс на кадр propagate (усі об'єкти разом)
    init_ms: float                               # init_state: декодування кадрів + енкодер першого кадру

    @property
    def objs(self) -> List[int]:
        return sorted(self.boxes)


def run_sam2_video(vp, clip: Clip, boxes0: np.ndarray, device: str = "cuda",
                   corrections: Optional[Dict[int, Dict[int, np.ndarray]]] = None,
                   keep_scale: float = 0.25, on_frame: Optional[Callable] = None) -> VideoRun:
    """
    Бокси на кадрі 0 → masklet'и по всьому кліпу. corrections = {кадр: {obj: бокс}} — промпт у середині ролика
    (слайд 29): SAM 2 бере його як ще один умовний кадр, і все після нього перераховується.
    on_frame(k, obj_ids, masks_bool_cpu) — для експорту повнороздільних масок без зберігання їх у RAM.
    """
    import torch
    import torch.nn.functional as F

    T = len(clip.frame_ids)
    n = len(boxes0)
    boxes = {o: np.full((T, 4), np.nan, np.float32) for o in range(n)}
    areas = {o: np.zeros(T, np.float32) for o in range(n)}
    small, ts = {}, []
    with torch.inference_mode(), autocast(device):
        t0 = time.perf_counter()
        state = vp.init_state(video_path=str(clip.dir), offload_video_to_cpu=True)
        for o, b in enumerate(boxes0):
            vp.add_new_points_or_box(state, frame_idx=0, obj_id=o, box=np.asarray(b, np.float32))
        for k, objs in (corrections or {}).items():
            for o, b in objs.items():
                vp.add_new_points_or_box(state, frame_idx=int(k), obj_id=int(o), box=np.asarray(b, np.float32))
        sync()
        init_ms = (time.perf_counter() - t0) * 1000
        it = iter(vp.propagate_in_video(state))
        while True:
            with timed(ts):
                try:
                    k, obj_ids, logits = next(it)
                except StopIteration:
                    break
                m = logits[:, 0] > 0                                  # (n_obj, H, W) на GPU
            rows, cols = m.any(2), m.any(1)
            area = m.sum((1, 2)).float().cpu().numpy()
            sm = F.interpolate(m[:, None].float(), scale_factor=keep_scale, mode="nearest")[:, 0] > 0
            sm = sm.cpu().numpy()
            rows, cols = rows.cpu().numpy(), cols.cpu().numpy()
            small[k] = {}
            for j, o in enumerate(obj_ids):
                o = int(o)
                areas[o][k] = area[j]
                small[k][o] = sm[j]
                if area[j] > 0:
                    ys, xs = np.flatnonzero(rows[j]), np.flatnonzero(cols[j])
                    boxes[o][k] = [xs[0], ys[0], xs[-1] + 1, ys[-1] + 1]
            if on_frame is not None:
                on_frame(k, [int(o) for o in obj_ids], m.cpu().numpy())
        vp.reset_state(state)
    return VideoRun(boxes, areas, small, ts[:-1], init_ms)


def track_eval(pred_boxes: np.ndarray, gt: pd.DataFrame, tid: int, iou_thr: float = 0.3) -> pd.DataFrame:
    """
    Покадрова оцінка одного masklet'а проти GT-треку tid (бокс маски vs GT-бокс):
    стан GT (видно / перекрито / немає в кадрі), IoU зі «своїм» GT, чи маска перескочила на інший об'єкт.
    """
    rows = []
    by_k = {k: g for k, g in gt.groupby("k")}
    for k in range(len(pred_boxes)):
        g = by_k.get(k, gt.iloc[:0])
        own = g[g.tid == tid]
        others = g[g.tid != tid]
        pb = pred_boxes[k]
        present = not np.isnan(pb[0])
        state = "немає" if own.empty else ("перекрито" if int(own.occ.iloc[0]) == 2 else "видно")
        iou_own = float(box_iou(pb, own[["x0", "y0", "x1", "y1"]].values)[0, 0]) if present and not own.empty else 0.0
        iou_oth = float(box_iou(pb, others[["x0", "y0", "x1", "y1"]].values).max()) if present and len(others) else 0.0
        rows.append(dict(k=k, state=state, present=present, iou=iou_own,
                         switch=present and iou_oth >= iou_thr and iou_oth > iou_own))
    return pd.DataFrame(rows)


def summarize_tracks(per_obj: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Зведення по об'єктах: середній IoU на кадрах, де GT видно, втрати, «домальовування» за перешкодою, перескоки."""
    out = {}
    for name, d in per_obj.items():
        vis = d[d.state == "видно"]
        hidden = d[d.state != "видно"]
        out[name] = {"кадрів": len(d), "IoU, де видно": vis.iou.mean() if len(vis) else np.nan,
                     "втрачено, де видно": int((~vis.present).sum()),
                     "маска, де об'єкт сховано / пішов": f"{int(hidden.present.sum())}/{len(hidden)}",
                     "перескок на сусіда, кадрів": int(d.switch.sum())}
    return pd.DataFrame(out).T


def flicker(area: np.ndarray) -> float:
    """Медіана |Δплощі| / площа між сусідніми кадрами, %: «маска дихає» (слайд 27)."""
    a = np.asarray(area, float)
    a = np.where(a > 0, a, np.nan)
    d = np.abs(np.diff(a)) / a[:-1]
    return float(np.nanmedian(d) * 100) if np.isfinite(d).any() else float("nan")


class ConstVelKF:
    """
    Фільтр Калмана зі сталою швидкістю для боксу (cx, cy, w, h) — модель руху в SAMURAI і ByteTrack (слайд 30).
    predict() → де бокс має бути; update(box) — коли спостереження прийняли.
    """

    def __init__(self, box, q: float = 1.0, r: float = 10.0):
        cx, cy, w, h = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2, box[2] - box[0], box[3] - box[1]
        self.x = np.array([cx, cy, w, h, 0, 0, 0, 0], float)
        self.P = np.eye(8) * 100
        self.F = np.eye(8)
        self.F[:4, 4:] = np.eye(4)
        self.H = np.eye(4, 8)
        self.Q, self.R = np.eye(8) * q, np.eye(4) * r

    @staticmethod
    def _xyxy(s):
        return np.array([s[0] - s[2] / 2, s[1] - s[3] / 2, s[0] + s[2] / 2, s[1] + s[3] / 2])

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self._xyxy(self.x)

    def update(self, box):
        z = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2, box[2] - box[0], box[3] - box[1]])
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P


# ======================================================================================
# demo3 · кодування маски для каналу (слайд 60)
# ======================================================================================
def rle_encode(m: np.ndarray) -> List[int]:
    """Довжини серій 0/1 по рядках (як COCO RLE без стиснення рядком): перша серія — нулі."""
    f = np.asarray(m, bool).ravel()
    idx = np.flatnonzero(np.diff(np.r_[False, f, False].astype(np.int8)))
    runs = np.diff(np.r_[0, idx, f.size])
    return runs.tolist()


def varint_bytes(values: Sequence[int]) -> int:
    """Скільки байтів займуть числа у varint-кодуванні (protobuf): малі серії — 1 байт."""
    return int(sum(max(1, math.ceil(max(int(v), 1).bit_length() / 7)) for v in values))


# ======================================================================================
# demo1 · замір енкодер / декодер по сімейству SAM (слайди 19, 24)
# ======================================================================================
def sam_bench_fns(name: str, img: np.ndarray, box: np.ndarray, device: str = "cuda"):
    """
    Для моделі із SAM_ZOO повертає (enc_fn, dec_fn, info, model): enc_fn() — енкодер кадру (раз на кадр),
    dec_fn() — prompt encoder + декодер на один бокс поверх уже порахованого embedding. Інтерфейс у всіх
    однаковий — різниться лише вартість енкодера (слайд 19).
    """
    import torch

    kind = SAM_ZOO[name]["kind"]
    box = np.asarray(box, np.float32)
    if kind == "sam2":
        pred = load_sam2_image(name, device)

        def enc():
            with torch.inference_mode(), autocast(device):
                pred.set_image(img)

        def dec():
            with torch.inference_mode(), autocast(device):
                pred.predict(box=box[None], multimask_output=False)

        enc()
        return enc, dec, dict(enc_params=n_params(pred.model.image_encoder), params=n_params(pred.model)), pred
    if kind in ("hf-sam", "hf-samhq"):
        proc, m = load_hf(name, device)
        inp = proc(img, input_boxes=[[box.tolist()]], return_tensors="pt").to(device)
        st = {}

        def enc():
            with torch.inference_mode():
                st["e"] = m.get_image_embeddings(inp["pixel_values"])

        def dec():
            e = st["e"]
            with torch.inference_mode():
                if isinstance(e, tuple):          # HQ-SAM: ще й ранні ознаки ViT для HQ-токена
                    m(image_embeddings=e[0], intermediate_embeddings=e[1], input_boxes=inp["input_boxes"],
                      multimask_output=False)
                else:
                    m(image_embeddings=e, input_boxes=inp["input_boxes"], multimask_output=False)

        enc()
        return enc, dec, dict(enc_params=n_params(m.vision_encoder), params=n_params(m)), m
    if name == "mobile-sam":                    # ultralytics: та сама архітектура SAM 1, дистильований TinyViT
        sam = load_ultra(name).model.to(device).eval()
        x = torch.randn(1, 3, 1024, 1024, device=device)
        b = torch.as_tensor(box * 1024 / max(img.shape[:2]), device=device)[None]
        st = {}

        def enc():
            with torch.inference_mode():
                st["e"] = sam.image_encoder(x)

        def dec():
            pe = sam.prompt_encoder
            with torch.inference_mode():
                sp, de = pe(points=None, boxes=b, masks=None)
                sam.mask_decoder(image_embeddings=st["e"], image_pe=pe.get_dense_pe(), sparse_prompt_embeddings=sp,
                                 dense_prompt_embeddings=de, multimask_output=False)

        enc()
        return enc, dec, dict(enc_params=n_params(sam.image_encoder), params=n_params(sam)), sam
    raise ValueError(f"{name}: немає промптового енкодера / декодера для заміру")
