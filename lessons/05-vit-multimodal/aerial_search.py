"""
aerial_search — бібліотека для Демо 2 (Заняття 5).

Текстовий пошук по аеровідео: ключові кадри -> тайли -> ембеддинги
(SigLIP 2 / RemoteCLIP / SkyCLIP) -> FAISS -> ранжування кадрів -> метрики.

Усі вектори L2-нормалізовані, тому скалярний добуток = сирий косинус.
Пороги (слайд 47) рахуються саме по ньому, а не по softmax.
"""
from __future__ import annotations

import hashlib
import json
import time
import urllib.request
import warnings
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image

# Шляхи від кореня репозиторію: не залежать від того, звідки запущено ноутбук чи tools/.
REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = REPO_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS / "models" / "clip"      # tools/fetch_clip_models.py
HF_HOME = MODELS_DIR / "hf_home"                # кеш HF (ваги й токенайзер SigLIP 2)
VISDRONE_DIR = ARTIFACTS / "visdrone"           # tools/fetch_visdrone.py
OWN_VIDEO_DIR = ARTIFACTS / "aerial_video"      # власне відео, DATASET="own"

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
VID_EXT = {".mp4", ".mov", ".avi", ".mkv", ".ts", ".m4v"}

# --------------------------------------------------------------------------------------
# Зоопарк моделей. Усі три завантажуються через OpenCLIP -> один інтерфейс.
# --------------------------------------------------------------------------------------
SKY_BASE_URL = "https://opendatasharing.s3.us-west-2.amazonaws.com/SkyScript/ckpt/"

MODEL_ZOO: Dict[str, Dict[str, dict]] = {
    # SigLIP 2 (Google, 2025): вебові дані, сигмоїдна втрата. Ваги з HF через open_clip.
    "siglip2": {
        "base": dict(arch="ViT-B-16-SigLIP2", pretrained="webli"),
        "large": dict(arch="ViT-L-16-SigLIP2-256", pretrained="webli"),
    },
    # RemoteCLIP (Liu et al., TGRS 2024): ваги у форматі OpenCLIP на HF.
    # Завантаження — як у README авторів (без quick_gelu).
    "remoteclip": {
        "base": dict(arch="ViT-B-32", hf_repo="chendelong/RemoteCLIP", hf_file="RemoteCLIP-ViT-B-32.pt"),
        "large": dict(arch="ViT-L-14", hf_repo="chendelong/RemoteCLIP", hf_file="RemoteCLIP-ViT-L-14.pt"),
    },
    # SkyCLIP (SkyScript, AAAI 2024): архіви з S3 авторів.
    # ViT-L-14 ініціалізовано з OpenAI CLIP -> потрібен quick_gelu (так у їхньому ноутбуці).
    "skyclip": {
        "base": dict(arch="ViT-B-32", url=SKY_BASE_URL + "SkyCLIP_ViT_B32_top50pct.zip"),
        "large": dict(arch="ViT-L-14", url=SKY_BASE_URL + "SkyCLIP_ViT_L14_top30pct.zip", quick_gelu=True),
    },
}

# Шаблони промптів із вказівкою ракурсу (слайд 36). "{}" — опис цілі.
PLAIN_TEMPLATES = ["{}"]
AERIAL_TEMPLATES = [
    "an aerial photo of {}",
    "a drone image of {}",
    "an overhead view of {}",
    "a photo of {} taken from a drone",
    "a high-altitude aerial view of {}",
    "a top-down aerial image showing {}",
]


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _download(url: str, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    print(f"[download] {url}\n           -> {out}")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done, last = 0, -1
        while True:
            buf = r.read(1 << 20)
            if not buf:
                break
            f.write(buf)
            done += len(buf)
            if total:
                pct = int(100 * done / total)
                if pct // 10 != last // 10:
                    print(f"           {pct}% ({done / 1e6:.0f} / {total / 1e6:.0f} MB)")
                    last = pct
    tmp.rename(out)
    return out


def resolve_checkpoint(name: str, size: str, models_dir: str | Path = MODELS_DIR, download: bool = True) -> Path:
    """Повертає локальний шлях до чекпойнта RemoteCLIP / SkyCLIP (за потреби завантажує)."""
    spec = MODEL_ZOO[name][size]
    models_dir = Path(models_dir)

    if "hf_repo" in spec:
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(spec["hf_repo"], spec["hf_file"], cache_dir=str(models_dir / "hf"),
                                    local_files_only=not download))

    if "url" in spec:
        zip_name = spec["url"].rsplit("/", 1)[-1]
        zdir = models_dir / "skyclip"
        zpath = zdir / zip_name
        xdir = zdir / zip_name.replace(".zip", "")
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
    Типи numpy, які трапляються в чекпойнтах тренування OpenCLIP (стан GradScaler у SkyCLIP):
    скаляр і dtype — це дані, а не код, тож дозволяємо їх у weights_only.
    Pickle посилається на numpy.core.multiarray.scalar, а в numpy 2 він живе в numpy._core —
    тому реєструємо його під старим ім'ям.
    """
    import numpy as np

    try:
        from numpy._core.multiarray import scalar
    except ImportError:  # numpy 1.x
        from numpy.core.multiarray import scalar
    dtypes = [type(np.dtype(c)) for c in "?bBhHiIlLqQefdgFD"]
    return [(scalar, "numpy.core.multiarray.scalar"), scalar, np.dtype, *dict.fromkeys(dtypes)]


def load_state_dict_safely(path: str | Path, trust_pickle: bool = False) -> dict:
    """
    Безпечне завантаження ваг.

    .pt — це pickle: при weights_only=False він може виконати довільний код.
    Тому за замовчуванням вантажимо лише тензори (+ numpy-скаляри, див. _numpy_safe_globals);
    якщо файл містить щось інше, падаємо з поясненням, а не мовчки довіряємо (тема для air-gapped контуру).
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
                    f"{path.name} не вантажиться з weights_only=True ({type(e).__name__}). "
                    "Якщо джерело перевірене — передайте trust_pickle=True і одразу збережіть "
                    "ваги в safetensors (python tools/fetch_clip_models.py --to-safetensors)."
                ) from e
            warnings.warn(f"{path.name}: завантаження з weights_only=False (pickle).")
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    if next(iter(sd)).startswith("module."):
        sd = {k[len("module."):]: v for k, v in sd.items()}
    return sd


def save_safetensors(state_dict: dict, out: str | Path) -> Path:
    from safetensors.torch import save_file

    out = Path(out)
    save_file({k: v.contiguous() for k, v in state_dict.items()}, str(out))
    return out


def pick_device(device: Optional[str] = None) -> str:
    import torch

    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _get_tokenizer(open_clip, arch: str, cache_dir: Path):
    """
    Токенайзер open_clip. Офлайн (HF_HUB_OFFLINE=1) get_tokenizer для SigLIP 2 падає: transformers шукає
    config.json, якого в репозиторії токенайзера немає. Тому офлайн будуємо той самий HFTokenizer
    з локальної копії репозиторію в кеші — з тими ж параметрами, що й open_clip.
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
    """Архітектура без ваг. open_clip при цьому пише в лог «Model initialized randomly» —
    для RemoteCLIP / SkyCLIP це хибна тривога (ваги вантажимо одразу після), тож глушимо."""
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


class Encoder:
    """Єдиний інтерфейс до SigLIP 2 / RemoteCLIP / SkyCLIP через OpenCLIP."""

    def __init__(
        self,
        name: str,
        size: str = "base",
        models_dir: str | Path = MODELS_DIR,
        device: Optional[str] = None,
        resize_mode: str = "longest",
        random_weights: bool = False,
        trust_pickle: bool = False,
        quick_gelu: Optional[bool] = None,
        download: bool = True,
    ):
        import open_clip
        import torch

        self.name, self.size = name, size
        self.key = f"{name}-{size}" + ("-RANDOM" if random_weights else "")
        self.device = pick_device(device)
        self.resize_mode = resize_mode
        models_dir = Path(models_dir)
        spec = dict(MODEL_ZOO[name][size])
        qg = spec.get("quick_gelu", False) if quick_gelu is None else quick_gelu

        if random_weights:
            # Лише для smoke-тесту без мережі: архітектура з вбудованим токенайзером.
            self.arch = "ViT-B-32"
            model = _create_bare_model(open_clip, self.arch)
        elif "pretrained" in spec:
            self.arch = spec["arch"]
            model = open_clip.create_model(self.arch, pretrained=spec["pretrained"],
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
            pp["size"], is_train=False, mean=pp["mean"], std=pp["std"],
            resize_mode=resize_mode, interpolation=pp.get("interpolation"), fill_color=pp.get("fill_color", 0),
        )
        self.input_size = pp["size"]
        self._torch = torch
        self.model = model.to(self.device).eval()
        self.n_params = sum(p.numel() for p in model.parameters())
        self.dim = int(self.encode_texts(["test"]).shape[1])
        self.img_per_s: Optional[float] = None

    def _autocast(self):
        torch = self._torch
        if self.device == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        import contextlib

        return contextlib.nullcontext()

    def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        torch = self._torch
        x = torch.stack([self.preprocess(im.convert("RGB")) for im in images]).to(self.device)
        with torch.no_grad(), self._autocast():
            f = self.model.encode_image(x, normalize=True)
        return f.float().cpu().numpy()

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        torch = self._torch
        t = self.tokenizer(list(texts)).to(self.device)
        with torch.no_grad(), self._autocast():
            f = self.model.encode_text(t, normalize=True)
        return f.float().cpu().numpy()

    def text_query(self, text: str, templates: Sequence[str] = AERIAL_TEMPLATES) -> np.ndarray:
        """Ансамбль промптів: усереднюємо нормовані вектори і знову нормуємо (слайд 36)."""
        e = self.encode_texts([tpl.format(text) for tpl in templates]).mean(0)
        return (e / np.linalg.norm(e)).astype("float32")

    def image_query(self, image: Image.Image) -> np.ndarray:
        return self.encode_images([image])[0].astype("float32")

    def __repr__(self) -> str:
        return f"Encoder({self.key}, arch={self.arch}, {self.n_params / 1e6:.0f}M, dim={self.dim}, {self.device})"


# --------------------------------------------------------------------------------------
# Джерела відео
# --------------------------------------------------------------------------------------
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


def discover_visdrone(root: str | Path, fps: float = 30.0) -> List[Source]:
    """VisDrone2019-VID-*: <split>/sequences/<seq>/*.jpg + <split>/annotations/<seq>.txt."""
    out = []
    for seq_root in sorted(Path(root).rglob("sequences")):
        if not seq_root.is_dir():
            continue
        ann_root = seq_root.parent / "annotations"
        split = seq_root.parent.name
        for seq in sorted(p for p in seq_root.iterdir() if p.is_dir()):
            ann = ann_root / f"{seq.name}.txt"
            out.append(Source(f"{split}/{seq.name}", "images", str(seq), fps, "visdrone",
                              str(ann) if ann.exists() else None))
    return out


def discover_generic(root: str | Path, image_seq_fps: float = 30.0, min_images: int = 10) -> List[Source]:
    """Власні дані: відеофайли + теки з послідовностями кадрів (UAVDT, AU-AIR тощо)."""
    root = Path(root)
    out = []
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
    """Тривалість джерела в секундах (для підсумку «скільки годин проіндексовано»)."""
    if src.kind == "video":
        cap = cv2.VideoCapture(src.path)
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        return float(n) / src.fps if n > 0 else 0.0
    return len(_image_files(Path(src.path))) / src.fps


def show_tiles(kf_row, tiles: pd.DataFrame, title: str = ""):
    """Як кадр нарізано на тайли (тайл 0 — увесь кадр)."""
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.imshow(Image.open(kf_row["path"]))
    for r in tiles[tiles["kf_id"] == kf_row["kf_id"]].itertuples(index=False):
        if r.tile_idx == 0:
            continue
        ax.add_patch(patches.Rectangle((r.x0, r.y0), r.x1 - r.x0, r.y1 - r.y0, fill=False, lw=1.5,
                                       ec=plt.cm.tab10(r.tile_idx % 10)))
        ax.text(r.x0 + 8, r.y0 + 28, str(r.tile_idx), color="w", fontsize=11, weight="bold")
    ax.set_title(title)
    ax.axis("off")
    plt.show()


def _iter_frames(src: Source, step: int) -> Iterable[Tuple[int, np.ndarray, Optional[Path]]]:
    """(індекс кадру, BGR, файл-оригінал або None для відео)."""
    if src.kind == "video":
        cap = cv2.VideoCapture(src.path)
        i = 0
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
    """Скільки кадрів переглянемо після прорідження — для прогрес-бару (у відео FRAME_COUNT приблизний)."""
    if src.kind == "video":
        cap = cv2.VideoCapture(src.path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    else:
        n = len(_image_files(Path(src.path)))
    return -(-max(n, 0) // step)


def extract_keyframes(
    sources: Sequence[Source],
    out_dir: str | Path,
    sample_fps: float = 1.0,
    dedup_thr: float = 2.0,
    max_side: int = 1920,
    jpeg_quality: int = 90,
    force: bool = False,
) -> pd.DataFrame:
    """
    Відбір ключових кадрів: прорідження до sample_fps + дедуплікація.

    Дедуплікація не залежить від моделі (мініатюра 32x18, середня абсолютна різниця),
    тож усі моделі індексують однаковий набір кадрів — порівняння чесне.
    Увага: для статичної камери (кейс 3) малий рухомий об'єкт майже не змінює мініатюру —
    там ставте dedup_thr=0 або дедуплікуйте по треку детектора.

    Що лягає в out_dir/frames:
      - кадри з відео — завжди: довільний доступ до кадру в H.264 повільний, а читаємо кожен кадр
        багато разів (3 моделі × тайли, видача, пошук за зразком);
      - кадри з послідовностей картинок (VisDrone, UAVDT) — лише якщо їх треба зменшити до max_side.
        Інакше в keyframes.csv іде шлях до оригіналу: без дубля на диску й без повторного стиснення JPEG.

    Кеш: якщо параметри й джерела ті самі, а всі кадри на місці, повертає збережений результат.
    Статистика прогону (скільки переглянуто / залишено / записано) — у df.attrs["stats"],
    і з кешу теж; stats["cached"] каже, звідки вона.
    """
    from tqdm.auto import tqdm

    out_dir = Path(out_dir)
    meta_csv = out_dir / "keyframes.csv"
    params_json = out_dir / "keyframes_params.json"
    stats_json = out_dir / "keyframes_stats.json"
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
    for old in frames_dir.glob("*.jpg"):  # кадри попереднього прогону з іншими параметрами
        old.unlink()

    steps = [max(1, int(round(s.fps / sample_fps))) for s in sources]
    rows, per_source, kf_id, n_copied = [], [], 0, 0
    with tqdm(total=sum(_n_sampled(s, st) for s, st in zip(sources, steps)), desc="ключові кадри",
              unit="кадр") as bar:  # leave=False не ставимо: VS Code на місці закритого віджета пише «model not found»
        for s, step in zip(sources, steps):
            bar.set_postfix_str(s.source_id.rsplit("/", 1)[-1], refresh=False)
            last_thumb, n_seen, n_kept = None, 0, 0
            for idx, fr, orig in _iter_frames(s, step):
                n_seen += 1
                bar.update()
                thumb = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), (32, 18), interpolation=cv2.INTER_AREA)
                if last_thumb is not None and np.abs(thumb.astype(np.float32) - last_thumb).mean() < dedup_thr:
                    continue
                last_thumb = thumb.astype(np.float32)
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
        # FRAME_COUNT відео приблизний: недобраний бар став би червоним, як при помилці.
        # reset(total) — бо віджет у Jupyter не підхоплює просте bar.total = ...
        n_done = bar.n
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
    """
    Тайл 0 — увесь кадр; далі сітка rows x cols з перекриттям (SAHI-подібно, слайд 23).
    Для 16:9 сітка 2x3 дає майже квадратні тайли, тож letterbox майже нічого не втрачає.
    """
    rows = []
    for r in kf.itertuples(index=False):
        rows.append((r.kf_id, 0, 0, 0, r.width, r.height))
        if grid:
            gr, gc = grid
            tw = r.width / (gc - (gc - 1) * overlap)
            th = r.height / (gr - (gr - 1) * overlap)
            t = 1
            for i in range(gr):
                for j in range(gc):
                    x0, y0 = j * tw * (1 - overlap), i * th * (1 - overlap)
                    rows.append((r.kf_id, t, int(round(x0)), int(round(y0)),
                                 int(round(min(r.width, x0 + tw))), int(round(min(r.height, y0 + th)))))
                    t += 1
    return pd.DataFrame(rows, columns=["kf_id", "tile_idx", "x0", "y0", "x1", "y1"])


def embed_tiles(enc: Encoder, kf: pd.DataFrame, tiles: pd.DataFrame, cache_dir: str | Path,
                batch_size: int = 64, force: bool = False, save_every: int = 10) -> np.ndarray:
    """
    Ембеддинги всіх тайлів (кеш на диску). Порядок рядків = порядок tiles.

    Ключ кешу — хеш тайлів, кадрів (kf_id, path), моделі й препроцесингу: змінили сітку, SAMPLE_FPS
    чи модель — рахуємо заново, інакше беремо готовий .npy.
    Рахуємо прямо у файл на диску (memmap) і кожні save_every батчів фіксуємо прогрес: якщо прогін
    обірвався (перезапуск ядра, Ctrl+C), наступний виклик продовжить з місця зупинки.
    enc.cached — чи взято результат з кешу; enc.img_per_s — швидкість останнього прогону.
    """
    from tqdm.auto import tqdm

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    sig = hashlib.md5()
    sig.update(pd.util.hash_pandas_object(tiles, index=False).values.tobytes())
    sig.update(pd.util.hash_pandas_object(kf[["kf_id", "path"]], index=False).values.tobytes())
    sig.update(f"{enc.key}|{enc.resize_mode}|{getattr(enc, 'checkpoint', '')}".encode())
    f = cache_dir / f"emb_{enc.key}_{sig.hexdigest()[:10]}.npy"
    meta_f = f.with_suffix(".json")
    part_f, prog_f = f.with_suffix(".partial.npy"), f.with_suffix(".partial.json")
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

    paths = kf.set_index("kf_id")["path"]
    boxes = tiles[["kf_id", "x0", "y0", "x1", "y1"]].to_numpy()
    cur_id, cur_img = None, None
    t0, n_new = time.time(), 0
    with tqdm(total=n, initial=done, desc=enc.key, unit="тайл",
              postfix="продовжуємо з кешу" if done else None) as bar:
        for b, i in enumerate(range(done, n, batch_size)):
            crops = []
            for k, x0, y0, x1, y1 in boxes[i:i + batch_size]:
                if k != cur_id:
                    cur_img, cur_id = Image.open(paths[k]).convert("RGB"), k
                crops.append(cur_img.crop((x0, y0, x1, y1)))
            out[i:i + len(crops)] = enc.encode_images(crops)
            n_new += len(crops)
            bar.update(len(crops))
            if (b + 1) % save_every == 0:
                out.flush()
                prog_f.write_text(json.dumps(dict(done=i + len(crops))))
    out.flush()
    del out  # закрити memmap, інакше Windows не дасть перейменувати файл
    dt = time.time() - t0
    enc.img_per_s = n_new / max(dt, 1e-6) if n_new else None
    part_f.replace(f)
    prog_f.unlink(missing_ok=True)
    meta_f.write_text(json.dumps(dict(img_per_s=enc.img_per_s, n=n, seconds=dt, resumed_from=done)))
    return np.load(f)


# --------------------------------------------------------------------------------------
# FAISS і ранжування
# --------------------------------------------------------------------------------------
def build_flat_index(emb: np.ndarray):
    """Точний пошук. На тисячах–десятках тисяч векторів — частки мілісекунди на CPU."""
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
    kid = tiles["kf_id"].to_numpy()
    mask = np.ones(len(tiles), bool) if use_tiles else (tiles["tile_idx"].to_numpy() == 0)
    out = np.full(n_frames, -np.inf, dtype=np.float32)
    np.maximum.at(out, kid[mask], tile_s[mask])
    return out


def search(index, q: np.ndarray, tiles: pd.DataFrame, kf: pd.DataFrame, k: int = 12,
           use_tiles: bool = True, exclude: Iterable[int] = ()) -> pd.DataFrame:
    """
    Інтерактивний пошук: FAISS top-k тайлів -> агрегація в кадри.
    Беремо тайлів із запасом; якщо унікальних кадрів не вистачило — повний перебір.
    exclude — kf_id, які не можна повертати (напр. кадр-зразок і його сусіди в пошуку за зразком).
    """
    exclude = set(exclude)
    tpf = int(tiles.groupby("kf_id").size().max()) if use_tiles else 1
    sub = tiles if use_tiles else tiles[tiles["tile_idx"] == 0]

    def _hits(kk):
        D, I = index.search(np.ascontiguousarray(q[None], dtype="float32"), kk)
        h = sub.reindex(I[0]).assign(score=D[0], tile_row=I[0]).dropna(subset=["kf_id"])
        return h[~h["kf_id"].isin(exclude)]

    kk = min(index.ntotal, (k + len(exclude)) * tpf * 4)
    hits = _hits(kk)
    if hits["kf_id"].nunique() < k and kk < index.ntotal:
        hits = _hits(index.ntotal)
    best = hits.sort_values("score", ascending=False).drop_duplicates("kf_id").head(k)
    best = best.astype({"kf_id": int, "tile_idx": int, "x0": int, "y0": int, "x1": int, "y1": int})
    res = best.merge(kf[["kf_id", "source_id", "t_sec", "path"]], on="kf_id", how="left")
    res.insert(0, "rank", np.arange(1, len(res) + 1))
    return res


# --------------------------------------------------------------------------------------
# Ground truth
# --------------------------------------------------------------------------------------
# Категорії VisDrone: 0 ignored, 1 pedestrian, 2 people, 3 bicycle, 4 car, 5 van, 6 truck,
# 7 tricycle, 8 awning-tricycle, 9 bus, 10 motor, 11 others.
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
    """
    Авто-GT з розмітки VisDrone-VID: кадр релевантний запиту, якщо на ньому є об'єкт
    потрібної категорії з короткою стороною >= min_px (у пікселях вихідного кадру)
    і оклюзією <= max_occlusion (0 немає, 1 часткова, 2 сильна).
    """
    cols = ["frame", "tid", "x", "y", "w", "h", "score", "cat", "trunc", "occ"]
    frame_cats: Dict[Tuple[str, int], set] = {}
    for s in sources:
        if not s.ann_path:
            continue
        a = pd.read_csv(s.ann_path, header=None, usecols=range(10), names=cols)
        a = a[(a["score"] == 1) & (a["cat"] > 0) & (np.minimum(a["w"], a["h"]) >= min_px) & (a["occ"] <= max_occlusion)]
        for fr, g in a.groupby("frame"):
            frame_cats[(s.source_id, int(fr) - 1)] = set(g["cat"].astype(int))  # кадри 1-based
    gt = {}
    for qid, q in queries.items():
        rel = [frame_cats.get((r.source_id, int(r.frame_idx)), set()) & set(q["cats"])
               for r in kf.itertuples(index=False)]
        sub = kf[[bool(x) for x in rel]]
        gt[qid] = dict(text=q["text"], relevant=set(sub["kf_id"].astype(int)), events=_group_events(sub, gap_s))
    return gt


def load_csv_gt(kf: pd.DataFrame, csv_path: str | Path, tol_s: float = 1.0) -> Dict[str, dict]:
    """
    Власний GT: CSV з колонками query_id,text,source_id,start_s,end_s.
    Кожен рядок — одна подія (інтервал часу, де ціль видно). source_id="*" — будь-яке джерело.
    """
    g = pd.read_csv(csv_path)
    gt: Dict[str, dict] = {}
    for qid, rows in g.groupby("query_id"):
        events, rel = [], set()
        for r in rows.itertuples(index=False):
            m = (kf["t_sec"] >= r.start_s - tol_s) & (kf["t_sec"] <= r.end_s + tol_s)
            if str(r.source_id) != "*":
                m &= kf["source_id"].astype(str) == str(r.source_id)
            ids = kf.loc[m, "kf_id"].astype(int).tolist()
            if ids:
                events.append(ids)
                rel |= set(ids)
            else:
                warnings.warn(f"{qid}: інтервал {r.source_id} {r.start_s}-{r.end_s} c не містить ключових кадрів")
        gt[str(qid)] = dict(text=str(rows["text"].iloc[0]), relevant=rel, events=events)
    return gt


def gt_summary(gt: Dict[str, dict], n_frames: int) -> pd.DataFrame:
    return pd.DataFrame([dict(query=q, text=v["text"], relevant_frames=len(v["relevant"]),
                              events=len(v["events"]), prevalence=len(v["relevant"]) / max(n_frames, 1))
                         for q, v in gt.items()]).sort_values("prevalence")


# --------------------------------------------------------------------------------------
# Метрики (слайд 44)
# --------------------------------------------------------------------------------------
def rank_metrics(fscore: np.ndarray, g: dict, ks: Sequence[int] = (10, 50)) -> dict:
    """
    EventRecall@k — частка подій, що мають хоча б один кадр у топ-k
                    («чи потрапила ціль у те, що побачить оператор»).
    P@k           — частка релевантних кадрів у топ-k («скільки сміття вгорі»).
    first_hit     — ранг першого релевантного кадру.
    AP            — average precision по кадрах.
    """
    n = len(fscore)
    order = np.argsort(-fscore, kind="stable")
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n)
    rel = np.zeros(n, bool)
    rel[list(g["relevant"])] = True
    out = dict(n_rel=int(rel.sum()), n_events=len(g["events"]))
    if rel.sum() == 0:
        return out
    ev_best = np.array([rank[e].min() for e in g["events"]])
    for k in ks:
        out[f"EventR@{k}"] = float((ev_best < k).mean())
        out[f"P@{k}"] = float(rel[order[:k]].mean())
    out["first_hit"] = int(rank[rel].min()) + 1
    hits = rel[order]
    prec = np.cumsum(hits) / np.arange(1, n + 1)
    out["AP"] = float((prec * hits).sum() / hits.sum())
    out["prevalence"] = float(rel.mean())
    return out


# --------------------------------------------------------------------------------------
# PQ — що втрачається, коли по каналу йдуть коди замість векторів (кейс 1)
# --------------------------------------------------------------------------------------
def build_pq(emb: np.ndarray, M: int = 64, nbits: int = 8, seed: int = 0, train: Optional[np.ndarray] = None):
    """
    IndexPQ (L2; для нормованих векторів порядок той самий, що й за cos).
    emb   — вектори, які кодуємо й шукаємо.
    train — на чому вчити кодову книгу (за замовчуванням emb). У реальній системі це великий
            архів на землі; тут — усі тайли, навіть якщо ранжуємо лише цілі кадри.

    k-means для 2^nbits центроїдів потребує щонайменше 2^nbits векторів — інакше зменшуємо nbits.
    FAISS радить >= 39 на центроїд; менше — працює, але кодова книга грубша: попереджаємо один раз
    (замість M однакових попереджень від самого FAISS).
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
        warnings.warn(f"PQ: {n} векторів для навчання — {n / (1 << nb):.0f} на центроїд (FAISS радить >= 39). "
                      f"Кодова книга грубша, ніж була б на великому архіві.")
    pq = faiss.IndexPQ(d, M, nb)
    pq.pq.cp.seed = seed
    pq.pq.cp.min_points_per_centroid = 1
    pq.train(xt)
    pq.add(x)
    return pq, dict(M=M, nbits=nb, code_bytes=M * nb / 8, fp16_bytes=d * 2, n_train=n)


def pq_tile_scores(pq, q: np.ndarray, emb: Optional[np.ndarray] = None, rerank_top: int = 0) -> np.ndarray:
    """
    Оцінки тайлів за PQ-кодами. cos ~= 1 - d^2/2 (FAISS L2 повертає квадрат відстані).
    rerank_top > 0: топ-R кандидатів переранжовуємо точним cos по повних векторах
    («дозапит повних векторів для топу видачі», слайд 49).
    """
    D, I = pq.search(np.ascontiguousarray(q[None], dtype="float32"), pq.ntotal)
    s = np.empty(pq.ntotal, dtype=np.float32)
    s[I[0]] = 1.0 - D[0] / 2.0
    if rerank_top and emb is not None:
        top = I[0][:rerank_top]
        s[top] = emb[top] @ q + 10.0  # кандидати гарантовано вище за решту, порядок — за точним cos
    return s


# --------------------------------------------------------------------------------------
# Візуалізація
# --------------------------------------------------------------------------------------
def show_results(res: pd.DataFrame, title: str = "", ncols: int = 4, relevant: Optional[set] = None,
                 thumb_w: float = 4.0):
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    n = len(res)
    if n == 0:
        print("порожня видача")
        return
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(thumb_w * ncols, thumb_w * 0.78 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, r in zip(axes, res.itertuples(index=False)):
        im = Image.open(r.path)
        ax.imshow(im)
        color = "#2ecc71" if relevant is None or r.kf_id in relevant else "#e74c3c"
        if r.tile_idx != 0:
            ax.add_patch(patches.Rectangle((r.x0, r.y0), r.x1 - r.x0, r.y1 - r.y0, fill=False, lw=2, ec="#f1c40f"))
        mark = "" if relevant is None else (" ✓" if r.kf_id in relevant else " ✗")
        src = str(r.source_id).rsplit("/", 1)[-1]
        ax.set_title(f"#{r.rank} cos={r.score:.3f}{mark}\n{src} · t={r.t_sec:.1f}s", fontsize=8, color=color)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    plt.show()


def show_image(im: Image.Image, title: str = "", size=(4, 3)):
    import matplotlib.pyplot as plt

    plt.figure(figsize=size)
    plt.imshow(im)
    plt.axis("off")
    plt.title(title)
    plt.show()


def plot_metrics(R: pd.DataFrame, metrics: Sequence[str] = ("EventR@50", "P@10"),
                 baselines: Optional[Dict[str, float]] = None):
    """Стовпчики: метрика, усереднена по запитах, для кожної моделі × (prompt, input).
    baselines — горизонтальна лінія «випадкове ранжування» для вибраних метрик."""
    import matplotlib.pyplot as plt

    baselines = baselines or {}
    n_q = R["query"].nunique()
    fig, axes = plt.subplots(1, len(metrics), figsize=(6.5 * len(metrics), 4.2))
    for ax, metric in zip(np.atleast_1d(axes), metrics):
        piv = R.groupby(["model", "prompt", "input"])[metric].mean().unstack(["prompt", "input"])
        piv.plot.bar(ax=ax, rot=0, width=0.8)
        ax.set_title(f"{metric} (середнє по {n_q} запитах)")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8, title="prompt / input")
        if metric in baselines:
            b = baselines[metric]
            ax.axhline(b, ls="--", c="k", lw=1)
            ax.text(-0.4, b + 0.02, "випадкове ранжування", fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_thresholds(panels: Sequence[dict], rel_mask: np.ndarray, title: str = ""):
    """
    Гістограми сирого cos «з ціллю» / «без цілі», по панелі на модель.
    Кожна панель: dict(title, scores, t_low, load) — поріг і навантаження рахує ноутбук.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 3.4))
    for ax, p in zip(np.atleast_1d(axes), panels):
        fs = p["scores"]
        bins = np.linspace(fs.min(), fs.max(), 50)
        ax.hist(fs[~rel_mask], bins, alpha=0.6, label="без цілі", log=True)
        ax.hist(fs[rel_mask], bins, alpha=0.7, label="з ціллю", log=True)
        ax.axvline(p["t_low"], c="k", ls="--")
        ax.set_title(f"{p['title']}\nT_low={p['t_low']:.3f}: оператору {p['load']:.0%} порожніх кадрів", fontsize=9)
        ax.set_xlabel("сирий cos")
        ax.legend(fontsize=8)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.tick_params(labelsize=8)
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def search_box(on_query: Callable[..., None], models: Sequence[str], gt_texts: Optional[Dict[str, str]] = None,
               value: str = "a truck parked next to a building"):
    """
    Панель пошуку (ipywidgets). Що робити із запитом, вирішує ноутбук:
        on_query(text, models=[...], k=int, use_tiles=bool, aerial=bool, gt_query=str|None)
    gt_texts — {query_id: text} із GT: вибір запиту підставляє його текст і вмикає позначки ✓/✗.
    """
    try:
        import ipywidgets as W
        from IPython.display import display
    except ImportError:
        print("ipywidgets не встановлено — викликайте ask('...') напряму")
        return
    gt_texts = gt_texts or {}
    box = W.Text(value=value, placeholder="опис цілі англійською", layout=W.Layout(width="55%"))
    btn = W.Button(description="Шукати", button_style="primary", icon="search")
    gt_dd = W.Dropdown(options=[("— без GT —", None)] + [(f"{q}: {t}", q) for q, t in gt_texts.items()],
                       description="GT", layout=W.Layout(width="45%"))
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
            on_query(box.value.strip(), models=sel, k=k_sl.value, use_tiles=tiles_tb.value,
                     aerial=prompt_tb.value, gt_query=gq)

    gt_dd.observe(_pick_gt, names="value")
    btn.on_click(_go)
    display(W.VBox([
        W.HBox([box, btn]),
        gt_dd,
        W.HBox([W.Label("моделі:"), *model_cb]),
        W.HBox([k_sl, tiles_tb, prompt_tb]),
        out,
    ]))


def image_search_panel(on_query: Callable[..., None], kf: pd.DataFrame, models: Sequence[str],
                       gt_texts: Optional[Dict[str, str]] = None, start_kf: Optional[int] = None,
                       start_box: Optional[Tuple[int, int, int, int]] = None):
    """
    Пошук за зразком (ipywidgets). Зображення — кадр з архіву або своє фото; регіон — рамкою мишею
    (потрібен ipyevents) або слайдерами. Що робити з вирізаним регіоном, вирішує ноутбук:
        on_query(crop: PIL.Image, models=[...], k=int, use_tiles=bool, src_kf=int|None, gt_query=str|None)
    src_kf — kf_id кадру з архіву (None для свого фото): ноутбук виключає його окіл з видачі.
    """
    try:
        import ipywidgets as W
        from IPython.display import display
    except ImportError:
        print("ipywidgets не встановлено — викликайте ask_image(crop) напряму")
        return None
    p = _ImageQueryPanel(W, on_query, kf, models, gt_texts or {}, start_kf, start_box)
    display(p.ui)
    return p


class _ImageQueryPanel:
    PREVIEW_W = 720     # ширина прев'ю в пікселях екрана
    MIN_BOX = 16        # мінімальна сторона рамки в пікселях кадру
    DRAG_MIN = 4        # менший рух миші — це клік, а не протягування

    def __init__(self, W, on_query, kf, models, gt_texts, start_kf, start_box):
        self.W, self.on_query, self.kf, self.models, self.gt_texts = W, on_query, kf.reset_index(drop=True), models, gt_texts
        self.photo: Optional[Image.Image] = None
        self.src_kf: Optional[int] = None
        self._drag: Optional[Tuple[float, float]] = None

        L = W.Layout
        self.frame_sl = W.IntSlider(min=0, max=max(len(self.kf) - 1, 0), description="кадр", continuous_update=False,
                                    layout=L(width="55%"))
        self.frame_lbl = W.Label()
        self.upload = W.FileUpload(accept="image/*", multiple=False, description="Обрати фото")
        self.preview = W.Image(format="jpeg", layout=L(width=f"{self.PREVIEW_W}px"))
        self.crop_img = W.Image(format="jpeg", layout=L(max_width="220px", max_height="220px"))
        self.crop_lbl = W.Label()
        self.x_rng = W.IntRangeSlider(description="x", continuous_update=False, layout=L(width=f"{self.PREVIEW_W}px"))
        self.y_rng = W.IntRangeSlider(description="y", continuous_update=False, layout=L(width=f"{self.PREVIEW_W}px"))
        self.full_btn = W.Button(description="Увесь кадр", icon="expand")
        self.go_btn = W.Button(description="Шукати схожі", button_style="primary", icon="search")
        self.model_cb = [W.Checkbox(value=True, description=m, indent=False, layout=L(width="auto")) for m in models]
        self.k_sl = W.IntSlider(value=8, min=4, max=24, step=4, description="top-k", continuous_update=False)
        self.tiles_tb = W.ToggleButtons(options=[("тайли", True), ("увесь кадр", False)], value=True)
        self.tiles_tb.style.button_width = "90px"
        self.gt_dd = W.Dropdown(options=[("— без GT —", None)] + [(f"{q}: {t}", q) for q, t in gt_texts.items()],
                                description="GT", layout=L(width="45%"))
        self.out = W.Output()

        try:
            from ipyevents import Event
            self.mouse = Event(source=self.preview, prevent_default_action=True,
                               watched_events=["mousedown", "mousemove", "mouseup", "mouseleave"],
                               throttle_or_debounce="throttle", wait=40)
            self.mouse.on_dom_event(self._on_mouse)
            hint = "Протягніть мишею по кадру, щоб виділити регіон пошуку; слайдери x / y — для точного налаштування."
        except ImportError:
            hint = "Миша недоступна: встановіть ipyevents, щоб виділяти регіон прямо на кадрі. Поки — слайдери x / y."

        self.frame_sl.observe(lambda ch: self._load_kf(ch["new"]), names="value")
        self.upload.observe(self._on_upload, names="value")
        for s in (self.x_rng, self.y_rng):
            s.observe(lambda _: self._redraw(), names="value")
        self.full_btn.on_click(lambda _: self._set_box(0, 0, *self.photo.size))
        self.go_btn.on_click(self._go)

        self.ui = W.VBox([
            W.HBox([self.frame_sl, self.frame_lbl]),
            W.HBox([self.upload, self.full_btn]),
            W.HTML(f"<i>{hint}</i>"),
            W.HBox([self.preview, W.VBox([W.Label("регіон-запит:"), self.crop_img, self.crop_lbl])]),
            self.x_rng, self.y_rng,
            W.HBox([W.Label("моделі:"), *self.model_cb]),
            W.HBox([self.k_sl, self.tiles_tb]),
            W.HBox([self.gt_dd, self.go_btn]),
            self.out,
        ])

        row = 0
        if start_kf is not None and (self.kf["kf_id"] == start_kf).any():
            row = int(np.flatnonzero(self.kf["kf_id"].to_numpy() == start_kf)[0])
        self.frame_sl.value = row
        self._load_kf(row)              # якщо row == 0, observe не спрацює
        if start_box is not None:
            self._set_box(*start_box)

    # ------------------------------------------------------------------ зображення
    def _set_photo(self, im: Image.Image, src_kf: Optional[int], label: str):
        self.photo, self.src_kf = im.convert("RGB"), src_kf
        w, h = self.photo.size
        self.frame_lbl.value = label
        with self.x_rng.hold_trait_notifications(), self.y_rng.hold_trait_notifications():
            self.x_rng.max, self.y_rng.max = w, h
            self.x_rng.value, self.y_rng.value = (0, w), (0, h)
        self._redraw()

    def _load_kf(self, row: int):
        if not len(self.kf):
            return
        r = self.kf.iloc[row]
        self._set_photo(Image.open(r["path"]), int(r["kf_id"]), f"{r['source_id']} · t={r['t_sec']:.1f}s · kf {r['kf_id']}")

    def _on_upload(self, ch):
        files = ch["new"]
        if not files:
            return
        f = files[0]
        import io

        try:
            im = Image.open(io.BytesIO(bytes(f["content"])))
        except Exception as e:  # noqa: BLE001
            self.frame_lbl.value = f"не вдалося відкрити {f['name']}: {type(e).__name__} (потрібен JPG або PNG)"
            return
        self._set_photo(im, None, f"своє фото: {f['name']} ({im.width}×{im.height})")

    # ------------------------------------------------------------------ рамка
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
        import io
        from PIL import ImageDraw

        w, h = self.photo.size
        sc = self.PREVIEW_W / w
        pv = self.photo.resize((self.PREVIEW_W, max(1, round(h * sc))), Image.BILINEAR)
        x0, y0, x1, y1 = self._box()
        full = (x0, y0, x1, y1) == (0, 0, w, h)
        if not full:
            # затемнюємо все поза регіоном, щоб рамку було видно на будь-якому фоні
            dim = Image.new("RGB", pv.size, (0, 0, 0))
            mask = Image.new("L", pv.size, 110)
            ImageDraw.Draw(mask).rectangle([x0 * sc, y0 * sc, x1 * sc, y1 * sc], fill=0)
            pv = Image.composite(dim, pv, mask)
            ImageDraw.Draw(pv).rectangle([x0 * sc, y0 * sc, x1 * sc, y1 * sc], outline=(241, 196, 15), width=2)
        buf = io.BytesIO()
        pv.save(buf, "JPEG", quality=85)
        self.preview.value = buf.getvalue()

        cr = self.crop()
        cr.thumbnail((220, 220))
        buf = io.BytesIO()
        cr.save(buf, "JPEG", quality=90)
        self.crop_img.value = buf.getvalue()
        self.crop_lbl.value = f"{x1 - x0}×{y1 - y0} px"

    # ------------------------------------------------------------------ миша
    def _event_xy(self, ev) -> Optional[Tuple[float, float]]:
        """Координати події в пікселях кадру; розмір рамки беремо з події — браузер міг відмасштабувати прев'ю."""
        rw, rh = ev.get("boundingRectWidth"), ev.get("boundingRectHeight")
        if not rw or not rh or self.photo is None:
            return None
        w, h = self.photo.size
        return (min(max(ev.get("relativeX", 0) / rw * w, 0), w), min(max(ev.get("relativeY", 0) / rh * h, 0), h))

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

    # ------------------------------------------------------------------ пошук
    def _go(self, _=None):
        sel = [m for m, cb in zip(self.models, self.model_cb) if cb.value]
        with self.out:
            self.out.clear_output(wait=True)
            if self.photo is None or not sel:
                print("виберіть зображення і хоча б одну модель")
                return
            self.on_query(self.crop(), models=sel, k=self.k_sl.value, use_tiles=self.tiles_tb.value,
                          src_kf=self.src_kf, gt_query=self.gt_dd.value)


# --------------------------------------------------------------------------------------
# Синтетичні дані для smoke-тесту (без мережі)
# --------------------------------------------------------------------------------------
def make_synthetic_visdrone(root: str | Path, n_seq: int = 3, n_frames: int = 90, size=(640, 360), seed: int = 0) -> Path:
    """Мінімальна копія структури VisDrone-VID: щоб перевірити конвеєр без даних і ваг."""
    rng = np.random.default_rng(seed)
    root = Path(root) / "VisDrone2019-VID-synthetic"
    W, H = size
    for s in range(n_seq):
        sd = root / "sequences" / f"uav{s:07d}_00000_v"
        sd.mkdir(parents=True, exist_ok=True)
        (root / "annotations").mkdir(parents=True, exist_ok=True)
        bg = rng.integers(60, 120, size=(H, W, 3), dtype=np.uint8)
        lines = []
        for f in range(n_frames):
            fr = np.roll(bg, shift=f * 3, axis=1).copy()
            for cat, color, period in [(6, (0, 0, 220), 30), (9, (220, 0, 0), 45)]:
                if (f // period) % 2 == 0:
                    x, y = (40 + 5 * f + 70 * cat) % (W - 60), 60 + 20 * cat % (H - 60)
                    cv2.rectangle(fr, (x, y), (x + 50, y + 30), color, -1)
                    lines.append(f"{f + 1},{cat},{x},{y},50,30,1,{cat},0,0")
            cv2.imwrite(str(sd / f"{f + 1:07d}.jpg"), fr)
        (root / "annotations" / f"{sd.name}.txt").write_text("\n".join(lines) + "\n")
    return root
