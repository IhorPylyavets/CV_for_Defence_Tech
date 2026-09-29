"""Інтерактивні панелі заняття 1 (demo1 і demo2).

Тут лише обв'язка: віджети, малювання, обробники подій. Уся оптика й підрахунки
лишаються в ноутбуках і приходять сюди аргументами launch_*(), щоб правки формул
діяли одразу, без перезапуску ядра.

Вміст:
    Panel, PhotoPanel   — спільна основа
    FovPanel            — demo1: фокусна відстань і поле зору
    ZoomPanel           — demo1: цифровий зум проти теле-сенсора
    RangePanel          — demo2: калькулятор дальності детекції
    ThresholdPanel      — demo2: поріг детекції справжнім детектором
    LeversPanel         — demo2: resize проти tiled проти теле
"""

import time
from pathlib import Path

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from ipywidgets import (Button, Checkbox, Dropdown, FileUpload, FloatSlider, FloatText, GridBox,
                        HBox, HTML, Image, IntSlider, Layout, Output, ToggleButtons, VBox)

try:
    from ipyevents import Event as MouseEvent
except ImportError:  # демо має працювати й без пакета, просто без миші
    MouseEvent = None

# ───────────────────────────────────────────────────────────────── артефакти

# шлях рахуємо від самого модуля, тож він не залежить від того, звідки запущено ноутбук
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = REPO_ROOT / "artifacts" / "models"
IMAGES_DIR = REPO_ROOT / "artifacts" / "images"
DEMO1_PHOTO = IMAGES_DIR / "lesson_1_demo_1_rgb.jpg"  # стартове фото demo1, поки не обрано своє
DEMO1_PHOTO_FOV_DEG = 120.0                           # горизонтальний кут, під яким його знято
DEMO2_THRESHOLD_PHOTO = IMAGES_DIR / "lesson_1_demo_2_2_1.jpg"  # стартові фото demo2, частина 2
DEMO2_LEVERS_PHOTO = IMAGES_DIR / "lesson_1_demo_2_2_2.jpg"


def model_path(name: str) -> str:
    """Шлях до ваг в artifacts/models.

    Якщо файлу там немає, ultralytics сам завантажить стандартні ваги саме за цим шляхом,
    а не в теку, звідки запущено ноутбук. В аудиторії без інтернету це не спрацює,
    тож ваги варто покласти заздалегідь.
    """
    return str(MODELS_DIR / name)


# ─────────────────────────────────────────────────────────────── оформлення

BG, FG, MUTED, ACCENT, WARN = "#17191d", "#e6e8ea", "#8a8f98", "#2fe08a", "#ff7a59"
SPINE, GRID = "#2a2e35", "#23272e"
ACCENT_BGR, MUTED_BGR = (122, 224, 47), (152, 143, 138)  # ті самі кольори для cv2
BASE_COLOR, TILED_COLOR, TELE_COLOR = "#7aa2f7", ACCENT, "#f7b26a"

DISPLAY_MAX_DIM = 1000
LABEL_STYLE = {"description_width": "150px"}
FIELD = Layout(width="250px")

# ───────────────────────────────────────────────────────────────── довідники

CUSTOM = "— своє —"
DEFAULT_PRESET = "1/2.8″ (5.6 × 3.1)"
SENSOR_PRESETS = {
    "1/4″ (3.6 × 2.7)": (3.6, 2.7),
    "1/3″ (4.8 × 3.6)": (4.8, 3.6),
    DEFAULT_PRESET: (5.6, 3.1),
    "1/2″ (6.4 × 4.8)": (6.4, 4.8),
    "1/1.8″ (7.2 × 5.4)": (7.2, 5.4),
    "2/3″ (8.8 × 6.6)": (8.8, 6.6),
    "1″ (13.2 × 8.8)": (13.2, 8.8),
    "4/3″ (17.3 × 13.0)": (17.3, 13.0),
    "APS-C (23.6 × 15.7)": (23.6, 15.7),
    "Full frame (36 × 24)": (36.0, 24.0),
    CUSTOM: None,
}

DEFAULT_RESOLUTION = "HD 1280×720"
SENSOR_RESOLUTIONS = {
    "VGA 640×480": (640, 480),
    DEFAULT_RESOLUTION: (1280, 720),
    "Full HD 1920×1080": (1920, 1080),
    "QHD 2560×1440": (2560, 1440),
    "4K 3840×2160": (3840, 2160),
    "4056 x 3040": (4056, 3040)
}

# Режим сенсора: (вихід w×h, область зчитування w×h у пікселях сенсора, fps або None).
# Біннінг = зчитування / вихід; зчитування менше за матрицю — кроп, FOV вужчий.
# Камери Raspberry Pi — з `rpicam-hello --list-cameras`; решта — з даташитів сенсорів.
# fps = None там, де частота залежить від плати й інтерфейсу, а не від сенсора.
SENSORS = {
    # IP-камери та CCTV
    "IMX307 · IP 1/2.8″": {"size": (1920, 1080), "pixel_um": 2.9,
                           "modes": [((1920, 1080), (1920, 1080), None)]},
    "IMX335 · IP 5MP": {"size": (2592, 1944), "pixel_um": 2.0,
                        "modes": [((2592, 1944), (2592, 1944), None),
                                  ((1920, 1080), (1920, 1080), None)]},
    "IMX415 · IP 4K": {"size": (3840, 2160), "pixel_um": 1.45,
                       "modes": [((3840, 2160), (3840, 2160), None),
                                 ((1920, 1080), (3840, 2160), None)]},
    "IMX678 · 4K 1/1.8″": {"size": (3840, 2160), "pixel_um": 2.0,
                           "modes": [((3840, 2160), (3840, 2160), None),
                                     ((1920, 1080), (3840, 2160), None)]},
    "IMX585 · 4K 1/1.2″": {"size": (3840, 2160), "pixel_um": 2.9,
                           "modes": [((3840, 2160), (3840, 2160), None),
                                     ((1920, 1080), (3840, 2160), None)]},
    # Raspberry Pi
    "OV5647 · Camera v1": {"size": (2592, 1944), "pixel_um": 1.4,
                           "modes": [((2592, 1944), (2592, 1944), 15), ((1920, 1080), (1920, 1080), 30),
                                     ((1296, 972), (2592, 1944), 43), ((640, 480), (2560, 1920), 58)]},
    "IMX219 · Camera v2": {"size": (3280, 2464), "pixel_um": 1.12,
                           "modes": [((3280, 2464), (3280, 2464), 21), ((1920, 1080), (1920, 1080), 47),
                                     ((1640, 1232), (3280, 2464), 41), ((640, 480), (1280, 960), 103)]},
    "IMX708 · Camera v3": {"size": (4608, 2592), "pixel_um": 1.4,
                           "modes": [((4608, 2592), (4608, 2592), 14), ((2304, 1296), (4608, 2592), 56),
                                     ((1536, 864), (3072, 1728), 120)]},
    "IMX477 · HQ Camera": {"size": (4056, 3040), "pixel_um": 1.55,
                           "modes": [((4056, 3040), (4056, 3040), 10), ((2028, 1520), (4056, 3040), 40),
                                     ((2028, 1080), (4056, 2160), 50), ((1332, 990), (2664, 1980), 120)]},
    "IMX500 · AI Camera": {"size": (4056, 3040), "pixel_um": 1.55,
                           "modes": [((4056, 3040), (4056, 3040), 10), ((2028, 1520), (4056, 3040), 30)]},
    "IMX296 · GS Camera": {"size": (1456, 1088), "pixel_um": 3.45,
                           "modes": [((1456, 1088), (1456, 1088), 60)]},
    # промислові, global shutter і великі матриці
    "AR0234 · GS 1/2.6″": {"size": (1920, 1200), "pixel_um": 3.0,
                           "modes": [((1920, 1200), (1920, 1200), 120)]},
    "IMX264 · GS 2/3″": {"size": (2448, 2048), "pixel_um": 3.45,
                         "modes": [((2448, 2048), (2448, 2048), None)]},
    "IMX183 · 1″ 20MP": {"size": (5472, 3648), "pixel_um": 2.4,
                         "modes": [((5472, 3648), (5472, 3648), None)]},
    # тепловізори (LWIR, неохолоджувані мікроболометри)
    "LWIR 640 · 12 мкм": {"size": (640, 512), "pixel_um": 12.0,
                          "modes": [((640, 512), (640, 512), None)]},
    "LWIR 384 · 12 мкм": {"size": (384, 288), "pixel_um": 12.0,
                          "modes": [((384, 288), (384, 288), None)]},
    "LWIR 160 · 12 мкм": {"size": (160, 120), "pixel_um": 12.0,
                          "modes": [((160, 120), (160, 120), None)]},
}
DEFAULT_SENSOR = "IMX307 · IP 1/2.8″"  # 1920 px × 2.9 мкм — сценарій зі слайда
MODEL_INPUTS = {"416 × 416": 416, "640 × 640": 640, "960 × 960": 960, "1280 × 1280": 1280}
THRESHOLDS = {"Детекція ~16 px": 16.0, "Класифікація ~32 px": 32.0, "Ідентифікація ~64 px": 64.0}

# ─────────────────────────────────────────────────────────────────── хелпери


def decode_upload(value) -> tuple[np.ndarray | None, str]:
    """RGB-зображення з FileUpload.value або (None, текст помилки)."""
    item = value[0] if isinstance(value, (list, tuple)) else next(iter(value.values()))
    img_bgr = cv2.imdecode(np.frombuffer(bytes(item["content"]), np.uint8), cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None, (f"Не вдалось прочитати «{item.get('name', 'файл')}»: OpenCV не читає HEIC/HEIF. "
                      "Пересохрани фото у JPG або PNG.")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), ""


def read_photo(path: Path) -> np.ndarray | None:
    """RGB-зображення з диска або None, якщо файлу немає чи OpenCV його не читає."""
    if not path.is_file():
        return None
    # np.fromfile + imdecode, бо cv2.imread на Windows не відкриває шляхи з кирилицею
    img_bgr = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    return None if img_bgr is None else cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def to_display(img: np.ndarray, max_dim: int = DISPLAY_MAX_DIM) -> tuple[np.ndarray, float]:
    """Зменшена копія для показу + коефіцієнт масштабу відносно оригіналу."""
    h, w = img.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale == 1.0:
        return img, 1.0
    small = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    return small, scale


def section(text: str) -> HTML:
    return HTML(f"<b style='color:{MUTED};letter-spacing:.09em;font-size:11px'>{text}</b>")


def photo_uploader() -> FileUpload:
    return FileUpload(accept="image/*", multiple=False, description="Обрати фото",
                      button_style="primary", icon="image", layout=Layout(width="220px"))


def dark_axes(*axes, hide: bool = False) -> None:
    """Темне тло панелей. hide=True — для картинок, де осі лише заважають."""
    for ax in axes:
        ax.set_facecolor(BG)
        if hide:
            ax.axis("off")
            continue
        for spine in ax.spines.values():
            spine.set_color(SPINE)
        ax.tick_params(colors=MUTED, labelsize=9)


def draw_viewfinder(ax, box, img_w: float, img_h: float, label: str) -> None:
    """Затемнює все поза рамкою, малює рамку з кутовими маркерами та підпис FOV."""
    x0, y0, w, h = box

    for rx, ry, rw, rh in ((0, 0, img_w, y0),
                           (0, y0 + h, img_w, img_h - y0 - h),
                           (0, y0, x0, h),
                           (x0 + w, y0, img_w - x0 - w, h)):
        ax.add_patch(patches.Rectangle((rx, ry), max(rw, 0), max(rh, 0),
                                       color="black", alpha=0.55, linewidth=0))

    ax.add_patch(patches.Rectangle((x0, y0), w, h, linewidth=1.6, edgecolor=ACCENT, facecolor="none"))

    tick = 0.07 * min(w, h)
    for cx, sx in ((x0, 1), (x0 + w, -1)):
        for cy, sy in ((y0, 1), (y0 + h, -1)):
            ax.plot((cx, cx + sx * tick), (cy, cy), color=ACCENT, linewidth=3)
            ax.plot((cx, cx), (cy, cy + sy * tick), color=ACCENT, linewidth=3)

    inside = y0 < 0.08 * img_h
    ax.text(x0 + (8 if inside else 0), y0 + (8 if inside else -8), label,
            color="#0d0f12", fontsize=10, fontweight="bold",
            ha="left", va="top" if inside else "bottom",
            bbox=dict(boxstyle="round,pad=0.35", facecolor=ACCENT, edgecolor="none"))


def centered_rect(ax, frame_w: float, frame_h: float, box_w: float, box_h: float,
                  color: str, dashed: bool, label: str, label_below: bool = False) -> None:
    """Рамка по центру кадру з підписом. Біля точки перемикання рамки майже збігаються,
    тому підписи розводимо: один над рамкою, інший під нею."""
    x0, y0 = (frame_w - box_w) / 2, (frame_h - box_h) / 2
    ax.add_patch(patches.Rectangle((x0, y0), box_w, box_h, linewidth=1.8, edgecolor=color,
                                   facecolor="none", linestyle="--" if dashed else "-"))

    if label_below:
        edge, outside_va, inside_va, step = y0 + box_h, "top", "bottom", 6
        outside_fits = edge < 0.94 * frame_h
    else:
        edge, outside_va, inside_va, step = y0, "bottom", "top", -6
        outside_fits = edge > 0.06 * frame_h

    y = edge + (step if outside_fits else -step)
    ax.text(x0 + (0 if outside_fits else 6), y, label,
            color="#0d0f12", fontsize=9, fontweight="bold",
            ha="left", va=outside_va if outside_fits else inside_va,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=color, edgecolor="none"))


# ──────────────────────────────────────────────────────────────────── основа


class Panel:
    """Спільна основа: вивід, придушення подій під час пакетних змін, безпечний рендер."""

    def __init__(self):
        self.out = Output()
        self._suspend = False

    def ui(self) -> VBox:
        raise NotImplementedError

    def problem(self) -> str | None:
        """Причина, чому малювати нічого. None — усе гаразд."""
        return None

    def draw(self) -> None:
        raise NotImplementedError

    def render(self, *_) -> None:
        if self._suspend:
            return
        self.out.clear_output(wait=True)
        with self.out:  # винятки з observer'ів не потрапляють у вивід клітинки — Output їх покаже
            problem = self.problem()
            if problem:
                print(problem)
                return
            self.draw()

    def show(self) -> None:
        display(self.ui())
        self.render()


class PhotoPanel(Panel):
    """Панель, яка працює із завантаженим фото."""

    upload_prompt = "Обери фото кнопкою вище."

    def __init__(self, default_photo: Path | None = None, photo_fov_deg: float = 70.0):
        super().__init__()
        self.photo = None
        self.default_photo = default_photo
        self.photo_fov_deg = photo_fov_deg  # стартове значення поля «FOV оригіналу»
        self.uploader = photo_uploader()
        self.uploader.observe(self._on_upload, names="value")

    def show(self) -> None:
        """Якщо є стартове фото — одразу малюємо з ним; своє фото кнопкою його замінить."""
        display(self.ui())
        self.photo = read_photo(self.default_photo) if self.default_photo else None
        if self.photo is None:
            self.render()
        else:
            self.on_photo()

    def _on_upload(self, *_) -> None:
        if not self.uploader.value:
            return
        self.out.clear_output(wait=True)
        loaded = None
        with self.out:
            loaded, error = decode_upload(self.uploader.value)
            if loaded is None:
                print(error)
        if loaded is None:  # і помилка декодування, і виняток лишають photo недоторканим
            return
        self.photo = loaded
        self.on_photo()

    def on_photo(self) -> None:
        """Реакція на нове фото. Типово — просто перемалювати."""
        self.render()

    def problem(self) -> str | None:
        return None if self.photo is not None else self.upload_prompt


# ───────────────────────────────────────── demo1: фокусна відстань і поле зору


class FovPanel(PhotoPanel):
    """Панель «оригінал + кадр камери». Уся оптика — у функціях, переданих ззовні."""

    def __init__(self, fov_from_focal, focal_from_fov, zoom_factor, crop_box, min_fov_deg,
                 default_photo: Path | None = None, photo_fov_deg: float = 70.0):
        super().__init__(default_photo, photo_fov_deg)
        self.fov_from_focal = fov_from_focal
        self.focal_from_fov = focal_from_fov
        self.zoom_factor = zoom_factor
        self.crop_box = crop_box
        self.min_fov = min_fov_deg

        self.preview = None
        self.preview_scale = 1.0
        self.fov_h = None

        self._build()
        self._wire()

    def _build(self) -> None:
        self.orig_fov = FloatText(value=self.photo_fov_deg, description="FOV оригіналу, °",
                                  style=LABEL_STYLE, layout=FIELD)
        self.preset = Dropdown(options=list(SENSOR_PRESETS), value=DEFAULT_PRESET,
                               description="Формат матриці", style=LABEL_STYLE,
                               layout=Layout(width="330px"))
        self.sensor_w = FloatText(value=SENSOR_PRESETS[DEFAULT_PRESET][0], description="Ширина, мм",
                                  style=LABEL_STYLE, layout=FIELD)
        self.sensor_h = FloatText(value=SENSOR_PRESETS[DEFAULT_PRESET][1], description="Висота, мм",
                                  style=LABEL_STYLE, layout=FIELD)

        f_min = self.focal_from_fov(self.sensor_w.value, self.orig_fov.value)
        f_max = self.focal_from_fov(self.sensor_w.value, self.min_fov)
        self.focal = FloatSlider(value=f_min, min=f_min, max=f_max, step=(f_max - f_min) / 300,
                                 description="f, мм", continuous_update=True, readout_format=".2f",
                                 style=LABEL_STYLE, layout=Layout(width="620px"))
        self.reset_btn = Button(description="Найширший кут", icon="undo", layout=Layout(width="170px"))

    def _wire(self) -> None:
        self.orig_fov.observe(self._rebuild_range, names="value")
        self.sensor_w.observe(self._on_manual_size, names="value")
        self.sensor_h.observe(self._on_manual_size, names="value")
        self.preset.observe(self._on_preset, names="value")
        self.focal.observe(self.render, names="value")
        self.reset_btn.on_click(lambda _: setattr(self.focal, "value", self.focal.min))

    def ui(self) -> VBox:
        return VBox([
            section("ФОТО"),
            HBox([self.uploader, self.orig_fov]),
            section("МАТРИЦЯ"),
            HBox([self.preset, self.sensor_w, self.sensor_h]),
            section("ОБ'ЄКТИВ"),
            HBox([self.focal, self.reset_btn]),
            self.out,
        ])

    def on_photo(self) -> None:
        self.preview, self.preview_scale = to_display(self.photo)
        self.render()

    # ------------------------------------------------------------------ шкала

    def _on_preset(self, change) -> None:
        size = SENSOR_PRESETS.get(change["new"])
        if size is None:
            return
        self._suspend = True
        self.sensor_w.value, self.sensor_h.value = size
        self._suspend = False
        self._rebuild_range()

    def _on_manual_size(self, *_) -> None:
        if self._suspend:
            return
        self.preset.value = CUSTOM
        self._rebuild_range()

    def _rebuild_range(self, *_) -> None:
        """Шкала f покриває FOV від min_fov до FOV оригіналу; поточний кут зберігаємо."""
        if self._suspend:
            return
        if self._params_problem():
            self.render()  # покаже, що саме не так, замість застиглого старого кадру
            return
        sensor_w = self.sensor_w.value
        f_min = self.focal_from_fov(sensor_w, self.orig_fov.value)
        f_max = self.focal_from_fov(sensor_w, self.min_fov)
        keep_fov = self.fov_h or self.orig_fov.value
        self._set_focal_range(f_min, f_max, self.focal_from_fov(sensor_w, keep_fov))
        self.render()

    def _set_focal_range(self, f_min: float, f_max: float, value: float) -> None:
        """Оновлює межі, не ловлячи проміжний стан min > max, і без зайвих перемальовок."""
        self._suspend = True
        try:
            if f_min > self.focal.max:
                self.focal.max, self.focal.min = f_max, f_min
            else:
                self.focal.min, self.focal.max = f_min, f_max
            self.focal.step = (f_max - f_min) / 300
            self.focal.value = float(np.clip(value, f_min, f_max))
        finally:
            self._suspend = False

    def _params_problem(self) -> str | None:
        if self.sensor_w.value <= 0 or self.sensor_h.value <= 0:
            return "Розміри матриці мають бути додатними."
        if not (self.min_fov < self.orig_fov.value < 180):
            return f"FOV оригіналу має бути в межах ({self.min_fov:.0f}°, 180°)."
        return None

    def problem(self) -> str | None:
        return super().problem() or self._params_problem()

    # ----------------------------------------------------------------- рендер

    def draw(self) -> None:
        img_h, img_w = self.photo.shape[:2]
        focal, orig_fov = self.focal.value, self.orig_fov.value

        fov_h = self.fov_from_focal(self.sensor_w.value, focal)
        fov_v = self.fov_from_focal(self.sensor_h.value, focal)
        self.fov_h = fov_h

        x0, y0, box_w, box_h, clipped = self.crop_box(
            img_w, img_h, orig_fov, fov_h, self.sensor_w.value / self.sensor_h.value)
        wanted_h, wanted_v = fov_h, fov_v
        if clipped:
            # рамка вперлась у верх і низ фото й звузилась, тож і кут у кадрі менший за розрахунковий:
            # це той самий кадр, який дала б довша фокусна без обрізання
            focal_seen = self.focal_from_fov(self.sensor_w.value, orig_fov) * img_w / box_w
            fov_h = self.fov_from_focal(self.sensor_w.value, focal_seen)
            fov_v = self.fov_from_focal(self.sensor_h.value, focal_seen)

        left, top = int(round(x0)), int(round(y0))
        crop = self.photo[top:top + max(1, int(round(box_h))), left:left + max(1, int(round(box_w)))]
        if crop.size == 0:
            print("Кроп вийшов порожнім — потрібне більше фото або менший зум.")
            return

        # кадр камери віддаємо в розмірі оригіналу: по ширині фото, висота — з аспекту матриці,
        # інакше картинку розтягнуло б, коли аспект матриці не збігається з аспектом фото
        out_res = (img_w, max(1, round(img_w / (self.sensor_w.value / self.sensor_h.value))))
        upscaled = out_res[0] > crop.shape[1]
        frame = cv2.resize(crop, out_res,
                           interpolation=cv2.INTER_CUBIC if upscaled else cv2.INTER_AREA)
        frame_preview, _ = to_display(frame)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), facecolor=BG)
        dark_axes(*axes, hide=True)

        preview_h, preview_w = self.preview.shape[:2]
        scale = self.preview_scale
        axes[0].imshow(self.preview)
        draw_viewfinder(axes[0], (x0 * scale, y0 * scale, box_w * scale, box_h * scale),
                        preview_w, preview_h, f"{fov_h:.1f}° × {fov_v:.1f}°")
        axes[0].set_title(f"FOV {orig_fov:.0f}° · Resolution {img_w}×{img_h} px", color=MUTED, fontsize=11)

        axes[1].imshow(frame_preview)
        source_note = f"\n · Crop {crop.shape[1]} × {crop.shape[0]} px" if upscaled else ""
        axes[1].set_title(f"FOV = {fov_h:.1f}° × {fov_v:.1f}° · Zoom: {self.zoom_factor(orig_fov, fov_h):.1f} · "
                          f"Resolution: {frame.shape[1]}×{frame.shape[0]} px{source_note}",
                          color=MUTED, fontsize=11)

        fig.suptitle(f"f = {focal:.2f} мм      FOV = {wanted_h:.1f}° × {wanted_v:.1f}°",
                     color=FG, fontsize=14, fontweight="bold")
        if clipped:
            fig.text(0.5, 0.015,
                     f"Матриця «вища» за фото: над і під центром фото бракує, тож рамка звужена "
                     f"й показує лише {fov_h:.1f}° × {fov_v:.1f}° замість {wanted_h:.1f}° × {wanted_v:.1f}°",
                     color=WARN, ha="center", fontsize=10)

        fig.tight_layout(rect=(0, 0.04, 1, 0.92))
        plt.show()
        plt.close(fig)


# ────────────────────────────────── demo1: цифровий зум проти теле-сенсора


class ZoomPanel(PhotoPanel):
    """Wide-сенсор із цифровим зумом проти пари wide + теле. Підрахунки — ззовні."""

    def __init__(self, sensor_capture, digital_zoom, hybrid_frame, crop_box,
                 default_photo: Path | None = None, photo_fov_deg: float = 70.0):
        super().__init__(default_photo, photo_fov_deg)
        self.sensor_capture = sensor_capture
        self.digital_zoom = digital_zoom
        self.hybrid_frame = hybrid_frame
        self.crop_box = crop_box

        self.wide_frame = None
        self.tele_frame = None

        self._build()
        self._wire()

    def _build(self) -> None:
        wide_field = Layout(width="320px")
        self.orig_fov = FloatText(value=self.photo_fov_deg, description="FOV оригіналу, °",
                                  style=LABEL_STYLE, layout=FIELD)
        # обидві камери з однаковою матрицею: так єдина різниця між половинами — об'єктив
        self.sensor_res = Dropdown(options=list(SENSOR_RESOLUTIONS), value=DEFAULT_RESOLUTION,
                                   description="Роздільність сенсорів", style=LABEL_STYLE,
                                   layout=wide_field)
        self.tele_fov = FloatText(value=40.0, description="FOV теле-лінзи, °",
                                  style=LABEL_STYLE, layout=FIELD)
        self.zoom = FloatSlider(value=1.0, min=1.0, max=6.0, step=0.05, description="Зум ×",
                                continuous_update=True, readout_format=".2f",
                                style=LABEL_STYLE, layout=Layout(width="620px"))

    def _wire(self) -> None:
        for widget in (self.orig_fov, self.tele_fov, self.sensor_res):
            widget.observe(self._on_param, names="value")
        self.zoom.observe(self.render, names="value")

    def ui(self) -> VBox:
        return VBox([
            section("ФОТО"),
            HBox([self.uploader, self.orig_fov]),
            section("КАМЕРИ"),
            HBox([self.sensor_res, self.tele_fov]),
            section("ЗУМ"),
            self.zoom,
            self.out,
        ])

    def on_photo(self) -> None:
        self._on_param()

    # ------------------------------------------------------ шкала та захоплення

    def _on_param(self, *_) -> None:
        if self._suspend:
            return
        self._rebuild_zoom_range()
        self._recompute_captures()
        self.render()

    def _crop_widths(self) -> tuple[float, float, bool]:
        """Ширина wide- і теле-кропу в пікселях фото + чи вперся wide-кроп у верх і низ фото."""
        img_h, img_w = self.photo.shape[:2]
        res = SENSOR_RESOLUTIONS[self.sensor_res.value]
        aspect, orig_fov = res[0] / res[1], self.orig_fov.value
        _, _, wide_w, _, clipped = self.crop_box(img_w, img_h, orig_fov, orig_fov, aspect)
        _, _, tele_w, _, _ = self.crop_box(img_w, img_h, orig_fov, self.tele_fov.value, aspect)
        return wide_w, tele_w, clipped

    def _switch_point(self) -> float | None:
        """Кратність, на якій кадр теле-камери зрівнюється з цифровим зумом wide-камери.

        Без обрізання це zoom_factor(FOV оригіналу, FOV теле). Але якщо матриця «вища» за фото,
        wide-кроп звужується, і теле стає ближчим до нього — тож рахуємо з реальних кропів.
        """
        if self.photo is None or self._params_problem():
            return None
        wide_w, tele_w, _ = self._crop_widths()
        return wide_w / tele_w

    def _rebuild_zoom_range(self) -> None:
        switch = self._switch_point()
        if switch is None:
            return
        self._suspend = True
        try:
            self.zoom.max = max(2.0, switch * 6.0)  # щоб точка перемикання завжди була в межах шкали
            self.zoom.value = float(np.clip(self.zoom.value, self.zoom.min, self.zoom.max))
        finally:
            self._suspend = False

    def _recompute_captures(self) -> None:
        """Кадри сенсорів залежать лише від фото й параметрів, тож рахуємо їх не на кожен рух повзунка."""
        if self.photo is None or self._params_problem():
            self.wide_frame = self.tele_frame = None
            return
        orig_fov, res = self.orig_fov.value, SENSOR_RESOLUTIONS[self.sensor_res.value]
        self.wide_frame = self.sensor_capture(self.photo, orig_fov, orig_fov, res)
        self.tele_frame = self.sensor_capture(self.photo, orig_fov, self.tele_fov.value, res)

    def _params_problem(self) -> str | None:
        if not (0 < self.orig_fov.value < 180):
            return "FOV оригіналу має бути в межах (0°, 180°)."
        if not (0 < self.tele_fov.value < self.orig_fov.value):
            return "FOV теле-лінзи має бути додатним і вужчим за FOV оригіналу."
        return None

    def problem(self) -> str | None:
        return super().problem() or self._params_problem()

    # ----------------------------------------------------------------- рендер

    def draw(self) -> None:
        zoom, switch = self.zoom.value, self._switch_point()
        res = SENSOR_RESOLUTIONS[self.sensor_res.value]

        left = self.digital_zoom(self.wide_frame, zoom, res)
        right, switched = self.hybrid_frame(self.wide_frame, self.tele_frame, zoom, switch, res)

        fig, axes = plt.subplot_mosaic([["wide", "wide"], ["digital", "hybrid"]],
                                       figsize=(13, 10), facecolor=BG,
                                       gridspec_kw={"height_ratios": [1.1, 1], "hspace": 0.16})
        dark_axes(*axes.values(), hide=True)

        preview, _ = to_display(self.wide_frame)
        pre_h, pre_w = preview.shape[:2]
        axes["wide"].imshow(preview)
        # матриця в обох камер однакова, тож кадр теле — просто зменшена копія wide-кадру
        centered_rect(axes["wide"], pre_w, pre_h, pre_w / switch, pre_h / switch,
                      WARN, True, f"теле-лінза {self.tele_fov.value:.0f}° (×{switch:.2f})",
                      label_below=True)
        centered_rect(axes["wide"], pre_w, pre_h, pre_w / zoom, pre_h / zoom,
                      ACCENT, False, f"поточний зум ×{zoom:.2f}")
        clipped_note = " · звужено: фото нижче за матрицю" if self._crop_widths()[2] else ""
        axes["wide"].set_title(f"Кадр wide-сенсора · {self.wide_frame.shape[1]}×{self.wide_frame.shape[0]} px "
                               f"· FOV {self.orig_fov.value:.0f}°{clipped_note}", color=MUTED, fontsize=11)

        axes["digital"].imshow(to_display(left)[0])
        axes["digital"].set_title(f"Тільки цифровий зум · {left.shape[1]}×{left.shape[0]} px",
                                  color=MUTED, fontsize=11)

        axes["hybrid"].imshow(to_display(right)[0])
        if switched:
            axes["hybrid"].set_title(f"ТЕЛЕ-СЕНСОР · {right.shape[1]}×{right.shape[0]} px",
                                     color=ACCENT, fontsize=11, fontweight="bold")
            axes["hybrid"].add_patch(patches.Rectangle((0, 0), 1, 1, transform=axes["hybrid"].transAxes,
                                                       fill=False, edgecolor=ACCENT, linewidth=3))
        else:
            axes["hybrid"].set_title(f"Wide + цифровий зум · {right.shape[1]}×{right.shape[0]} px",
                                     color=MUTED, fontsize=11)

        fig.suptitle(f"Зум ×{zoom:.2f}", color=FG, fontsize=15, fontweight="bold")
        self._draw_status(fig, zoom, switch, switched, res)

        # не tight_layout: він перерахував би відступи й з'їв заданий hspace між рядами
        fig.subplots_adjust(left=0.03, right=0.97, top=0.90, bottom=0.10, hspace=0.18)
        plt.show()
        plt.close(fig)

    def _draw_status(self, fig, zoom: float, switch: float, switched: bool, res) -> None:
        if switched:
            status = (f"Теле-сенсор увімкнувся на ×{switch:.2f} — далі йде цифровий зум "
                      f"поверх нього (×{zoom / switch:.2f})")
        else:
            status = f"Поки що обидві половини однакові: теле-сенсор увімкнеться на ×{switch:.2f}"
        fig.text(0.5, 0.035, status, color=ACCENT if switched else MUTED, ha="center", fontsize=11,
                 fontweight="bold" if switched else "normal")

        source_px = self._crop_widths()[1]  # скільки пікселів оригіналу лягає в теле-кадр
        if source_px < res[0]:
            fig.text(0.5, 0.008,
                     f"Фото замало: теле-кадр береться з {source_px:.0f} px і розтягується до "
                     f"{res[0]} px — перевага теле тут частково домальована",
                     color=WARN, ha="center", fontsize=10)


# ──────────────────────────────────── demo2: калькулятор дальності детекції


class RangePanel(Panel):
    """Дальність детекції з оптичного рівняння проєкції. Підрахунки — ззовні."""

    COLUMN, GAP = 300, 16  # px; сітка з трьох однакових стовпчиків
    LABELS = {"description_width": "140px"}

    def __init__(self, pixels_on_target, detection_range, focal_for_range, resize_factor,
                 field_of_view, effective_pixel):
        super().__init__()
        self.pixels_on_target = pixels_on_target
        self.detection_range = detection_range
        self.focal_for_range = focal_for_range
        self.resize_factor = resize_factor
        self.field_of_view = field_of_view
        self.effective_pixel = effective_pixel

        self._build()
        self._wire()

    def _build(self) -> None:
        # усі поля в одній сітці 3 × COLUMN з однаковою шириною підписів — тож стовпчики рівні
        cell, style = Layout(width=f"{self.COLUMN}px"), self.LABELS
        self.target = FloatText(value=2.5, description="Розмір цілі H, м", style=style, layout=cell)
        self.sensor = Dropdown(options=list(SENSORS), value=DEFAULT_SENSOR,
                               description="Сенсор", style=style, layout=cell)
        self.pixel = FloatText(value=SENSORS[DEFAULT_SENSOR]["pixel_um"], description="Піксель p, мкм",
                               style=style, layout=cell)
        self.mode = Dropdown(options=self._mode_options(DEFAULT_SENSOR), value=0,
                             description="Режим", style=style, layout=cell)
        self.mode_info = HTML(layout=Layout(width=f"{2 * self.COLUMN + self.GAP}px"))
        self.model_input = Dropdown(options=list(MODEL_INPUTS.items()), value=640,
                                    description="Вхід моделі", style=style, layout=cell)
        self.min_px = Dropdown(options=list(THRESHOLDS.items()), value=16.0,
                               description="Поріг N, px", style=style, layout=cell)
        # indent=True відсуває прапорець на ширину підпису — він стає в один стовпчик із полями
        self.tiled = Checkbox(value=False, description="Tiled inference (k = 1)", indent=True,
                              style=style, layout=cell)
        self.focal = FloatSlider(value=25.0, min=0.5, max=200.0, step=0.5, description="Фокусна f, мм",
                                 continuous_update=True, readout_format=".1f", style=style,
                                 layout=Layout(width=f"{2 * self.COLUMN + self.GAP}px"))
        self.fov_label = HTML(description="Кут огляду", style=style,
                              layout=Layout(width=f"{self.COLUMN}px", overflow="visible"))
        self.want_range = FloatText(value=500.0, description="Потрібна дальність, м",
                                    style=style, layout=cell)

    @staticmethod
    def _mode_options(sensor: str) -> list[tuple[str, int]]:
        """Підписи режимів; кроп позначаємо одразу — він звужує FOV. fps — у рядку під списком."""
        size = SENSORS[sensor]["size"]
        return [(f"{w}×{h}" + (" · кроп" if readout != size else ""), i)
                for i, ((w, h), readout, _) in enumerate(SENSORS[sensor]["modes"])]

    def _wire(self) -> None:
        self.sensor.observe(self._on_sensor, names="value")
        for widget in (self.target, self.pixel, self.mode, self.model_input,
                       self.tiled, self.min_px, self.focal, self.want_range):
            widget.observe(self.render, names="value")

    def _on_sensor(self, change) -> None:
        """Новий сенсор — його піксель і список режимів; малюємо один раз, а не на кожну зміну."""
        self._suspend = True
        try:
            self.pixel.value = SENSORS[change["new"]]["pixel_um"]
            self.mode.options = self._mode_options(change["new"])
            self.mode.value = 0
        finally:
            self._suspend = False
        self.render()

    def _row(self, *widgets, spans: tuple[int, ...] = ()) -> GridBox:
        """Рядок сітки. spans — скільки стовпчиків займає кожен віджет (за замовчуванням по одному).

        Ширину колонок задаємо самим шаблоном, а не grid_column: статичний рендер віджетів його ігнорує.
        """
        spans = spans or (1,) * 3
        columns = " ".join(f"{n * self.COLUMN + (n - 1) * self.GAP}px" for n in spans)
        return GridBox(list(widgets), layout=Layout(grid_template_columns=columns,
                                                    grid_gap=f"4px {self.GAP}px", align_items="center"))

    def ui(self) -> VBox:
        return VBox([
            section("ЦІЛЬ І СЕНСОР"),
            self._row(self.target, self.sensor, self.pixel),
            self._row(self.mode, self.mode_info, spans=(1, 2)),
            section("ПАЙПЛАЙН"),
            self._row(self.model_input, self.min_px, self.tiled),
            section("ОПТИКА"),
            self._row(self.focal, self.fov_label, spans=(2, 1)),
            section("ЗВОРОТНА ЗАДАЧА"),
            self._row(self.want_range),
            self.out,
        ])

    # ------------------------------------------------------------------ логіка

    def _mode(self) -> tuple[tuple[int, int], tuple[int, int], int | None]:
        return SENSORS[self.sensor.value]["modes"][self.mode.value]

    def _pixel_eff(self) -> float:
        """Крок пікселя на виході режиму: з біннінгом 2×2 він удвічі більший за фізичний."""
        (out_w, _), (read_w, _), _ = self._mode()
        return self.effective_pixel(self.pixel.value, read_w, out_w)

    def _resize_k(self) -> float:
        if self.tiled.value:
            return 1.0
        return self.resize_factor(self._mode()[0][0], self.model_input.value)

    def problem(self) -> str | None:
        if self.target.value <= 0 or self.pixel.value <= 0:
            return "Розмір цілі та крок пікселя мають бути додатними."
        if self.want_range.value <= 0:
            return "Потрібна дальність має бути додатною."
        return None

    def _fov(self, focal: float) -> tuple[float, float, float]:
        """Кут огляду по горизонталі, вертикалі й діагоналі — з області зчитування, а не з виходу.

        Біннінг FOV не змінює, кроп — звужує.
        """
        _, (read_w, read_h), _ = self._mode()
        return tuple(self.field_of_view(side, self.pixel.value, focal)
                     for side in (read_w, read_h, np.hypot(read_w, read_h)))

    def _mode_note(self) -> str:
        (out_w, _), (read_w, read_h), fps = self._mode()
        size_w, size_h = SENSORS[self.sensor.value]["size"]
        binning = read_w / out_w
        parts = [f"біннінг {binning:.0f}×{binning:.0f}" if binning > 1 else "без біннінгу",
                 f"кроп {read_w}×{read_h}" if (read_w, read_h) != (size_w, size_h) else "уся матриця",
                 f"піксель на виході {self._pixel_eff():.2f} мкм"]
        if fps:
            parts.append(f"{fps} fps")
        return f"<span style='color:{MUTED}'>{' · '.join(parts)}</span>"

    def _levers(self, k: float) -> list[tuple[str, float, float]]:
        """Поточна конфігурація і два важелі зі слайда: tiled прибирає resize, довша лінза множить f.

        Третє значення — горизонтальний FOV: tiled його не чіпає, а f × 2 звужує майже вдвічі.
        """
        target, pixel, n, f = self.target.value, self._pixel_eff(), self.min_px.value, self.focal.value
        return [
            ("Поточна", self.detection_range(target, f, pixel, k, n), self._fov(f)[0]),
            ("+ tiled (k = 1)", self.detection_range(target, f, pixel, 1.0, n), self._fov(f)[0]),
            ("+ tiled і f × 2", self.detection_range(target, f * 2, pixel, 1.0, n), self._fov(f * 2)[0]),
        ]

    # ----------------------------------------------------------------- рендер

    def draw(self) -> None:
        h, p, n = self.target.value, self._pixel_eff(), self.min_px.value
        f, k = self.focal.value, self._resize_k()
        d_max = self.detection_range(h, f, p, k, n)
        fov_h, fov_v, fov_d = self._fov(f)
        self.fov_label.value = f"<b>{fov_h:.1f}° × {fov_v:.1f}°</b> · діаг. {fov_d:.1f}°"
        self.mode_info.value = self._mode_note()

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), facecolor=BG,
                                 gridspec_kw={"width_ratios": [1.25, 1]})
        dark_axes(*axes)

        self._draw_curve(axes[0], h, f, p, k, n, d_max)
        self._draw_levers(axes[1], k, d_max)

        fig.suptitle(f"D max = {d_max:,.0f} м".replace(",", " "),
                     color=ACCENT, fontsize=18, fontweight="bold")
        # скільки метрів кадр накриває на дистанції D: 2·D·tan(FOV/2)
        swath_w, swath_h = (2 * d_max * np.tan(np.radians(fov / 2)) for fov in (fov_h, fov_v))
        fig.text(0.5, 0.905, f"FOV {fov_h:.1f}° × {fov_v:.1f}° (діаг. {fov_d:.1f}°) · на D max кадр охоплює "
                             f"{swath_w:,.0f} × {swath_h:,.0f} м".replace(",", " "),
                 color=MUTED, ha="center", fontsize=11)
        self._draw_status(fig, h, p, k, n, f)

        fig.subplots_adjust(left=0.08, right=0.97, top=0.82, bottom=0.22, wspace=0.25)
        plt.show()
        plt.close(fig)

    def _draw_curve(self, ax, h: float, f: float, p: float, k: float, n: float, d_max: float) -> None:
        distances = np.linspace(d_max * 0.08, d_max * 2.5, 400)
        pixels = [self.pixels_on_target(h, f, p, k, d) for d in distances]

        ax.plot(distances, pixels, color=ACCENT, linewidth=2)
        ax.axhline(n, color=WARN, linestyle="--", linewidth=1.4)
        ax.axvline(d_max, color=MUTED, linestyle=":", linewidth=1.2)
        ax.plot([d_max], [n], "o", color=WARN, markersize=7)

        ax.text(distances[0], n, f" поріг {n:.0f} px", color=WARN, fontsize=9, va="bottom", ha="left")
        ax.set_ylim(0, n * 4)
        ax.set_xlim(0, distances[-1])
        ax.set_xlabel("Дистанція, м", color=MUTED, fontsize=10)
        ax.set_ylabel("Пікселів на цілі (вхід моделі)", color=MUTED, fontsize=10)
        ax.set_title("Скільки пікселів лишається від цілі", color=MUTED, fontsize=11)
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)

    def _draw_levers(self, ax, k: float, d_max: float) -> None:
        levers = self._levers(k)
        labels = [name for name, _, _ in levers]
        values = [value for _, value, _ in levers]
        fovs = [fov for _, _, fov in levers]
        colors = [ACCENT if i == 0 else "#3b7f5e" for i in range(len(levers))]

        bars = ax.barh(labels, values, color=colors, height=0.55)
        ax.invert_yaxis()
        ax.set_xlim(0, max(values) * 1.5)  # праворуч від стовпчиків — дальність, виграш і FOV
        ax.set_xlabel("Дальність, м", color=MUTED, fontsize=10)
        ax.set_title("Що дають важелі", color=MUTED, fontsize=11)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for label in ax.get_yticklabels():
            label.set_color(FG)

        for bar, value, fov in zip(bars, values, fovs):
            gain = f"  ×{value / d_max:.1f}" if value > d_max * 1.01 else ""
            x, y = bar.get_width() * 1.02, bar.get_y() + bar.get_height() / 2
            ax.text(x, y, f"{value:,.0f} м".replace(",", " ") + gain, color=FG, va="bottom", fontsize=10)
            ax.text(x, y, f"FOV {fov:.1f}°", color=MUTED, va="top", fontsize=9)

    def _draw_status(self, fig, h: float, p: float, k: float, n: float, f: float) -> None:
        if self.tiled.value:
            k_note = "k = 1.00 — tiled inference прибирає стиснення кадру"
        else:
            k_note = (f"k = {k:.2f} — вихід сенсора {self._mode()[0][0]} px тиснеться до входу "
                      f"{self.model_input.value} px")
        fig.text(0.5, 0.10, k_note, color=MUTED, ha="center", fontsize=10)

        need_f = self.focal_for_range(h, p, k, n, self.want_range.value)
        fig.text(0.5, 0.045,
                 f"Щоб дістати {self.want_range.value:,.0f} м".replace(",", " ") +
                 f", потрібна фокусна {need_f:.1f} мм — це ×{need_f / f:.2f} до поточних {f:.1f} мм",
                 color=FG, ha="center", fontsize=11, fontweight="bold")

        if k < 1:
            fig.text(0.5, 0.005,
                     "k < 1: вхід моделі більший за кадр. N росте на інтерполяції, "
                     "але деталі не додається — поріг у px тут уже не працює",
                     color=WARN, ha="center", fontsize=9)


# ─────────────────────────────────────────── demo2: поріг детекції на практиці

TARGET_SIZES_PX = (64, 48, 40, 32, 24, 20, 16, 14, 12, 10, 8)  # розміри цілі, від близької до далекої
STRIP_PX = (48, 32, 24, 16, 10)
DETECTED_CONF = 0.25   # нижче цього вважаємо, що ціль втрачено
THUMB_WINDOW = 160     # скільки пікселів кадру показуємо у смужці прикладів
MIN_BOX_PX = 8         # менший ручний бокс не має сенсу
PLACEHOLDER_MAX = 10000
PREVIEW_W = 900        # ширина інтерактивного прев'ю в пікселях
DRAG_MIN_PX = 12       # коротший рух вважаємо кліком, а не малюванням

FROM_DETECTION, FROM_BOX = "З детекції", "Свій бокс"


class ThresholdPanel(PhotoPanel):
    """Поріг детекції: впевненість детектора залежно від розміру цілі в пікселях. Підрахунки — ззовні.

    Ціль обирається мишею просто на фото (клік по рамці або протягування нової).
    Якщо ipyevents не встановлено, лишаються слайдери — демо працює в будь-якому разі.
    """

    def __init__(self, detect, scene_at_size, matched_confidence, detection_range, canvas: int,
                 default_photo: Path | None = None):
        super().__init__(default_photo)
        self.detect = detect
        self.scene_at_size = scene_at_size
        self.matched_confidence = matched_confidence
        self.detection_range = detection_range
        self.canvas = canvas

        self.candidates = []
        self.results = []
        self.target_name = ""
        self._base_preview = None
        self._preview_scale = 1.0
        self._drag_start = None

        self._build()
        self._wire()

    def _build(self) -> None:
        slider = Layout(width="330px")
        self.mode = ToggleButtons(options=[FROM_DETECTION, FROM_BOX], value=FROM_DETECTION,
                                  layout=Layout(width="330px"))
        self.target_dd = Dropdown(options=[], description="Ціль", style=LABEL_STYLE,
                                  layout=Layout(width="430px"))
        self.run_btn = Button(description="Виміряти поріг", icon="play", button_style="success",
                              layout=Layout(width="200px"))
        self.preview_img = Image(format="jpeg", layout=Layout(width=f"{PREVIEW_W}px",
                                                              border=f"1px solid {MUTED}"))
        self.caption = HTML()

        # справжні межі підставить _reset_box() з розмірів фото; тут лише валідні заглушки
        self.box_x = IntSlider(value=0, min=0, max=PLACEHOLDER_MAX, description="Центр X",
                               style=LABEL_STYLE, continuous_update=True, layout=slider)
        self.box_y = IntSlider(value=0, min=0, max=PLACEHOLDER_MAX, description="Центр Y",
                               style=LABEL_STYLE, continuous_update=True, layout=slider)
        self.box_w = IntSlider(value=MIN_BOX_PX, min=MIN_BOX_PX, max=PLACEHOLDER_MAX,
                               description="Ширина", style=LABEL_STYLE, continuous_update=True,
                               layout=slider)
        self.box_h = IntSlider(value=MIN_BOX_PX, min=MIN_BOX_PX, max=PLACEHOLDER_MAX,
                               description="Висота", style=LABEL_STYLE, continuous_update=True,
                               layout=slider)
        # рядок збирається тут, а не в ui(): на нього дивляться обробники,
        # тож панель має бути справною ще до побудови розкладки
        self.box_row = HBox([self.box_x, self.box_y, self.box_w, self.box_h],
                            layout=Layout(flex_flow="row wrap", display="none"))

        self.target_m = FloatText(value=1.8, description="Розмір цілі H, м",
                                  style=LABEL_STYLE, layout=FIELD)
        self.focal = FloatText(value=25.0, description="Фокусна f, мм", style=LABEL_STYLE, layout=FIELD)
        self.pixel = FloatText(value=2.9, description="Піксель p, мкм", style=LABEL_STYLE, layout=FIELD)
        self.resize_k = FloatText(value=3.0, description="Стиснення k", style=LABEL_STYLE, layout=FIELD)

    def _wire(self) -> None:
        self.mode.observe(self._on_target_change, names="value")
        self.target_dd.observe(self._on_target_change, names="value")
        for slider in (self.box_x, self.box_y, self.box_w, self.box_h):
            slider.observe(self._on_target_change, names="value")
        self.run_btn.on_click(self._on_run)
        for widget in (self.target_m, self.focal, self.pixel, self.resize_k):
            widget.observe(self._on_optics, names="value")

        if MouseEvent is not None:
            self.mouse = MouseEvent(source=self.preview_img, prevent_default_action=True,
                                    watched_events=["mousedown", "mousemove", "mouseup", "mouseleave"],
                                    throttle_or_debounce="throttle", wait=40)
            self.mouse.on_dom_event(self._on_mouse)

    def ui(self) -> VBox:
        hint = ("Клік по рамці — обрати ціль, протягування — намалювати свою."
                if MouseEvent is not None else
                "Миша недоступна: постав <code>ipyevents</code>, щоб обирати ціль прямо на фото. "
                "Поки що користуйся слайдерами.")
        return VBox([
            section("ФОТО"),
            HBox([self.uploader, self.run_btn]),
            section("ЦІЛЬ"),
            HTML(f"<span style='color:{MUTED};font-size:12px'>{hint}</span>"),
            HBox([self.mode, self.target_dd]),
            self.preview_img,
            self.caption,
            self.box_row,
            section("ПЕРЕВЕДЕННЯ ПІКСЕЛІВ У МЕТРИ"),
            HBox([self.target_m, self.focal, self.pixel, self.resize_k]),
            self.out,
        ])

    def on_photo(self) -> None:
        self.results = []
        self._find_candidates()
        self._build_base_preview()
        self._reset_box()
        self._on_target_change()
        self.render()  # прибирає графік попереднього фото й показує підказку

    def _optics_problem(self) -> str | None:
        fields = {"Розмір цілі H": self.target_m, "Фокусна f": self.focal,
                  "Піксель p": self.pixel, "Стиснення k": self.resize_k}
        wrong = [name for name, field in fields.items() if field.value <= 0]
        return f"Мають бути додатними: {', '.join(wrong)}." if wrong else None

    def problem(self) -> str | None:
        base = super().problem()
        if base:
            return base
        if not self.results:
            return "Обери ціль і тисни «Виміряти поріг»."
        return self._optics_problem()

    # --------------------------------------------------------------- обробники

    def _find_candidates(self) -> None:
        """Опорний прогін на повному кадрі: з нього беремо ціль та її розмір при масштабі 1."""
        detections = self.detect(self.photo, imgsz=1280)
        self.candidates = sorted(detections, key=lambda d: -(d[2][3] - d[2][1]))
        self.target_dd.options = [
            (f"#{i + 1} · {cls} · conf {conf:.2f} · {box[3] - box[1]:.0f} px", i)
            for i, (cls, conf, box) in enumerate(self.candidates)
        ]
        # режим міняємо без обробників: прев'ю нового фото ще не зібране
        self._suspend = True
        try:
            if self.candidates:
                self.target_dd.value = 0
            self.mode.value = FROM_DETECTION if self.candidates else FROM_BOX
        finally:
            self._suspend = False

    def _reset_box(self) -> None:
        """Ручний бокс за замовчуванням — найбільша детекція, інакше чверть кадру по центру."""
        img_h, img_w = self.photo.shape[:2]
        if self.candidates:
            x1, y1, x2, y2 = self.candidates[0][2]
            cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
        else:
            cx, cy, w, h = img_w / 2, img_h / 2, img_w / 4, img_h / 4
        self._set_box(cx, cy, w, h)

    def _set_box(self, cx: float, cy: float, w: float, h: float) -> None:
        """Ставить усі чотири слайдери за один раз, без чотирьох перемальовок."""
        img_h, img_w = self.photo.shape[:2]
        self._suspend = True
        try:
            for slider, maximum, value in ((self.box_x, img_w, cx), (self.box_y, img_h, cy),
                                           (self.box_w, img_w, w), (self.box_h, img_h, h)):
                slider.max = max(slider.min + 1, int(maximum))
                slider.value = int(min(max(value, slider.min), slider.max))
        finally:
            self._suspend = False

    def _on_target_change(self, *_) -> None:
        if self._suspend:
            return
        self.box_row.layout.display = "flex" if self.mode.value == FROM_BOX else "none"
        self.target_dd.disabled = self.mode.value == FROM_BOX
        if self.photo is not None:
            self._render_preview()

    def _on_optics(self, *_) -> None:
        if self.results:
            self.render()

    def _on_run(self, *_) -> None:
        self.out.clear_output(wait=True)
        measured = False
        with self.out:
            if self.photo is None:
                print("Спочатку обери фото.")
                return
            if self.mode.value == FROM_DETECTION and not self.candidates:
                print("Детектор нічого не знайшов — намалюй ціль мишею або слайдерами.")
                return
            self.results = []
            self._measure()
            measured = True
        # якщо _measure упав, Output уже показав traceback — render() його лише затер би
        if measured:
            self.render()

    # -------------------------------------------------------------------- миша

    def _event_to_photo(self, event) -> tuple[float, float] | None:
        """Координати події у пікселях оригіналу. Беремо розмір рамки з самої події,
        щоб не залежати від того, як браузер відмасштабував картинку."""
        rect_w, rect_h = event.get("boundingRectWidth"), event.get("boundingRectHeight")
        if not rect_w or not rect_h or self.photo is None:
            return None
        img_h, img_w = self.photo.shape[:2]
        x = event.get("relativeX", 0) / rect_w * img_w
        y = event.get("relativeY", 0) / rect_h * img_h
        return min(max(x, 0), img_w), min(max(y, 0), img_h)

    def _on_mouse(self, event) -> None:
        point = self._event_to_photo(event)
        if point is None:
            return
        kind = event.get("type")

        if kind == "mousedown":
            self._drag_start = point
        elif kind == "mousemove" and self._drag_start is not None:
            start = self._drag_start
            if abs(point[0] - start[0]) >= DRAG_MIN_PX or abs(point[1] - start[1]) >= DRAG_MIN_PX:
                self._apply_drag(start, point)
        elif kind in ("mouseup", "mouseleave") and self._drag_start is not None:
            start, self._drag_start = self._drag_start, None
            if kind == "mouseleave":
                return
            if abs(point[0] - start[0]) < DRAG_MIN_PX and abs(point[1] - start[1]) < DRAG_MIN_PX:
                self._select_at(point)
            else:
                self._apply_drag(start, point)

    def _apply_drag(self, start, end) -> None:
        x1, x2 = sorted((start[0], end[0]))
        y1, y2 = sorted((start[1], end[1]))
        self._set_box((x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, MIN_BOX_PX), max(y2 - y1, MIN_BOX_PX))
        if self.mode.value != FROM_BOX:
            self.mode.value = FROM_BOX  # само перемалює
        else:
            self._on_target_change()

    def _select_at(self, point) -> None:
        """Клік: беремо найменшу рамку, що містить точку — так вкладені об'єкти теж доступні."""
        x, y = point
        hits = [i for i, (_, _, (x1, y1, x2, y2)) in enumerate(self.candidates)
                if x1 <= x <= x2 and y1 <= y <= y2]
        if not hits:
            return
        smallest = min(hits, key=lambda i: ((self.candidates[i][2][2] - self.candidates[i][2][0]) *
                                            (self.candidates[i][2][3] - self.candidates[i][2][1])))
        self._suspend = True
        try:
            self.mode.value = FROM_DETECTION
            self.target_dd.value = smallest
        finally:
            self._suspend = False
        self._on_target_change()

    # ----------------------------------------------------------------- прев'ю

    def _build_base_preview(self) -> None:
        """Фото з рамками кандидатів — малюємо один раз, далі лише накладаємо ціль."""
        img_h, img_w = self.photo.shape[:2]
        scale = min(1.0, PREVIEW_W / img_w)
        size = (max(1, int(img_w * scale)), max(1, int(img_h * scale)))
        base = cv2.resize(self.photo, size, interpolation=cv2.INTER_AREA)

        for i, (cls, conf, box) in enumerate(self.candidates):
            x1, y1, x2, y2 = (int(v * scale) for v in box)
            cv2.rectangle(base, (x1, y1), (x2, y2), MUTED_BGR, 1)
            # cv2 не вміє кирилицю, тож на картинці лишаємо тільки ASCII-підписи
            cv2.putText(base, f"#{i + 1} {cls}", (x1 + 3, max(12, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, MUTED_BGR, 1, cv2.LINE_AA)

        self._base_preview, self._preview_scale = base, scale

    def _target_box(self) -> tuple[str | None, tuple[float, float, float, float]]:
        """Клас і рамка цілі. Для ручного боксу класу немає: приймаємо будь-яку детекцію."""
        if self.mode.value == FROM_DETECTION and self.candidates:
            cls, _, box = self.candidates[self.target_dd.value]
            return cls, box
        half_w, half_h = self.box_w.value / 2, self.box_h.value / 2
        return None, (self.box_x.value - half_w, self.box_y.value - half_h,
                      self.box_x.value + half_w, self.box_y.value + half_h)

    def _render_preview(self) -> None:
        if self._base_preview is None:
            return
        frame = self._base_preview.copy()
        target_cls, box = self._target_box()
        scale = self._preview_scale
        x1, y1, x2, y2 = (int(v * scale) for v in box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), ACCENT_BGR, 2)

        ok, buffer = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                                  [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            self.preview_img.value = buffer.tobytes()

        height_px = box[3] - box[1]
        selected = self.target_dd.value if self.mode.value == FROM_DETECTION and self.candidates else None
        name = f"#{selected + 1} {target_cls}" if selected is not None else "свій бокс"
        note = ""
        if not self.candidates:
            note = f" · <span style='color:{WARN}'>детектор нічого не знайшов — намалюй ціль</span>"
        elif height_px < max(TARGET_SIZES_PX):
            note = (f" · <span style='color:{WARN}'>ціль лише {height_px:.0f} px: більші розміри "
                    "будуть домальовані інтерполяцією</span>")
        self.caption.value = (
            f"<span style='color:{MUTED};font-size:12px'>Ціль: <b style='color:{ACCENT}'>{name}</b>"
            f" · {height_px:.0f} px заввишки · знайдено об'єктів: {len(self.candidates)}{note}</span>")

    # ------------------------------------------------------------ вимір порогу

    def _measure(self) -> None:
        target_cls, (x1, y1, x2, y2) = self._target_box()
        object_px = y2 - y1
        center = ((x1 + x2) / 2, (y1 + y2) / 2)

        results = []
        for size_px in TARGET_SIZES_PX:
            frame = self.scene_at_size(self.photo, object_px, size_px, center, self.canvas)
            conf = self.matched_confidence(self.detect(frame), target_cls, size_px, self.canvas)
            results.append((size_px, conf, frame))
        self.results = results  # лише повний набір: з половиною графік не намалюється
        self.target_name = target_cls or "свій бокс"

    def _loss_band(self) -> tuple[float | None, float | None]:
        """Між якими розмірами модель втрачає ціль: останній впевнений і перший втрачений."""
        last_ok = None
        for size_px, conf, _ in self.results:  # TARGET_SIZES_PX іде від великих до малих
            if conf >= DETECTED_CONF:
                last_ok = size_px
            elif last_ok is not None:
                return last_ok, size_px
        return last_ok, None

    # ----------------------------------------------------------------- рендер

    def draw(self) -> None:
        mosaic = [[f"t{i}" for i in range(len(STRIP_PX))], ["curve"] * len(STRIP_PX)]
        fig, axes = plt.subplot_mosaic(mosaic, figsize=(13, 8.5), facecolor=BG,
                                       gridspec_kw={"height_ratios": [1, 1.5], "hspace": 0.32})

        by_size = {size: (conf, frame) for size, conf, frame in self.results}
        for i, size_px in enumerate(STRIP_PX):
            ax = axes[f"t{i}"]
            dark_axes(ax, hide=True)
            conf, frame = by_size[size_px]
            half = THUMB_WINDOW // 2
            mid = self.canvas // 2
            ax.imshow(frame[mid - half:mid + half, mid - half:mid + half], interpolation="nearest")
            found = conf >= DETECTED_CONF
            ax.set_title(f"{size_px} px · {conf:.2f}" if found else f"{size_px} px · —",
                         color=ACCENT if found else WARN, fontsize=10,
                         fontweight="bold" if found else "normal")

        self._draw_curve(axes["curve"])
        fig.suptitle(f"Поріг детекції для «{self.target_name}»", color=FG, fontsize=15, fontweight="bold")
        self._draw_status(fig)

        fig.subplots_adjust(left=0.07, right=0.97, top=0.90, bottom=0.17)
        plt.show()
        plt.close(fig)

    def _draw_curve(self, ax) -> None:
        sizes = [size for size, _, _ in self.results]
        confs = [conf for _, conf, _ in self.results]
        dark_axes(ax)

        ax.plot(sizes, confs, "-o", color=ACCENT, linewidth=2, markersize=5)
        ax.axhline(DETECTED_CONF, color=WARN, linestyle="--", linewidth=1.2)
        # вісь інвертована, тож найбільший розмір — це лівий край
        ax.text(max(sizes), DETECTED_CONF, f" поріг conf {DETECTED_CONF}", color=WARN,
                fontsize=9, va="bottom", ha="left")

        for rule_px, label in ((16, "правило: детекція ~16 px"), (32, "класифікація ~32 px")):
            ax.axvline(rule_px, color=MUTED, linestyle=":", linewidth=1.2)
            ax.text(rule_px, 1.02, label, color=MUTED, fontsize=9, ha="center")

        ax.set_xlim(max(sizes) + 2, min(sizes) - 1)  # зліва направо = ціль віддаляється
        ax.set_ylim(0, 1.08)
        ax.set_xlabel("Розмір цілі на вході моделі, px", color=MUTED, fontsize=10)
        ax.set_ylabel("Впевненість", color=MUTED, fontsize=10)
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)

    def _draw_status(self, fig) -> None:
        last_ok, first_lost = self._loss_band()
        if last_ok is None:
            text, color = "Модель не побачила ціль на жодному розмірі — візьми більший об'єкт або інше фото.", WARN
        elif first_lost is None:
            text, color = f"Ціль трималась аж до {min(TARGET_SIZES_PX)} px — додай менші розміри в TARGET_SIZES_PX.", MUTED
        else:
            text = f"Модель втрачає ціль між {last_ok} px і {first_lost} px"
            color = ACCENT
        fig.text(0.5, 0.075, text, color=color, ha="center", fontsize=12, fontweight="bold")

        if last_ok is not None and first_lost is not None:
            distance = self.detection_range(self.target_m.value, self.focal.value,
                                            self.pixel.value, self.resize_k.value, first_lost)
            fig.text(0.5, 0.035,
                     f"За оптики f = {self.focal.value:.0f} мм, p = {self.pixel.value} мкм, "
                     f"k = {self.resize_k.value:g} це станеться на " + f"{distance:,.0f} м".replace(",", " "),
                     color=FG, ha="center", fontsize=11)

        fig.text(0.5, 0.005,
                 "Симуляція чисто геометрична: зникають пікселі, але не з'являються серпанок і втрата MTF. "
                 "Тому це оптимістична верхня межа.",
                 color=MUTED, ha="center", fontsize=9)


# ───────────────────────────────── demo2: resize проти tiled проти теле


class LeversPanel(PhotoPanel):
    """Базовий resize проти tiled inference проти довшої фокусної. Підрахунки — ззовні."""

    def __init__(self, detect, run_whole, run_tiled, run_tele, default_photo: Path | None = None):
        super().__init__(default_photo)
        self.detect = detect
        self.run_whole = run_whole
        self.run_tiled = run_tiled
        self.run_tele = run_tele

        self.results = []

        self._build()
        self._wire()

    def _build(self) -> None:
        field = Layout(width="300px")
        self.run_btn = Button(description="Порівняти режими", icon="play", button_style="success",
                              layout=Layout(width="220px"))
        self.tile = Dropdown(options=[("512", 512), ("640", 640), ("768", 768), ("1024", 1024)],
                             value=640, description="Розмір тайла", style=LABEL_STYLE, layout=field)
        self.overlap = FloatSlider(value=0.2, min=0.0, max=0.5, step=0.05, description="Перекриття",
                                   readout_format=".0%", style=LABEL_STYLE, layout=field)
        self.tele_zoom = IntSlider(value=3, min=2, max=8, description="Теле ×",
                                   style=LABEL_STYLE, layout=field)

    def _wire(self) -> None:
        self.run_btn.on_click(self._on_run)

    def ui(self) -> VBox:
        return VBox([
            section("ФОТО"),
            HBox([self.uploader, self.run_btn]),
            section("ПАРАМЕТРИ РЕЖИМІВ"),
            HBox([self.tile, self.overlap, self.tele_zoom]),
            self.out,
        ])

    def on_photo(self) -> None:
        self.results = []
        self.render()

    def problem(self) -> str | None:
        base = super().problem()
        if base:
            return base
        if not self.results:
            return (f"Фото {self.photo.shape[1]}×{self.photo.shape[0]} px. "
                    "Тисни «Порівняти режими».")
        return None

    # --------------------------------------------------------------- обробники

    def _on_run(self, *_) -> None:
        self.out.clear_output(wait=True)
        compared = False
        with self.out:
            if self.photo is None:
                print("Спочатку обери фото.")
                return
            print("Рахую три режими…")
            self.results = []
            self._compare()
            compared = True
        # якщо _compare упав, Output уже показав traceback — render() його лише затер би
        if compared:
            self.render()

    def _compare(self) -> None:
        img_w = self.photo.shape[1]
        tile = self.tile.value
        zoom = self.tele_zoom.value

        runs = [
            ("Базовий resize", BASE_COLOR,
             lambda: self.run_whole(self.photo, self.detect, tile), img_w / tile, None),
            (f"Tiled {tile}px / {self.overlap.value:.0%}", TILED_COLOR,
             lambda: self.run_tiled(self.photo, self.detect, tile, self.overlap.value), 1.0, None),
            (f"Теле ×{zoom} + широкий", TELE_COLOR,
             lambda: self.run_tele(self.photo, self.detect, zoom, tile), img_w / zoom / tile, zoom),
        ]

        results = []
        for name, color, run, k, tele_zoom in runs:
            started = time.perf_counter()
            detections, passes, *gained = run()  # теле повертає ще й те, що знайшло лише воно
            results.append({"name": name, "color": color, "detections": detections,
                                 "passes": passes, "seconds": time.perf_counter() - started,
                                 "k": k, "tele_zoom": tele_zoom,
                                 "gained": gained[0] if gained else []})
        self.results = results

    # ----------------------------------------------------------------- рендер

    def draw(self) -> None:
        preview, scale = to_display(self.photo, 620)
        # пропорції рядів підігнані під ландшафтне фото: інакше кадри центруються
        # у зависокому ряду й між ними та стовпчиками лишається порожнеча
        fig, axes = plt.subplots(2, 3, figsize=(14, 7.2), facecolor=BG,
                                 gridspec_kw={"height_ratios": [2.2, 1]})

        for column, result in enumerate(self.results):
            self._draw_frame(axes[0, column], preview, scale, result)
        self._draw_summary(axes[1, 0], "Знайдено об'єктів", [len(r["detections"]) for r in self.results])
        self._draw_summary(axes[1, 1], "Проходів детектора", [r["passes"] for r in self.results])
        self._draw_summary(axes[1, 2], "Час, с", [r["seconds"] for r in self.results], fmt="{:.2f}")

        base, tiled, tele = self.results
        gain = len(tiled["detections"]) - len(base["detections"])
        fig.suptitle(f"Tiled: {gain:+d} об'єктів до базового ціною "
                     f"×{tiled['passes'] / max(base['passes'], 1):.0f} обчислень   ·   "
                     f"Теле ×{tele['tele_zoom']}: +{len(tele['gained'])} у центрі за 2 проходи",
                     color=FG, fontsize=13, fontweight="bold")
        fig.text(0.5, 0.012,
                 "k — у скільки разів кадр тиснеться під вхід моделі. Tiled прибирає стиснення (k = 1), "
                 "теле — лише в центрі, тому й іде поверх широкого. Це той самий k, що у формулі з частини 1.",
                 color=MUTED, ha="center", fontsize=9)

        fig.subplots_adjust(left=0.04, right=0.98, top=0.90, bottom=0.08, hspace=0.35, wspace=0.18)
        plt.show()
        plt.close(fig)

    def _draw_frame(self, ax, preview, scale: float, result: dict) -> None:
        dark_axes(ax, hide=True)
        ax.imshow(preview)

        gained = set(result["gained"])
        # там, де проходів два, приглушуємо спільні бокси: яскравими лишаються тільки нові
        shared_color = MUTED if gained else result["color"]
        for detection in result["detections"]:
            _, _, (x1, y1, x2, y2) = detection
            new = detection in gained
            ax.add_patch(patches.Rectangle((x1 * scale, y1 * scale), (x2 - x1) * scale,
                                           (y2 - y1) * scale, linewidth=2.4 if new else 1.2,
                                           edgecolor=result["color"] if new else shared_color,
                                           facecolor="none"))

        if result["tele_zoom"]:  # показуємо, якою ціною дістається деталь
            img_h, img_w = self.photo.shape[:2]
            w, h = img_w / result["tele_zoom"] * scale, img_h / result["tele_zoom"] * scale
            ax.add_patch(patches.Rectangle(((preview.shape[1] - w) / 2, (preview.shape[0] - h) / 2),
                                           w, h, linewidth=1.6, edgecolor=result["color"],
                                           facecolor="none", linestyle="--"))

        found = f"{len(result['detections'])} об'єктів"
        if result["tele_zoom"]:
            found += f" · +{len(result['gained'])} у теле"
            k_note = f"k = {result['k']:.2f} у центрі"
        else:
            k_note = f"k = {result['k']:.2f}"
        ax.set_title(f"{result['name']}\n{found} · {k_note}",
                     color=result["color"], fontsize=11, fontweight="bold")

    def _draw_summary(self, ax, title: str, values, fmt: str = "{:.0f}") -> None:
        dark_axes(ax)
        names = [r["name"].split()[0] for r in self.results]
        colors = [r["color"] for r in self.results]
        bars = ax.bar(names, values, color=colors, width=0.6)
        ax.set_title(title, color=MUTED, fontsize=10)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_ylim(0, max(values) * 1.35 if max(values) else 1)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.04,
                    fmt.format(value), color=FG, ha="center", fontsize=9, fontweight="bold")


# ───────────────────────────────────────────────────────────────── запускачі


def launch_fov(fov_from_focal, focal_from_fov, zoom_factor, crop_box, min_fov_deg: float = 10.0,
               photo: Path | None = DEMO1_PHOTO,
               photo_fov_deg: float = DEMO1_PHOTO_FOV_DEG) -> None:
    """demo1: фокусна відстань і поле зору."""
    FovPanel(fov_from_focal, focal_from_fov, zoom_factor, crop_box, min_fov_deg, photo, photo_fov_deg).show()


def launch_zoom(sensor_capture, digital_zoom, hybrid_frame, crop_box,
                photo: Path | None = DEMO1_PHOTO,
                photo_fov_deg: float = DEMO1_PHOTO_FOV_DEG) -> None:
    """demo1: цифровий зум проти теле-сенсора."""
    ZoomPanel(sensor_capture, digital_zoom, hybrid_frame, crop_box, photo, photo_fov_deg).show()


def launch_range(pixels_on_target, detection_range, focal_for_range, resize_factor,
                 field_of_view, effective_pixel) -> None:
    """demo2: калькулятор дальності детекції."""
    RangePanel(pixels_on_target, detection_range, focal_for_range, resize_factor,
               field_of_view, effective_pixel).show()


def launch_threshold(detect, scene_at_size, matched_confidence, detection_range,
                     canvas: int = 640, photo: Path | None = DEMO2_THRESHOLD_PHOTO) -> None:
    """demo2: поріг детекції справжнім детектором."""
    ThresholdPanel(detect, scene_at_size, matched_confidence, detection_range, canvas, photo).show()


def launch_levers(detect, run_whole, run_tiled, run_tele,
                  photo: Path | None = DEMO2_LEVERS_PHOTO) -> None:
    """demo2: resize проти tiled проти теле."""
    LeversPanel(detect, run_whole, run_tiled, run_tele, photo).show()
