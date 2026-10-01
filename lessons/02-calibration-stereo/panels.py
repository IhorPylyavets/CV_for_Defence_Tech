"""Інтерактивні панелі заняття 2.

Тут лише обв'язка: віджети, малювання, обробники подій. Уся математика — вартість
відповідності, методи disparity, оцінка, глибина — лишається в ноутбуці й приходить сюди
аргументами launch_*(), щоб формули можна було правити просто в клітинці.

Базовий клас Panel свідомо продубльовано з заняття 1: за домовленістю в README спільний
шар виносимо на третьому повторенні, а це лише друге.

Вміст:
    MethodsPanel      — demo4: шість методів disparity на реальних парах Middlebury
    CostCurvePanel    — demo4: вартість відповідності в одній точці, звідки беруться помилки
    SgbmTuningPanel   — demo4: штрафи гладкості SGBM і L–R перевірка
    show_depth_law    — demo4: від disparity до міліметрів, закон Z² на реальних даних
"""

import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from ipywidgets import (Checkbox, Dropdown, FloatSlider, HBox, HTML, IntSlider, Layout, Output,
                        ToggleButtons, VBox)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

BG, FG, MUTED, ACCENT, WARN = "#17191d", "#e6e8ea", "#8a8f98", "#2fe08a", "#ff7a59"
SPINE, GRID = "#2a2e35", "#23272e"
TRUE_COLOR = "#7aa2f7"

LABEL_STYLE = {"description_width": "170px"}
FIELD = Layout(width="300px")


def section(text: str) -> HTML:
    return HTML(f"<b style='color:{MUTED};letter-spacing:.09em;font-size:11px'>{text}</b>")


def dark_axes(*axes, hide: bool = False) -> None:
    """Темне тло панелей. hide=True — для картинок, де осі лише заважають."""
    for ax in axes:
        ax.set_facecolor(BG)
        if hide:
            ax.axis("off")
            continue
        for spine in ax.spines.values():
            spine.set_color(SPINE)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)


class Panel:
    """Спільна основа: вивід, придушення подій під час пакетних змін, безпечний рендер."""

    def __init__(self):
        self.out = Output()
        self._suspend = False

    def ui(self) -> VBox:
        raise NotImplementedError

    def problem(self) -> str | None:
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


# ─────────────────────────── demo4: методи disparity на реальних парах Middlebury


MISSING_COLOR = "#4c566a"
OCCLUDED_WEIGHT = 0.45   # у зоні перекриття кольори приглушені: відповідності там не існує


def _rgb(hex_color: str) -> np.ndarray:
    return np.array([int(hex_color[i:i + 2], 16) for i in (1, 3, 5)], np.float32) / 255


def _disparity_image(ax, disparity, vmax, title) -> None:
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad(BG)
    ax.imshow(np.ma.masked_invalid(disparity), cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
    ax.set_title(title, color=MUTED, fontsize=9.5)


def error_overlay(scene: dict, disparity: np.ndarray, bad_px: float) -> np.ndarray:
    """Карта помилок поверх приглушеного лівого кадру: зелений — правильно, помаранчевий —
    хибно, сірий — відповіді немає. У зоні перекриття ті самі кольори, але бліді."""
    gray = cv2.cvtColor(scene["left"], cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    out = np.repeat(gray[..., None] * 0.30, 3, axis=2)
    found = np.isfinite(disparity)
    known = np.isfinite(scene["gt"])
    error = np.abs(np.where(found & known, disparity, 0) - np.where(known, scene["gt"], 0))
    wrong = found & known & (error > bad_px)
    correct = found & known & ~wrong
    for region, weight in ((scene["nocc"], 1.0), (scene["occluded"], OCCLUDED_WEIGHT)):
        for mask, color in ((correct, ACCENT), (wrong, WARN), (~found, MISSING_COLOR)):
            sel = region & mask
            out[sel] = out[sel] * (1 - weight) + _rgb(color) * weight
    return np.clip(out, 0, 1)


ERROR_LEGEND = ("зелений — правильно (|Δd| ≤ τ) · помаранчевий — хибно · сірий — відповіді немає · "
                "бліді кольори — зона перекриття, де відповідності не існує в принципі")


def _pct(value: float) -> str:
    return "—" if not np.isfinite(value) else f"{value * 100:.0f}%"


def _legend(ax, loc="upper right", fontsize=8) -> None:
    legend = ax.legend(facecolor=BG, edgecolor=SPINE, fontsize=fontsize, labelcolor=FG, loc=loc)
    legend.get_frame().set_alpha(0.9)


class _ScenePicker:
    """Сцена + варіант правого кадру: спільне для трьох панелей demo4."""

    def _build_scene_widgets(self, scenes: dict, about: dict | None = None) -> None:
        self.scenes = scenes
        self.about = about or {}
        self.scene_name = Dropdown(options=list(scenes), description="Сцена",
                                   style=LABEL_STYLE, layout=FIELD)
        self.variant = Dropdown(description="Правий кадр", style=LABEL_STYLE, layout=FIELD)
        self.about_html = HTML()
        self._sync_variants()

    def _sync_variants(self) -> None:
        options = list(self.scenes[self.scene_name.value])
        self._suspend = True
        self.variant.options = options
        self.variant.value = options[0]
        self._suspend = False
        text = self.about.get(self.scene_name.value, "")
        self.about_html.value = f"<span style='color:{MUTED};font-size:12px'>{text}</span>"

    def _source(self) -> str:
        return self.scenes[self.scene_name.value][self.variant.value]


class MethodsPanel(_ScenePicker, Panel):
    """Шість методів на одній реальній парі: карти disparity, карти помилок і таблиця чисел.

    Метод — функція (left, right, num_disp, block) → disparity у px, NaN там, де відповіді
    немає. Якщо результат узято з кешу, метод може повернути (disparity, секунди) — тоді
    в таблиці буде справжній час обчислення, а не час читання з диска.
    """

    def __init__(self, scenes, about, load_scene, methods, evaluate, shift_rows, vertical_probe=None):
        super().__init__()
        self.load_scene = load_scene
        self.methods = methods
        self.evaluate = evaluate
        self.shift_rows = shift_rows
        self.vertical_probe = vertical_probe
        self._results = {}
        self._build_scene_widgets(scenes, about)
        self.shift = FloatSlider(value=0.0, min=0.0, max=4.0, step=0.5,
                                 description="Вертикальний зсув, px", style=LABEL_STYLE,
                                 readout_format=".1f", continuous_update=False, layout=FIELD)
        self.block = IntSlider(value=9, min=5, max=21, step=2, description="blockSize",
                               style=LABEL_STYLE, continuous_update=False, layout=FIELD)
        self.bad = Dropdown(options=[0.5, 1.0, 2.0, 4.0], value=2.0,
                            description="Поріг хибності τ, px", style=LABEL_STYLE, layout=FIELD)
        self.view = ToggleButtons(options=[("Карта помилок", "error"), ("Disparity", "disp")],
                                  value="error")
        self.status = HTML()
        self.scene_name.observe(self._on_scene, names="value")
        for widget in (self.variant, self.shift, self.block, self.bad, self.view):
            widget.observe(self.render, names="value")

    def ui(self) -> VBox:
        return VBox([
            section("СТЕРЕОПАРА"),
            HBox([self.scene_name, self.variant]),
            self.about_html,
            section("НЕІДЕАЛЬНА РЕКТИФІКАЦІЯ: ПРАВИЙ КАДР ЗСУНУТО ВГОРУ"),
            self.shift,
            section("МЕТОДИ Й ОЦІНКА"),
            HBox([self.block, self.bad]),
            HBox([self.view, self.status]),
            self.out,
        ])

    def _on_scene(self, *_) -> None:
        self._sync_variants()
        self.render()

    def _compute(self, scene, right) -> dict:
        results = {}
        for name, method in self.methods.items():
            key = (self._source(), self.shift.value, self.block.value, name)
            if key not in self._results:
                self.status.value = f"<span style='color:{MUTED}'>&nbsp;рахую {name}…</span>"
                started = time.perf_counter()
                value = method(scene["left"], right, scene["num_disp"], self.block.value)
                elapsed = time.perf_counter() - started
                if isinstance(value, tuple):
                    value, elapsed = value
                self._results[key] = (value, elapsed)
            results[name] = self._results[key]
        self.status.value = ""
        return results

    def draw(self) -> None:
        scene = self.load_scene(self._source())
        right = self.shift_rows(scene["right"], self.shift.value)
        results = self._compute(scene, right)
        bad = self.bad.value
        vmax = np.nanpercentile(scene["gt"], 99.5)

        names = list(results)
        rows = [["left", "right", "gt"]]
        rows += [names[i:i + 3] + ["_"] * (3 - len(names[i:i + 3])) for i in range(0, len(names), 3)]
        fig, axes = plt.subplot_mosaic(rows, figsize=(14, 3.35 * len(rows)), facecolor=BG,
                                       empty_sentinel="_")
        dark_axes(*axes.values(), hide=True)

        axes["left"].imshow(scene["left"])
        axes["left"].set_title("Лівий кадр", color=MUTED, fontsize=9.5)
        axes["right"].imshow(right)
        right_title = "Правий"
        if len(self.scenes[self.scene_name.value]) > 1:
            right_title += f" · {self.variant.value}"
        if self.shift.value:
            right_title += f" · зсув {self.shift.value:.1f} px"
        axes["right"].set_title(right_title, color=MUTED, fontsize=9.5)
        _disparity_image(axes["gt"], scene["gt"], vmax, "Істина (структуроване світло)")

        stats = {}
        for name, (disparity, elapsed) in results.items():
            stats[name] = self.evaluate(disparity, scene, bad)
            title = f"{name} · {elapsed * 1000:.0f} мс · хибних {_pct(stats[name]['bad_all'])}"
            if self.view.value == "disp":
                _disparity_image(axes[name], disparity, vmax, title)
            else:
                axes[name].imshow(error_overlay(scene, disparity, bad))
                axes[name].set_title(title, color=MUTED, fontsize=9.5)

        fig.suptitle(f"{self.scene_name.value} · {scene['left'].shape[1]}×{scene['left'].shape[0]} · "
                     f"діапазон пошуку {scene['num_disp']} px · τ = {bad:g} px",
                     color=FG, fontsize=13, fontweight="bold")
        if self.view.value == "error":
            fig.text(0.5, 0.005, ERROR_LEGEND, color=MUTED, ha="center", fontsize=9)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.03, hspace=0.16, wspace=0.03)
        plt.show()
        plt.close(fig)
        display(HTML(self._table(results, stats)))
        self._vertical_note(scene, right)

    def _table(self, results, stats) -> str:
        best = min(s["bad_all"] for s in stats.values())
        columns = (("Метод", "left"), ("Час", "right"), ("Щільність", "right"),
                   ("Хибні серед знайдених", "right"), ("Хибні, дірка = помилка", "right"),
                   ("Сер. |Δd|, px", "right"), ("Перекриття: відповідь є", "right"),
                   ("…і вона хибна", "right"))
        head = "".join(f"<th style='padding:4px 10px;text-align:{align};color:{MUTED}'>{title}</th>"
                       for title, align in columns)
        body = ""
        for name, (_, elapsed) in results.items():
            s = stats[name]
            color = ACCENT if s["bad_all"] == best else FG
            cells = (f"{elapsed * 1000:.0f} мс", _pct(s["density"]), _pct(s["bad_found"]),
                     f"<b style='color:{color}'>{_pct(s['bad_all'])}</b>", f"{s['avg_error']:.2f}",
                     _pct(s["occl_filled"]), _pct(s["occl_wrong"]))
            body += (f"<tr><td style='padding:3px 10px;color:{FG}'>{name}</td>"
                     + "".join(f"<td style='padding:3px 10px;text-align:right;color:{FG}'>{c}</td>"
                               for c in cells) + "</tr>")
        return (f"<table style='background:{BG};border-collapse:collapse;font-family:monospace;"
                f"font-size:12px;margin-top:4px'><tr>{head}</tr>{body}</table>"
                f"<div style='color:{MUTED};font-size:11px;margin-top:4px'>Щільність і хибні рахуємо на "
                f"пікселях, видимих обом камерам. «Дірка = помилка» — так рахує рейтинг Middlebury: "
                f"метод, що чесно мовчить, штрафується так само, як той, що бреше.</div>")

    def _vertical_note(self, scene, right) -> None:
        if self.vertical_probe is None:
            return
        measured = self.vertical_probe(scene["left"], right)
        color = WARN if abs(measured) > 0.3 else ACCENT
        display(HTML(f"<div style='color:{FG};font-size:12px;margin-top:6px'>Вертикальна складова, "
                     f"яку бачить RAFT (медіана): <b style='color:{color}'>{measured:+.2f} px</b> при "
                     f"заданому зсуві {self.shift.value:.1f} px. Класичні методи шукають лише вздовж "
                     f"рядка й такого зсуву не помічають — вони просто помиляються. Це і є "
                     f"health-метрика зі слайда «Як це ловити онлайн».</div>"))


class CostCurvePanel(_ScenePicker, Panel):
    """Вартість відповідності для одного пікселя лівого кадру вздовж його рядка в правому.

    Будь-який block matching — це пошук мінімуму цієї кривої. Видно, чому він помиляється:
    на слабкій текстурі мінімум пологий, на повторюваному патерні мінімумів кілька.
    """

    BLOCKS = (5, 11, 21)
    BLOCK_COLORS = ("#e0af68", FG, "#bb9af7")
    SHOWN_BLOCK = 11          # вікно для маркерів на кадрах і латок
    MARGIN = max(BLOCKS) // 2 + 4   # півсмуги рядків навколо y: вистачає на вікно й Census 7×7
    MANUAL = "вручну (x, y)"

    def __init__(self, scenes, load_scene, cost_volume, pick_points):
        super().__init__()
        self.load_scene = load_scene
        self.cost_volume = cost_volume
        self.pick_points = pick_points
        self._points = {}
        self._build_scene_widgets(scenes)
        self.point = Dropdown(description="Точка", style=LABEL_STYLE, layout=Layout(width="460px"))
        self.x = IntSlider(description="x", style=LABEL_STYLE, continuous_update=False, layout=FIELD)
        self.y = IntSlider(description="y", style=LABEL_STYLE, continuous_update=False, layout=FIELD)
        self.scene_name.observe(self._on_scene, names="value")
        self.variant.observe(self._on_variant, names="value")
        self.point.observe(self._on_point, names="value")
        for widget in (self.x, self.y):
            widget.observe(self._on_xy, names="value")
        self._on_variant()

    def ui(self) -> VBox:
        return VBox([
            section("СТЕРЕОПАРА"),
            HBox([self.scene_name, self.variant]),
            section("ПІКСЕЛЬ ЛІВОГО КАДРУ"),
            self.point,
            HBox([self.x, self.y]),
            self.out,
        ])

    def _on_scene(self, *_) -> None:
        self._sync_variants()
        self._on_variant()

    def _on_variant(self, *_) -> None:
        if self._suspend:
            return
        source = self._source()
        scene = self.load_scene(source)
        if source not in self._points:
            self._points[source] = self.pick_points(scene)
        points = self._points[source]
        height, width = scene["gt"].shape
        margin = self.MARGIN
        self._suspend = True
        self.x.min, self.x.max = 0, width - margin
        self.x.min = scene["num_disp"]
        self.y.min, self.y.max = 0, height - margin - 1    # смуга y ± margin цілком у кадрі
        self.y.min = margin
        self.point.options = list(points) + [self.MANUAL]
        self.point.value = next(iter(points), self.MANUAL)
        self._suspend = False
        self._on_point()

    def _on_point(self, *_) -> None:
        if self._suspend:
            return
        points = self._points[self._source()]
        if self.point.value in points:
            self._suspend = True
            self.x.value, self.y.value = points[self.point.value]
            self._suspend = False
        self.render()

    def _on_xy(self, *_) -> None:
        if self._suspend:
            return
        points = self._points[self._source()]
        if points.get(self.point.value) != (self.x.value, self.y.value):
            self._suspend = True
            self.point.value = self.MANUAL
            self._suspend = False
        self.render()

    def _curves(self, left, right, x, y, num_disp) -> dict:
        """Криві вартості в точці (x, y): об'єм рахуємо лише на смузі рядків навколо y."""
        margin = self.MARGIN
        rows = slice(y - margin, y + margin + 1)
        return {(kind, block): self.cost_volume(left[rows], right[rows], num_disp, block, kind)[:, margin, x]
                for kind in ("sad", "census") for block in self.BLOCKS}

    def draw(self) -> None:
        scene = self.load_scene(self._source())
        x, y, nd = self.x.value, self.y.value, scene["num_disp"]
        left = cv2.cvtColor(scene["left"], cv2.COLOR_RGB2GRAY)
        right = cv2.cvtColor(scene["right"], cv2.COLOR_RGB2GRAY)
        curves = self._curves(left, right, x, y, nd)
        truth = scene["gt"][y, x]
        best = {kind: int(np.argmin(curves[kind, self.SHOWN_BLOCK])) for kind in ("sad", "census")}
        occluded = bool(scene["occluded"][y, x])

        fig, axes = plt.subplot_mosaic([["left", "right", "patch"], ["sad", "census", "strip"]],
                                       figsize=(14, 8.8), facecolor=BG,
                                       gridspec_kw={"height_ratios": [1.15, 1]})
        dark_axes(axes["left"], axes["right"], axes["patch"], axes["strip"], hide=True)
        dark_axes(axes["sad"], axes["census"])
        marks = [("істина", truth, TRUE_COLOR), ("SAD", best["sad"], WARN),
                 ("Census", best["census"], ACCENT)]

        axes["left"].imshow(scene["left"])
        axes["left"].axhline(y, color=FG, linewidth=0.6, alpha=0.6)
        axes["left"].plot([x], [y], "+", color=WARN, markersize=16, markeredgewidth=2)
        axes["left"].set_title(f"Лівий кадр · піксель ({x}, {y})", color=MUTED, fontsize=9.5)
        axes["right"].imshow(scene["right"])
        axes["right"].axhline(y, color=FG, linewidth=0.6, alpha=0.6)
        axes["right"].axvspan(x - nd, x, color=FG, alpha=0.10)
        for label, d, color in marks:
            if np.isfinite(d):
                axes["right"].plot([x - d], [y], "o", markersize=9, markerfacecolor="none",
                                   markeredgecolor=color, markeredgewidth=2, label=label)
        axes["right"].set_title("Правий кадр: пошук лише в цьому рядку",
                                color=MUTED, fontsize=9.5)
        _legend(axes["right"], loc="lower right")

        self._draw_patches(axes["patch"], left, right, x, y, marks)
        self._draw_strip(axes["strip"], scene["right"], x, y, nd, marks)
        self._draw_curve(axes["sad"], curves, "sad", truth, "SAD — сума |різниць| яскравості у вікні")
        self._draw_curve(axes["census"], curves, "census", truth, "Census — Геммінг між бітовими масками")

        where = "зона перекриття: правій камері цей піксель не видно" if occluded else "видно обом камерам"
        truth_text = f"{truth:.1f} px" if np.isfinite(truth) else "невідома"
        fig.suptitle(f"{self.point.value} · істина d = {truth_text} · SAD {best['sad']} px · "
                     f"Census {best['census']} px · {where}", color=FG, fontsize=12, fontweight="bold")
        fig.subplots_adjust(left=0.05, right=0.99, top=0.92, bottom=0.07, hspace=0.25, wspace=0.12)
        plt.show()
        plt.close(fig)

    def _draw_patches(self, ax, left, right, x, y, marks) -> None:
        r = self.SHOWN_BLOCK // 2
        tiles = [("ліва", left[y - r:y + r + 1, x - r:x + r + 1], WARN)]
        for label, d, color in marks:
            if np.isfinite(d) and x - int(round(d)) - r >= 0:
                xr = x - int(round(d))
                tiles.append((label, right[y - r:y + r + 1, xr - r:xr + r + 1], color))
        gap = np.zeros((2 * r + 1, 2), np.uint8)
        pieces = []
        for _, tile, _ in tiles:
            pieces += [tile, gap]
        ax.imshow(np.hstack(pieces[:-1]), cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        step = 2 * r + 3
        for i, (label, _, color) in enumerate(tiles):
            ax.text(i * step + r, -1.2, label, color=color, ha="center", fontsize=8.5)
        ax.set_title(f"Вікна {self.SHOWN_BLOCK}×{self.SHOWN_BLOCK}: ліве й кандидати в правому",
                     color=MUTED, fontsize=9.5, pad=16)

    def _draw_strip(self, ax, right, x, y, nd, marks) -> None:
        x0, x1 = max(0, x - nd - 15), min(right.shape[1], x + 15)
        y0, y1 = max(0, y - 25), min(right.shape[0], y + 26)
        ax.imshow(right[y0:y1, x0:x1], interpolation="nearest",
                  extent=(x0 - 0.5, x1 - 0.5, y1 - 0.5, y0 - 0.5))
        for _, d, color in marks:
            if np.isfinite(d):
                ax.axvline(x - d, color=color, linewidth=1.6)
        ax.set_title("Правий кадр навколо діапазону пошуку, збільшено", color=MUTED, fontsize=9.5)

    def _draw_curve(self, ax, curves, kind, truth, title) -> None:
        for block, style, color in zip(self.BLOCKS, (":", "-", "--"), self.BLOCK_COLORS):
            cost = curves[kind, block]
            finite = np.isfinite(cost)
            lo, hi = cost[finite].min(), cost[finite].max()
            norm = np.where(finite, (cost - lo) / max(hi - lo, 1e-6), np.nan)
            d_best = int(np.nanargmin(norm))
            ax.plot(np.arange(len(cost)), norm, style, color=color, linewidth=1.5,
                    label=f"вікно {block}×{block} · мінімум {d_best} px")
            ax.plot([d_best], [0], "o", color=color, markersize=6)
        if np.isfinite(truth):
            ax.axvline(truth, color=TRUE_COLOR, linestyle="--", linewidth=1.2, label="істина")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(title, color=MUTED, fontsize=9.5)
        ax.set_xlabel("кандидат disparity d, px", color=MUTED, fontsize=9)
        ax.set_ylabel("вартість (нормована)", color=MUTED, fontsize=9)
        _legend(ax, fontsize=7.5)


class SgbmTuningPanel(_ScenePicker, Panel):
    """Штрафи гладкості SGBM, фільтри й L–R перевірка — і що вони роблять із помилками."""

    def __init__(self, scenes, load_scene, sgbm_disparity, evaluate):
        super().__init__()
        self.load_scene = load_scene
        self.sgbm_disparity = sgbm_disparity
        self.evaluate = evaluate
        self._last = (None, None)   # (параметри, (disparity, секунди)): рядок і τ не перераховують SGBM
        self._build_scene_widgets(scenes)
        self.mode = Dropdown(options=[("SGBM · 5 напрямків", "sgbm"), ("3WAY · швидкий", "3way"),
                                      ("HH · 8 напрямків", "hh")], value="3way",
                             description="Режим", style=LABEL_STYLE, layout=FIELD)
        self.block = IntSlider(value=5, min=3, max=15, step=2, description="blockSize",
                               style=LABEL_STYLE, continuous_update=False, layout=FIELD)
        self.p1 = IntSlider(value=8, min=0, max=64, description="P1, ×3·blockSize²",
                            style=LABEL_STYLE, continuous_update=False, layout=FIELD)
        self.p2 = IntSlider(value=32, min=0, max=256, step=4, description="P2, ×3·blockSize²",
                            style=LABEL_STYLE, continuous_update=False, layout=FIELD)
        self.uniqueness = IntSlider(value=10, min=0, max=30, description="uniquenessRatio, %",
                                    style=LABEL_STYLE, continuous_update=False, layout=FIELD)
        self.speckle = Dropdown(options=[("вимкнено", 0), ("50 px", 50), ("100 px", 100), ("200 px", 200)],
                                value=100, description="Фільтр плям", style=LABEL_STYLE, layout=FIELD)
        self.lr_check = Checkbox(value=False, description="L–R перевірка")
        self.row = IntSlider(description="Рядок профілю", style=LABEL_STYLE,
                             continuous_update=False, layout=FIELD)
        self.bad = Dropdown(options=[0.5, 1.0, 2.0, 4.0], value=2.0,
                            description="Поріг хибності τ, px", style=LABEL_STYLE, layout=FIELD)
        self.scene_name.observe(self._on_scene, names="value")
        for widget in (self.variant, self.mode, self.block, self.p1, self.p2, self.uniqueness,
                       self.speckle, self.lr_check, self.row, self.bad):
            widget.observe(self.render, names="value")
        self._sync_row()

    def ui(self) -> VBox:
        return VBox([
            section("СТЕРЕОПАРА"),
            HBox([self.scene_name, self.variant]),
            section("ФУНКЦІЯ ЕНЕРГІЇ: P1 — ШТРАФ ЗА СХОДИНКУ ±1 px, P2 — ЗА БІЛЬШИЙ СТРИБОК"),
            HBox([self.mode, self.block]),
            HBox([self.p1, self.p2]),
            section("ФІЛЬТРИ ПІСЛЯ ПОШУКУ: ЧЕСНО ПРОМОВЧАТИ ЗАМІСТЬ ВИГАДАТИ"),
            HBox([self.uniqueness, self.speckle, self.lr_check]),
            HBox([self.row, self.bad]),
            self.out,
        ])

    def _on_scene(self, *_) -> None:
        self._sync_variants()
        self._sync_row()
        self.render()

    def _sync_row(self) -> None:
        height = self.load_scene(self._source())["gt"].shape[0]
        self._suspend = True
        self.row.max = height - 1
        self.row.value = height // 2
        self._suspend = False

    def _compute(self, scene) -> tuple[np.ndarray, float]:
        params = dict(p1=self.p1.value, p2=self.p2.value, uniqueness=self.uniqueness.value,
                      speckle=self.speckle.value, lr_check=self.lr_check.value, mode=self.mode.value)
        key = (self._source(), self.block.value, *params.values())
        if self._last[0] != key:
            started = time.perf_counter()
            disparity = self.sgbm_disparity(scene["left"], scene["right"], scene["num_disp"],
                                            self.block.value, **params)
            self._last = (key, (disparity, time.perf_counter() - started))
        return self._last[1]

    def draw(self) -> None:
        scene = self.load_scene(self._source())
        disparity, elapsed = self._compute(scene)
        stats = self.evaluate(disparity, scene, self.bad.value)
        vmax = np.nanpercentile(scene["gt"], 99.5)
        row = self.row.value

        fig, axes = plt.subplot_mosaic([["disp", "error"], ["profile", "profile"]], figsize=(14, 8.6),
                                       facecolor=BG, gridspec_kw={"height_ratios": [1.25, 0.75]})
        dark_axes(axes["disp"], axes["error"], hide=True)
        dark_axes(axes["profile"])
        _disparity_image(axes["disp"], disparity, vmax, f"Disparity · {elapsed * 1000:.0f} мс")
        axes["disp"].axhline(row, color=TRUE_COLOR, linewidth=0.9, alpha=0.8)
        axes["error"].imshow(error_overlay(scene, disparity, self.bad.value))
        axes["error"].axhline(row, color=TRUE_COLOR, linewidth=0.9, alpha=0.8)
        axes["error"].set_title("Карта помилок", color=MUTED, fontsize=9.5)

        profile = axes["profile"]
        columns = np.arange(disparity.shape[1])
        profile.fill_between(columns, 0, vmax * 1.05, where=scene["occluded"][row], color=WARN,
                             alpha=0.13, linewidth=0, label="перекриття")
        profile.plot(columns, scene["gt"][row], color=TRUE_COLOR, linewidth=2.4, label="істина")
        profile.plot(columns, disparity[row], color=ACCENT, linewidth=1.1, label="SGBM")
        profile.set_xlim(0, disparity.shape[1] - 1)
        profile.set_ylim(0, vmax * 1.05)
        profile.set_title(f"Профіль рядка {row}: малий P2 — шум на площинах, великий — розмиті сходинки "
                          f"на краях", color=MUTED, fontsize=9.5)
        profile.set_xlabel("x, px", color=MUTED, fontsize=9)
        profile.set_ylabel("d, px", color=MUTED, fontsize=9)
        _legend(profile, loc="upper left")

        fig.suptitle(f"{self.scene_name.value} · щільність {_pct(stats['density'])} · хибні серед знайдених "
                     f"{_pct(stats['bad_found'])} · дірка = помилка {_pct(stats['bad_all'])} · "
                     f"у перекритті відповідь є для {_pct(stats['occl_filled'])}",
                     color=FG, fontsize=12, fontweight="bold")
        fig.text(0.5, 0.005, ERROR_LEGEND, color=MUTED, ha="center", fontsize=9)
        fig.subplots_adjust(left=0.05, right=0.99, top=0.91, bottom=0.08, hspace=0.22, wspace=0.03)
        plt.show()
        plt.close(fig)


def show_depth_law(scene: dict, disparities: dict, depth_mm, depth_error_theory, title: str = "") -> None:
    """Від disparity до міліметрів: карта глибини, закон Z² на реальних даних і ціна doffs."""
    known = scene["nocc"] & np.isfinite(scene["gt"])
    z_true = depth_mm(scene["gt"], scene)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), facecolor=BG,
                             gridspec_kw={"width_ratios": [1.1, 1, 1]})
    dark_axes(axes[0], hide=True)
    dark_axes(*axes[1:])

    shown = axes[0].imshow(np.where(known, z_true / 1000, np.nan), cmap="viridis")
    axes[0].set_title("Істинна глибина Z = f·B / (d + doffs), м", color=MUTED, fontsize=9.5)
    bar = fig.colorbar(shown, ax=axes[0], fraction=0.035, pad=0.02)
    bar.ax.tick_params(colors=MUTED, labelsize=8)

    edges = np.quantile(z_true[known], np.linspace(0.02, 0.98, 13))
    centers = (edges[:-1] + edges[1:]) / 2
    bins = [known & (z_true >= lo) & (z_true < hi) for lo, hi in zip(edges[:-1], edges[1:])]
    for name, disparity in disparities.items():
        z = depth_mm(disparity, scene)
        error = np.abs(z - z_true)
        medians = [np.median(error[b & np.isfinite(z)]) for b in bins]
        axes[1].plot(centers / 1000, medians, "o-", markersize=3.5, linewidth=1.5, label=name)
    grid = np.linspace(edges[0], edges[-1], 100)
    for dd, style in ((0.25, "--"), (0.5, ":")):
        axes[1].plot(grid / 1000, depth_error_theory(grid, scene, dd), style, color=MUTED,
                     linewidth=1.2, label=f"Z²·Δd/(f·B), Δd = {dd} px")
    axes[1].set_title("Медіана |ΔZ| проти істинної дальності", color=MUTED, fontsize=9.5)
    axes[1].set_xlabel("Z, м", color=MUTED, fontsize=9)
    axes[1].set_ylabel("|ΔZ|, мм", color=MUTED, fontsize=9)
    _legend(axes[1], loc="upper left", fontsize=7.5)

    naive = depth_mm(scene["gt"], scene, with_doffs=False)
    relative = (naive - z_true) / z_true * 100
    axes[2].plot(centers / 1000, [np.median(relative[b]) for b in bins], "o-", color=WARN,
                 markersize=3.5, linewidth=1.6)
    axes[2].axhline(0, color=SPINE, linewidth=1)
    axes[2].set_title(f"Забули doffs = {scene['doffs']:.1f} px: наївна Z = f·B/d",
                      color=MUTED, fontsize=9.5)
    axes[2].set_xlabel("Z, м", color=MUTED, fontsize=9)
    axes[2].set_ylabel("похибка дальності, %", color=MUTED, fontsize=9)

    fig.suptitle(f"{title or scene['name']} · f = {scene['focal']:.0f} px · B = {scene['baseline_mm']:.0f} мм",
                 color=FG, fontsize=12.5, fontweight="bold")
    fig.subplots_adjust(left=0.02, right=0.99, top=0.86, bottom=0.13, wspace=0.22)
    plt.show()
    plt.close(fig)


# ───────────────────────────────────────────────────────────────── запускачі


def launch_disparity_methods(scenes, about, load_scene, methods, evaluate, shift_rows,
                             vertical_probe=None) -> None:
    """demo4: методи disparity поруч на реальній парі — карти, помилки, час."""
    MethodsPanel(scenes, about, load_scene, methods, evaluate, shift_rows, vertical_probe).show()


def launch_cost_curve(scenes, load_scene, cost_volume, pick_points) -> None:
    """demo4: крива вартості відповідності в одному пікселі."""
    CostCurvePanel(scenes, load_scene, cost_volume, pick_points).show()


def launch_sgbm_tuning(scenes, load_scene, sgbm_disparity, evaluate) -> None:
    """demo4: штрафи гладкості SGBM, фільтри й L–R перевірка."""
    SgbmTuningPanel(scenes, load_scene, sgbm_disparity, evaluate).show()
