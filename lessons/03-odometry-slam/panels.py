"""
Заняття 3 · Panel-віджети для демо-ноутбуків.

Конвенція курсу: математика живе в клітинках ноутбука, а тут лише інтерактивні обгортки.
Тому launch_fusion отримує обчислювальну функцію з ноутбука demo2 як аргумент.
"""

import param
import panel as pn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

pn.extension()

BG, FG, MUT, GRID = "#1E2833", "#E8EAED", "#8E99A8", "#2E3A48"


def _dark(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=MUT)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.grid(True, color=GRID, lw=0.6, which="both")


# ---------------------------------------------------------------- demo2: VO + колеса + гіроскоп
# Ноутбук малює на світлому тлі; тут ті самі відтінки, але кроки палітри під темне тло
_DARK_STEP = {"#eb6834": "#d95926", "#1baf7a": "#199e70", "#2a78d6": "#3987e5", "#4a3aa7": "#9085e9"}


class FusionPanel(param.Parameterized):
    """Злиття VO з імітованими колесами й гіроскопом: що буде з масштабом і курсом, якщо змінити сенсори."""

    wheel_scale_err = param.Number(2.0, bounds=(-10.0, 10.0), doc="Похибка радіуса колеса, %")
    wheel_noise = param.Number(0.1, bounds=(0.0, 1.0), doc="Шум швидкості коліс, м/с")
    gyro_bias_dps = param.Number(0.2, bounds=(0.0, 1.0), doc="Зсув гіроскопа, °/с")
    gyro_noise_dps = param.Number(0.1, bounds=(0.01, 1.0), doc="Шум гіроскопа, °/с")
    calib_s = param.Number(0.0, bounds=(0.0, 60.0), doc="Калібрування нерухомим перед стартом, с")
    vo_noise_deg = param.Number(0.1, bounds=(0.01, 2.0), doc="Довіра до VO: σ приросту курсу за кадр, °")
    seed = param.Integer(0, bounds=(0, 999))

    def __init__(self, run_fn, **kw):
        super().__init__(**kw)
        self._run_fn = run_fn
        self._plot = pn.pane.Matplotlib(None, dpi=100, tight=True, sizing_mode="stretch_width")
        self._stats = pn.pane.Markdown("")
        self.param.watch(self._update, list(self.param)[1:])
        self._update()

    def _update(self, *_):
        res = self._run_fn(wheel_scale_err=self.wheel_scale_err / 100, wheel_noise=self.wheel_noise,
                           gyro_bias_dps=self.gyro_bias_dps, gyro_noise_dps=self.gyro_noise_dps,
                           calib_s=self.calib_s, vo_noise_deg=self.vo_noise_deg, seed=self.seed)
        style = {k: {**s, "color": _DARK_STEP.get(s["color"], s["color"])} for k, s in res["style"].items()}
        fig, (a0, a1) = plt.subplots(1, 2, figsize=(13, 5.6), facecolor=BG, gridspec_kw=dict(width_ratios=[1, 1.3]))
        for ax in (a0, a1):
            _dark(ax)
        gt = res["gt_xyz"]
        a0.plot(gt[:, 0], gt[:, 2], color=MUT, ls="--", lw=1.4, label="GT")
        for k in ["колеса + гіроскоп", "2D-2D + колеса", "2D-2D + колеса + гіроскоп"]:
            a0.plot(res["xyz"][k][:, 0], res["xyz"][k][:, 2], **style[k], label=k)
        a0.set_aspect("equal", adjustable="datalim")
        a0.set_xlabel("x, м", color=MUT); a0.set_ylabel("z, м", color=MUT)
        a0.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=8, loc="best")
        for k, e in res["yaw_err"].items():
            a1.plot(res["dist"], e, **style[k], label=f"{k} · {e[-1]:.1f}°")
        a1.set_ylim(0, 25)          # 3D-3D без гіроскопа виходить за шкалу — його значення в легенді
        a1.set_xlabel("пройдено, м", color=MUT); a1.set_ylabel("похибка курсу, °", color=MUT)
        a1.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=7.5, ncol=2, loc="lower right")
        fig.tight_layout()
        self._plot.object = fig
        plt.close(fig)
        rows = "\n".join(f"| {k} | {s['ate']:.2f} | {s['end']:.2f} | {s['yaw_end']:.2f} |" for k, s in res["stats"].items())
        self._stats.object = "| комбінація | ATE, м | дрейф, % | курс у кінці, ° |\n|---|---:|---:|---:|\n" + rows

    def view(self):
        return pn.Row(pn.Param(self, show_name=False, width=320), pn.Column(self._stats, self._plot, sizing_mode="stretch_width"))


def launch_fusion(run_fn):
    """run_fn(**params) -> dict з gt_xyz, dist, xyz, yaw_err, stats, style (fusion_experiment з demo2)."""
    return FusionPanel(run_fn).view()
