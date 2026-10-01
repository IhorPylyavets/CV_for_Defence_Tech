"""Синтетичні «фото» шахівниці й ChArUco в artifacts/images/calibration/ (заняття 2).

Параметри дошок збігаються з першою клітинкою коду lesson_02_demo3_calibration.ipynb:
    шахівниця 10×7 клітинок по 34 мм (9×6 внутрішніх кутів),
    ChArUco 9×6 клітинок по 40 мм, маркер 30 мм, DICT_5X5_250.

Камера з ВІДОМИМИ K і дисторсією — вони записуються в truth.json, тож результат
калібрування можна звірити з істиною. Для репетиції розділу «Як повторити на власній
камері»; на занятті краще власні знімки. Див. artifacts/images/calibration/README.md.

Запуск з кореня репозиторію:  python tools/make_synthetic_calibration.py
"""
import json
from pathlib import Path

import cv2
import numpy as np
W, H = 1280, 720
K = np.array([[1000., 0, 645.], [0, 1000., 356.], [0, 0, 1]])
D = np.array([-0.25, 0.09, 0.0005, -0.0003, -0.015])
OUT = Path(__file__).resolve().parent.parent / "artifacts" / "images" / "calibration"
PX_MM = 6                                            # роздільність текстури дошки

def chess_texture():
    sq = 34; nx, ny = 10, 7; m = 20                   # 10×7 клітинок, біле поле 20 мм
    w, h = (nx * sq + 2 * m) * PX_MM, (ny * sq + 2 * m) * PX_MM
    img = np.full((h, w), 235, np.uint8)
    for j in range(ny):
        for i in range(nx):
            if (i + j) % 2 == 0:
                x0, y0 = (m + i * sq) * PX_MM, (m + j * sq) * PX_MM
                img[y0:y0 + sq * PX_MM, x0:x0 + sq * PX_MM] = 25
    # початок системи дошки — перший ВНУТРІШНІЙ кут (як у findChessboardCornersSB)
    origin_mm = (m + sq, m + sq)
    return img, origin_mm

def charuco_texture():
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
    b = cv2.aruco.CharucoBoard((9, 6), 0.040, 0.030, d)
    m = 20
    inner = b.generateImage((9 * 40 * PX_MM, 6 * 40 * PX_MM), marginSize=0)
    img = np.full((inner.shape[0] + 2 * m * PX_MM, inner.shape[1] + 2 * m * PX_MM), 235, np.uint8)
    img[m * PX_MM:m * PX_MM + inner.shape[0], m * PX_MM:m * PX_MM + inner.shape[1]] = np.where(inner > 127, 235, 25)
    return img, (m, m)                                # ChArUco: початок — кут дошки

def render(tex, origin_mm, rvec, tvec, rng):
    R, _ = cv2.Rodrigues(rvec)
    Hm = np.column_stack([R[:, 0], R[:, 1], tvec])    # ідеальні норм. координати ← дошка (м)
    us, vs = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    pts = np.stack([us.ravel(), vs.ravel()], 1).reshape(-1, 1, 2)
    n = cv2.undistortPoints(pts, K, D).reshape(-1, 2)
    b = np.linalg.solve(Hm, np.column_stack([n, np.ones(len(n))]).T).T
    bx, by = b[:, 0] / b[:, 2], b[:, 1] / b[:, 2]     # координати на дошці, м
    mx = ((bx * 1000 + origin_mm[0]) * PX_MM).reshape(H, W).astype(np.float32)
    my = ((by * 1000 + origin_mm[1]) * PX_MM).reshape(H, W).astype(np.float32)
    bg = cv2.GaussianBlur(rng.integers(70, 150, (H, W)).astype(np.uint8), (0, 0), 3)
    img = cv2.remap(tex, mx, my, cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    inside = cv2.remap(np.ones_like(tex), mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
    img = np.where(inside, img, bg).astype(np.float32)
    shade = 0.85 + 0.25 * np.linspace(0, 1, W)[None, :] * rng.uniform(0.5, 1)
    img = cv2.GaussianBlur(img * shade, (0, 0), 0.9) + rng.normal(0, 2.0, (H, W))
    return np.clip(img, 0, 255).astype(np.uint8)

def poses(n, rng, board_wh, center_bias=False, frontal=False):
    out = []
    while len(out) < n:
        z = rng.uniform(0.55, 1.05)
        tilt = 0.05 if frontal else rng.uniform(0.15, 0.85)
        axis = rng.normal(size=3); axis[2] *= 0.3; axis /= np.linalg.norm(axis)
        rvec = axis * tilt
        R, _ = cv2.Rodrigues(rvec)
        spread = 0.08 if center_bias else 0.9
        u = W / 2 + rng.uniform(-spread, spread) * W / 2
        v = H / 2 + rng.uniform(-spread, spread) * H / 2
        c = np.array([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z])
        tvec = c - R @ np.array([board_wh[0] / 2, board_wh[1] / 2, 0.0])
        out.append((rvec, tvec))
    return out

if __name__ == "__main__":
    rng = np.random.default_rng(3)
    for kind, (tex, org), wh in (("chessboard", chess_texture(), (8 * 0.034, 5 * 0.034)),
                                 ("charuco", charuco_texture(), (9 * 0.040, 6 * 0.040))):
        folder = OUT / kind; folder.mkdir(parents=True, exist_ok=True)
        for f in folder.glob("*.png"): f.unlink()
        plan = poses(18, rng, wh) + poses(4, rng, wh, center_bias=True) + poses(4, rng, wh, frontal=True)
        for i, (rv, tv) in enumerate(plan):
            cv2.imwrite(str(folder / f"{kind}_{i:02d}.png"), render(tex, org, rv, tv, rng))
        print(f"{kind}: {len(plan)} кадрів → {folder}")
    (OUT / "truth.json").write_text(json.dumps(
        {"K": K.tolist(), "dist_k1_k2_p1_p2_k3": D.tolist(), "image_size": [W, H]}, indent=2))
    print(f"істина: {OUT / 'truth.json'}")
