"""Стандартні знімки шахівниці з репозиторію OpenCV (Apache 2.0) в artifacts/images/calibration/ (заняття 2).

13 кадрів 640×480 однієї камери, шахівниця 9×6 внутрішніх кутів — збігається з
CORNERS у lesson_02_demo3_calibration.ipynb. Розмір клітинки невідомий, але на K і дисторсію
він не впливає. ChArUco тут немає — лише шахівниця.

Запуск з кореня репозиторію:  python tools/fetch_opencv_calibration_samples.py
"""
from pathlib import Path
from urllib.request import urlretrieve

BASE = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/"
NAMES = [f"left{i:02d}.jpg" for i in range(1, 15) if i != 10]   # left10.jpg у репозиторії немає
OUT = Path(__file__).resolve().parent.parent / "artifacts" / "images" / "calibration" / "chessboard"

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name in NAMES:
        urlretrieve(BASE + name, OUT / name)
        print(f"  {name}")
    print(f"{len(NAMES)} кадрів → {OUT}")
