"""Стереопари Middlebury 2014 з істинною disparity для demo 3b заняття 2.

Завантажує MiddEval3 у чверті роздільності (≈45 МБ): 15 навчальних сцен ~740×500 з
еталоном disp0GT.pfm і маскою перекриттів mask0nocc.png. Тестові сцени без еталона
демо не потрібні, тож їх не розпаковуємо.

З ключем --full ще й Motorcycle-perfect у повній роздільності (≈105 МБ). Лише в ній є
правий кадр з іншою експозицією (im1E) та іншим освітленням (im1L) для тієї самої сцени.
Ноутбук зменшує її вчетверо, тож за розміром кадру вона не відрізняється від решти.

Запуск з кореня репозиторію:  python tools/fetch_middlebury.py [--full]
"""
import io
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen

QUARTER = "https://vision.middlebury.edu/stereo/submit3/zip/"
FULL = "https://vision.middlebury.edu/stereo/data/scenes2014/zip/Motorcycle-perfect.zip"
FULL_FILES = {"calib.txt", "im0.png", "im1.png", "im1E.png", "im1L.png", "disp0.pfm", "disp1.pfm"}
OUT = Path(__file__).resolve().parent.parent / "artifacts" / "middlebury"


def fetch(url: str) -> zipfile.ZipFile:
    print(f"  {url}")
    with urlopen(url) as response:
        return zipfile.ZipFile(io.BytesIO(response.read()))


def extract(archive: zipfile.ZipFile, keep) -> int:
    names = [n for n in archive.namelist() if not n.endswith("/") and keep(n)]
    for name in names:
        target = OUT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(archive.read(name))
    return len(names)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    count = 0
    for name in ("MiddEval3-data-Q.zip", "MiddEval3-GT0-Q.zip"):
        count += extract(fetch(QUARTER + name), lambda n: n.startswith("MiddEval3/trainingQ/"))
    print(f"{count} файлів → {OUT / 'MiddEval3' / 'trainingQ'}")

    if "--full" in sys.argv:
        count = extract(fetch(FULL), lambda n: Path(n).name in FULL_FILES)
        print(f"{count} файлів → {OUT / 'Motorcycle-perfect'}")
