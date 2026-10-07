"""KITTI odometry, послідовності 00 і 05, для заняття 3 — з Google Drive курсу в artifacts/kitti/.

    core      сірі стерео image_0/1 (усі 4541 кадр), calib.txt, times.txt, poses/00.txt  ≈ 2,4 ГБ   demo2, demo3
    velodyne  перші 500 сканів LiDAR                                                      ≈ 0,9 ГБ   demo1
    vocab     05: ліва сіра image_0 (2761 кадр), calib.txt, times.txt, poses/05.txt     ≈ 0,7 ГБ   demo3, словник BoW
    reference траєкторії-еталони KISS-ICP і ORB-SLAM3 stereo (необов'язково)               ≈ 2 МБ     demo3

Це копія офіційних даних (https://www.cvlibs.net/datasets/kitti/eval_odometry.php, ліцензія
CC BY-NC-SA 3.0 — лише некомерційне використання), перепакована tools/pack_kitti.py, щоб не
качати десятки гігабайт заради двох послідовностей.

Вже розпаковані частини пропускаються, тож скрипт можна перезапускати. Обірване завантаження
починається спочатку.

Запуск з кореня репозиторію:  python tools/fetch_kitti.py [--no-lidar] [--keep-zip]
"""
import argparse
import re
import shutil
import sys
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

# Посилання «Усі, хто має посилання» з Google Drive (можна вставити повне посилання або лише id файлу).
PARTS = {
    "core": dict(file="kitti00_core.zip", drive="1cFwzOa_woW17J1RfydhyJmShrVKiagpE",
                 done="sequences/00/image_1/004540.png"),
    "velodyne": dict(file="kitti00_velodyne_500.zip", drive="1EOiVPF7TXVZWJXSvV2Ut-tks9xEx6zE2",
                     done="sequences/00/velodyne/000499.bin"),
    "vocab": dict(file="kitti05_vocab.zip", drive="https://drive.google.com/file/d/1di0PtKWpRR73Ee8R6UEdHk-Gg3U4vmXE/view?usp=drive_link", optional=True,
                  done="sequences/05/image_0/002760.png"),
    # необов'язкова: без посилання просто пропускається; збирає викладач (demo3 + tools/orbslam3_kitti.sh)
    "reference": dict(file="kitti_reference.zip", drive="", optional=True,
                      done="reference/kiss_icp_00.txt"),
}
OUT = Path(__file__).resolve().parent.parent / "artifacts" / "kitti"
DOWNLOAD = "https://drive.usercontent.google.com/download"
HEADERS = {"User-Agent": "Mozilla/5.0"}


def drive_id(link: str) -> str:
    m = re.search(r"/d/([\w-]{20,})|[?&]id=([\w-]{20,})", link)
    return (m.group(1) or m.group(2)) if m else link.strip()


class _Form(HTMLParser):
    # Для великих файлів Drive замість файлу віддає сторінку «не вдалося перевірити на віруси» з формою.
    def __init__(self):
        super().__init__()
        self.action, self.fields = None, {}

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "form" and "download" in (a.get("action") or ""):
            self.action = a["action"]
        elif tag == "input" and a.get("name"):
            self.fields[a["name"]] = a.get("value", "")


def open_drive(file_id: str):
    response = urlopen(Request(f"{DOWNLOAD}?{urlencode(dict(id=file_id, export='download', confirm='t'))}", headers=HEADERS))
    if not response.headers.get_content_type().startswith("text/html"):
        return response
    form = _Form()
    form.feed(response.read().decode("utf-8", "replace"))
    if not form.action:
        sys.exit(f"Google Drive не віддав файл {file_id}. Перевірте, що доступ відкрито «усім, хто має посилання», "
                 "і що денний ліміт завантажень файлу не вичерпано (тоді спробуйте пізніше).")
    response = urlopen(Request(f"{form.action}?{urlencode(form.fields)}", headers=HEADERS))
    if response.headers.get_content_type().startswith("text/html"):
        sys.exit(f"Google Drive знову віддав сторінку замість файлу {file_id} — спробуйте пізніше.")
    return response


def download(file_id: str, target: Path) -> None:
    tmp = target.with_suffix(".zip.part")
    with open_drive(file_id) as response, open(tmp, "wb") as f:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        while chunk := response.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            of = f" / {total / 2**30:.2f}" if total else ""
            print(f"\r  {target.name}: {done / 2**30:.2f}{of} ГБ", end="", flush=True)
    print()
    tmp.replace(target)


def extract(archive: Path) -> None:
    with zipfile.ZipFile(archive) as z:
        names = z.infolist()
        for i, info in enumerate(names, 1):
            z.extract(info, OUT)
            if i % 500 == 0 or i == len(names):
                print(f"\r  розпаковую: {i}/{len(names)}", end="", flush=True)
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--no-lidar", action="store_true", help="лише core, без сканів LiDAR (demo1 тоді не запрацює)")
    ap.add_argument("--keep-zip", action="store_true", help="не видаляти архіви після розпакування")
    args = ap.parse_args()

    staging = OUT / "_download"
    for name, part in PARTS.items():
        if name == "velodyne" and args.no_lidar:
            continue
        if (OUT / part["done"]).exists():
            print(f"{name}: вже є, пропускаю")
            continue
        if not part["drive"] and part.get("optional"):
            print(f"{name}: посилання не вказано, пропускаю (необов'язкова частина)")
            continue
        if not part["drive"]:
            sys.exit(f"У tools/fetch_kitti.py не вказано посилання на {part['file']} (PARTS['{name}']['drive']).")
        archive = staging / part["file"]
        print(f"{name}: завантажую {part['file']}")
        if not archive.exists():
            staging.mkdir(parents=True, exist_ok=True)
            download(drive_id(part["drive"]), archive)
        try:
            extract(archive)
        except zipfile.BadZipFile:
            archive.unlink()
            sys.exit(f"{part['file']} пошкоджений — видалив його, запустіть скрипт ще раз.")
        if not args.keep_zip:
            archive.unlink()
    if staging.exists() and not any(staging.iterdir()):
        shutil.rmtree(staging)
    print(f"Готово -> {OUT}")
