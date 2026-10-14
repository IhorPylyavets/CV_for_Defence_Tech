#!/usr/bin/env python
"""
VisDrone для занять 5 (VID, усі три демо) і 6 (VID val, DET val) — в artifacts/visdrone/.

    python tools/fetch_visdrone.py --splits train val test-dev  # усе, що потрібно ноутбукам заняття 5, ≈ 11 ГБ
    python tools/fetch_visdrone.py                              # VID val + test-dev, ≈ 3,6 ГБ
    python tools/fetch_visdrone.py --splits val                 # лише VID val, ≈ 1,5 ГБ (заняття 6, демо 3)
    python tools/fetch_visdrone.py --dataset visdrone-det       # DET val, 548 кадрів, ≈ 80 МБ (заняття 6, демо 1–2)
    python tools/fetch_visdrone.py --dataset own                # підказка, куди класти власне відео
    python tools/fetch_visdrone.py --dataset info               # UAVDT / AU-AIR: де брати вручну

VisDrone-VID має покадрову розмітку (bbox + категорія), з якої ноутбук автоматично
будує ground truth для текстових запитів «truck», «bus», «tricycle» тощо.
Посилання — з README репозиторію github.com/VisDrone/VisDrone-Dataset (Google Drive).
Ліцензія VisDrone — для академічного використання; перевірте умови для свого курсу.

Вже розпаковані спліти пропускаються, тож скрипт можна перезапускати.
Архів після розпакування видаляється (--keep-zip, щоб лишити).

Запуск з кореня репозиторію.
"""
import argparse
import re
import sys
import zipfile
from pathlib import Path

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"

VISDRONE_VID = {
    # split: (google drive id, розмір)
    "train": ("1UDmtMh1k-WOns4grLKV8fYfGtlE-Eg_2", "7.53 GB"),
    "val": ("1H1JTaUTlCbncNDpGl_rQTr8TsJDpmkcx", "1.49 GB"),
    "test-dev": ("15QSZUFRDqceFG6xxe-G0_6cLP9sW9iPQ", "2.14 GB"),
}

# VisDrone2019-DET-val: дзеркало Ultralytics на GitHub Releases (той самий архів, що й офіційний на Google Drive).
VISDRONE_DET_VAL_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip"

INFO = """
Інші відкриті аеровідео (завантаження вручну, умови використання — на сторінках проєктів):

  UAVDT     — ~80 тис. кадрів, 100 послідовностей; різні висоти, ніч, туман.
              Сторінка: https://sites.google.com/view/grli-uavdt
              Покладіть теки з кадрами в artifacts/aerial_video/uavdt/<seq>/*.jpg
  AU-AIR    — ~2 год відео над трафіком з телеметрією (GPS, висота, IMU).
              Репозиторій: https://github.com/bozcani/auair-dataset
              Кадри або .mp4 — у artifacts/aerial_video/auair/

Для обох ноутбук індексує дані в режимі DATASET="own"; ground truth задається CSV
(lessons/05-vit-multimodal/gt/own_queries_template.csv), бо готових текстових описів у цих датасетів немає.
"""

OWN = """
Власне відео (рекомендований варіант для «години польоту»):
  1. Покладіть *.mp4 / *.mov у artifacts/aerial_video/   (підтеки дозволені)
  2. Скопіюйте lessons/05-vit-multimodal/gt/own_queries_template.csv -> gt/own_queries.csv і заповніть інтервали,
     де кожна ціль видно (source_id = шлях відео відносно artifacts/aerial_video без розширення).
  3. У ноутбуці: DATASET = "own"
"""


def drive_id(link: str) -> str:
    m = re.search(r"/d/([\w-]{20,})|[?&]id=([\w-]{20,})", link)
    return (m.group(1) or m.group(2)) if m else link.strip()


MANUAL = """
[fail] {sp}: Google Drive не віддав файл.
       {err}
       Найчастіше це квота Drive на популярний файл — минає за кілька годин, до доби. Варіанти:
         1. Завантажте в браузері (з увімкненим входом у Google квота часто не діє):
              https://drive.google.com/file/d/{gid}/view
            і покладіть архів сюди, не розпаковуючи:  {zpath}
            Потім перезапустіть скрипт — він розпакує.
         2. «Додати ярлик / Зробити копію» на свій Drive, увімкнути доступ за посиланням і передати:
              --mirror {sp}=<посилання на копію>
"""


def split_ready(out_dir: Path, sp: str) -> bool:
    """Спліт вважається наявним, якщо розпакування завершилось (.done) і послідовності реально лежать на диску."""
    seq_dir = out_dir / f"VisDrone2019-VID-{sp}" / "sequences"
    return (out_dir / f".{sp}.done").exists() and seq_dir.is_dir() and any(seq_dir.iterdir())


def download_visdrone(splits, out_dir: Path, keep_zip: bool, mirrors: dict):
    try:
        import gdown
    except ImportError:
        sys.exit("pip install gdown")
    out_dir.mkdir(parents=True, exist_ok=True)
    failed = []
    for sp in splits:
        gid, size = VISDRONE_VID[sp]
        gid = drive_id(mirrors.get(sp, gid))
        zpath = out_dir / f"VisDrone2019-VID-{sp}.zip"
        done_flag = out_dir / f".{sp}.done"
        if split_ready(out_dir, sp):
            print(f"[skip] {sp}: вже розпаковано")
            continue
        done_flag.unlink(missing_ok=True)  # прапорець без даних (теку видалили) — качаємо заново
        if not zpath.exists():
            print(f"[download] VisDrone-VID {sp} ({size})")
            try:
                # gdown 5 кидає FileURLRetrievalError, старіші версії повертають None
                res = gdown.download(id=gid, output=str(zpath), quiet=False)
                err = "" if res else "gdown повернув None"
            except Exception as e:  # noqa: BLE001
                res, err = None, " ".join(str(e).split())[:240]
            if not res or not zpath.exists():
                print(MANUAL.format(sp=sp, err=err, gid=gid, zpath=zpath))
                failed.append(sp)
                continue
        print(f"[unzip] {zpath.name}")
        try:
            with zipfile.ZipFile(zpath) as z:
                z.extractall(out_dir)
        except zipfile.BadZipFile:
            print(f"[fail] {sp}: {zpath} пошкоджений або це HTML-сторінка Drive замість архіву — видаліть і завантажте знову")
            failed.append(sp)
            continue
        done_flag.touch()
        if not keep_zip:
            zpath.unlink()
    if failed:
        sys.exit(f"Не завантажено: {', '.join(failed)}. Решта — в {out_dir}")
    print(f"Готово: {out_dir}")


def det_ready(out_dir: Path) -> bool:
    d = out_dir / "VisDrone2019-DET-val"
    return (d / "images").is_dir() and (d / "annotations").is_dir() and any((d / "images").iterdir())


def download_visdrone_det(out_dir: Path, keep_zip: bool):
    """VisDrone2019-DET-val: 548 кадрів, 38 759 об'єктів. Кадри різних сцен і висот — для замірів на зображеннях."""
    import urllib.request

    out_dir.mkdir(parents=True, exist_ok=True)
    if det_ready(out_dir):
        print("[skip] DET-val: вже розпаковано")
        return
    zpath = out_dir / "VisDrone2019-DET-val.zip"
    if not zpath.exists():
        print(f"[download] {VISDRONE_DET_VAL_URL}")
        tmp = zpath.with_suffix(".part")
        urllib.request.urlretrieve(VISDRONE_DET_VAL_URL, tmp)
        tmp.rename(zpath)
    print(f"[unzip] {zpath.name}")
    with zipfile.ZipFile(zpath) as z:
        z.extractall(out_dir)
    if not det_ready(out_dir):
        sys.exit(f"Неочікувана структура архіву {zpath}: чекаю VisDrone2019-DET-val/images і /annotations")
    if not keep_zip:
        zpath.unlink()
    print(f"Готово: {out_dir / 'VisDrone2019-DET-val'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="visdrone-vid", choices=["visdrone-vid", "visdrone-det", "own", "info"])
    ap.add_argument("--splits", nargs="+", default=["val", "test-dev"], choices=list(VISDRONE_VID))
    ap.add_argument("--keep-zip", action="store_true", help="не видаляти архів після розпакування")
    ap.add_argument("--mirror", nargs="+", default=[], metavar="SPLIT=LINK",
                    help="своя копія на Google Drive замість офіційної, напр. train=https://drive.google.com/file/d/.../view")
    args = ap.parse_args()
    mirrors = dict(m.split("=", 1) for m in args.mirror)
    if bad := set(mirrors) - set(VISDRONE_VID):
        ap.error(f"--mirror: невідомий спліт {bad}")
    if args.dataset == "visdrone-vid":
        download_visdrone(args.splits, ARTIFACTS / "visdrone", args.keep_zip, mirrors)
    elif args.dataset == "visdrone-det":
        download_visdrone_det(ARTIFACTS / "visdrone", args.keep_zip)
    elif args.dataset == "own":
        (ARTIFACTS / "aerial_video").mkdir(parents=True, exist_ok=True)
        print(OWN)
    else:
        print(INFO)


if __name__ == "__main__":
    main()
