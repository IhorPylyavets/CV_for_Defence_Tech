#!/usr/bin/env python
"""
VisDrone-VID для заняття 5 (усі три демо) — в artifacts/visdrone/.

    python tools/fetch_visdrone.py --splits train val test-dev  # усе, що потрібно ноутбукам заняття 5, ≈ 11 ГБ

    python tools/fetch_visdrone.py                              # val + test-dev, ≈ 3,6 ГБ
    python tools/fetch_visdrone.py --splits val                 # лише val, ≈ 1,5 ГБ
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="visdrone-vid", choices=["visdrone-vid", "own", "info"])
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
    elif args.dataset == "own":
        (ARTIFACTS / "aerial_video").mkdir(parents=True, exist_ok=True)
        print(OWN)
    else:
        print(INFO)


if __name__ == "__main__":
    main()
