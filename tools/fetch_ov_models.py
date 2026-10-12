#!/usr/bin/env python
"""
Open-vocabulary детектори й VLM для заняття 6 (демо 2 і 3) — в artifacts/models/ov/ і artifacts/models/ultralytics/.

    python tools/fetch_ov_models.py                                  # GDINO-T/B, MM-GDINO-T, OWLv2, Florence-2, YOLOE, YOLO-World, YOLO11
    python tools/fetch_ov_models.py --models gdino-base owlv2 yoloe  # мінімум для демо 2
    python tools/fetch_ov_models.py --models qwen2.5-vl              # опційно: VLM-grounding, ≈ 7.5 ГБ

Ultralytics-моделі тягнуть ще й текстовий енкодер: YOLOE — MobileCLIP (mobileclip_blt.ts, ≈ 570 МБ),
YOLO-World — CLIP ViT-B/32. Усе кладеться в artifacts/models/ultralytics/, а не в теку, звідки запущено ноутбук.

Після запуску в artifacts/models/ov/manifest.json лежать файли й SHA-256. Запуск з кореня репозиторію.
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "lessons" / "06-zero-shot-segmentation"))
import panels as Z  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=Z.OV_DEFAULT, choices=list(Z.OV_ZOO))
    args = ap.parse_args()
    failed = Z.fetch_models(args.models, Z.OV_DIR)
    print("Ноутбуки заняття 6 тепер працюють офлайн (HF_HUB_OFFLINE=1 вмикається автоматично).")
    if failed:
        print(f"Не завантажено: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
