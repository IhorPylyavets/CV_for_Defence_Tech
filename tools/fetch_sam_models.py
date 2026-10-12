#!/usr/bin/env python
"""
Ваги сімейства SAM для заняття 6 (демо 1 і 3) — в artifacts/models/sam/ (+ MobileSAM і FastSAM в artifacts/models/ultralytics/).

    python tools/fetch_sam_models.py                                   # SAM ViT-B/H, HQ-SAM, SAM 2.1 T/S/B+/L, MobileSAM, FastSAM
    python tools/fetch_sam_models.py --models sam2.1-small sam2.1-large  # мінімум для демо 1 і 3 (≈ 1 ГБ)
    python tools/fetch_sam_models.py --models sam3                     # gated, SAM License

SAM 3 на Hugging Face — gated (SAM License, не Apache 2.0): прийміть умови на https://huggingface.co/facebook/sam3,
дочекайтесь підтвердження, потім `hf auth login`. Без SAM 3 демо 3 просто пропускає розділ «Композиція чи одна модель».

Після запуску в artifacts/models/sam/manifest.json лежать файли й SHA-256 — ноутбук бачить, що все є,
і далі працює з HF_HUB_OFFLINE=1. Запуск з кореня репозиторію.
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "lessons" / "06-zero-shot-segmentation"))
import panels as Z  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=Z.SAM_DEFAULT, choices=list(Z.SAM_ZOO))
    args = ap.parse_args()
    failed = Z.fetch_models(args.models, Z.SAM_DIR)
    print("Ноутбуки заняття 6 тепер працюють офлайн (HF_HUB_OFFLINE=1 вмикається автоматично).")
    if failed:
        print(f"Не завантажено: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
