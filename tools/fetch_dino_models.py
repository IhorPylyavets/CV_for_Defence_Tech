#!/usr/bin/env python
"""
Ваги DINOv2 / DINOv2-reg / DINOv3 для заняття 5 (Демо 1) — в artifacts/models/dino/.

    python tools/fetch_dino_models.py                              # dinov2, dinov2reg, dinov3 · base
    python tools/fetch_dino_models.py --size small                 # для CPU
    python tools/fetch_dino_models.py --models dinov3sat --size large

DINOv3 на Hugging Face — gated (DINOv3 License): спершу прийміть умови на сторінці моделі
(https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m), дочекайтесь підтвердження,
потім `hf auth login` або змінна HF_TOKEN. DINOv2 відкриті (Apache 2.0).

Після запуску в artifacts/models/dino/manifest.json лежать файли й SHA-256 —
ноутбук бачить, що все є, і далі працює з HF_HUB_OFFLINE=1.

Запуск з кореня репозиторію.
"""
import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "lessons" / "05-dino-clip"))
import dino_features as D  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["dinov2", "dinov2reg", "dinov3"], choices=list(D.DINO_ZOO))
    ap.add_argument("--size", default="base", choices=["small", "base", "large"])
    args = ap.parse_args()

    mdir = D.MODELS_DIR
    mdir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(D.HF_HOME))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError

    manifest_f = mdir / "manifest.json"
    manifest = json.loads(manifest_f.read_text()) if manifest_f.exists() else {}
    failed = []
    for name in args.models:
        size = args.size if args.size in D.DINO_ZOO[name] else next(iter(D.DINO_ZOO[name]))
        repo = D.repo_id(name, size)
        key = f"{name}-{size}"
        print(f"\n=== {key}: {repo}")
        try:
            local = Path(snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.txt"]))
        except GatedRepoError:
            print(f"  ✗ gated: прийміть ліцензію на https://huggingface.co/{repo} і залогіньтесь (`hf auth login`)")
            failed.append(key)
            continue
        # Перевірка: модель реально вантажиться й рахує.
        enc = D.DinoEncoder(name, size, device="cpu")
        print(f"  OK: {enc}")
        files = sorted(p for p in local.rglob("*") if p.is_file())
        manifest[key] = dict(repo=repo, patch=enc.patch, registers=enc.n_reg, dim=enc.dim, n_params=enc.n_params,
                             files=[dict(path=Path(os.path.relpath(p.resolve(), REPO)).as_posix(),
                                         sha256=D.A._sha256(p.resolve())) for p in files])

    manifest_f.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\nManifest: {manifest_f}")
    print(f"Для роботи офлайн: HF_HOME={os.environ['HF_HOME']} HF_HUB_OFFLINE=1")
    if failed:
        print(f"Не завантажено: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
