#!/usr/bin/env python
"""
Ваги CLIP / SigLIP 2 / RemoteCLIP / SkyCLIP для заняття 5 (демо 2 і 3) — в artifacts/models/clip/.

    python tools/fetch_clip_models.py                       # усі чотири моделі, base
    python tools/fetch_clip_models.py --size large
    python tools/fetch_clip_models.py --models clip siglip2     # лише для lesson05_demo2_clip_basics
    python tools/fetch_clip_models.py --to-safetensors      # SkyCLIP/RemoteCLIP -> .safetensors

Після запуску в artifacts/models/clip/manifest.json лежать шляхи й SHA-256 усіх файлів:
їх можна закріпити й перевіряти в air-gapped контурі.
Кеш HF (ваги й токенайзер SigLIP 2) теж кладемо туди, тож тека самодостатня:
ноутбуки бачать маніфест і самі переходять в офлайн (panels.setup_hf).

Запуск з кореня репозиторію.
"""
import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "lessons" / "05-vit-multimodal"))
import panels as A  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["clip", "siglip2", "remoteclip", "skyclip"], choices=list(A.CLIP_ZOO))
    ap.add_argument("--size", default="base", choices=["base", "large"])
    ap.add_argument("--to-safetensors", action="store_true",
                    help="зберегти чекпойнти RemoteCLIP/SkyCLIP у safetensors поруч з оригіналом")
    ap.add_argument("--trust-pickle", action="store_true",
                    help="дозволити torch.load(weights_only=False), якщо файл містить не лише тензори")
    args = ap.parse_args()

    mdir = A.CLIP_DIR
    mdir.mkdir(parents=True, exist_ok=True)
    # Windows без Developer Mode: HF кешує копіями замість symlink — працює, лише займає більше місця.
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    manifest_f = mdir / "manifest.json"
    manifest = json.loads(manifest_f.read_text()) if manifest_f.exists() else {}

    for name in args.models:
        key = f"{name}-{args.size}"
        spec = A.CLIP_ZOO[name][args.size]
        print(f"\n=== {key}: {spec}")
        files = []
        if "pretrained" not in spec:
            ckpt = A.resolve_checkpoint(name, args.size, mdir, download=True)
            files.append(ckpt)
            if args.to_safetensors and ckpt.suffix != ".safetensors":
                sd = A.load_state_dict_safely(ckpt, trust_pickle=args.trust_pickle)
                st = A.save_safetensors(sd, ckpt.with_suffix(".safetensors"))
                print(f"  safetensors: {st}")
                files.append(st)
        # Повна ініціалізація: тягне ваги SigLIP 2 і токенайзери в кеш, перевіряє, що все вантажиться.
        enc = A.ClipEncoder(name, args.size, models_dir=mdir, device="cpu", trust_pickle=args.trust_pickle)
        print(f"  OK: {enc}")
        manifest[key] = dict(spec=spec, arch=enc.arch, dim=enc.dim, n_params=enc.n_params,
                             files=[dict(path=Path(os.path.relpath(f, REPO)).as_posix(), sha256=A._sha256(Path(f)))
                                    for f in files])

    manifest_f.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\nManifest: {manifest_f}")
    print("Ноутбуки заняття 5 тепер працюють офлайн (HF_HUB_OFFLINE=1 вмикається автоматично).")


if __name__ == "__main__":
    main()
