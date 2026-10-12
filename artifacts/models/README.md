# Ваги моделей

Сюди кладемо `*.pt`, `*.onnx`, `*.engine` та інші ваги. У git не потрапляють.

| Тека | Що | Звідки |
|---|---|---|
| `dino/` | DINOv2, DINOv2-reg, DINOv3 (заняття 5) | `python tools/fetch_dino_models.py` |
| `clip/` | CLIP, SigLIP 2, RemoteCLIP, SkyCLIP (заняття 5) | `python tools/fetch_clip_models.py` |
| `sam/` | SAM ViT-B / H, HQ-SAM, SAM 2.1 T / S / B+ / L, SAM 3 (заняття 6) | `python tools/fetch_sam_models.py` |
| `ov/` | Grounding DINO T / B, OWLv2, Florence-2, Qwen2.5-VL (заняття 6) | `python tools/fetch_ov_models.py` |
| `ultralytics/` | YOLOE, YOLO-World, YOLO11, MobileSAM, FastSAM і текстові енкодери MobileCLIP / CLIP (заняття 6) | ті самі скрипти |

У `dino/`, `clip/`, `sam/` і `ov/` лежить `manifest.json` (ваги з `ultralytics/` записані в маніфести `sam/` і `ov/`) зі шляхами й SHA-256 файлів: ноутбуки за ним вирішують,
що можна працювати офлайн (`HF_HUB_OFFLINE=1`), а в закритому контурі за ним перевіряють, що ваги не підмінено.
