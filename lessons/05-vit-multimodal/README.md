# Заняття 5. Vision Transformers та мультимодальні моделі

**Блок 2:** Foundation Models та Advanced Generative AI

## План
- Від згорток до уваги: одна формула агрегації, Q/K/V, multi-head
- Vision Transformer: токени, позиції, роздільність, O(N²), Swin, розрахунок пікселів на цілі
- Навчання без міток: DINOv2 / DINOv3, робочий рецепт «архів → 200 міток → kNN», PEFT / LoRA
- Мультимодальні моделі: CLIP і SigLIP, межі zero-shot, ІЧ-домен, метрики, векторний пошук
- Застосування в DefTech: людина в контурі, три зони впевненості, вузький канал і split-computing

## Демо

| Файл | Слайд | Про що | Дані |
|---|---|---|---|
| [demo1](lesson05_demo1_dino.ipynb) | LIVE DEMO 1 (29) | DINOv2 / DINOv3 без міток: PCA патч-фіч, регістри, увага [CLS], пошук кліком по патчу, скільки міток потрібно, витік train / test | VisDrone-VID train, val, test-dev |
| [demo2](lesson05_demo2_clip_basics.ipynb) | LIVE DEMO 2 (40) | CLIP проти SigLIP 2: дві вежі, InfoNCE і сигмоїда, температура, zero-shot з класом «фон», пісочниця, батарея відмов, zero-shot проти DINOv2 | VisDrone-VID, пара RGB / LWIR з `artifacts/images` |
| [demo3](lesson05_demo3_video_search.ipynb) | LIVE DEMO 3 (44) | Пошук по відеоархіву: ключові кадри й тайли, FAISS, текстовий пошук і пошук за зразком з текстом, метрики, три зони, PQ для вузького каналу | VisDrone-VID або власне відео |

Увесь спільний код — у [`panels.py`](panels.py): дані VisDrone, моделі DINO і CLIP-сімейства, кеші, метрики, графіки й
віджети. Формули зі слайдів (косинус, InfoNCE, сигмоїдна втрата, softmax / sigmoid, злиття оцінок) рахуються в клітинках ноутбуків.

```bash
# з кореня репозиторію
./scripts/setup.sh                                          # середовище cv-advanced; пакети — у кореневому requirements.txt
python tools/fetch_visdrone.py --splits train val test-dev  # VisDrone-VID -> artifacts/visdrone/ (~9 ГБ)
python tools/fetch_dino_models.py                           # demo1, demo2: DINOv2, DINOv2-reg, DINOv3 -> artifacts/models/dino/
python tools/fetch_clip_models.py                           # demo2, demo3: CLIP, SigLIP 2, RemoteCLIP, SkyCLIP -> artifacts/models/clip/
```
- **Офлайн.** Скрипти `fetch_*` пишуть `manifest.json` із SHA-256 файлів; ноутбук бачить, що всі потрібні ваги на місці,
  і сам вмикає `HF_HUB_OFFLINE=1` (`panels.setup_hf`). Після завантаження інтернет не потрібен.
- **DINOv3 — gated** (DINOv3 License, не Apache 2.0): прийміть умови на
  [сторінці моделі](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m), `hf auth login`,
  потім `python tools/fetch_dino_models.py --models dinov3`. Без доступу ноутбуки пропускають DINOv3 з підказкою.
- **Пристрій.** `DEVICE = None` у першій клітинці — cuda / mps / cpu автоматично. demo1 на CPU бере `small`-моделі й менший кадр.
- **Кеш.** Кропи, ембеддинги й ключові кадри — в `outputs/demo1..3/` (у git не потрапляє). Перший прогін demo3 рахує
  ембеддинги ~60 тис. тайлів (кілька хвилин на GPU), далі все береться з кешу.
- **Smoke-тест без мережі** (синтетичні кадри, випадкові ваги — лише перевірка, що все запускається):
  `DEMO1_SMOKE=1 jupyter nbconvert --to notebook --execute lesson05_demo1_dino.ipynb --output smoke1.ipynb`,
  так само `DEMO3_SMOKE=1` для demo3.
  Для повного прогону через `nbconvert` додайте `PANELS_AUTORUN=0`: панелі не малюватимуть перший результат самі.
  Інакше nbclient (він імітує віджет Output) на великих картинках зависає до тайм-ауту клітинки; у JupyterLab і VS Code проблеми немає.
- **Інтерактив** — ipywidgets; клік і рамка мишею по кадру — ipyevents (обидва в `requirements.txt`).

## Слайди → демо

| Слайди | Демо |
|---|---|
| 15–17 «ViT», «Позиційне кодування», «Нова роздільність»; 19 «O(N²)» | demo1, розділ 1 |
| 22 «Розрахунок пікселів», 23 «Стратегії для високої роздільності» | demo1, розділи 1–3; demo3, розділ 1 |
| 25–28 «Дані в обороні», «DINOv2», «Колапс», «DINOv3» | demo1, розділи 1, 2, 4 |
| 30 «Робочий рецепт» | demo1, розділ 3; demo2, розділ 8 |
| 34–36 «Спільний простір», «Контрастивна втрата», «Zero-shot» | demo2, розділи 1–4 |
| 37–39 «Де CLIP помиляється», «Доменний розрив ракурсу», «Тепловізор» | demo2, розділи 4–7 |
| 41–43 «Text-to-Image», «Image+Text-to-Image», «FAISS» | demo3, розділи 3–5 |
| 46 «Як міряти мультимодальну систему» | demo1, розділи 3–4; demo2, розділи 4, 7; demo3, розділ 6 |
| 48–49 «Human-in-the-loop», «Три зони впевненості» | demo2, розділ 6; demo3, розділ 7 |
| 50–53 «Вузький канал», «Split-computing», кейс 1 | demo3, розділ 8 |

## Звідки ваги

| Модель | Джерело | Як вантажиться |
|---|---|---|
| DINOv2, DINOv2-reg, DINOv3 | HF `facebook/dinov2-*`, `facebook/dinov3-*` | transformers, кеш `artifacts/models/dino/hf_home/hub` |
| CLIP | OpenAI, `ViT-B-16` / `ViT-L-14`, `pretrained="openai"` | open_clip з `force_quick_gelu=True`: без нього open_clip 3.x тихо ставить GELU, і модель гіршає |
| SigLIP 2 | HF `timm/ViT-B-16-SigLIP2`, `pretrained="webli"` | open_clip |
| RemoteCLIP | HF `chendelong/RemoteCLIP`, `RemoteCLIP-ViT-B-32.pt` | open_clip + `load_state_dict`, як у README авторів |
| SkyCLIP | S3 авторів SkyScript, `SkyCLIP_ViT_B32_top50pct.zip` | open_clip; для ViT-L-14 — `quick_gelu=True` |

`.pt` — це pickle. Ваги вантажаться з `weights_only=True`; якщо файл містить щось, крім тензорів, код зупиниться й попросить
явно `--trust-pickle`. `python tools/fetch_clip_models.py --to-safetensors` збереже перевірену копію в safetensors.

## Дані
- **VisDrone-VID** (train 7.5 ГБ, val + test-dev ~3.6 ГБ): покадрова розмітка дає кропи для класифікації (demo1, demo2)
  і авто-GT для запитів `truck`, `bus`, `van`, `tricycle`, `bicycle`, `motorcycle`, `person`, `car` (demo3).
  Це хвилини відео, а не година: для порівняння моделей цього досить, для «години польоту» — власне відео.
- **Власне відео** для demo3: `artifacts/aerial_video/*.mp4` + `gt/own_queries.csv` (формат —
  [`gt/own_queries_template.csv`](gt/own_queries_template.csv)), у ноутбуці `DATASET = "own"`, `SAMPLE_FPS = 1.0`.
- **Пара RGB / LWIR** з заняття 1 (`artifacts/images/lesson_1_demo_1_*.jpg`) — ілюстративна, для тестів напису й полярності в demo2.
- Ліцензії VisDrone / UAVDT / AU-AIR — академічні; перевірте умови для платного курсу.

## Джерела
- A. Dosovitskiy et al. [An Image is Worth 16×16 Words](https://arxiv.org/abs/2010.11929), ICLR 2021.
- M. Oquab et al. [DINOv2](https://arxiv.org/abs/2304.07193), TMLR 2024; T. Darcet et al.
  [Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588), ICLR 2024; O. Siméoni et al. [DINOv3](https://arxiv.org/abs/2508.10104), 2025.
- A. Radford et al. [CLIP](https://arxiv.org/abs/2103.00020), ICML 2021; X. Zhai et al. [SigLIP](https://arxiv.org/abs/2303.15343), ICCV 2023;
  M. Tschannen et al. [SigLIP 2](https://arxiv.org/abs/2502.14786), 2025.
- F. Liu et al. [RemoteCLIP](https://arxiv.org/abs/2306.11029), IEEE TGRS 2024; Z. Wang et al. [SkyScript](https://arxiv.org/abs/2312.12856), AAAI 2024.
- J. Johnson, M. Douze, H. Jégou. [Billion-scale similarity search with GPUs (FAISS)](https://arxiv.org/abs/1702.08734), 2017.
- P. Zhu et al. [VisDrone](https://github.com/VisDrone/VisDrone-Dataset), IEEE TPAMI 2021.
