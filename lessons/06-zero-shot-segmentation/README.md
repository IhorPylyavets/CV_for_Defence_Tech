# Заняття 6. Zero-shot сегментація та Open-Vocabulary детекція

**Блок 2:** Foundation Models та Advanced Generative AI

## План
Лекція — 67 слайдів, три live-демо (слайди 25, 43, 55).

- Мотивація: новий клас у полі, архів без розмітки, зміна спектра — без жодного розміченого кадру
- Парадигма: zero-shot / open-vocabulary / open-set, чому foundation-модель узагальнює і де межа
- SAM: важкий енкодер і легкий декодер, неоднозначність промпта, три режими, де зникає час
- SAM 2 і відео: пам'ять, голова оклюзії, SAMURAI, SAM 2 чи класичний трекер, SAM 3
- Відкритий словник: S = O·Pᵀ, OWLv2, GLIP, Grounding DINO, YOLO-World / YOLOE, VLM-grounding
- ISR і Grounded-пайплайн: пікселі до моделі, тайлінг, ІЧ, метрики, каталог провалів
- Продакшн: машина розмітки, а не борт; робочий рецепт, бюджет каналу, матриця рішень

## Демо

| Файл | Слайд | Про що | Дані |
|---|---|---|---|
| [demo1](lesson06_demo1_sam.ipynb) | LIVE DEMO 1 (25) | SAM / SAM 2.1: пікселі до енкодера, параметри по вузлах, клік → три маски (з мишею), промпти й `mask_input`, `predicted_iou` ≠ впевненість, дрібні цілі й кроп, автоматичний режим і FastSAM, енкодер vs декодер по сімейству, T(k) | VisDrone-DET-val |
| [demo2](lesson06_demo2_ov_detectors.ipynb) | LIVE DEMO 2 (43) | OV-детектори: S = O·Pᵀ руками в OWLv2, ціна нового запиту, крапка проти коми й два пороги, пікселі на цілі, бенчмарк AP / AP_small / мс із тайлами, промпт як конфігурація (dev / test), запит зразком, VLM-grounding і пастка координат | VisDrone-DET-val |
| [demo3](lesson06_demo3_grounded_sam2_video.ipynb) | LIVE DEMO 3 (55) | Grounded-SAM 2 на 4K-відео: ключовий кадр і тайли, фільтр боксів, SAM покадрово vs SAM 2 за GT-треками, оклюзія, модель руху й корекція, вартість від n_obj vs ByteTrack, **каркас ДЗ** (розділ 6), кошики розмітки, бюджет каналу, ІЧ, SAM 3 | VisDrone-VID-val, пара RGB / LWIR з `artifacts/images` |

Увесь спільний код — у [`panels.py`](panels.py): дані VisDrone, реєстр і завантаження моделей, обгортки детекторів
з однаковим виходом `Dets`, тайлінг, метрики, запуск SAM 2 на відео, графіки й віджет кліку. Формули зі слайдів
(кількість токенів, S = O·Pᵀ, T(k), формула тайлів, GSD, T_frame, байти на об'єкт) рахуються в клітинках ноутбуків.

```bash
# з кореня репозиторію
./scripts/setup.sh                                       # середовище cv-advanced; пакети — у кореневому requirements.txt
python tools/fetch_visdrone.py --dataset visdrone-det    # demo1, demo2: VisDrone2019-DET-val -> artifacts/visdrone/ (~80 МБ)
python tools/fetch_visdrone.py --splits val              # demo3: VisDrone2019-VID-val (~1.5 ГБ; є, якщо вже робили заняття 5)
python tools/fetch_sam_models.py                         # demo1, demo3: SAM, HQ-SAM, SAM 2.1 T/S/B+/L, MobileSAM, FastSAM (~4.7 ГБ)
python tools/fetch_ov_models.py                          # demo2, demo3: GDINO-T/B, OWLv2, Florence-2, YOLOE, YOLO-World, YOLO11 (~4.8 ГБ)
python tools/fetch_ov_models.py --models qwen2.5-vl      # опційно, demo2 розділ 8 (~7.5 ГБ)
python tools/fetch_sam_models.py --models sam3           # опційно, gated — demo3 розділ 9 (~3.3 ГБ)
```
- **Офлайн.** Скрипти `fetch_*` пишуть `manifest.json` із SHA-256 файлів в `artifacts/models/{sam,ov}/`; ноутбук бачить,
  що всі потрібні ваги на місці, і сам вмикає `HF_HUB_OFFLINE=1` (`panels.setup_hf`). Ultralytics-ваги й їхні текстові
  енкодери (MobileCLIP для YOLOE, CLIP ViT-B/32 для YOLO-World) лежать в `artifacts/models/ultralytics/`, а не в теці,
  з якої запущено ноутбук (`panels.ultra_home`).
- **SAM 3 — gated** (SAM License, не Apache 2.0): прийміть умови на [сторінці моделі](https://huggingface.co/facebook/sam3),
  `hf auth login`, потім `python tools/fetch_sam_models.py --models sam3`. Без ваг demo3 пропускає розділ 9 з підказкою.
- **Пристрій.** `DEVICE = None` у першій клітинці — cuda / mps / cpu автоматично. VRAM: вистачає 12 ГБ (моделі
  вивантажуються між розділами); на 8 ГБ приберіть `sam-vit-huge` і `sam2.1-large` з `BENCH` у demo1. На CPU demo1 сам
  бере `sam2.1-tiny`; demo2 і demo3 на CPU працюють, але хвилинами на розділ.
- **Кеш.** Кадри кліпу, експорт ДЗ і всі картинки / таблиці для слайдів — в `outputs/demo1..3/` (у git не потрапляє).
  Бенчмарк demo2 (~10 хв на GPU) зберігається в `outputs/demo2/02_ov_bench.csv`, і повторний запуск з `RECOMPUTE = False`
  бере його. На лекції або показуйте збережений, або ставте `N_IMG = 30`.
- **Прогін без браузера:** `PANELS_AUTORUN=0 jupyter nbconvert --to notebook --execute lesson06_demo1_sam.ipynb --output run1.ipynb`.
  `PANELS_AUTORUN=0` вимикає перший автоматичний рендер віджета кліку в demo1, як і в заняття 5.
- **Інтерактив** — ipywidgets; клік мишею по кадру в demo1 — ipyevents (обидва в `requirements.txt`).

## Слайди → демо

| Слайди | Демо |
|---|---|
| 12–13 «SAM: архітектура по вузлах», «Image encoder: найдорожчий вузол» | demo1, розділи 1–2 |
| 14 «Скільки пікселів лишається від цілі» | demo1, розділ 1 (`01_object_size_hist.png`); demo1, розділ 5 |
| 15 «Prompt encoder і mask decoder» | demo1, розділи 2, 4 |
| 16–17 «Проблема неоднозначності», «Як SAM вирішує неоднозначність» | demo1, розділ 3 (`01_ambiguity_3masks.png`); розділ 5 |
| 18 «Чого SAM принципово не вміє» | demo1, розділи 5–6; demo3, розділ 2 |
| 19 «Сімейство похідних від SAM» | demo1, розділи 6–7 |
| 21–23 «Три режими», «Три режими в коді», «Гіперпараметри автоматичного режиму» | demo1, розділи 3, 4, 6 |
| 24 «Latency-бюджет: де зникає час» | demo1, розділ 7 (`bench_sam.csv`, `01_latency_budget.png`); demo3, розділ 5 |
| 27–29 «Відео — це не SAM покадрово», «SAM 2: архітектура з пам'яттю», «Promptable Visual Segmentation» | demo3, розділи 3–4 |
| 30 «Коли пам'ять стає проблемою: SAMURAI» | demo3, розділ 4 |
| 31 «SAM 2 чи класичний трекер?» | demo3, розділ 5 |
| 32, 54 «SAM → SAM 2 → SAM 3», «Композиція чи одна модель?» | demo3, розділ 9 |
| 34–36 «Open-vocabulary: постановка», «Чотири гілки», «OWL-ViT / OWLv2» | demo2, розділи 1–2 |
| 37–39 «GLIP», «Grounding DINO», «Механіка текстового промпта» | demo2, розділи 2–3 |
| 40 «Prompt engineering під домен» | demo2, розділ 6 (`02_prompt_variants.csv`) |
| 41 «YOLO-World і YOLOE» | demo2, розділи 2, 5, 7 |
| 42 «VLM, яка пише бокси текстом» | demo2, розділ 8 |
| 45–46 «Розрахунок до вибору моделі», «Дрібні об'єкти на великому кадрі» | demo2, розділи 4–5; demo3, розділи 1–2 |
| 47 «Не-RGB спектр» | demo3, розділ 8 |
| 49–50 «Домен-шифт і чесна оцінка», «Як чесно виміряти» | demo2, розділи 5–6 (`02_ov_bench.csv`); demo3, розділи 1, 7 |
| 51 «Каталог провалів» | demo1, розділ 5; demo2, розділи 5, 7; demo3, розділи 2–4, 8 |
| 52–53 «Grounded-SAM: класична композиція», «Композиція в 15 рядках і її latency-бюджет» | demo3, розділи 1–2, 5–6 |
| 57–58 «Правильне місце zero-shot», «Робочий рецепт» | demo3, розділ 7 |
| 60 «Бюджет каналу» | demo3, розділ 7 |
| 63 «Домашнє завдання» | demo3, розділ 6 — каркас пайплайну й експорту |

Числа на слайдах 14, 24 і 49 («? AP») ноутбуки рахують на вашому залізі; картинки й таблиці для слайдів пишуться
в `outputs/demo1..3/` з тими самими іменами файлів, що згадані на слайдах.

## Звідки ваги

| Модель | Джерело | Як вантажиться |
|---|---|---|
| SAM 2.1 T / S / B+ / L | HF `facebook/sam2.1-hiera-*`, `sam2.1_hiera_*.pt` | офіційний пакет `sam2` (`build_sam2`, API слайдів 22 і 53), кеш `artifacts/models/sam/hf_home/hub` |
| SAM ViT-B / ViT-H, HQ-SAM ViT-B | HF `facebook/sam-vit-*`, `syscv-community/sam-hq-vit-base` | transformers `SamModel`, `SamHQModel` |
| SAM 3 | HF `facebook/sam3` (gated, SAM License) | transformers `Sam3Model` |
| MobileSAM, FastSAM-s | GitHub Releases Ultralytics | ultralytics, `artifacts/models/ultralytics/` |
| Grounding DINO T / B | HF `IDEA-Research/grounding-dino-{tiny,base}` | transformers, кеш `artifacts/models/ov/hf_home/hub` |
| OWLv2 B/16 | HF `google/owlv2-base-patch16-ensemble` | transformers |
| Florence-2-L | HF `florence-community/Florence-2-large` | transformers (нативна реалізація, без `trust_remote_code`) |
| Qwen2.5-VL-3B (опційно) | HF `Qwen/Qwen2.5-VL-3B-Instruct` | transformers |
| YOLOE-11L-seg, YOLO-World v2-X, YOLO11m | GitHub Releases Ultralytics | ultralytics; текстові енкодери MobileCLIP / CLIP — туди ж |

- **`grounding-dino-base` бачив COCO в тренуванні**, тож «zero-shot на COCO» для нього не існує — demo2 порівнює з
  паперовими 48.4 AP саме `grounding-dino-tiny` (O365, GoldG, Cap4M).
- **MM-Grounding DINO** (наступник зі слайду 38) у transformers 5.19 повертає вироджені бокси нульової ширини, хоча всі ваги
  завантажуються без розбіжностей, — тому його немає в реєстрі.
- Без CUDA-розширення `sam2._C` SAM 2 лише пропускає заповнення дрібних дірок у масках; `scripts/setup.py` (його викликають
  `setup.sh` і `setup.ps1`) ставить `SAM2_BUILD_CUDA=0`, і попередження про `_C` у ноутбуках приглушене.

## Дані
- **VisDrone2019-DET-val** — 548 кадрів, 38 759 об'єктів, здебільшого 1360 × 765; для замірів на зображеннях (demo1, demo2).
  Дзеркало Ultralytics, той самий архів, що й офіційний.
- **VisDrone2019-VID-val** — покадрова розмітка з ID треків і рівнем перекриття: це дає чесні цифри для мерехтіння, оклюзій і
  перескоків у demo3. Ролик demo3 — `uav0000268_05773_v`, 4K, кадри 41–339 через один: машина з першого плану ховається за
  лісосмугою з 161-го кадру.
- **Пара RGB / LWIR** з заняття 1 (`artifacts/images/lesson_1_demo_1_*.jpg`) — ілюстративна й не зареєстрована попіксельно,
  тому ІЧ-розділ demo3 порівнює кожен кадр окремо.
- **Для ДЗ:** VisDrone2019-VID або UAV123 (RGB), HIT-UAV (ІЧ, бонус) — завантажуються вручну з офіційних сторінок.
- Ліцензії VisDrone — академічні; ваги мають різні ліцензії (SAM 3 — SAM License, YOLO — AGPL-3.0), перевірте умови
  перед використанням поза навчанням (слайд 61).

## Джерела
- A. Kirillov et al. [Segment Anything](https://arxiv.org/abs/2304.02643), ICCV 2023; L. Ke et al. [HQ-SAM](https://arxiv.org/abs/2306.01567), NeurIPS 2023.
- N. Ravi et al. [SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714), ICLR 2025;
  C.-Y. Yang et al. [SAMURAI](https://arxiv.org/abs/2411.11922), 2024; Meta AI [SAM 3: Segment Anything with Concepts](https://github.com/facebookresearch/sam3), 2025.
- C. Zhang et al. [MobileSAM](https://arxiv.org/abs/2306.14289), 2023; X. Zhao et al. [FastSAM](https://arxiv.org/abs/2306.12156), 2023.
- M. Minderer et al. [Simple Open-Vocabulary Object Detection (OWL-ViT)](https://arxiv.org/abs/2205.06230), ECCV 2022;
  [Scaling Open-Vocabulary Object Detection (OWLv2)](https://arxiv.org/abs/2306.09683), NeurIPS 2023.
- L. H. Li et al. [GLIP](https://arxiv.org/abs/2112.03857), CVPR 2022; S. Liu et al. [Grounding DINO](https://arxiv.org/abs/2303.05499), ECCV 2024.
- T. Cheng et al. [YOLO-World](https://arxiv.org/abs/2401.17270), CVPR 2024; A. Wang et al. [YOLOE](https://arxiv.org/abs/2503.07465), 2025.
- B. Xiao et al. [Florence-2](https://arxiv.org/abs/2311.06242), CVPR 2024; S. Bai et al. [Qwen2.5-VL](https://arxiv.org/abs/2502.13923), 2025.
- F. C. Akyon et al. [SAHI: Slicing Aided Hyper Inference](https://arxiv.org/abs/2202.06934), ICIP 2022; Y. Zhang et al. [ByteTrack](https://arxiv.org/abs/2110.06864), ECCV 2022.
- P. Zhu et al. [VisDrone](https://github.com/VisDrone/VisDrone-Dataset), IEEE TPAMI 2021.
