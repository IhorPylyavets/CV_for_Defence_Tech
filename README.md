# Computer Vision для Defence Tech та автономних систем

Матеріали курсу для Middle/Senior CV Engineers, ML Engineers та AI Tech Leads.
19 занять, 7 блоків, 8 домашніх завдань, фінальний проєкт.

## Програма

Матеріали з'являються по ходу курсу: посилання в таблиці є лише в тих занять, які
вже викладені.

| № | Заняття | Блок | ДЗ |
|---|---------|------|-----|
| 1 | [Вступ до курсу та як підібрати камеру](lessons/01-camera-selection/) | 1 |  |
| 2 | [Калібрування камер та стереозір](lessons/02-calibration-stereo/) | 1 |  |
| 3 | [Оцінка руху та основи SLAM](lessons/03-odometry-slam/) | 1 | [✔](homework/01-features-icp/) |
| 4 | [3D-реконструкція: NeRF та 3D Gaussian Splatting](lessons/04-nerf-3dgs/) | 1 |  |
| 5 | [Vision Transformers та мультимодальні моделі](lessons/05-vit-multimodal/) | 2 |  |
| 6 | [Zero-shot сегментація та Open-Vocabulary детекція](lessons/06-zero-shot-segmentation/) | 2 | [✔](homework/02-grounded-sam2-video/) |
| 7 | Генеративні моделі в CV: принципи | 2 |  |
| 8 | Адаптація, керування та синтез даних | 2 | [✔](homework/03-diffusion-lora/) |
| 9 | Multi-Object Tracking та Re-Identification | 3 | ✔ |
| 10 | Відео-пайплайни та low-latency стрімінг | 3 |  |
| 11 | Крос-сенсорна інтеграція: RGB + LiDAR / Radar | 4 | ✔ |
| 12 | Мультиспектральне злиття: RGB + тепловізори | 4 |  |
| 13 | Active Learning, DVC та XAI | 5 | ✔ |
| 14 | Продакшн-сервінг моделей | 5 |  |
| 15 | Компресія нейронних мереж | 6 |  |
| 16 | Квантизація моделей | 6 | ✔ |
| 17 | Інференс на Edge-девайсах | 6 |  |
| 18 | System Design: архітектура CV-систем | 7 | ✔ |
| 19 | Захист курсових проєктів | 7 |  |

**Блоки:**
1. Апаратний фундамент, просторовий аналіз та 3D
2. Foundation Models та Advanced Generative AI
3. Advanced Tracking, стрімінг та відеоаналітика
4. Multi-Sensor Fusion
5. MLOps, Data-Centric AI та Model Serving
6. Edge Deployment та апаратна оптимізація
7. Індустріальні кейси та фінал

## Як влаштований репозиторій

```
lessons/NN-slug/   одне заняття — одна папка: README з планом, демо-ноутбуки, модулі
homework/NN-slug/  одне ДЗ — одна папка: ноутбук для студента, README, solution/ для викладача
artifacts/         ваги моделей, зображення й датасети для демо (у git — лише зображення)
scripts/           налаштування середовища
tools/             завантаження датасетів і ваг (KITTI, Middlebury, VisDrone, DINO, CLIP, SAM і OV-детектори) і допоміжні генератори даних
environment.yml    conda-середовище: лише Python 3.11 і pip
requirements.txt   усі пакети курсу, що вживаються в одному середовищі; згруповані за заняттями
```

**Спільних шарів коду немає навмисно:** коли щось почне повторюватись утретє,
тоді й винесемо. `artifacts/` — виняток і не суперечить цьому правилу: там не код,
а бінарники, і ті самі ваги потрібні кільком заняттям — дублювати їх по дев'ятнадцяти
теках немає сенсу.

**Ноутбуки містять лише підрахунки.** Віджети, малювання й обробники подій живуть
у сусідньому модулі заняття, а формули приходять у нього аргументами. Так формулу
можна правити просто в клітинці й одразу перезапускати — панель рахує саме нею.

Що не комітимо: датасети, ваги моделей, відео, зібрані `.pptx`, вивід ноутбуків.
Перед комітом очищай вивід: **Kernel → Restart Kernel and Clear Outputs of All Cells**
у JupyterLab.

## Середовище

Потрібна [Miniconda](https://docs.conda.io/en/latest/miniconda.html) або Anaconda.
Один скрипт створює conda-середовище `cv-advanced`, ставить залежності, реєструє
ядро Jupyter і наприкінці перевіряє, що імпортуються базові пакети (torch, OpenCV, ultralytics, ipywidgets).

**Windows** (PowerShell, з кореня репозиторію):

```powershell
.\scripts\setup.ps1
```

Якщо PowerShell не бачить `conda`, виконай один раз `conda init powershell` і
перезапусти термінал (або запускай з Anaconda Prompt). Якщо заборонене виконання
скриптів, дозволь його для поточного вікна:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

**Ubuntu та macOS:**

```bash
./scripts/setup.sh           # якщо файл не виконуваний: chmod +x scripts/setup.sh
```

Скрипт сам добирає збірку torch під машину: CUDA за наявності NVIDIA (на Windows —
найновішу, яку підтримує драйвер), CPU — якщо відеокарти немає, Apple MPS — на Silicon.
Повторний запуск оновлює наявне середовище й сам замінює torch, якщо стоїть не та
збірка — наприклад, CPU на машині з GPU. Чи бачить torch відеокарту, видно в розділі
«ПЕРЕВІРКА» наприкінці виводу.

Прапорці однакові на всіх платформах і пишуться з двома дефісами, зокрема
в PowerShell (`.\scripts\setup.ps1 --dry-run`):

| Прапорець | Що робить |
|---|---|
| `--dry-run` | показати команди, нічого не виконувати |
| `--force` | видалити середовище й створити з нуля |
| `--cpu` | CPU-збірка torch навіть за наявності GPU |
| `--cuda cu126` | конкретна CUDA-збірка torch |
| `--name NAME` | інша назва середовища |
| `--no-kernel` | не реєструвати ядро Jupyter |

Після встановлення:

```bash
conda activate cv-advanced
jupyter lab
```

**У ноутбуці треба обрати ядро «Python (cv-advanced)»** — інакше Jupyter візьме
base-середовище, де немає torch і ultralytics.

Без conda працює і звичайний venv на Python 3.11:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

У цьому разі збірку torch під свою відеокарту обираєш сам: на Windows `pip` з PyPI
ставить лише CPU-збірку. Команду для своєї CUDA бери з
[pytorch.org/get-started](https://pytorch.org/get-started/locally/) і виконуй **до**
`pip install -r requirements.txt`. Перед нею задай `SAM2_BUILD_CUDA=0` (Windows:
`$env:SAM2_BUILD_CUDA = "0"`): інакше SAM 2 для заняття 6 намагається скомпілювати
необов'язкове CUDA-розширення і без `nvcc` падає. `scripts/setup.*` роблять це самі.

Пакети окремих занять (наприклад, `open3d`, `scipy`, `kiss-icp` для заняття 3) теж у
кореневому `requirements.txt`, згруповані за заняттями, і ставляться тією самою командою.
`open3d` потребує Python 3.10–3.12; у `cv-advanced` (Python 3.11) він уживається з torch.
Важке й конфліктне (`diffusers`, `tensorrt`, GStreamer) ставиться окремим середовищем
під конкретні заняття — див. коментарі в кореневому `requirements.txt` і README заняття.

## Дані та ваги

Усе лежить в `artifacts/`. Зображення для демо є в репозиторії одразу після
клонування, а ваги й датасети в git не потрапляють. Що куди класти й звідки брати,
описано в README всередині:

- [`artifacts/models/`](artifacts/models/) — ваги. YOLO-ваги ultralytics докачує
  сам під час першого запуску, якщо файлу тут немає.
- [`artifacts/images/`](artifacts/images/) — фото для демо, зокрема знімки
  калібрувальних дошок для заняття 2.
- [`artifacts/middlebury/`](artifacts/middlebury/) — стереопари Middlebury 2014 для
  заняття 2: `python tools/fetch_middlebury.py` (≈45 МБ).
- [`artifacts/kitti/`](artifacts/kitti/) — KITTI odometry для демо заняття 3:
  `python tools/fetch_kitti.py` (≈3 ГБ з Google Drive курсу — завантажуй заздалегідь).
- [`artifacts/visdrone/`](artifacts/visdrone/) — VisDrone-VID для демо занять 5 і 6:
  `python tools/fetch_visdrone.py --splits train val test-dev` (≈11 ГБ; заняттю 6 досить `--splits val`),
  і VisDrone-DET-val для заняття 6: `python tools/fetch_visdrone.py --dataset visdrone-det` (≈80 МБ).
- `artifacts/models/dino/`, `artifacts/models/clip/` — DINOv2 / DINOv3 і CLIP, SigLIP 2,
  RemoteCLIP, SkyCLIP для заняття 5: `python tools/fetch_dino_models.py` і
  `python tools/fetch_clip_models.py`. Скрипти пишуть `manifest.json` із SHA-256, і після
  завантаження ноутбуки працюють офлайн. DINOv3 — gated-модель: спершу прийми ліцензію на
  Hugging Face і виконай `hf auth login` (деталі — у [README заняття](lessons/05-vit-multimodal/)).
- `artifacts/models/sam/`, `artifacts/models/ov/`, `artifacts/models/ultralytics/` — сімейство SAM (SAM, HQ-SAM,
  SAM 2.1, MobileSAM, FastSAM, gated SAM 3) і open-vocabulary детектори (Grounding DINO, OWLv2, Florence-2, YOLOE,
  YOLO-World, YOLO11, опційно Qwen2.5-VL) для заняття 6: `python tools/fetch_sam_models.py` і
  `python tools/fetch_ov_models.py`. Так само з маніфестами й офлайн-режимом
  (деталі — у [README заняття](lessons/06-zero-shot-segmentation/)).

Невеликі датасети для домашніх завдань (наприклад, Redwood і Fountain у ДЗ 3) Open3D
завантажує сам під час першого запуску в `~/open3d_data`, а не в `artifacts/`.

## Домашні завдання

Заняття 3, 6, 8, 9, 11, 13, 16, 18. Завдання лежать у [`homework/`](homework/), по теці на ДЗ;
у README кожної — кроки, бали й що здати. Виконаний ноутбук (з виводами) і файли, які просить
завдання, студенти пакують в один архів і прикріплюють його в особистому кабінеті LMS; комітити в репозиторій не потрібно.
Дедлайн — початок наступного заняття.

Якщо README завдання не каже інакше, оцінка така: коректність 40, метрики та чесність
замірів 20, якість коду 20, звіт 20. Еталонні розв'язки лежать у `homework/NN-slug/solution/`;
перед тим як відкрити репозиторій студентам, їх переносять у гілку `solutions`, яку
відкривають після дедлайну.

## Фінальний проєкт

Наскрізна робота, захист на занятті 19. Треки:

1. **Семантичний пошук зображень** — CLIP + Grounding DINO + FAISS/Qdrant, метрика Recall@K.
2. **Синтетичні дані для рідкісних класів** — SAM 2 + ControlNet + SD, доведений приріст F1.
3. **YOLO на RGB + Thermal** — 4-канальний вхід, порівняння mAP50-95 на нічній вибірці.
4. **Трекінг на Edge** — ByteTrack + ONNX/TensorRT, приріст FPS у 2–3 рази.

Контрольні точки: вибір треку після заняття 8, baseline з метриками після 13,
архітектура на 18, захист на 19.
