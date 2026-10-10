# Демо 1 · DINOv2 vs DINOv3 на VisDrone (Заняття 5)

Фічі без жодної мітки, чотири блоки (слайди 17, 19, 22, 26–30, 46):
1. **Що бачить модель**: PCA патч-фіч, високонормні токени, увага [CLS]; DINOv2 → DINOv2 + регістри → DINOv3; patch 14 vs 16.
2. **Клікни на ціль**: косинус одного патча до всіх патчів кадрів з інших відео (ipyevents).
3. **«200 кадрів проти 2000»**: kNN і linear probe на кропах об'єктів VisDrone; випадковий vs кластерний добір міток; точність за розміром цілі.
4. **Архів як актив**: t-SNE-карта сцен, найсхожіші сцени з різних відео, витік даних при розбитті за кадром / послідовністю / відео.

```
demo1_dino.ipynb             ноутбук демо
dino_features.py             моделі (transformers), щільні фічі, панелі, кропи, kNN / probe, k-means-добір, витік
tools/fetch_dino_models.py   (у корені) ваги -> artifacts/models/dino/ + manifest.json (SHA-256)
```

```bash
# з кореня репозиторію
pip install -r lessons/05-dino-clip/requirements.txt
python tools/fetch_dino_models.py                       # dinov2, dinov2reg, dinov3 · base;  --size small для CPU
python tools/fetch_visdrone.py --splits train val test-dev   # блокам 3–4 потрібен train (7.5 ГБ)
jupyter lab lessons/05-dino-clip/demo1_dino.ipynb
```
- **Пристрій:** у першій клітинці `DEVICE = "cuda" | "cpu" | None` (авто). Без CUDA ноутбук попереджає і йде на CPU.
  На CPU за замовчуванням беруться `small`-моделі, менший кадр (672 px) і менше кропів.
- **DINOv3 — gated** (DINOv3 License, не Apache 2.0): прийміть умови на
  [сторінці моделі](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m), дочекайтесь підтвердження, `hf auth login`,
  потім `python tools/fetch_dino_models.py --models dinov3`. Без доступу ноутбук пропускає DINOv3 з підказкою.
- Кеш кропів і ембедінгів — `outputs/demo1/`. Smoke без мережі: `DEMO1_SMOKE=1 jupyter nbconvert --to notebook --execute demo1_dino.ipynb --output smoke1.ipynb`.

---

# Демо 2 · Текстовий пошук по аеровідео (Заняття 5)

SigLIP 2 vs RemoteCLIP vs SkyCLIP на тих самих ключових кадрах: FAISS-індекс, авто-GT,
метрики пошуку (EventRecall@k, P@k, AP), тайли проти цілого кадру, промпти з ракурсом,
пороги по сирому cos і PQ-коди для вузького каналу.

## Структура
```
demo2_text_search.ipynb      ноутбук демо
aerial_search.py             конвеєр: кадри, тайли, моделі, FAISS, GT, метрики, PQ, візуалізація
gt/own_queries_template.csv  формат GT для власного польоту

tools/fetch_clip_models.py   (у корені) ваги трьох моделей -> artifacts/models/clip/ + manifest.json (SHA-256)
tools/fetch_visdrone.py      (у корені) VisDrone-VID -> artifacts/visdrone/; підказки для UAVDT / AU-AIR / власного відео
```
Кеш ключових кадрів і ембеддингів ноутбук пише в `outputs/demo2/` (у git не потрапляє).

## Запуск
```bash
# з кореня репозиторію
pip install -r lessons/05-dino-clip/requirements.txt
python tools/fetch_clip_models.py                 # --size large для L-моделей
python tools/fetch_visdrone.py                    # --splits val, щоб узяти лише 1.5 ГБ
jupyter lab lessons/05-dino-clip/demo2_text_search.ipynb
```
Smoke-тест без мережі (синтетичні кадри, випадкові ваги — лише перевірка, що все запускається):
```bash
DEMO2_SMOKE=1 jupyter nbconvert --to notebook --execute demo2_text_search.ipynb --output smoke.ipynb
```

## Звідки ваги
| Модель | Джерело | Як вантажиться |
|---|---|---|
| SigLIP 2 | HF `timm/ViT-B-16-SigLIP2` через open_clip (`pretrained="webli"`) | open_clip |
| RemoteCLIP | HF `chendelong/RemoteCLIP`, `RemoteCLIP-ViT-B-32.pt` | open_clip + `load_state_dict`, як у README авторів |
| SkyCLIP | S3 авторів SkyScript, `SkyCLIP_ViT_B32_top50pct.zip` | open_clip; для ViT-L-14 — `quick_gelu=True`, як у їхньому ноутбуці |

`.pt` — це pickle. За замовчуванням ваги вантажаться з `weights_only=True`; якщо файл містить щось,
крім тензорів, код зупиниться й попросить явно `--trust-pickle`. Офлайн: `HF_HOME=artifacts/models/clip/hf_home HF_HUB_OFFLINE=1`. `--to-safetensors` збереже
перевірену копію в safetensors.

## Дані
- **VisDrone-VID val + test-dev** (~3.6 ГБ): покадрова розмітка → авто-GT для запитів `truck`, `bus`,
  `tricycle`, `van`, `bicycle`, `motorcycle`, `person`, `car`. Це хвилини відео, а не година:
  для кількісного порівняння моделей цього достатньо, для «години польоту» — власне відео.
- **Власне відео**: `artifacts/aerial_video/*.mp4` + `gt/own_queries.csv`, у ноутбуці `DATASET = "own"`, `SAMPLE_FPS = 1.0`.
- Ліцензії VisDrone / UAVDT / AU-AIR — академічні; перевірте умови для платного курсу.

## Що перевірено
Smoke-прогін усього ноутбука (синтетика + випадкові ваги): конвеєр, кеші, FAISS, метрики, PQ, графіки.
Завантаження реальних ваг і VisDrone в середовищі збірки перевірити не було як (мережа обмежена) —
перший повний прогін зробіть до заняття.
