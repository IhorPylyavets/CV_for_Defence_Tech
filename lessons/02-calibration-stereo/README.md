# Заняття 2. Калібрування камер та стереозір

**Блок 1:** Апаратний фундамент, просторовий аналіз та 3D

## План
- Intrinsic / extrinsic параметри, матриця проекції, моделі дисторсії
- Епіполярна геометрія, ректифікація, RANSAC
- Побудова карт глибини (disparity) зі стереопари

## Результат для студента
Розумію математику формування зображення, усуваю дисторсію, рахую глибину сцени.

## Демо

| Демо | Про що | До слайдів |
|---|---|---|
| [demo1](lesson_02_demo1_pinhole_demo.html) | Камера обскура: повзунки `K` і пози камери, та сама сцена в 3D і в кадрі 1280×720, живі `K`, `[R \| t]` і `M = K[R \| t]`, шлях однієї точки від світу до пікселя; чому `t = −R·C` — не положення камери | — |
| [demo2](distortion_demo.html) | Модель Брауна — Конраді з OpenCV (`k1, k2, k3, p1, p2`, пресети лінз): куди лінза зсуває точку, чому «виправити» тією самою формулою не можна, 5 ітерацій `undistortPoints` проти збіжності, справжнє поле зору і де поліном «складається» | «Radial distortion», «Fisheye» |
| [demo3](lesson_02_demo3_calibration.ipynb) | Чотири типові помилки калібрування в цифрах на віртуальній камері з відомою `K`: замало знімків, дошка лише в центрі, фронтальні кадри, довіра до RMS; паспорт калібрування | «Пайплайн калібрування», «Типові помилки калібрування» |
| [demo4](lesson_02_demo4_disparity_methods.ipynb) | Шість методів disparity (SAD, Census, BM, SGBM, SGBM + L–R, RAFT) на **реальних** парах Middlebury з істиною; крива вартості в одному пікселі; штрафи SGBM; `doffs` і закон Z² | «Алгоритми», «Де стереозір ламається», «Епіполюси на нескінченності», «Ректифікація на практиці» |

**demo1 і demo2** — самодостатні HTML-сторінки: досить відкрити файл у браузері, сервер
не потрібен. Без інтернету лише шрифт IBM Plex заміниться системним.

**demo3 і demo4** — ноутбуки, ядро **Python (cv-advanced)**, OpenCV 5.

demo4 бере стереопари Middlebury 2014 з `artifacts/middlebury/`: з кореня репозиторію
`python tools/fetch_middlebury.py` (≈45 МБ; з `--full` ще й повна `Motorcycle-perfect`,
≈105 МБ, — вона додає варіант «інше освітлення»). Деталі — у
[`artifacts/middlebury/README.md`](../../artifacts/middlebury/README.md).
RAFT іде з `torchvision`, ваги (~20 МБ) завантажаться самі в `artifacts/models/torch`.
Віджети й малювання demo4 живуть у [`panels.py`](panels.py), а формули приходять туди
аргументами: їх можна правити просто в клітинці й одразу перезапускати. Після правок
самого `panels.py` потрібен перезапуск ядра. demo3 самодостатній і `panels.py` не імпортує.

## demo3: калібрування з відомою відповіддю

Для реальної камери істинної `K` не знає ніхто, тому demo3 знімає **віртуальною**
камерою: `K` і дисторсію задано, кадри шахівниці й ChArUco рендеряться, а детектор кутів
і `cv2.calibrateCameraExtended` — справжні. Метрика одна на весь ноутбук — **промах
променя** в мрад (1 мрад = 1 м на 1 км; при `f = 1000 px` 1 px ≈ 1 мрад).

Кожен розділ бере свій генератор випадкових чисел (`reseed`), тож числа відтворюються
й при перезапуску окремої клітинки. Повний прогін — ~30 с. Паспорт калібрування
записується в `artifacts/calibration_demo/` (у git не потрапляє).

Що показує (10 кадрів на набір, медіана з 12 випадкових підмножин):

| Набір | RMS перепроекції, px | Промах RMS, мрад | `fx` − істина, px |
|---|---|---|---|
| GOOD — нахили 10–45°, по всьому кадру | 0.11 | 1.6 | −1.9 |
| MILD — нахили 2–15° | 0.06 | 1.1 | +0.3 |
| CENTER — дошка лише в центрі | 0.19 | ≈ 390 | +2.3 |
| FRONTAL — нахил < 2° | 0.10 | 72 | −177 |
| CHARUCO — частина обрізана краєм | 0.17 | 2.2 | +1.3 |

- RMS перепроекції не впорядковує калібрування за якістю: у FRONTAL він нижчий, ніж
  у GOOD, а промах у ~45 разів більший.
- Фронтальні кадри не визначають `fx`: у 12 повторах 794 ± 71 px проти 999 ± 1 у GOOD.
- На 3–4 кадрах промах утричі більший, ніж на 10+; MILD на 3 кадрах розвалюється
  (σ(`fx`) — тисячі px).
- Помірних нахилів (2–15°) вистачає, якщо кадрів багато й вони покривають увесь кадр:
  промах MILD на великих `n` навіть трохи нижчий, ніж у GOOD, бо круті нахили дають
  шумніші кути. Але розкид `fx` у MILD у 2–4 рази більший.
- Промах CENTER у кутах кадру сягає сотень мрад, і в третьому знаку він трохи
  відрізняється від прогону до прогону: там ітерація інверсії розходиться.

Як повторити на власній камері, описано в кінці ноутбука. Знімки й готові набори для
цього — в [`artifacts/images/calibration/`](../../artifacts/images/calibration/README.md).

## demo4: числа

`blockSize = 9`; хибний — якщо |Δd| > 2 px, дірка вважається помилкою.

| Сцена | SAD | Census | BM | SGBM | SGBM + L–R | RAFT |
|---|---|---|---|---|---|---|
| Motorcycle | 21 % | 5 % | 22 % | 12 % | 20 % | 7 % |
| Motorcycle, інша експозиція | 85 % | 5 % | 23 % | 23 % | 47 % | 7 % |
| Motorcycle, інше освітлення (`--full`) | 78 % | 21 % | 52 % | 42 % | 54 % | 14 % |
| Motorcycle, правий зсунуто на 2 px | 61 % | 67 % | 73 % | 36 % | 45 % | 7 % |
| Shelves (слабка текстура) | 50 % | 36 % | 50 % | 34 % | 39 % | 9 % |
| Pipes (повтор) | 25 % | 12 % | 26 % | 16 % | 25 % | 14 % |

- RAFT вимірює заданий вертикальний зсув з точністю ±0.05 px, а на `Playtable` з реальною
  ректифікацією знаходить −0.27 px залишку (`dyavg=0.24` у `calib.txt` сцени).
- Серед знайденого L–R перевірка знижує частку хибних: на Vintage у SGBM їх 18 %,
  у SGBM + L–R — 6 %.
- Без `doffs` дальність на Motorcycle завищена на 60–280 %.
- RAFT на CPU рахує ~2–15 с на пару, далі результат береться з кешу
  `artifacts/middlebury/cache`. З теплим кешем увесь ноутбук виконується за ~30 с.

## Що можна додати

- Справжні стерео-мережі (RAFT-Stereo, IGEV) замість RAFT для потоку. Їх немає
  в `torchvision`, потрібен окремий репозиторій з вагами.
- Монокулярна глибина (Depth Anything) на тих самих сценах: відносна глибина без масштабу,
  для контрасту зі стерео.

## Джерела

**Модель камери та проекція**
- R. Hartley, A. Zisserman. [*Multiple View Geometry in Computer Vision*](https://www.robots.ox.ac.uk/~vgg/hzbook/),
  2nd ed., Cambridge University Press, 2004 — розд. 6 «Camera Models»: `K`, `[R | t]`, `P = K[R | t]`,
  `t = −R·C` (demo1); розд. 9–11 — епіполярна геометрія й фундаментальна матриця.
- R. Szeliski. [*Computer Vision: Algorithms and Applications*](https://szeliski.org/Book/), 2nd ed., 2022 —
  розд. 2 «Image formation», розд. 11 (калібрування), розд. 12 «Depth estimation» (стерео).
  PDF безкоштовний для особистого використання.

**Дисторсія та калібрування**
- D. C. Brown. Decentering Distortion of Lenses. *Photogrammetric Engineering* 32(3), 1966 —
  радіальна й тангенціальна модель, яку реалізує OpenCV (`k1, k2, k3, p1, p2`, demo2).
- J. Kannala, S. S. Brandt. [A Generic Camera Model and Calibration Method for Conventional, Wide-Angle, and Fish-Eye Lenses](https://doi.org/10.1109/TPAMI.2006.153).
  IEEE TPAMI 28(8), 2006 — модель fisheye, на яку переходять, коли поліном Брауна «складається».
- Z. Zhang. [A Flexible New Technique for Camera Calibration](https://doi.org/10.1109/34.888718).
  IEEE TPAMI 22(11), 2000 — калібрування за кількома кадрами плоскої мішені; основа
  `cv2.calibrateCamera` (demo3), зокрема чому фронтальні кадри вироджені.
- S. Garrido-Jurado et al. [Automatic generation and detection of highly reliable fiducial markers under occlusion](https://doi.org/10.1016/j.patcog.2014.01.005).
  Pattern Recognition 47(6), 2014 — маркери ArUco, з яких складається ChArUco.
- OpenCV: [Camera Calibration](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html),
  [ChArUco Board Detection](https://docs.opencv.org/4.x/df/d4a/tutorial_charuco_detection.html),
  [модуль calib3d](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html) — формули моделі,
  `calibrateCameraExtended`, `undistortPoints`.

**Стерео та disparity**
- D. Scharstein, R. Szeliski. [A Taxonomy and Evaluation of Dense Two-Frame Stereo Correspondence Algorithms](https://doi.org/10.1023/A:1014573219977).
  IJCV 47, 2002 — вартість відповідності, агрегація, WTA: та сама схема, що в demo4.
- R. Zabih, J. Woodfill. Non-parametric Local Transforms for Computing Visual Correspondence. ECCV 1994 —
  Census-трансформ, стійкий до зміни експозиції.
- H. Hirschmüller. [Stereo Processing by Semiglobal Matching and Mutual Information](https://doi.org/10.1109/TPAMI.2007.1166).
  IEEE TPAMI 30(2), 2008 — SGM; у OpenCV це `StereoSGBM` зі штрафами `P1`, `P2`.
- Z. Teed, J. Deng. [RAFT: Recurrent All-Pairs Field Transforms for Optical Flow](https://arxiv.org/abs/2003.12039).
  ECCV 2020 — потік, яким demo4 рахує disparity; ваги з [torchvision](https://pytorch.org/vision/stable/models/raft.html).

**Дані**
- D. Scharstein et al. High-Resolution Stereo Datasets with Subpixel-Accurate Ground Truth. GCPR 2014 —
  [Middlebury 2014](https://vision.middlebury.edu/stereo/data/scenes2014/) і
  [MiddEval3](https://vision.middlebury.edu/stereo/eval3/): пари з істиною, `calib.txt` з `doffs` (demo4).

**Що можна додати**
- L. Lipson, Z. Teed, J. Deng. [RAFT-Stereo: Multilevel Recurrent Field Transforms for Stereo Matching](https://arxiv.org/abs/2109.07547). 3DV 2021.
- G. Xu et al. [Iterative Geometry Encoding Volume for Stereo Matching](https://arxiv.org/abs/2303.06615) (IGEV-Stereo). CVPR 2023.
- L. Yang et al. [Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data](https://arxiv.org/abs/2401.10891). CVPR 2024.
