# Заняття 1. Вступ до курсу та як підібрати камеру

**Блок 1:** Апаратний фундамент, просторовий аналіз та 3D

## План
- Огляд програми, формату роботи та архітектури наскрізного фінального проєкту
- Вплив заліза на алгоритми: motion blur, шуми, фотонний бюджет
- Global vs Rolling Shutter, характеристики сенсорів (матриця, FPS)
- Основи оптики: фокусна відстань, FOV, MTF, DoF, GSD

## Результат для студента
Вмію обґрунтовано підібрати оптику та сенсор під конкретні технічні вимоги.

## Демо

| Ноутбук | Про що | Потрібно |
|---|---|---|
| [demo1](lesson_01_demo1_digital_vs_optical_zoom.ipynb) | Фокусна відстань і поле зору; цифровий зум проти другого сенсора з теле-лінзою | базове середовище |
| [demo2, частина 1](lesson_01_demo2_part1_detection_range.ipynb) | Калькулятор дальності детекції: рівняння проєкції в обидва боки | базове середовище |
| [demo2, частина 2](lesson_01_demo2_part2_detection_threshold.ipynb) | Поріг детекції справжнім YOLO; resize проти tiled inference проти теле | `ultralytics`, ваги в `artifacts/models/`, бажано `ipyevents` |

У ноутбуках — **лише обчислення**: формули оптики, імітація дистанції, нарізка на тайли.
Віджети й графіки — у [`panels.py`](panels.py), формули передаються туди аргументами.
Змінив формулу → перезапусти її клітинку й клітинку з панеллю: панель бере функції в момент
запуску. Змінив `panels.py` → перезапусти ядро.

## Обмеження моделей у демо

- **Прямолінійна (rectilinear) оптика**: `tan(FOV/2) = sensor / (2f)`. Для fisheye
  (`r ∝ θ`) модель не працює.
- **Імітація дистанції суто геометрична**: з відстанню зникають пікселі, але не додаються
  серпанок, турбулентність, розмиття оптики й шум. Тому виміряний поріг детекції —
  **оптимістична верхня межа**, а не обіцянка для поля.

## Джерела

**Оптика та модель камери**
- R. Szeliski. [*Computer Vision: Algorithms and Applications*](https://szeliski.org/Book/), 2nd ed., 2022 —
  розд. 2 «Image formation»: pinhole-модель, фокусна відстань, FOV. PDF безкоштовний для особистого використання.
- J. Kannala, S. S. Brandt. [A Generic Camera Model and Calibration Method for Conventional, Wide-Angle, and Fish-Eye Lenses](https://doi.org/10.1109/TPAMI.2006.153).
  IEEE TPAMI 28(8), 2006 — чому `tan(FOV/2) = sensor / (2f)` не працює для fisheye і що замість неї.

**Сенсор**
- [EMVA 1288](https://www.emva.org/standards-technology/emva-1288/), Release 4.0, 2021 — стандарт характеристик
  камер і сенсорів: квантова ефективність, шуми, динамічний діапазон. Корисний, щоб порівнювати даташити.
- C.-K. Liang, L.-W. Chang, H. H. Chen. [Analysis and Compensation of Rolling Shutter Effect](https://doi.org/10.1109/TIP.2008.925384).
  IEEE TIP 17(8), 2008 — звідки береться викривлення rolling shutter і як його компенсують.
- [Raspberry Pi Camera: документація](https://www.raspberrypi.com/documentation/accessories/camera.html) —
  режими IMX477 (HQ Camera) з demo2: 4056×3040, 2028×1520, 1332×990.

**Дрібні об'єкти та дальність детекції**
- T.-Y. Lin et al. [Microsoft COCO: Common Objects in Context](https://arxiv.org/abs/1405.0312), ECCV 2014.
  Поділ на small / medium / large (small — площа менше 32² px) задано в
  [метриках COCO](https://cocodataset.org/#detection-eval).
- G. Cheng et al. [Towards Large-Scale Small Object Detection: Survey and Benchmarks](https://arxiv.org/abs/2207.14096).
  IEEE TPAMI 45(11), 2023 — огляд методів для дрібних об'єктів і датасети SODA-D / SODA-A.
- F. C. Akyon, S. O. Altinuc, A. Temizel. [Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection](https://arxiv.org/abs/2202.06934),
  ICIP 2022 — tiled inference (SAHI) з demo2, частина 2. Код: [obss/sahi](https://github.com/obss/sahi).
- [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/) — детектор із demo2, частина 2.
