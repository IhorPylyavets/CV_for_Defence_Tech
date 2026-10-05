# Заняття 3. Оцінка руху та основи SLAM

**Блок 1:** Апаратний фундамент, просторовий аналіз та 3D

## План
- Хмари точок: джерела (LiDAR, стерео, ToF), вокселі й нормалі, Rerun і Open3D, датасет KITTI
- Візуальна одометрія: ознаки ORB/SIFT, ratio test, RANSAC; 2D-2D, 3D-3D (ICP), 3D-2D (PnP); масштаб, дрейф, діагностика калібрування
- SLAM: front-end / back-end, факторний граф, bundle adjustment, loop closure, маргіналізація, метрики ATE/RPE, фреймворки під Deftech-задачі

## Результат для студента
Розумію архітектурні принципи SLAM, маніпулюю просторовими даними.

## Демо

| Файл | Про що | Дані |
|---|---|---|
| [demo1](lesson03_demo1_kitti_rerun.py) | KITTI у Rerun: скан лідара, карта за відомою траєкторією, проєкція лідара в стереопару, своя траєкторія поверх GT | KITTI 00 з лідаром |
| [demo2](lesson03_demo2_vo_methods.ipynb) | Три способи оцінити рух на одних кадрах: 2D-2D, 3D-3D, PnP; масштаб, дрейф, злиття з колесами й гіроскопом | KITTI 00, перші 800 кадрів |
| [demo3](lesson03_demo3_slam_kitti.ipynb) | Свій стерео-SLAM сходинками: ключові кадри, мішок слів, граф поз; порівняння з KISS-ICP і ORB-SLAM3 | KITTI 00 повністю, 05 для словника |

Дані завантажує `python tools/fetch_kitti.py` з кореня репозиторію; що саме потрібно кожному демо,
описано в [artifacts/kitti/README.md](../../artifacts/kitti/README.md). Пакети заняття (`scipy`, `open3d`,
`rerun-sdk`, `panel`, `kiss-icp`) уже є в кореневому `requirements.txt`.

demo1 запускається з терміналу:

```
python lessons/03-odometry-slam/lesson03_demo1_kitti_rerun.py --frames 500
python lessons/03-odometry-slam/lesson03_demo1_kitti_rerun.py --frames 800 --est-poses lessons/03-odometry-slam/outputs/vo_3d2d_00.txt
```

Друга команда накладає траєкторію, яку порахував demo2. Важкі проходи demo2 і demo3 кешуються в
`outputs/` (у git не потрапляє): перший запуск demo3 на 00 триває ≈15–20 хв, далі все береться з кешу.
Віджет злиття сенсорів у demo2 живе в [`panels.py`](panels.py), формули приходять туди з ноутбука.

## Слайди → демо

| Слайди | Демо |
|---|---|
| «Бібліотека Rerun», «KITTI dataset», «Конфігурація сенсорів» | demo1 |
| «2. Оцінка відносного руху» … «Який метод коли?», «Дрейф», «ATE проти RPE» | demo2 |
| «Архітектура типової системи SLAM» … «Рекомендовані SLAM фреймворки», «ATE проти RPE» | demo3 |

## Домашнє завдання
[`homework/01-features-icp`](../../homework/01-features-icp/): ознаки, відсіювання аутлаєрів і ICP
(OpenCV + Open3D), ≈ 1,5 год. Спирається лише на слайди й не повторює демо: у демо ознаки трекаються KLT,
а Open3D та ICP не використовуються.

## Джерела
- D. Scaramuzza, F. Fraundorfer. [Visual Odometry, Part I](https://doi.org/10.1109/MRA.2011.943233) і
  [Part II](https://doi.org/10.1109/MRA.2012.2182810). IEEE Robotics & Automation Magazine, 2011–2012 — 2D-2D, 3D-3D, 3D-2D, дрейф.
- R. Mur-Artal, J. D. Tardós. [ORB-SLAM2](https://arxiv.org/abs/1610.06475), IEEE T-RO 2017; C. Campos et al.
  [ORB-SLAM3](https://arxiv.org/abs/2007.11898), IEEE T-RO 2021 — ключові кадри, мішок слів, loop closure.
- I. Vizzo et al. [KISS-ICP](https://arxiv.org/abs/2209.15397), IEEE RA-L 2023 — лідарний еталон у demo3.
- A. Geiger, P. Lenz, R. Urtasun. [Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite](https://www.cvlibs.net/publications/Geiger2012CVPR.pdf), CVPR 2012.
- J. Sturm et al. [A Benchmark for the Evaluation of RGB-D SLAM Systems](https://doi.org/10.1109/IROS.2012.6385773), IROS 2012 — ATE і RPE.
- [evo](https://github.com/MichaelGrupp/evo) — оцінка траєкторій з командного рядка.
