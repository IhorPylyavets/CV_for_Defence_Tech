# Middlebury Stereo 2014

Стереопари з істинною disparity для demo4 заняття 2 (`lesson_02_demo4_disparity_methods.ipynb`).
У git не потрапляють — лише цей README.

## Завантажити

З кореня репозиторію:

```
python tools/fetch_middlebury.py          # ≈ 45 МБ: MiddEval3, чверть роздільності
python tools/fetch_middlebury.py --full   # + Motorcycle-perfect у повній роздільності, ≈ 105 МБ
```

| Набір | Що всередині | Для чого |
|---|---|---|
| `MiddEval3/trainingQ/` | 15 навчальних сцен ~740×500: `im0.png`, `im1.png`, `calib.txt`, еталон `disp0GT.pfm`, маска перекриттів `mask0nocc.png` | усі порівняння методів у demo4 |
| `Motorcycle-perfect/` (`--full`) | та сама сцена з правим кадром за іншої експозиції (`im1E`) та іншого освітлення (`im1L`) | рядок «інше освітлення» в таблиці demo4 |

Тестові сцени MiddEval3 без еталона демо не потрібні, тому скрипт їх не розпаковує.
Повну Motorcycle-perfect ноутбук зменшує вчетверо, тож за розміром кадру вона не відрізняється від решти.

## Очікувана структура

```
artifacts/middlebury/
    MiddEval3/trainingQ/<Сцена>/{im0.png, im1.png, calib.txt, disp0GT.pfm, mask0nocc.png}
    Motorcycle-perfect/{im0.png, im1.png, im1E.png, im1L.png, calib.txt, disp0.pfm, disp1.pfm}   (лише з --full)
    cache/      результати RAFT (~2–15 с на пару на CPU); з'являється після першого запуску demo4
```

## Ліцензія та джерело

[Middlebury Stereo Datasets](https://vision.middlebury.edu/stereo/data/scenes2014/);
D. Scharstein et al., *High-resolution stereo datasets with subpixel-accurate ground truth*, GCPR 2014.
Набори відкриті для досліджень і навчання; у публікаціях автори просять посилатися на статтю.
