# KITTI odometry

Дані для заняття 3. У git не потрапляють — лише цей README.

## Швидко: з Google Drive курсу

```
python tools/fetch_kitti.py              # ≈ 3,8 ГБ: core + 500 сканів LiDAR + 05 для словника + еталон ORB-SLAM3
python tools/fetch_kitti.py --no-lidar   # ≈ 3,1 ГБ: без сканів LiDAR (demo2, demo3)
```

| Архів | Розмір | Що всередині | Для чого |
|---|---|---|---|
| `kitti00_core.zip` | 2,2 ГБ | `image_0`, `image_1` (усі 4541 кадр), `calib.txt`, `times.txt`, `poses/00.txt` | demo2, demo3 |
| `kitti00_velodyne_500.zip` | 0,7 ГБ | `velodyne/000000–000499.bin` | demo1 |
| `kitti05_vocab.zip` | 0,7 ГБ | 05: `image_0` (усі 2761 кадр), `calib.txt`, `times.txt`, `poses/05.txt` | demo3, словник BoW |
| `kitti_reference.zip` | ≈ 2 МБ | `reference/orbslam3_stereo_00_run*.txt` — траєкторії ORB-SLAM3 | demo3, еталон (необов'язково) |

Скрипт качає архіви, розпаковує їх сюди й видаляє (`--keep-zip`, щоб лишити).
Уже наявні частини пропускає. Архіви збирає `tools/pack_kitti.py` з офіційних
даних. Кольорових камер (`image_2/3`) в архівах немає, тож у demo1 є лише сірі.

Ліцензія KITTI: CC BY-NC-SA 3.0. Для навчання підходить, для комерційного продукту ні.

## Повністю: з офіційного сайту

Завантажити: <https://www.cvlibs.net/datasets/kitti/eval_odometry.php>
(потрібна безкоштовна реєстрація).

### Що саме качати

Архіви монолітні — вибрати одну послідовність при завантаженні не можна. Тому
беремо потрібні частини й **після розпакування лишаємо тільки послідовність 00**,
решту видаляємо.

| Архів | Розмір | Для чого |
|---|---|---|
| `data_odometry_poses` | ~4 МБ | ground truth траєкторії — потрібен усім трьом демо |
| `data_odometry_calib` | ~1 МБ | `K`, стереобаза, `Tr` — потрібен усім трьом демо |
| `data_odometry_gray` | десятки ГБ | кадри для візуальної одометрії (demo 2, demo 3) |
| `data_odometry_velodyne` | десятки ГБ | хмари точок (demo 1) |

Послідовність **00** обрана навмисно: у ній машина повертається у вже пройдені
місця, тож на ній видно і дрейф одометрії, і ефект замикання петлі.

## Що потрібно кожному демо

| Демо | Дані | Звідки |
|---|---|---|
| demo1 (`lesson03_demo1_kitti_rerun.py`) | 00: `velodyne` (стільки сканів, скільки `--frames`), `calib.txt`, `poses`; кадри — за бажанням | `fetch_kitti.py` (500 сканів) |
| demo2 (`lesson03_demo2_vo_methods.ipynb`) | 00: `image_0`, `image_1`, перші 800 кадрів | `fetch_kitti.py --no-lidar` |
| demo3 (`lesson03_demo3_slam_kitti.ipynb`) | 00: уся послідовність (петлі є лише на повному маршруті) | `fetch_kitti.py --no-lidar` |
| demo3, словник BoW | 05: `image_0` — словник навчається на **іншій** послідовності | `fetch_kitti.py` (архів `kitti05_vocab.zip`); після першого запуску кеш `outputs/bow_05.npz` |
| demo3, еталон KISS-ICP (опційно) | `velodyne` усієї послідовності | офіційний `data_odometry_velodyne`; результат кешується в `outputs/` |
| demo3, еталон ORB-SLAM3 (опційно) | траєкторії `reference/orbslam3_stereo_00_run*.txt` | `fetch_kitti.py` (архів `kitti_reference.zip`) |

Для швидкого прогону на лекції в demo3 є `SEQUENCE = '07'` (коротша послідовність,
теж з петлею) — її кадри беруться з офіційного архіву.

## Очікувана структура

```
artifacts/kitti/
    sequences/00/
        velodyne/000000.bin ...
        image_0/000000.png  ...      ліва сіра камера
        image_1/000000.png  ...      права сіра камера
        calib.txt
        times.txt
    sequences/05/image_0/...         лише для словника BoW у demo3
    poses/00.txt, poses/05.txt
    reference/orbslam3_stereo_00_run*.txt   опційно, еталон ORB-SLAM3 для demo3
```

## Перевірити, що все на місці

```python
from pathlib import Path
seq = Path("artifacts/kitti/sequences/00")
for d in ("image_0", "image_1", "velodyne"):
    print(d, len(list((seq / d).glob("*"))))      # 4541, 4541, 500 (або 4541 з офіційного архіву)
print("poses", Path("artifacts/kitti/poses/00.txt").exists())
```

Демо самі скажуть, чого бракує, і не впадуть із незрозумілою помилкою.

## Чому саме KITTI

Через `poses/00.txt`. Це **справжня** ground truth траєкторія, тож дрейф
одометрії можна не ілюструвати, а вимірювати в метрах на 100 м шляху — і так
само виміряти, наскільки його прибирає замикання петлі.

## Еталон ORB-SLAM3 (для викладача)

Студентам збирати ORB-SLAM3 не треба: траєкторії приходять архівом `kitti_reference.zip`.
Щоб їх перерахувати або додати послідовність:

```
bash tools/orbslam3_kitti.sh                       # Ubuntu 22.04 або WSL2; ≈ 15–40 хв збірка + 8 хв на прогін 00
python tools/pack_kitti.py --reference-only --out <тека>   # -> kitti_reference.zip для Google Drive
```

На macOS скрипт не працює (немає `apt`, ORB-SLAM3 офіційно не збирається) і зупиняється з поясненням.
Посилання на новий архів вписується в `PARTS['reference']['drive']` у `tools/fetch_kitti.py`.
