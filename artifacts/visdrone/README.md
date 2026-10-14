# VisDrone

Дані для занять 5 і 6. У git не потрапляють — лише цей README.

```
python tools/fetch_visdrone.py --splits train val test-dev  # усе для ноутбуків заняття 5, ≈ 11 ГБ
python tools/fetch_visdrone.py                              # VID val + test-dev, ≈ 3,6 ГБ
python tools/fetch_visdrone.py --splits val                 # лише VID val, ≈ 1,5 ГБ (заняття 6, демо 3)
python tools/fetch_visdrone.py --dataset visdrone-det       # DET val, 548 кадрів, ≈ 80 МБ (заняття 6, демо 1–2)
```

Після розпакування: `VisDrone2019-VID-<split>/sequences/<seq>/*.jpg` і `.../annotations/<seq>.txt` (розмітка з ID треків
і рівнем перекриття); `VisDrone2019-DET-val/images/*.jpg` і `.../annotations/*.txt`.
Ліцензія VisDrone — для академічного використання.

Якщо Google Drive відповідає «Too many users have viewed or downloaded this file recently» — це квота Drive
на сам файл. Завантажте архів у браузері й покладіть сюди як `VisDrone2019-VID-<split>.zip` (не розпаковуючи),
потім перезапустіть скрипт. Або зробіть копію на свій Drive і передайте `--mirror <split>=<посилання>`.
