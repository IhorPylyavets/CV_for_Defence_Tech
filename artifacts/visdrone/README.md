# VisDrone-VID

Дані для заняття 5 (Демо 2, текстовий пошук по аеровідео). У git не потрапляють — лише цей README.

```
python tools/fetch_visdrone.py                  # val + test-dev, ≈ 3,6 ГБ
python tools/fetch_visdrone.py --splits val     # лише val, ≈ 1,5 ГБ
```

Після розпакування: `VisDrone2019-VID-<split>/sequences/<seq>/*.jpg` і `.../annotations/<seq>.txt`.
Ліцензія VisDrone — для академічного використання.

Якщо Google Drive відповідає «Too many users have viewed or downloaded this file recently» — це квота Drive
на сам файл. Завантажте архів у браузері й покладіть сюди як `VisDrone2019-VID-<split>.zip` (не розпаковуючи),
потім перезапустіть скрипт. Або зробіть копію на свій Drive і передайте `--mirror <split>=<посилання>`.
