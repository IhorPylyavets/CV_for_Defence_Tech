# Власне аеровідео

Відео для demo3 заняття 5 у режимі `DATASET = "own"`. У git не потрапляє — лише цей README.

- `*.mp4` / `*.mov` — сюди, підтеки дозволені.
- Кадри UAVDT — `uavdt/<seq>/*.jpg`, AU-AIR — `auair/`.
- Ground truth — `lessons/05-vit-multimodal/gt/own_queries.csv` за шаблоном `own_queries_template.csv`;
  `source_id` — шлях відео відносно цієї теки без розширення.

Підказка: `python tools/fetch_visdrone.py --dataset own` (власне відео) або `--dataset info` (UAVDT, AU-AIR).
