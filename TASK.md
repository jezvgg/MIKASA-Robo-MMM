# TASK

## Что это за работа

Рефакторинг и слияние трёх линий работы над сбором данных MIKASA:

- `wt/takeitwithhead-worktree-20260921` — текущая рабочая линия задач;
- `origin/train/vla` — линия, на которой записаны опубликованные датасеты;
- эта ветка, `tmp/season-dish-consolidation`, — место, где слияние и чистка
  выполняются.

**Основной метод — удаление кода.** Планировщики и Oracle-модули
(`planners/`, `planners/oracle/`) накопили несколько поколений ветвей: fallback'и
на случай, которого не бывает, повторные реализации одного примитива в разных
задачах, CLI-утилиты, которыми никто не пользуется, и тесты, дублирующие друг
друга. Задача — оставить один экземпляр каждого поведения, подтверждённого
прогонами, и удалить остальное, а не «улучшать» планировщик.

Границы правки: `my_scenes/`, `planners/`, `utils/`, `robots/fetch/`, `scripts/`,
`tests/`. Upstream ManiSkill — read-only. `README.md`, `BIBLE.md`,
`DATASET_CHECKLIST.md` не менялись.

## Что уже сделано

### Слияние планировщиков SeasonDish

`planners/season_dish_{candidates,paths,transfer,stance}.py` слиты в
`planners/season_dish_planner.py` (1851 строка). Удалены stance-scan
(`planners/oracle/stance_scan.py`), его CLI (`scripts/dr_stance_scan.py`),
флаг `--stance-scan` в `scripts/collection_episode.py` и тесты
(`utils/mikasa/test_stance_scan.py`).

### Разделение `planners/oracle`

Общие примитивы, нужные более чем одной задаче:

- `oracle_common.py` — статусный контракт (`say`, `fail`, `stopped_by_horizon`),
  `plan_joints`, `arm_move`, contact/keep-out, планировочный бюджет;
- `paths.py` — `straight_plan`, `execute_straight`, `curved_rest`,
  `drive_rounded_route`, `RollPathPlanner`, `ReleaseSettling`, `reference_goals`;
- `collection_solver.py` — `CollectionMotionPlanner` (13-мерный контракт,
  чётные шаги гриппера, roll-окно);
- `head_tracking.py`, `ik_chain.py`.

По-задачные модули (каждый — только своя задача):

- `oracle_same_drawer.py`, `oracle_season_dish.py`, `oracle_depth_recall.py`,
  `oracle_cabinet_retrieval.py`, `oracle_water_plants.py`, `oracle_takeitback.py`.

Удалены как невызываемые: `door_swing.py`, `path_clearance.py`, `rest_paths.py`,
`roll_paths.py`, `search_budget.py`, `settling.py`, `smooth_base.py`,
`straight_paths.py`, `upright_payload.py`, `wrist_pour.py`, `cabinet_place.py`,
`upright_cone.py`, `planners/takeitback_common.py`.

### Удалённые тесты и документация

Слиты в шесть модулей под `utils/collection/`: `test_collection_data.py`,
`test_motion_execution.py`, `test_motion_planning.py`, `test_oracle_depth_recall.py`,
`test_task_motion.py`, `test_task_scenes.py`. Удалены ~40 узких тест-файлов
(`test_roll_paths.py`, `test_rest_paths.py`, `test_smooth_base.py`,
`test_wrist_pour.py`, `test_search_budget.py` и т.д.), qualification-хелперы
(`qualify_*.py`), `audit.py`, `render_condiment_pictures.py`, `merge_replays.py`,
`utils/mikasa/point_cloud.py` и Markdown-файлы, которые они обслуживали
(`utils/DATA_CONTRACT.md`, `utils/collection/*.md`, `utils/mikasa/*.md`,
`planners/oracle/CHAIN_AND_STANCE.md`, `planners/DEPTH_RECALL_CHECKLIST.md`).

### Перенос из `origin/train/vla` (0864fb12)

Без этого валидационные сиды датасета не воспроизводятся: профили записаны с
прямым приводом «рука → объект», а не с прокси.

- `utils/mikasa/assisted_grasp.py` — прокси-драйв, `AssistedGraspConfig.hand_collides`;
- профили: `same_drawer_profile.json` 21, `depth_recall_profile.json` 6,
  `season_dish_profile.json` 26, `cabinet_search_profile.json` 19,
  `cabinet_search5_profile.json` 7;
- `my_scenes/depth_recall_v1.py` — `INSTRUCTION_VERSIONS` и `instruction_version`;
- `my_scenes/cabinet_search.py` — пятиотсечная `MikasaCabinetSearch5-v0`,
  коврик вместо диска (`my_scenes/assets/home_rug/`);
- `utils/collection/profile.py` — запись `cabinet_search5` в `TASKS` и
  `profile_instructions()` с учётом версии инструкции.

### Восстановленные guard'ы и fallback'и

- `planners/season_dish_planner.py` — проверки отказа/горизонта после планирования
  и исполнения подхода, захвата, подъёма, загрузки и док-драйва; физический промах
  возвращает Gym-кортеж, а не `-1`;
- `planners/depth_recall_v1_planner.py` — те же проверки в `grasp_ladder`,
  `release_and_clear`, `extract_one`, `restore_one`, `_solve`;
- `planners/oracle/oracle_common.py` — `plan_joints(..., line_only=...)`,
  RRT-fallback и `tries`;
- `planners/cabinet_search_planner.py` — `_opening_turn` не падает на `-1`,
  fallback `lower_the_hand`, отказ tuck/nudge завершает эпизод как missed,
  `retrace_the_door`, вернувший `None`, отдаёт закрытие «лестнице»;
- `planners/cabinet_door_bimbo.py` — восстановлена `lower_the_hand`.

## Результаты на валидационных сидах (oracle-фаза, 100 сидов)

| Задача | success | missed | no_plan | error | baseline (индекс до рефакторинга) |
|---|---|---|---|---|---|
| SameDrawer 21 | 81 | 3 | 16 | 0 | 80 / 4 / 16 / 0 |
| DepthRecall 6 | 98 | 2 | 0 | 0 | 98 / 2 / 0 / 0 |
| SeasonDish 26 (repo v2) | 83 | 1 | 16 | 0 | 83 / 0 / 17 / 0 |
| CabinetSearch5 7 | 83 | 17 | 0 | 0 | нет (сцены не было в ветке) |

Вывод: регрессий нет — множества успешных сидов совпадают с baseline, у
SameDrawer один сид добавлен (8250082). Оставшиеся `no_plan` — те же сиды, что
отказывали и до рефакторинга; чинить их не требуется.

Не покрыто:

- ~~SeasonDish из HF помечен profile 28, такого профиля нет ни здесь, ни в
  `origin/train/vla` (там 26) — эти сиды не прогнать;~~ profile 28 перенесён,
  см. следующий раздел;
- TakeItBack и Veggie не проверялись (нет collection-профиля ни в одной ветке).

## Перенос SeasonDish profile 28 и CabinetSearch5 из `train/vla` (001503f)

Что записано в `seasondish-shake-ag-1000ep` и `cabinetsearch5-ag-1000ep`,
в раскладке этой ветки.

- SeasonDish profile 28 (карточка с числом 1–9, столько встряхиваний над миской,
  затем вертикально и неподвижно): сцена, счётчик встряхиваний
  (`utils/mikasa/shake_counter.py`), карточки, профиль; `pour_and_shake` и его
  геометрия — в `oracle_season_dish.py`, изменения подхода к миске — в
  `season_dish_planner.py`; исключение D6 для финального удержания — в
  `checklist.py`. Заливка одним наклоном (profile 26) удалена: ни один профиль
  её больше не исполняет. Датасет `seasondish-ag-1000ep` пересобирается на
  `train/vla` 0864fb1.
- CabinetSearch5 profile 7: `planners/cabinet_door_fist.py` (`cab_1`),
  закрывающий кулак в `cabinet_retrieval_planner.py`, `release_and_clear` в
  `cabinet_door_bimbo.py`, взгляд одной камерой в `head_tracking.py`. Вызовы
  `door_swing` идут к копиям, встроенным в `cabinet_door_bimbo.py`.
- Функции шкафов возвращены к записывающим версиям: в вычищенных менялся
  записанный эпизод. `grasp_the_bar` терял шум точки захвата ручки (первое
  расхождение около 200-го шага в каждом из 20 проверенных эпизодов),
  `fold_arm_to_rest` — `tries=3` и `trace_rrt=False`, `plan_joints` —
  `tries`/`line_only`, `retrace_the_door` — подъезд базы, другие — проверки
  горизонта. Вместе с ними вернулись их помощники и `oracle_common.fold_via_tcp`
  (запасной путь `go_home`). Мёртвым при значениях по умолчанию осталось
  удалённым: выравнивание стойки (`STANCE_TURN` 0), `close_by_the_bar`
  (`CLOSE_BY_BAR` выключен), открывание не через BIMBO.

Проверка на поде RunPod (5 воркеров, его квота CPU):

| Проверка | До переноса (эта ветка) | После |
|---|---|---|
| CabinetSearch5, эпизоды 5, 55, …, 955: побитово с `train/vla` на том же поде (действия, состояния, инструкция) | — (кулака нет) | 20/20 |
| SeasonDish 28, те же эпизоды: побитово с `train/vla` | профиля нет | 20/20 |
| DepthRecall: побитово с этой веткой до переноса | — | 20/20 |
| SameDrawer: с этой веткой до переноса | — | 15/20 побитово, 5/20 `no_plan` на обеих |
| CabinetSearch5, критерий этой ветки: успехи oracle-фазы на эпизодах 0, 10, …, 990 | 73/100 | 100/100* |
| Тесты `utils/collection/` (без медленного) | 252 passed, 4 skipped | 270 passed, 4 skipped |

\* 99 на первом прогоне; сотый сид (7921509) упал на `common.pull_hinge_arc`,
которую эта ветка перенесла в `oracle_cabinet_retrieval`. Вызов исправлен, повтор —
успех. CabinetSearch5 и SeasonDish 28 побитово и тесты сняты на итоговом коммите;
DepthRecall, SameDrawer и 99 из 100 сидов — на предыдущем, который отличался только
этим вызовом и отсутствием `oracle_common.fold_via_tcp`.

Промежуточный вариант — кулак перенесён, функции шкафов вычищенные — дал 0/20:
все эпизоды расходятся на 192–270-м шаге. На этом поде `train/vla` совпадает с HF побитово для
CabinetSearch5 и SeasonDish 28 (20/20), поэтому «с `train/vla`» здесь значит
«с опубликованными эпизодами». На этой ветке до переноса из 27 промахов 19 —
отказ подталкивания банки (8 в `cab_1`), 8 — дверь не закрылась.

Критерий «множества успешных сидов совпадают» не видит изменения траектории:
DepthRecall здесь совпадает с `train/vla` побитово в 13 эпизодах из 20,
SameDrawer — в 2 из 20, хотя оба успешны. Для датасетов, которые надо
пересобирать, нужна побитовая проверка.

Уже было на этой ветке до переноса, не трогал: `door_from_cfg` не определён
(`door_rad_now` и `close_the_door` в `cabinet_retrieval_planner.py`, вызов без `door`),
`counter_grasp_chains` и `entry_chain_rank` в `depth_recall_v1_planner.py`,
`common.CARRY_YAWS_DEG` в `water_plants_planner.py`.

## Проверки

- `253 passed, 4 skipped` — focused-набор `utils/collection/test_*.py`;
- `git diff --check` чистый;
- инвентарь: 70 файлов / 35 651 строка в `my_scenes` + `planners` + `robots`
  (было 164 файла / 30 909 строк production-кода в начале работы).

## Инструменты

`scripts/tree_loc.py` — дерево проекта с LOC:

```bash
python3 scripts/tree_loc.py                      # my_scenes planners robots
python3 scripts/tree_loc.py --git planners       # с пометками git status
python3 scripts/tree_loc.py --assets robots      # с бинарными ассетами
```

## Как продолжать

1. Не менять upstream ManiSkill и не трогать `README.md`/`BIBLE.md`/
   `DATASET_CHECKLIST.md` без отдельного решения.
2. Любое удаление подтверждать прогоном или ссылкой на baseline: код, который
   не исполнялся ни в одном профиле, удаляется; код, который исполнялся, — нет.
3. Проверять регрессии только сравнением с baseline на одних и тех же сидах,
   а не по абсолютному числу успехов.
4. Запускать через `uv`, окружение `.venv-sim`, `MS_ASSET_DIR=$PWD/.venv-sim/maniskill-assets`,
   `OMP_NUM_THREADS=1`, все `MIKASA_*` сбрасывать перед прогоном.
