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
  `season_dish_profile.json` 26 (с тех пор перенесён на 28, см. ниже),
  `cabinet_search5_profile.json` 7 (профиль 19 четырёхотсечной задачи и её
  датасет `cabinetsearch-ag-1000ep` сняты, см. ниже);
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

### SeasonDish: число на карточке и встряхивания (profile 28)

Задача переопределена в `origin/train/vla` и перенесена сюда точечно, без слияния
её планировщика (там всё ещё `season_dish_{candidates,paths,transfer,stance}.py`):

- `my_scenes/season_dish.py` — карточка с числом 1–9 рядом с картинкой
  кондимента, счётчик встряхиваний в `evaluate`, `hold_steps` для финального
  удержания, `INSTRUCTION_VERSIONS` 1/2/3 (по умолчанию 3);
- `my_scenes/assets/season_dish/digit_1..9.png` + `tools/make_digit_cards.py` —
  сама карточка и генератор её текстур;
- `utils/mikasa/shake_counter.py` — `step_shakes`, `lean_from_down_deg`,
  `step_upright_hold`: встряхивание считается, когда шейкер вернулся к наклону
  `<= shake_return_deg`;
- `planners/oracle/wrist_pour.py` — `plan_shakes` (четыре ноги `in/out/back/home`,
  все планируются до первого движения) и `pour_and_shake`; из одного и того же
  модуля наш `preview_pour` берёт окно, которое обязан оставить hover;
- `utils/collection/season_dish_profile.json` 28 (`horizon 2000`, `cue_steps 50`,
  `hold_steps 10`, `shake_*`, `digit_card_*`), `utils/collection/checklist.py`
  (50-шаговый cue и исключение финального удержания), `utils/fridge_picture.py`
  (`z_offset`, `name` — вторая группа картинок на двери),
  `utils/collection/test_season_shakes.py`;
- `planners/season_dish_planner.py` — `preview_pour` через
  `wrist_pour.plan_shakes`, число встряхиваний из `info["num_shakes"]`, финал -
  `pour_and_shake`; `pour_with_wrist` и `tilt_angles` удалены вместе с тестами
  one-tip-наклона: остался один путь налива.

### Снята четырёхотсечная задача cabinet search

Канон — пятиотсечная `MikasaCabinetSearch5-v0` (профиль 7, датасет
`cabinetsearch5-ag-1000ep`, пересобран 20/20 идентично); четырёхотсечная
(`MikasaCabinetSearch-v0`, профиль 19, датасет `cabinetsearch-ag-1000ep`)
снята как superseded (владелец зафиксировал пять отсеков 2026-10-25):

- `my_scenes/cabinet_search.py` — с `CabinetSearchTask` снят
  `@register_env("MikasaCabinetSearch-v0", ...)`; класс остался общим телом
  для `CabinetSearch5Task`, `CabinetSearchConfig` — базой
  `CabinetSearch5Config`;
- `utils/collection/profile.py` — из `TASKS` убрана запись `cabinet_search`,
  `load_profile()` по умолчанию грузит `cabinet_search5`, `PROFILE`
  указывает на `cabinet_search5_profile.json`;
- удалён `utils/collection/cabinet_search_profile.json`;
- `utils/collection/pipeline.py` (`--profile` по умолчанию), тесты
  (`test_collection_data.py`, `test_task_scenes.py`), `utils/mikasa/preflight.py`,
  `utils/mikasa/test_assisted_grasp.py` (пять листьев вместо четырёх),
  `utils/validate_seeded_replay.py`, `README.md` — переведены на
  `cabinet_search5`;
- `tools/collection_gangway/driver/*` не тронуты: это архив кампании на H100
  (пути `~/sd-collect/repo-cab`), он описывает снятый датасет, а не запускает env.

## Результаты на валидационных сидах (oracle-фаза, 100 сидов)

Сиды 0–99, `MIKASA_IK_BACKEND=mplib`, восемь параллельных процессов.

| Задача | success | missed | no_plan | error | baseline (индекс до рефакторинга) |
|---|---|---|---|---|---|
| SameDrawer 21 | 81 | 3 | 16 | 0 | 80 / 4 / 16 / 0 |
| DepthRecall 6 | 98 | 2 | 0 | 0 | 98 / 2 / 0 / 0 |
| SeasonDish 26 (repo v2) | 83 | 1 | 16 | 0 | 83 / 0 / 17 / 0 |
| SeasonDish 28 (карточка+встряхивания) | 91 | 0 | 9 | 0 | 91 / 0 / 9 / 0 на `origin/train/vla` (`c6a429d`), те же сиды |
| CabinetSearch5 7 | 98 | 2 | 0 | 0 | 100 / 0 / 0 / 0 на `origin/train/vla` (`c6a429d`), те же сиды |

CabinetSearch5 до порта потерянных тактик давал 83/17/0/0: 17 сидов падали
`cube nudge refused` (не было подготовительной стадии «ready arm за 0.10 м до
дока, затем roll in» и ветки кулака по линии «стойка → куб») либо
`the door did not close` (`close_and_verify` не возвращал успех после удачной
попытки, а закрытие ручкой оставляло кисть на штанге, из-за чего латч
`is_open` не снимался). Остались 2 сида — 39 и 74: ручка закрыта по углу,
эпизод кончается `home credited ... any_open`; снимать кисть с закрытой штанги
нельзя — и отъезд базой, и вытягивание руки распахивают створку до ~1.07 рад,
а без этого `is_open` не снимается (нужны 10 подряд тиков с TCP > 0.25 м от
штанги).

Вывод: регрессий нет — множества успешных сидов совпадают с baseline, у
SameDrawer один сид добавлен (8250082). Оставшиеся `no_plan` — те же сиды, что
отказывали и до рефакторинга; чинить их не требуется.

Не покрыто:

- TakeItBack и Veggie не проверялись (нет collection-профиля ни в одной ветке).

Отказы SeasonDish 28 (9 `no_plan`): 5 — `no checked loaded route with a
complete bowl pour` (сиды 21, 60, 76, 94, 99), 3 — `bounded horizontal
grasp/lift/continuation refused` (41, 42, 71), 1 — сид 86 (эпизод кончился без
сообщения об отказе на 790-м шаге). Тот же набор отказов у `origin/train/vla`:
при переносе мы временно потеряли 6 сидов (11, 37, 44, 78, 79, 98) вместе с
fallback'ом hover'а `more=True` (`EXTRA_SPINS` и `rest_shifts`), он восстановлен.
Ни один отказ не вызван `preview_pour`.

## Проверки

- `266 passed, 4 skipped` — focused-набор `utils/collection/test_*.py`;
- `git diff --check` чистый;
- эпизод вручную: `scripts/collection_episode.py season_dish 435` ->
  `verdict=success steps=791` (`shake_count=2`, `pour_hold=12`);
- инвентарь: 71 файл / 36 011 строк в `my_scenes` + `planners` + `robots`
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
