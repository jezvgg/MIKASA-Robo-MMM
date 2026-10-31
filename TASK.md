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

### Ветки сведены: обучение, RoboMME и legacy-планировщики

Ветка `tmp/season-dish-consolidation` теперь содержит и то, что осталось
в `origin/train/vla` (`c6a429d`), и то, что наработано на
`origin/wt/takeitwithhead-worktree-20260921`. Сливалось точечно, по файлам,
чтобы не воскресить код, удалённый на этой ветке.

Взято из `origin/train/vla` (`f46cedd`, `ed9c668`): `vla/memer`,
`vla/memoryvla`, `vla/TRAINING_DEVIATIONS.md`,
`vla/mem_guard.py`, `vla/mem_log.py`, `vla/debug_4gpu.sh`, новые
`vla/pi05_first_frame/src/mikasa_pi05/{configs,dataset,provenance,transforms}.py`,
`third_party/` (только gitlink'и: `memer` `fb6dbd0`, `Qwen3-VL` `9658872`,
`MemoryVLA` `082ce25` branch `mikasa-robo`, `openpi` `215abfb`, `robomme`
`ecf086c`, `hamlet` `63b4055`; файлов чужих репозиториев в дереве нет),
`utils/policies/{pi05,memer,hamlet,memory}_runtime.py` — по одному кадру на
политику,
`my_scenes/my_robocasa_takeit.py`, `MyRoboCasaSceneTakeItBack`/
`MyRoboCasaFridgeVeggies` на общем миксине `AssistedGraspTask`,
`planners/oracle/head_tracking.py` с `aim_one_eye`, `utils/target_visibility.py`,
`utils/tray_episode_checkers.py`, `utils/collection/test_one_branch_profiles.py`,
`tools/` целиком.

Взято из `origin/wt/takeitwithhead-worktree-20260921` (`6b09933`):
`utils/policies/` (RoboMME-слой и `recipes/author_recipes.json`), `research/`,
`OPENPI.md`, `utils/openpi_*`, `utils/serve_openpi.py`, `utils/evaluate_*`,
`utils/trajectory_metadata.py`, `utils/combine_lerobot_stream.py`,
`utils/merge_replays.py`, `robots/fetch/forward_rrt.py`,
`planners/myrobocasa_takeit_planner.py`, `planners/takeitback_common.py`,
`scripts/collect_tray_{dataset,demos}.py`, `tests/*` (14 файлов) и версии
`planners/myrobocasa_{fridge_veggies,takeitback}_planner.py`, к которым эти
тесты написаны.

Не взято: `pyproject.toml`, `uv.lock` (наш lock новее и проверен прогоном),
`BIBLE.md`, `DATASET_CHECKLIST.md`, `README.md` (наш уже упоминает
`MikasaCabinetSearch5-v0`), `utils/convert_to_lerobot_stream.py` (наш пишет
потоком с постоянной памятью), `utils/mikasa/preflight.py`,
`utils/validate_seeded_replay.py`, `planners/cabinet_door_fist.py` (вторая
копия кулачковых тактик, которые уже внутри `cabinet_search_planner.py`),
`robots/fetch/{extand,utils}.py` целиком (наш IK-бэкенд новее — порт только
нужных методов), а также 88 файлов, которые `train/vla` влила бы заново из
числа удалённых здесь (пять `season_dish_*`, три `same_drawer_*`,
`depth_recall_v1_perception.py`, 15 oracle-модулей, `season_{salad,postures}.py`,
`qualify_*.py`, `audit.py`, `merge_replays.py`, `point_cloud.py`, ~40 узких
тестов, пять одноразовых скриптов).

Три переноса не файлами: `SapienPlannerV2.pad_move_group_qpos` расширяет по
индексу сустава, а не по префиксу (Fetch чередует головные суставы, и
префиксный паддинг сдвигал их — тест падал на 7 из 11 значений);
`solver.set_execution_noise_seed`/`_apply_execution_noise` (нужны legacy
планировщикам, которые задают шум сами; при `action_noise = 0.0` это
тождество, покрытые профилями задачи не меняются); обсерверы шага в
`utils/logging_utils.py`.

Снято как лишнее: `tests/test_lerobot_quantiles.py` (импортирует
`vector_quantiles` старого конвертера) и `tests/test_assisted_grasp.py`
(ждёт `AssistedGrasp(task, objs)` без `AssistedGraspConfig`; покрытие даёт
`utils/mikasa/test_assisted_grasp.py`). `tests/test_fridge_veggies_closings.py`
переписан под то, что осталось от закрывания: раньше он проверял перебор
нескольких направлений схвата по длине винтового пути, теперь — измерение
узкой оси по мешу (`_narrow_closing_axis`).

Сабмодули: для `openpi` gitlink'а достаточно (`vla/pi05_first_frame/scripts/
setup_server.sh` клонирует его в `$OPENPI_DIR` и ставит отдельный venv,
`third_party/openpi` в `train/vla` никем не читается), а `memer`,
`Qwen3-VL` и `MemoryVLA` их же setup-скрипты **патчат и ставят `-e`** из
`third_party/`, поэтому для обучения нужен `git submodule update --init`.
В основном venv нет `jax`, `flax`, `openpi.shared`, `mme_vla_suite` —
тесты `utils/policies` проходят не импортируя их на верхнем уровне.

### Зависимости: ManiSkill из PyPI, без соседнего клона

`pyproject.toml` и `uv.lock` взяты из `origin/wt/takeitwithhead-worktree-20260921`
(коммиты `174db21` и `dc48eed`): `mani_skill==3.0.0b22` и `mplib==0.2.1`
(`override-dependencies`, потому что ManiSkill объявляет `mplib==0.1.1`), плюс
группа `openpi-eval = ["openpi-client"]` с путём
`third_party/openpi/packages/openpi-client` (ставится только при явном
запросе группы). До этого в ветке жил `[tool.uv.sources] mani_skill = { path =
"../ManiSkill", editable = true }` со времён миграции (`b2c098f`), из-за него
установка требовала соседний клон ManiSkill.

`uv sync --frozen` по этому локу ставит `mani-skill 3.0.0b22` колесом (97 MB)
и ровно те версии, что запинены в профилях: gymnasium 0.29.1, sapien 3.0.3,
torch 2.14.0, numpy 1.26.4, h5py 3.16.0, mplib 0.2.1. `load_profile()`
проходит для всех четырёх профилей, `423 passed, 6 skipped`, эпизод
`depth_recall 7` — `verdict=success steps=2742`. Соседний клон
`../ManiSkill` больше не нужен: ни один файл репозитория на него не ссылается.

## Результаты на валидационных сидах (oracle-фаза, 100 сидов)

Сиды 0–99, `MIKASA_IK_BACKEND=mplib`, восемь параллельных процессов.

| Задача | success | missed | no_plan | error | baseline (индекс до рефакторинга) |
|---|---|---|---|---|---|
| SameDrawer 21 | 98 | 1 | 1 | 0 | 79 / 2 / 19 / 0 на `origin/train/vla` (`c6a429d`) |
| DepthRecall 6 | 100 | 0 | 0 | 0 | 98 / 1 / 1 / 0 на `origin/train/vla`, те же сиды |
| SeasonDish 26 (repo v2) | 83 | 1 | 16 | 0 | 83 / 0 / 17 / 0 |
| SeasonDish 28 (карточка+встряхивания) | 96 | 0 | 4 | 0 | 91 / 0 / 9 / 0 на `origin/train/vla` (`c6a429d`), те же сиды |
| CabinetSearch5 7 | 99 | 1 | 0 | 0 | 100 / 0 / 0 / 0 на `origin/train/vla` (`c6a429d`), те же сиды |

Числа выше — прогон 2026-10-31 по сидам из HF-датасета (`<task>/validation_seeds.json`,
правило `first_100_successful_scene_seeds_in_ascending_order`): samedrawer 8 250 000+,
depthrecall 7 965 000+, seasondish 6 280 000+, cabinetsearch5 7 847 000+, по 100 сидов
на задачу, 32 воркера, `MIKASA_IK_BACKEND=mplib`. Ранее опубликованные здесь числа
(91/98/83/91/100) сняты на сидах 0–99 и больше не действуют: HF-сиды — те, на которых
собирались датасеты.

Два поведения записи, потерянные при сведении веток, восстановлены в
`robots/fetch/extand.py` (коммиты `ec025ac`, `448ed17`): суб-`1e-6` базовые
скорости снова обнуляются в `_compose`/`_from_abs`, а голова паркуется на
приводах базы — но только там, где это было при записи (`head_zero_on_hold`
включают `planners/myrobocasa_takeit_planner.py` и
`planners/myrobocasa_fridge_veggies_planner.py`; MIKASA-оракулы целят голову
сами, и общая парковка роняла DepthRecall 100→97).

CabinetSearch5 до порта потерянных тактик давал 83/17/0/0: 17 сидов падали
`cube nudge refused` (не было подготовительной стадии «ready arm за 0.10 м до
дока, затем roll in» и ветки кулака по линии «стойка → куб») либо
`the door did not close` (`close_and_verify` не возвращал успех после удачной
попытки, а закрытие ручкой оставляло кисть на штанге, из-за чего латч
`is_open` не снимался). Остались были 2 сида — 39 и 74 (диагноз снят по трассировке
`is_open`/`closed_count`): закрытие ручкой оставляет кисть на штанге, кисть
при отходе распахивает створку до **0.54 рад** (больше `theta_closed`, меньше
`theta_count`: нового `rising` нет, `reopened` тоже False), поэтому
`closed_count` компартмента так и остаётся нулём, `is_open` не снимается, а
следующий круг открывает вторую дверь — `two_open` + `skipped_home` = провал.
Обходы, которые не работают: отъезд базой по курсу (1.01 рад), отъезд по
нормали панели (1.07), подъём на 0.18 м (через раз), вытягивание руки на
0.20 м (планировщик отказывает).

Решение — «уводить кисть вдоль створки»: сила вдоль плоскости закрытой
створки не даёт момента относительно вертикальной петли (плечо и сила
параллельны), поэтому кисть сначала едет вдоль панели к петле
(`BAR_SLIDE_M = 0.20`) и сходит со штанги, потом выдерживается dwell и латч
снимается; после этого кисть опускается под нижнюю кромку створки
(`BAR_CLEAR_DROP_M = 0.35`) — снова вдоль плоскости, — чтобы следующий круг не
проносил её сквозь створку (`rotation sweep hits
...ds_fetch_r_gripper_finger_link<->...cab_main_main_group_0_hingerightdoor`,
сид 74). Итог: 100/100 на сидах 0–99, 99/100 на HF-сидах (остаётся 7 847 060,
где створка встаёт на 0.022 рад при `theta_closed = 0.01`).

Вывод: регрессий нет. Там, где строки расходятся с baseline, виноват не
рефакторинг, а потерянная тактика, и она возвращена: SameDrawer — 14 сидов
(`DRAWER_CLOSE_STARTS` с `close_starts`/`reference_ik_starts` вокруг
`approach_for_contact`, `contact_context`/`_drawer_initial_wrist_budget` как в
upstream, `move(..., contact=, contact_stretch=)`, fallback «align with apple»,
`try_grasp(..., resync_before_grasp/close_on_contact/approach_by_line)`);
CabinetSearch — порт `nudge_the_cube`, `if closed: return res, True` в
`close_and_verify`, поворот кисти под кулак и свободная ось проката.

Остающиеся отказы SameDrawer на HF-сидах: 8 250 025 (`no_plan`: после
отпускания яблока ни один из трёх кандидатов уборки открытой кисти не
планируется) и 8 250 096 (`missed`: `wait for final drawer hold waited_steps=20
reached=False`). На сидах 0–99 отказов девять (22, 30, 43, 52, 60, 64, 66, 72,
78) — строгое подмножество отказов `origin/train/vla`, где их 21.

TakeItBack и Veggie: collection-профиля нет ни в одной ветке, поэтому
эталон — команда из `collection_provenance.json` датасета
(`python -m utils.test_planner -n 1 --assisted-grasp --control-mode pd_joint_pos
--sim-backend physx_cpu`) плюс `MIKASA_AG_LEGACY_DRIVE=1`. После двух
восстановлений выше из 19 ранее падавших HF-сидов проходят 12: TakeItBack
92→96 из 100 (остаются 100006, 100098 — планировщик доходит до тарелки, но
`returned_to_initial` у сцены требует возврата чашки), Veggie 89→93 из 100
(остаются 100050, 100079, 100111, 100128, 100145 — объект доведён до тарелки
позиционно, `xy_error 0.0003`, но вращается в assisted-grasp захвате,
`dbg_av ≈ 10`, и не становится статичным).

Отказы SeasonDish 28 (9 `no_plan`): 5 — `no checked loaded route with a
complete bowl pour` (сиды 21, 60, 76, 94, 99), 3 — `bounded horizontal
grasp/lift/continuation refused` (41, 42, 71), 1 — сид 86 (эпизод кончился без
сообщения об отказе на 790-м шаге). Тот же набор отказов у `origin/train/vla`:
при переносе мы временно потеряли 6 сидов (11, 37, 44, 78, 79, 98) вместе с
fallback'ом hover'а `more=True` (`EXTRA_SPINS` и `rest_shifts`), он восстановлен.
Ни один отказ не вызван `preview_pour`.

Отказы SeasonDish 28 на HF-сидах (6280002, 6280020, 6280092, 6280095) — упор в
геометрию, а не в поиск пути: позы встряхивания требуют, чтобы ось запястья
была не дальше `90 - lean/2` от горизонтали, то есть при `shake_return_deg 10`
и `SHAKE_LEAN_MARGIN_DEG 3` — не дальше 3.5°. Замер на 6280002: наш захват
даёт 6.1°, `origin/train/vla` — 3.1°; при 6.1° `rest_angles` возвращает пустой
список и задача невыполнима ни при каком пути (проверено численно,
`planners/oracle/wrist_pour.py:rest_angles`). Это же объясняет, почему такие
сиды «плавают» между прогонами: сдвиг на микрон в записи меняет остаточный
наклон бутылки в захвате на градусы.

## Проверки

- `423 passed, 6 skipped` — весь `tests/` и `utils/collection` вместе,
  `-m "not gpu_sim"` (из них 266/4 — focused-набор `utils/collection/test_*.py`);
- `git diff --check` чистый;
- эпизод вручную: `scripts/collection_episode.py season_dish 435` ->
  `verdict=success steps=791` (`shake_count=2`, `pour_hold=12`);
- инвентарь `scripts/tree_loc.py`: 75 файлов / 39 077 строк в `my_scenes` +
  `planners` + `robots` (было 164 файла / 30 909 строк production-кода в
  начале работы; +2 650 строк добавили legacy-планировщики и сцены из
  `takeitwithhead`).

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
