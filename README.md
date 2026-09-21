# MIKASA-Robo-MMM

Benchmark VLA-памяти на базе ManiSkill. ManiSkill находится в `../ManiSkill` и подключается как editable-зависимость.

## Установка

```bash
cd ~/Projects/MIKASA-Robo-MMM
uv sync
uv run python -c 'import mani_skill; import my_scenes'
```

RoboCasa assets должны быть установлены в `MS_ASSET_DIR` ManiSkill. Проверка CLI:

```bash
uv run python -m planners.myrobocasa_takeitback_planner --help
```

`my_scenes` нужно импортировать до `gym.make(...)`: импорт регистрирует MIKASA environments и `ds_fetch`.

## Планировщики

| Environment | Planner module | Control mode |
|---|---|---|
| `MyRoboCasa-v1` | `myrobocasa_planner` | `pd_joint_pos` |
| `MyRoboCasa_TakeItBack-v1` | `myrobocasa_takeitback_planner` | `pd_joint_delta_pos` |
| `MyRoboCasa_TakeItBackTray-v1` | `myrobocasa_takeitback_tray_planner` | `pd_joint_delta_pos` |
| `MyRoboCasa_FridgeVeggies-v1` | `myrobocasa_fridge_veggies_planner` | `pd_joint_pos` |
| `MikasaSeasonDish-v0` | `season_dish_planner` | `pd_joint_pos` |
| `MikasaWaterPlants-v0` | `water_plants_planner` | `pd_joint_pos` |
| `MikasaCabinetRetrieval-v0` | `cabinet_retrieval_planner` | `pd_joint_pos` |
| `MikasaCabinetSearch-v0` | `cabinet_search_planner` | `pd_joint_pos` |
| `MikasaDepthRecall-v1` | `depth_recall_v1_planner` | `pd_joint_pos` |

Standalone запуск одного сида:

```bash
uv run python -m planners.<planner_module> --seed 3 --no-video
```

Например:

```bash
uv run python -m planners.myrobocasa_takeitback_tray_planner \
  --seed 3 --no-video --log-dir runs/planner_logs
```

Полный список аргументов конкретного планировщика:

```bash
uv run python -m planners.<planner_module> --help
```

Standalone-скрипты имеют разный формат выходных данных. Для единых sweep, HDF5-траекторий и текстовых trace используйте `utils.test_planner`.

## Валидация на 100 сидах

Пример для `MikasaCabinetSearch-v0`, сида `0..99`, CPU simulation/rendering и без RGB-наблюдений:

```bash
uv run python -m utils.test_planner \
  --scene MikasaCabinetSearch-v0 \
  --planner cabinet_search_planner \
  --num-episodes 100 \
  --start-seed 0 \
  --seed-step 1 \
  --scene-idx 0 \
  --control-mode pd_joint_pos \
  --obs-mode state \
  --sim-backend cpu \
  --render-backend cpu
```

Для другого задания замените `--scene` и `--planner` по таблице. Для двух `TakeItBack` используйте `--control-mode pd_joint_delta_pos`:

```bash
uv run python -m utils.test_planner \
  --scene MyRoboCasa_TakeItBackTray-v1 \
  --planner myrobocasa_takeitback_tray_planner \
  --num-episodes 100 --start-seed 0 --seed-step 1 \
  --control-mode pd_joint_delta_pos \
  --obs-mode state --sim-backend cpu --render-backend cpu
```

`--scene-idx 0` фиксирует кухню для задач, которые принимают этот аргумент; для остальных harness сообщает, что аргумент проигнорирован. GPU-вариант: уберите backend-флаги или задайте backend, доступный в установленной версии ManiSkill.

В конце sweep печатается:

```text
=== RESULTS === success HITS/100 (missed M, no plan P, errored E, truncated T)
```

Сохраняйте именно `HITS/100`, а не только процент: текущий harness не восстанавливает сцену между эпизодами, поэтому сравнивать нужно одинаковые `N`, seed range и scene. `--blind` запускает memory-free arm только у планировщиков, у которых есть параметр `blind`.

## Запись траекторий

`--traj-dir` записывает state-only trajectory: actions, `env_states`, flags и observations в режиме `state`. RGB сюда не добавляется.

```bash
uv run python -m utils.test_planner \
  --scene MyRoboCasa_TakeItBackTray-v1 \
  --planner myrobocasa_takeitback_tray_planner \
  --num-episodes 1 --start-seed 3 \
  --control-mode pd_joint_delta_pos \
  --obs-mode state --sim-backend cpu --render-backend cpu \
  --traj-dir runs/trajectories/takeitback_tray_seed3 \
  --log-dir runs/traces/takeitback_tray
```

Результат:

```text
runs/trajectories/takeitback_tray_seed3/
├── trajectory.h5       # traj_0, actions, env_states, obs, rewards/flags
└── trajectory.json     # env config, seed, control mode, episode metadata

runs/traces/takeitback_tray/seed_3/
├── events.jsonl
└── *_trajectory.csv
```

Для 100 сидов поменяйте `--num-episodes 1` на `--num-episodes 100`; все непустые эпизоды будут `traj_0`, `traj_1`, ... в одном HDF5. `--log-dir` необязателен. `--video-dir` — отдельная запись MP4, не RGB-данные в HDF5.

## Перезапись траекторий с RGB

### Один эпизод: `utils.replay_rgb`

Источник — директория с `trajectory.h5` и `trajectory.json`, созданная в режиме `--obs-mode state`:

```bash
uv run python -m utils.replay_rgb \
  runs/trajectories/takeitback_tray_seed3 \
  --output-dir runs/trajectories/takeitback_tray_seed3_rgb \
  --episode 0 \
  --robot fetch \
  --stride 10
```

Скрипт восстанавливает `env_states`, рендерит RGB каждые 10 control steps и пишет новый `trajectory.h5`/`trajectory.json`. Исходный файл не изменяется. `--stride 1` сохраняет каждый шаг. `--episode` — индекс `traj_N` в исходном HDF5.

### Все эпизоды: upstream replayer

Для перезаписи всех эпизодов одним новым HDF5 используйте upstream replay с предварительной регистрацией `my_scenes`:

```bash
uv run python -c \
  'import my_scenes; from mani_skill.trajectory.replay_trajectory import main, parse_args; main(parse_args())' \
  --traj-path runs/trajectories/takeitback_tray_seed3/trajectory.h5 \
  --use-env-states \
  --obs-mode rgb \
  --save-traj \
  --allow-failure \
  --sim-backend physx_cpu
```

Новый файл получает имя с суффиксом observation/control/backend рядом с исходным. `--use-env-states` нужен для восстановления сохранённого состояния, `--allow-failure` сохраняет также неуспешные replay. Исходная траектория не перезаписывается.

## Seeded action replay

RoboMME-style проверка запускает каждый planner в свежем CPU env, пишет `EpisodeSpec` в `<run>/planner_events.jsonl` и `trajectory.json`, затем повторяет actions на новом env без `env_states`:

```bash
uv run python -m utils.validate_seeded_replay \
  --all-planners --num-episodes 10 \
  --obs-mode none --sim-backend physx_cpu --render-backend cpu \
  --output-dir runs/seeded_replay
```

Для action replay `obs-mode=none` достаточно; state observations остаются в обычном `utils.test_planner` пути.

Результаты находятся в `runs/seeded_replay/<planner>/seed_<seed>/`; общий итог — `replay_summary.json`. `utils.episode_replay.SeededEpisodeBuilder` — action-replay builder. State replay остаётся в `utils.replay_rgb` для точного post-pass рендеринга.
