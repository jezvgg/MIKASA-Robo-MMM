# Запуск обучения: pi0.5 + первый кадр, OpenSameDrawer, 4×H100

Инструкция для того, кто запускает основное обучение на своей машине. Всё ставится без root, в одну
папку (`~/mikasa-pi05`), из GitHub и Hugging Face. Подробности о самой модели — в [README](README.md).

**Что получится.** pi0.5 (openpi, JAX) дообучается на 1000 эпизодах OpenSameDrawer
(`nurtayev-d/samedrawer-1000ep`). Модель получает 4 картинки: три текущие камеры и первый кадр эпизода.
Во время обучения каждые 1000 шагов пишется SR на 20 отдельных сидах. В конце — SR на 100
валидационных сидах и контрольный прогон, где вместо первого кадра подаётся чёрная картинка.
Ожидается SR > 80%.

## 1. Что нужно от машины

| | |
|---|---|
| GPU | 4 × H100 80 GB, NVIDIA с Vulkan: в контейнере `NVIDIA_DRIVER_CAPABILITIES` содержит `graphics` или равно `all` |
| Диск | ~175 ГБ с полными чекпойнтами (можно продолжить после обрыва) или ~90 ГБ в режиме «только веса» (см. п. 5) |
| Время | установка ~1 ч; проверка ~15 мин; обучение 30 000 шагов ~13–20 ч; итоговая оценка ~30 мин |
| Сеть | github.com, huggingface.co (+ его CDN), PyPI или его зеркало; download.pytorch.org — если драйвер старше 580 |
| ПО | git, curl, python3; `uv` поставится сам, если его нет |

## 2. Код и проверка доступа (2 минуты)

```bash
git clone -b feat/pi05-first-frame-samedrawer https://github.com/pa40l/MIKASA-Robo-MMM.git ~/mikasa-pi05/MIKASA-Robo-MMM
~/mikasa-pi05/MIKASA-Robo-MMM/vla/pi05_first_frame/scripts/setup_server.sh preflight
```

В конце должно быть `all downloads reachable`. Если какой-то адрес помечен `BLOCKED`, его нужно
открыть в сети машины. Строка `blocked storage.googleapis.com (optional ...)` не мешает: веса pi0.5
берутся с Hugging Face и сверяются по контрольным суммам с оригиналом.

## 3. Установка (~1 час, ~55 ГБ)

```bash
mkdir -p ~/mikasa-pi05/logs
nohup ~/mikasa-pi05/MIKASA-Robo-MMM/vla/pi05_first_frame/scripts/setup_server.sh \
    > ~/mikasa-pi05/logs/setup.log 2>&1 < /dev/null &
tail -f ~/mikasa-pi05/logs/setup.log        # Ctrl+C закрывает только просмотр
```

Готово, когда в конце лога `== setup finished`. Если установка упала, запустите ту же команду
снова: готовые шаги пропускаются.

Что ставится:
- openpi с правкой (две маленькие правки: 4-я картинка и своё чтение данных) и его venv;
- окружение симулятора — ровно то, на котором записан набор;
- сцены RoboCasa, набор, веса pi05_base;
- файл `~/mikasa-pi05/server.env`. Его нужно подключать перед каждой командой: `. ~/mikasa-pi05/server.env`.

## 4. Проверка машины (~15 минут)

```bash
. ~/mikasa-pi05/server.env
$REPO_DIR/vla/pi05_first_frame/scripts/check_server.sh --base 2>&1 | tee ~/mikasa-pi05/logs/check.log
cd $REPO_DIR && $SIM_VENV/bin/python -W ignore vla/pi05_first_frame/eval_samedrawer.py --policy recorded \
    --seeds train:20 --dataset-dir $SAMEDRAWER_DATASET_DIR --out $PI05_WORK/results/a5-replay > /dev/null
grep '"success_rate"' $PI05_WORK/results/a5-replay/summary.json
```

Ожидается:
- **после первой команды** — `all checks passed`. По дороге: тесты `5 passed`; рендер против набора около 1–1,5 для `live vs frame0`; pi05_base грузится и выдаёт действия;
- **после второй** — `"success_rate": 1.0` или не ниже 0.95. Это записанные действия 20 эпизодов, прогнанные через цикл оценки (пункт A5 чеклиста данных).

Если что-то из этого не так, обучение не запускайте (см. п. 7).

## 5. Обучение (одна команда)

```bash
. ~/mikasa-pi05/server.env
nohup $REPO_DIR/vla/pi05_first_frame/scripts/run_4xh100.sh run1 > ~/mikasa-pi05/logs/run1.log 2>&1 < /dev/null &
```

Скрипт делает всё сам:
- 30 000 шагов обучения на GPU 0–3;
- SR каждые 1000 шагов: проверка идёт на GPU 3 рядом с обучением;
- в конце — итоговая оценка.

Если обучение прервалось, та же команда продолжит с последнего полного чекпойнта. Они пишутся каждые 5000 шагов.

Настройки (переменные окружения перед `nohup`):

| Переменная | По умолчанию | Когда менять |
|---|---|---|
| `GPUS` | `0,1,2,3` | на машине больше 4 карт или заняты другие |
| `SR_GPU` | `3` | карта для проверок по ходу обучения (одна из `GPUS`) |
| `TRAIN_ARGS` | — | `"--params-only-checkpoints"`, если диска меньше ~175 ГБ: тогда ~90 ГБ, но после обрыва обучение начнётся заново |

## 6. Что смотреть

```bash
grep -E "Step [0-9]+:" ~/mikasa-pi05/logs/train-run1.log | tail -2   # через 10–15 мин: шаги и loss
cat ~/mikasa-pi05/logs/sr-run1.txt                                    # через ~45 мин: первая строка step 1000: SR ...
cat ~/mikasa-pi05/logs/report-run1.txt                                # итог в самом конце
```

- **`sr-run1.txt`** — SR на 20 отдельных сидах после каждых 1000 шагов. Эти сиды не входят ни в обучение, ни в валидацию. Разброс на 20 эпизодах около ±10 п.п., поэтому смотрите на тренд.
- **`report-run1.txt`** — в конце два итоговых блока: `first frame: reset` (основной SR на 100 валидационных сидах) и `first frame: black` (контроль). С чёрным кадром SR должен заметно упасть: модель не знает, какой ящик был открыт.
- **Видео:** первые 10 эпизодов итоговой оценки лежат в `~/mikasa-pi05/results/run1-final-reset/shard-*/videos/`.

Что прислать после окончания: `report-run1.txt` и `sr-run1.txt`, а также сводку `~/mikasa-pi05/results/run1-paper.md` (см. п. 7).

## 7. Что записывается (для статьи и воспроизведения)

Всё пишется само, ничего включать не нужно. Пути — относительно `~/mikasa-pi05`.

| Файл | Что в нём |
|---|---|
| `checkpoints/pi05_sd_ff_4xh100/run1/run_meta.json` | весь конфиг обучения; коммиты MIKASA-Robo-MMM и openpi, контрольная сумма правки openpi; набор (HF, ревизия, контрольные суммы), статистика нормировки, веса; версии пакетов; GPU и драйвер; пачка на карту, число эпох. При продолжении после обрыва добавляется `run_meta.resume-*.json` |
| `checkpoints/pi05_sd_ff_4xh100/run1/metrics.jsonl` | шаг, время, loss, норма градиента — каждые 100 шагов, с wandb и без |
| `logs/sr-run1.tsv` | SR по ходу обучения: шаг, успехи, 95% интервал, этапы |
| `results/run1-final-reset/`, `results/run1-final-black/` | итоговая оценка и контроль: `summary.json` (SR, интервал, разбивка по ящикам и этапам), `shard-*/episodes.jsonl` (строка на каждый сид), `shard-*/run.json` (команда, сиды, какой чекпойнт оценивался, отпечатки кода и движка, GPU), видео |
| **`results/run1-paper.md`**, `results/run1-paper.json` | **сводка для статьи**: постановка (код, данные, железо, гиперпараметры); вычисления (секунды на шаг, часы, GPU-часы); итоговый SR с интервалом, по ящикам и этапам; контроль; таблица SR по ходу обучения |

Сводка собирается в конце `run_4xh100.sh`. Пересобрать её в любой момент, в том числе по незавершённому обучению:
```bash
. ~/mikasa-pi05/server.env
python3 $REPO_DIR/vla/pi05_first_frame/scripts/collect_results.py run1
```

Метаданные самого набора (сиды, длины, награды, `success`, коммиты сбора) лежат на HF вместе с набором. `run_meta.json` ссылается на его точную ревизию.

## 8. Если что-то не так

| Что видно | Что делать |
|---|---|
| установка обрывается на загрузке пакета (`timed out`) | запустить установку ещё раз: она продолжит; можно увеличить `UV_HTTP_TIMEOUT` (по умолчанию 600) |
| `No Vulkan driver renders on the NVIDIA GPU` | в контейнере нет `graphics` в `NVIDIA_DRIVER_CAPABILITIES`; без этого симулятор не рисует |
| SAPIEN висит на первом кадре или `vk::DeviceLostError` | нода не умеет рисовать; на кластерах с `qd-gpucheck` проверить им и перейти на другую ноду |
| `torch ... cannot use CUDA with this driver` | установить заново шаг `sim`: `setup_server.sh sim` (выбирает сборку torch под драйвер) |
| `Simulator differs from the one that recorded the dataset` | окружение симулятора не то; переустановить шаг `sim`, не ставить `mani_skill` вручную |
| в отчёте `training FAILED` и `RESOURCE_EXHAUSTED` | не хватило памяти GPU; запустить с `TRAIN_ARGS="--batch-size 64"` и написать нам |
| оценка `SR 0/0` | прислать `tail -n 40 ~/mikasa-pi05/results/sr-run1/eval-*.log` (или `.../run1-final-*/logs/eval-0.log`) |
| диск заканчивается | старые снимки удаляются сами (держатся 3); на маленьком диске — `TRAIN_ARGS="--params-only-checkpoints"` |

Не задавайте переменные с префиксом `MIKASA_`: профиль симулятора их запрещает.
