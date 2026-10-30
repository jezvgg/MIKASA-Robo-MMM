# RoboMME / π0.5: проверенный авторский fine-tuning

Дата проверки: 2026-10-20. Это исследование статьи и исходников, не запуск обучения.

## Первичные источники

- Dai et al., **RoboMME: Benchmarking and Understanding Memory for Robotic Generalist Policies**, [arXiv:2603.04639](https://arxiv.org/abs/2603.04639), [HTML v1](https://arxiv.org/html/2603.04639v1): §4; Appendix A.1–A.3, B.3–B.4, Table7.
- Physical Intelligence, **π0.5: a Vision-Language-Action Model with Open-World Generalization**, [arXiv:2504.16054](https://arxiv.org/abs/2504.16054), §§IV-A–D. Не путать foundation pre/post-training с downstream RoboMME SFT.
- [Официальный OpenPI README](https://github.com/Physical-Intelligence/openpi): открытый pipeline π0.5 поддерживает flow-matching head; JAX допускает LoRA/full, но это не свидетельство LoRA в опытах RoboMME.
- RoboMME training repo зафиксирован на **ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b**. Небольшие source snapshots: `research/sources/robomme/`; `REVISION` фиксирует SHA.

## 1. Что именно обучали авторы

**Статья RoboMME, B.4:** заморожен vision backbone (назван SigLIP2 в тексте), визуальные признаки предварительно вычисляются; обучаются **VLM expert + action expert + все новые memory-параметры**. Это dense/full fine-tuning разрешённых модулей, **не full unfreeze абсолютно всех весов** и не LoRA. Отдельной обязательной projector-only alignment стадии для MME-VLA не описано.

**Table7:** action horizon20, proprioception=False, batch64 для symbolic/perceptual моделей, AdamW β1=.9/β2=.95, weight decay0, clip norm1, constant LR5e-5 после warmup5%, EMA=.999, 80,000 optimizer steps, сохранение каждые10,000, четыре FSDP devices. Для recurrent batch8; это не наши FrameSamp/TokenDrop. Аппаратная настройка авторов — четыре A40, примерно3–4 дня.

π0.5 baseline использует тот же принцип обучения без memory-модуля. В статье речь идёт о единой multi-task политике на RoboMME, а не о переобучении всего foundation model с нуля.

**Важное отличие исходного π0.5:** в оригинальной статье foundation pretraining использует разнородные robot/web данные и FAST-token CE; post-training добавляет action expert и continuous flow matching вместе с текстовыми задачами. Proprioception дискретизируется в текст. В RoboMME (§A.1) VLM используется как feature extractor без high-level text generation, а proprioception исключена. Ни дополнительный pretraining всей visual-language проекции, ни полный foundation recipe не требуется повторять для downstream SFT из `pi05_base`.

## 2. Какие «проекции» здесь есть

### 2.1 Обычная π0.5

У неё уже есть предобученный visual/VLM интерфейс и action adapters. В [history_pi0.py:365–389](https://github.com/RoboMME/robomme_policy_learning/blob/ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b/src/mme_vla_suite/models/integration/history_pi0.py#L365-L389):

- `action_in_proj`: action channels → action-expert width1024;
- `action_out_proj`: width1024 → action channels;
- для `pi05=True` используются time MLP; отдельный непрерывный `state_proj` в этой ветке **не создаётся** — он находится в `else` для другой архитектуры.

Поэтому «обучить state projection12→…» нельзя автоматически объявлять обязательной стадией π0.5: сначала нужно проверить дискретный state-input путь. Существующая VLM-проекция не заменяется только потому, что робот теперь Fetch.

### 2.2 FrameSamp+Modul и TokenDrop+Modul

Новые проекции необходимы для **memory pathway**, а не для переобучения всего vision-language connector:

1. В статье B.3.1: SigLIP features2048 → memory feature encoder1024.
2. Точнее в опубликованном коде: positional `pos_proj`768→768, SiLU; конкатенация с image2048; затем `encoder_static`**2816→1024** при `use_pos_emb=True,use_state_emb=False`. Это не просто один линейный2048→1024 слой. См. [mem_encoder.py:25–52,66–104](https://github.com/RoboMME/robomme_policy_learning/blob/ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b/src/mme_vla_suite/models/representation/mem_encoder.py#L25-L104).
3. В каждом action-expert слое: action-query cross-attention к memory; его результат через conditioning projection превращается в scale/shift для adaptive normalization. Статья A.3.2, eqs21–24; B.3.2: width1024, head dimension256, один KV head, примерно80M дополнительных параметров.
4. Эти новые линейные слои, cross-attention и AdaLN/modulation layers **обучаются целиком** на robot action flow-matching loss совместно с разрешёнными pretrained модулями. В статье нет отдельного contrastive или image-text loss для их предварительного alignment.

Это одинаковый тип обучаемой проекции у двух вариантов. Разница — **выбор истории**, а не full vs LoRA:

- FrameSamp:512 tokens =32 равномерно выбранных кадра ×16 spatial tokens, одна front camera.
- TokenDrop:8×8=64 tokens/frame; RGB-difference saliency с temporal stride8, threshold>1e-4, budget512; первый кадр сохраняется как context. Само правило выбора кадров/патчей не является trainable projection. Градиент идёт в последующие learned memory layers, а не в hard top-k/RGB selection.

## 3. Как именно обучать новые проекции

**Авторский рецепт:** загрузить `pi05_base`, создать новые memory modules, freeze vision encoder, вычислять causal history features из нужной frozen-версии vision encoder и минимизировать action flow-matching MSE. Градиент от action expert должен доходить в memory feature encoder и modulation blocks. Не следует `detach`/кэшировать выход **обучаемого** memory projector: допустим кэш входных frozen visual features, но не фиксирование trainable bridge.

В обозначениях приложения A.1: `A_tau=(1-tau)*A_data+tau*noise`, target velocity=`noise-A_data`; MSE по валидным action channels/timesteps. Другой разворот времени допустим, только если одновременно изменён target и inference ODE; нельзя смешивать формулы разных implementations.

**Практическая рекомендация для нового Fetch embodiment (не эксперимент авторов):** новые memory bridges и несовместимые/new action adapters обучать dense. При нехватке VRAM использовать LoRA на pretrained VLM/action expert, но оставить новые memory/projector weights fully trainable. Отдельный краткий connector warmup допустим как исследовательская гипотеза, а не обязательная опубликованная стадия; затем совместное SFT. Не замораживать случайно инициализированный projector и не ограничивать его только low-rank поправкой без обоснования.

## 4. Что реально поддерживает код: LoRA — опция, не paper default

[history_pi0.py:203–233](https://github.com/RoboMME/robomme_policy_learning/blob/ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b/src/mme_vla_suite/models/integration/history_pi0.py#L203-L233) возвращает freeze mask `.*img.*` без LoRA. В LoRA-вариантах исключает из заморозки `.*lora.*` и `.*mem.*`.

В [integration/utils.py:52–77](https://github.com/RoboMME/robomme_policy_learning/blob/ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b/src/mme_vla_suite/models/integration/utils.py#L52-L77) `gemma_2b_lora` и `gemma_300m_lora` имеют **rank32/alpha32** для attention и FFN. Если использовать эту экономную ветку, нужно пересчитать freeze filter **после** выбора model variants; в зарегистрированном TrainConfig исходный filter создаётся заранее из дефолтного config.

`rank16` из другой таблицы статьи относится к SAM2Act, а LoRA для Qwen subgoal predictor — к отдельному symbolic VLM. Приписывать их π0.5/FrameSamp/TokenDrop нельзя.

## 5. Расхождения paper и released snapshot, которые нельзя скрывать

- Table7: warmup5%=4,000 из80k. [config.py:570–619](https://github.com/RoboMME/robomme_policy_learning/blob/ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b/src/mme_vla_suite/training/config.py#L570-L619): warmup10,000, peak=end LR5e-5; baseline batch128, memory batch64.
- Table7 weight decay0; [optimizer.py:66–75](https://github.com/RoboMME/robomme_policy_learning/blob/ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b/src/openpi/training/optimizer.py#L66-L75) использует negligible1e-10, с комментарием об OOM при точном0.
- Статья A.3.2 задаёт identity modulation при инициализации (gamma1,beta0). [history_gemma.py:39–42](https://github.com/RoboMME/robomme_policy_learning/blob/ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b/src/mme_vla_suite/models/integration/history_gemma.py#L39-L42) использует small-random output projection; [representation/utils.py:10](https://github.com/RoboMME/robomme_policy_learning/blob/ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b/src/mme_vla_suite/models/representation/utils.py#L10) std=.002, не точный zero-init.
- В статье pooling описан как max pooling; опубликованные YAML содержат `pool_type: mean`. Это конфигурационное расхождение; отдельно проверяется, где именно ключ используется preprocessor, прежде чем заявлять побитовое воспроизведение.

Поэтому воспроизводимый отчёт должен отдельно фиксировать **paper settings** и **конкретный запущенный config/SHA**. Предыдущие локальные трёхшаговые LoRA smoke tests проверяли работоспособность адаптеров, а не воспроизводили dense авторское обучение и не оценивали качество памяти.
