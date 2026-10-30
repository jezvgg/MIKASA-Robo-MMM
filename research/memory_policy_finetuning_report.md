# Как правильно fine-tune memory-политики: авторские рецепты и адаптация к Fetch

**Проверено 20 октября 2026.** Исследование статей, приложений и официального кода; новые обучения не запускались. Ниже **«в статье»**, **«в опубликованном коде»** и **«рекомендация для проекта»** намеренно разделены. Предыдущие трёхшаговые проверки в нашем проекте — smoke tests, а не воспроизведение авторских результатов.

## 1. Оригинальные статьи

| Модель | Оригинальная работа / проверенная версия |
|---|---|
| RoboMME / FrameSamp / TokenDrop | Dai et al., **RoboMME: Benchmarking and Understanding Memory for Robotic Generalist Policies**, [2603.04639v1](https://arxiv.org/abs/2603.04639v1), §4, Appendix A–B, особенно B.3–B.4/Table7 |
| Базовая π0.5 | Physical Intelligence et al., **π0.5: a Vision-Language-Action Model with Open-World Generalization**, [2504.16054](https://arxiv.org/abs/2504.16054), §IV; отдельно [официальный OpenPI](https://github.com/Physical-Intelligence/openpi) |
| HAMLET | Koo et al., **HAMLET: Switch your Vision-Language-Action Model into a History-Aware Policy**, [2510.00695v3](https://arxiv.org/abs/2510.00695v3), ICLR2026, §3, Appendix A.3–A.4 |
| MemER | Sridhar et al., **MemER: Scaling Up Memory for Robot Control via Experience Retrieval**, [2510.20328v1](https://arxiv.org/abs/2510.20328v1), §3.3, Appendix A/Tables4–5 |
| MemoryVLA | Shi et al., **MemoryVLA: Perceptual-Cognitive Memory in Vision-Language-Action Models for Robotic Manipulation**, [2508.19236v2](https://arxiv.org/abs/2508.19236v2), §3–4, Appendix C |
| PonderPounce | Choi et al., **PonderPounce: A Pretrained MLLM as an Episode Context Engine for Robot Control**, [2608.24115v2](https://arxiv.org/abs/2608.24115v2), §III–IV |

`RoboMME/MemoryVLA` — адаптация MemoryVLA к другому benchmark, **не отдельная новая модельная статья**.

## 2. Короткий ответ: какую проекцию обучать и нужен ли LoRA

«Проекция» здесь может обозначать четыре разных объекта: pretrained **vision→language connector**; новый **memory/cognition→policy bridge**; **state/action adapters**; вспомогательную **contrastive head**. Их нельзя объединять в одну обязательную стадию «обучить проекцию».

| Вариант | Что обучать из проекций | Что делали авторы с основной моделью | Отдельный alignment/pretraining |
|---|---|---|---|
| RoboMME π0.5 | Action input/output adapters вместе с policy; новой memory-проекции нет | Dense VLM + action expert, vision frozen | Нет новой обязательной projector-only стадии |
| FrameSamp+Modul | Memory feature/position projection, cross-attention, AdaLN conditioning | То же + dense новые memory weights | Совместно на action loss, не отдельный alignment |
| TokenDrop+Modul | Те же learned bridges; hard token selection не обучается градиентом | То же | То же |
| HAMLET / GR00T N1.5 из статьи | Сначала TCL head + moment embeddings; затем memory/action компоненты | На action SFT VLM **и moment tokens frozen**; не LoRA | **Да, TCL**, но это не обучение VLM visual connector |
| HAMLET / CogACT из статьи | TCL head; далее memory/action и moment embeddings | На action SFT VLM **и moment tokens trainable** | TCL, другая freeze-карта |
| MemER | High-level visual connector **заморожен**; low-level adapters относятся к π0.5 | High-level dense LLM; low-level отдельно π0.5-DROID SFT, LoRA не заявлена | Нет нового learned memory bridge |
| MemoryVLA | Pretrained VLM connector + новые perceptual, retrieval/gating, DiT conditioning projections | Released LIBERO recipe: full VLM + memory + DiT | Отдельный projector-only этап не обязателен в этом recipe |
| PonderPounce | Новые carrier embeddings, cognition projection, age projection/null cognition | Joint end-to-end; RoboMME: Pounce vision frozen, LM/action expert trainable; RoboCasa: Ponder full, явно без LoRA | Авторы **прямо исключают отдельный bridge pretraining** |

**Важно:** dense/full fine-tuning разрешённых частей не означает разморозку всех весов. Наличие LoRA-флага в репозитории не доказывает, что LoRA использовалась в статье.

## 3. RoboMME: π0.5, FrameSamp+Modul, TokenDrop+Modul

### 3.1. Авторское обучение

В [Appendix B.4/Table7](https://arxiv.org/html/2603.04639v1#A2.SS4) vision backbone заморожен; visual tokens предварительно вычисляются. Обучаются **VLM expert, action expert и все новые memory parameters**. Это не опубликованный LoRA-рецепт.

Для perceptual моделей: **80k steps, global batch64, LR5e-5, constant после warmup5%, AdamW(.9,.95), clip1, weight decay0, EMA.999, horizon20, четыре FSDP devices**. Авторы не используют proprioception в этом наборе опытов. Baseline π0.5 — без memory layers; сохранение каждые10k.

Не путать с foundation π0.5: оригинальная работа описывает heterogeneous FAST-token pretraining и последующий continuous-action/semantic post-training. Для downstream RoboMME берут готовый `pi05_base`, а не повторяют всё foundation обучение. RoboMME использует VLM как feature extractor, без самостоятельного high-level text generation.

### 3.2. Проекции и loss

**Обычная π0.5:** pretrained visual interface уже есть. `action_in_proj` переводит action coordinates в width1024, `action_out_proj` — обратно. В π0.5 proprioception может идти дискретизированными текстовыми токенами: наличие отдельного непрерывного `state_proj` нельзя предполагать по названию VLA.

**Два Modul-варианта:** создаются memory feature encoder и layer-wise modulation. Статья описывает feature mapping2048→1024. Точнее, [опубликованный FeatureEncoder](https://github.com/RoboMME/robomme_policy_learning/blob/ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b/src/mme_vla_suite/models/representation/mem_encoder.py#L25-L104) с position features делает:

`pos768 → Linear768 → SiLU; concat(image2048,pos768) → Linear2816→1024`.

Затем action queries cross-attend к memory, а conditioning projection выдаёт scale/shift adaptive normalization. Новых modulation-параметров около80M. Все эти новые слои обучаются **dense**, совместно с policy на **action flow-matching loss**. Отдельные image-text пары для alignment этой проекции в статье не требуются. Кэшировать допустимо frozen visual features; кэшировать/detach выход обучаемого bridge вместо его обучения — уже другой метод.

Разница вариантов:

- **FrameSamp:** 32 равномерных кадра ×16 spatial tokens =512 memory tokens.
- **TokenDrop:** grid8×8, temporal stride8, RGB-difference threshold>1e-4, budget512, сохраняется initial-frame context. Hard selection не дифференцируется; learned layers после него — дифференцируются.

### 3.3. LoRA и расхождения code/paper

Fork поддерживает `gemma_2b_lora`/`gemma_300m_lora`, attention+FFN **rank32/alpha32**. Это альтернатива dense recipe. При таком выборе новые memory projections/modulators всё равно оставляют fully trainable; freeze filter нужно пересчитать после выбора variants. [Freeze logic](https://github.com/RoboMME/robomme_policy_learning/blob/ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b/src/mme_vla_suite/models/integration/history_pi0.py#L203-L233).

Есть реальные несовпадения: paper warmup5% против code10k/80k; paper exact identity modulation против small-random init в code; pooling max в тексте против `pool_type: mean` в YAML. Таблица задаёт batch64 для symbolic/perceptual, а отдельный baseline batch в ней не уточнён: released baseline config использует128, memory config64. Нужно зафиксировать выбранную версию, а не называть любой launcher точным воспроизведением статьи. Подробные ссылки и строки: [заметка RoboMME](finetuning_sources_robomme.md).

## 4. HAMLET: обязательное различие TCL, backbone и release

### 4.1. Как обучается «проекция»

[Статья, §3 и Appendix A.3–A.4](https://arxiv.org/html/2510.00695v3): к VLM добавляются четыре learned moment tokens. **TCL-stage** обучает их вместе с отдельной contrastive head `g`; VLM frozen. Положительные пары — аугментации одного кадра; отрицательные — кадры той же траектории дальше16 timesteps; temperature.07.

В коде `g`: **mean-pool moment features → Linear(d,d) → SiLU → Linear(d,d) → L2 norm**, cosine contrastive CE. Это **не** visual-language connector. После TCL переносятся moment embeddings; contrastive head не используется для выдачи действий. Action policy получает двухслойный causal memory Transformer, history length4.

### 4.2. Точные авторские стадии

| Backbone | TCL | Action SFT | Freeze на action SFT |
|---|---|---|---|
| GR00T N1.5 | До30k steps, batch64, LR1e-5 | 60k, batch32, LR1e-5 | **VLM и moment tokens frozen** |
| CogACT | До30k, фактически stop10k, batch64, LR2e-5 | 20k, batch32, LR2e-5 | **VLM и moment tokens trainable** |

Не существует одного ответа «HAMLET обучается full/LoRA» вне выбранного backbone. Paper recipes не являются LoRA-рецептами.

### 4.3. Что изменилось в N1.6 release

[Официальный N1.6 launcher](https://github.com/myungkyuKoo/HAMLET-Isaac-GR00T/blob/63b40557503e32dd93baebb24923e8ebf42121ef/run_scripts/robomme/train_hamlet_n1d6.sh): default **single-stage fine-tune-only**, random moment tokens обучаются, без TCL; есть optional загрузка TCL tokens и их freeze. Поэтому наш ранний single-stage smoke test нельзя назвать воспроизведением двухстадийной статьи.

В N1.6 особенно легко ошибиться с `tune_projector`: этот flag включает **action-head state_encoder/action_encoder/action_decoder**, а не TCL `g` и не автоматически VLM connector. [Реализация](https://github.com/myungkyuKoo/HAMLET-Isaac-GR00T/blob/63b40557503e32dd93baebb24923e8ebf42121ef/gr00t/model/gr00t_n1d6/gr00t_n1d6.py#L155-L172). Для нового Fetch embodiment эти adapters должны учиться. Default release: frozen visual/LLM backbones, trainable action projectors и memory; четыре GPU, batch32,60k, K4/stride16. Подробности: [заметка HAMLET/MemER](finetuning_sources_hamlet_memer.md).

## 5. MemER: две отдельные политики, а не memory-projector в VLA

[Статья §3.3, Appendix A](https://arxiv.org/html/2510.20328v1):

1. **High-level Qwen2.5-VL-7B-Instruct:** recent frames + remembered keyframes + instruction → subtask и keyframe selection. Vision encoder **и multimodal projection layer frozen**; dense обучение LLM на текстовых targets. Table4: **4500 updates, batch256, AdamW(.9,.999), wd0, clip1, cosine/warmup.05**. Основной текст округляет до~5000. LR в таблице **не указан**.
2. **Low-level π0.5-DROID:** current image/state + subtask text → действия; отдельное action SFT. Table5: **18k updates, batch128, AdamW(.9,.95), wd0, clip1, cosine/warmup.033**. LR, action horizon и полный freeze-map таблицей не определены; LoRA не заявлена. Нельзя дописать отсутствующие параметры из другого recipe.
3. После high-level SFT авторы смешивают веса: **.2 pretrained + .8 finetuned**. Это model merging, не LoRA.

Новой learned «retrieval projection» для ключевых кадров здесь нет: память представлена выбранными изображениями. Есть другая обязательная работа — supervision. Авторы размечают subtask boundaries и задают правила first/last/no-frame per subtask; затем автоматически получают keyframe targets. **Текущих общих инструкций MyRoboCasa на эпизод недостаточно.** Нужны эти labels и subtask-conditioned low-level training.

Публичный [Qwen3 training recipe](https://github.com/memer-policy/memer/blob/fb6dbd0e37a8dd76c965ff3f1344f9b448b61911/TRAINING_QWEN3VL.md) — **другая версия**: Qwen3-VL-4B,1500steps, LR6e-5, frozen visual+connector, dense LLM; effective batch **2×8×8=128**. Не выдавать его за исходные Qwen2.5 настройки статьи.

## 6. MemoryVLA и RoboMME/MemoryVLA

### 6.1. Что загружают и что обучают

[Оригинальная статья §3–4, Appendix C](https://arxiv.org/html/2508.19236v2) и [official OpenVLA branch](https://github.com/shihao1895/MemoryVLA/tree/openvla-codebase): pretrained VLM с DINOv2+SigLIP и LLaMA7B, perceptual/cognitive memory banks и diffusion DiT action expert. Для LIBERO released script стартует из **OpenVLA-7B Prismatic**; для Bridge/Fractal опубликованы также CogACT-Large-based инструкции.

**Pretrained:** vision backbones, LLM, Prismatic vision→language projector. **Новые:** perceptual SE bottleneck до feature-width256, memory retrieval Q/K/V/FFN, gates `Linear(2D,D)`, temporal embeddings и diffusion expert/conditioning.

DiT-L: width1024,24blocks,16heads; action embedding `action_dim→1024`, cognitive conditioning `LLM hidden→1024`, perceptual embedding `256→1024`, output `1024→action_dim`. Это несколько разных проекций, не одна «проекция памяти».

В [released LIBERO training path](https://github.com/shihao1895/MemoryVLA/blob/d732ea9072bc063399ccc817aed74ab172eb50be/train.py#L179-L200) выбран **full-finetune**: vision, LLM и pretrained Prismatic projector trainable; PCMB/perceptual/DiT также trainable; EMA-copy frozen. Loss — action diffusion/noise-prediction MSE. Отдельная mandatory projector alignment стадия не используется. В коде есть `align`/`finetune` варианты, но наличие `align` не означает, что этот этап обязательно повторяют после OpenVLA initialization. Готовый LoRA recipe в проверенных scripts не найден.

### 6.2. Авторские параметры

Paper §4.1/C: **8×A100, global batch256=8×32, LR2e-5, clip1, chunk16, четыре diffusion repeats**. LIBERO Spatial script: **20k steps**, `future_action_window_size=15`; расхождения нет — текущий action +15 future =16. Inference: DDIM10, CFG1.5. Это diffusion schedule, не π0.5 flow matching.

[RoboMME fork](https://github.com/RoboMME/MemoryVLA/tree/f7befabcddafc72738b23c78cf965ae39fbd22f9): OpenVLA-based адаптация, **batch64, LR2e-5,160k steps**. Авторы README прямо называют адаптацию базовой и параметры потенциально неоптимальными. Это не новый отдельный pretraining/projection method и не подтверждение, что160k оптимальны для Fetch.

### 6.3. Что менять для Fetch

Оригинальный pipeline: один third-person RGB224, instruction, **7D EEF delta + gripper**. Нет готового использования наших state12/трёх камер; serving hardcodes gripper channel6. Нужно адаптировать multi-view/state путь при необходимости и action expert/normalization/postprocessing для13D смешанного joint/base control. Дополнить7D output нулями нельзя. Новые/изменённые adapters учить dense вместе с memory/action stack; LoRA на pretrained LLM — возможный **наш эксперимент**, не авторский baseline. Детальные модули/источники: [заметка MemoryVLA/PonderPounce](finetuning_sources_memoryvla_ponderpounce.md).

## 7. PonderPounce: joint bridge learning и необходимый SFT-код

### 7.1. Авторская модель

[Статья v2 §III](https://arxiv.org/html/2608.24115v2#S3): **Ponder=Qwen3.5-9B**, Pounce=π0.5 на RoboMME или GR00T N1.5 на RoboCasa-DC. Final hidden state **одного carrier token** служит cognition. K=8 относится к latency profiling, **training/eval K=1**.

Случайно инициализируются carrier embeddings, cognition bridge и age projector; null cognition начинается с нулей. Интерфейс:

`carrier hidden H_Qwen → cognition_proj → H_Pounce`,

`age_ms → sinusoidal encoding → learned Linear(H_Pounce,H_Pounce)`.

В π0.5-Pounce prefix — `[age token, projected cognition tokens]`; дальше обычная PaliGemma обработка. Это не external keyframe retrieval. **Все новые bridge weights обучаются joint end-to-end; отдельного bridge pretraining нет**, что прямо сказано авторами.

### 7.2. Full или LoRA и точные параметры

| Benchmark | Настройки статьи v2 |
|---|---|
| RoboMME | **4320 episode-based updates**, global batch32, AdamW, clip1, warmup100→constant **LR2e-5**; Pounce SigLIP frozen, PaliGemma LM/action expert trainable; `L=L_fm+.1 L_ground`; chunk20; no cognition dropout |
| RoboCasa-DC | **4000 episode-based updates**, batch32, LR1e-5; **Ponder full fine-tune без LoRA**; action-only loss, cognition dropout15%, chunk16 |

Для RoboMME статья описывает joint dense training разрешённых модулей, не публикует LoRA rank/targets. Нельзя переносить явный full-Ponder statement RoboCasa на произвольный freeze-map RoboMME без дополнительной спецификации. **Episode-based update не эквивалентен frame-based update RoboMME**; сравнение4320 с80k по одному числу шагов некорректно.

RoboMME training schedule: Pounce prediction times в среднем каждые100ms, Ponder queries≈1s, simulated delay≈300ms. RoboCasa: Pounce2s/Ponder4s/delay300ms. Train/eval schedules не тождественны простому независимому sampling кадров.

### 7.3. Что именно надо написать для SFT

В [публичном snapshot](https://github.com/worv-ai/PonderPounce/tree/723df35762bb641e1d520e4fa9359b98644adc21) есть inference/eval/integration, но **не найден reproduce-ready trainer/dataset/loss pipeline**. Наличие checkpoint и model config не означает, что SFT уже реализован.

Минимально faithful план реализации:

1. **Episode dataset/collator:** causal observation sequence, camera timestamps, task text, action targets; demonstrations/grounding labels — только если реально присутствуют. Не превращать episode-SFT в shuffled independent frames.
2. **Differentiable training forward:** teacher-forced causal context и carrier hidden states. Inference StaticCache/`inference_mode` нельзя автоматически считать пригодными для autograd.
3. **Schedule:** для каждого Pounce tick выбрать latest-*ready* cognition, а не самую новую source observation; age считать от её реального времени. Будущий ещё не готовый cognition — утечка.
4. **Loss:** masked continuous action flow-matching MSE; при наличии labels — CE transition/subgoal/demo-reasoning. Без labels допустим `w_ground=0`, но это ablation, не полный RoboMME recipe.
5. **Gradient routing:** несколько Pounce calls могут использовать один cognition tensor; action gradients накапливаются в нём. В Ponder action-derived gradient масштабируется **.5**, LM-grounding gradient не масштабируется. Само forward значение cognition при этом не меняется.
6. **Temporal graph:** статья требует градиент через earlier context **без temporal detachment**. Detached carrier cache или truncated BPTT — сознательное изменение метода, а не faithful reproduction. Memory-saving checkpointing можно делать без обрыва этого графа.
7. **Parameter groups/checkpoints:** новые bridges trainable dense; backbone freeze/full/PEFT явно задан; сохранение processor, config, masks, normalization, model+optimizer/scheduler/RNG и проверяемый resume.

Сначала реализовать и проверить этот forward/loss/causality, затем выбирать длительность SFT. Начать с «запустить inference forward в цикле и обучить один Linear» — не воспроизвести PonderPounce.

## 8. Рекомендуемый путь для нашего Fetch — отдельно от авторских опытов

1. **Сохранить контракт:** реальные state12/action13, три синхронные камеры; actions = arm7 + gripper1 + body3 + base forward/yaw. Это absolute joint targets вместе с base velocities, **не EEF delta**.
2. **Не переинициализировать всё только из-за13D.** Native π0.5/PonderPounce-Pi05 сохраняет padded internal action width32: совместимые action layers можно оставить, настроив13 valid channels/masks и новые stats. Но logical8D checkpoint config, delta masks и decoder всё равно нужно изменить. MemoryVLA с реально7-wide head — другой случай.
3. **Развести три задачи:** авторская reproduction; resource-efficient adaptation; tiny integration smoke. Для первой — paper freeze-map. Для второй — допустим LoRA pretrained LM/action expert **плюс dense новые projectors/memory/state-action adapters**. Это уже новый эксперимент; не подписывать его авторскими результатами.
4. **Выбор практического старта:** FrameSamp+Modul имеет готовый action SFT и требует меньше нового кода; HAMLET — два варианта, paper-style TCL+SFT или явно помеченный N1.6 fine-tune-only. MemER потребует недостающую subtask/keyframe разметку. MemoryVLA требует более глубокого embodiment adapter. PonderPounce — отдельная разработка episode SFT.
5. **Физическое время обязательно:** наши данные10Hz, контроль20Hz/action-repeat2. У авторского PonderPounce H20@20Hz=1s; H20 на наших10Hz labels=2s. Согласовать horizon, query periods, age и исполнение; не объявлять дублирование кадров реальными новыми20Hz observations.
6. **Данные/валидация:** разделять episodes/seeds до статистик, считать normalization только на train split, исключить будущие кадры и cross-episode memory. Для MemER/grounded PonderPounce сначала подготовить необходимые labels; случайные кадры или общие task prompts их не заменяют.
7. **Контроль обучения:** логировать фактические trainable parameter names/count, gradient norm и изменение **самих новых projections/memory weights**, а не только общий loss; проверить reload, action semantics и real episode reset. Три шага доказывают работоспособность графа, но не сходимость и не task success.

## 9. Уровень подтверждения и дополнительные материалы

Подтверждено первоисточниками: основные train/freeze-карты, роли проекций, этапы, числа выше; PDF-таблицы HAMLET/MemER и MemoryVLA извлечены локально, а не восстановлены по чужим обзорам. Не установлены: отсутствующие в MemER таблицах LR/horizon/full low-level freeze-map; универсальный optimal Fetch LR/LoRA rank; отсутствующий публичный PonderPounce trainer. Эти пробелы не заполнены догадками.

Source notes: [RoboMME](finetuning_sources_robomme.md), [HAMLET/MemER](finetuning_sources_hamlet_memer.md), [MemoryVLA/PonderPounce](finetuning_sources_memoryvla_ponderpounce.md). Зафиксированные SHA находятся в них; небольшие source snapshots и PDF — `research/sources/`. Новых GPU jobs, model weights и dataset downloads в ходе этого исследования не было.
