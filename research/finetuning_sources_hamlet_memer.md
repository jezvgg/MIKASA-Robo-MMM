# Авторские рецепты: HAMLET и MemER

Проверено 2026-10-20 по статьям, PDF-таблицам и официальным исходникам. Обучение не запускалось. Исправления относительно первоначального извлечения: HAMLET N1.5 действительно **замораживает moment tokens на второй стадии**; Appendix A MemER доступен в PDF, поэтому его параметры ниже подтверждены, а не объявлены неизвестными.

## 1. HAMLET

**Статья:** Koo et al., *HAMLET: Switch your Vision-Language-Action Model into a History-Aware Policy*, [arXiv:2510.00695v3](https://arxiv.org/abs/2510.00695v3), ICLR2026; §3, Appendix A.3–A.4. PDF и `pdftotext -layout` сохранены в `research/sources/papers/2510.00695.{pdf,txt}`; строки текста991–1007 содержат обе авторские freeze-карты.

### Проекции и этапы

1. **TCL initialization.** Замороженный VLM обрабатывает изображения и четыре новых learnable moment tokens. Обучаются moment-token embeddings и contrastive projection head `g`. Положительные пары — разные аугментации одного кадра (photometric/blur/noise/small-patch occlusion); отрицательные — кадры той же траектории на расстоянии >16 timesteps. Температура `.07` (Appendix A.3). Замороженные веса backbone не означают `no_grad` на всём forward: нужен градиент к token embeddings через его операции.
2. **Точная голова в release:** mean pooling moment representations → `Linear(d,d) → SiLU → Linear(d,d)` → L2 normalization; cosine contrastive CE. Это `moment_to_repr`, не исходный vision-language projector. N1.6 [tcl_head.py:31–45,59–86](https://github.com/myungkyuKoo/HAMLET-Isaac-GR00T/blob/63b40557503e32dd93baebb24923e8ebf42121ef/gr00t/model/gr00t_n1d6/tcl_head.py#L31-L86). `d` определяется backbone; N1.5 implementation default1536. Инициализация Linear Normal(std=.02), bias0. В TCL заменяется action head и обучаются только tokens+contrastive head: [gr00t_n1d6.py:735–747](https://github.com/myungkyuKoo/HAMLET-Isaac-GR00T/blob/63b40557503e32dd93baebb24923e8ebf42121ef/gr00t/model/gr00t_n1d6/gr00t_n1d6.py#L735-L747).
3. **Action SFT.** TCL head не является управляющей головой и не требуется для inference: переносятся инициализированные moment tokens, восстанавливается pretrained action head, обучается memory-conditioned action policy. Два слоя causal memory Transformer, четыре history observations, четыре moment tokens на observation.
4. **GR00T N1.5, опубликованный рецепт:** TCL до30k steps, batch64, LR1e-5; затем action SFT60k, batch32, LR1e-5. На второй стадии **VLM и moment tokens frozen**, обучаются memory/action компоненты. Это не full tuning всего VLM и не LoRA.
5. **CogACT, опубликованный рецепт:** TCL до30k, фактически early-stop10k после сходимости, batch64, LR2e-5; action SFT20k, batch32, LR2e-5. Здесь **VLM и moment tokens остаются trainable**. Нельзя переносить freeze-карту GR00T на CogACT.

Точный универсальный LoRA rank/alpha для HAMLET не заявлен. В статье отсылаются к исходным backbone training recipes для остальных настроек; не следует угадывать отсутствующий optimizer или полный список узких подмодулей по одному слову fine-tune.

### Отдельно: более поздний N1.6 release и значение tune_projector

Main repo [HAMLET-Isaac-GR00T](https://github.com/myungkyuKoo/HAMLET-Isaac-GR00T) pin **63b40557503e32dd93baebb24923e8ebf42121ef**; branch `n1.5` pin **3ec4cd65c996614adf14a94db0f879798699d8de**. Они не тождественны исходным экспериментам N1.5.

`run_scripts/robomme/train_hamlet_n1d6.sh:17–58`: GR00T-N1.6-3B, четыре GPU, global batch32, accumulation1,60k steps, K4/stride16, memory layers2. **Default single-stage:** случайные moment tokens trainable, без TCL. Paper-style две стадии включаются загрузкой TCL embeddings и `FREEZE_MOMENT_TOKENS=1`. Замораживать случайные токены без TCL checkpoint — ошибка; launcher предупреждает об этом.

Особенно важно для вопроса о проекции: в N1.6 `tune_projector` относится к **action-head** `state_encoder`, `action_encoder`, `action_decoder` и дополнительным positional/mask параметрам, а не к TCL `g` и не автоматически к visual-language connector! [gr00t_n1d6.py:155–172](https://github.com/myungkyuKoo/HAMLET-Isaac-GR00T/blob/63b40557503e32dd93baebb24923e8ebf42121ef/gr00t/model/gr00t_n1d6/gr00t_n1d6.py#L155-L172). Defaults `tune_llm=False,tune_visual=False,tune_projector=True` в `gr00t/configs/finetune_config.py:36–42`; memory transformer trainable. Такие state/action adapters нужно обучать для нового Fetch embodiment.

## 2. MemER

**Статья:** Sridhar, Pan, Sharma, Finn, *MemER: Scaling Up Memory for Robot Control via Experience Retrieval*, [arXiv:2510.20328v1](https://arxiv.org/abs/2510.20328v1), §3.3 и Appendix A Tables4–5. PDF и layout text сохранены в `research/sources/papers/2510.20328.{pdf,txt}`; строки текста764–801 содержат таблицы.

### Две отдельно обучаемые политики

- High-level **Qwen2.5-VL-7B-Instruct** предсказывает current subtask и keyframes для запоминания. Existing vision encoder **и multimodal projection layer frozen**, dense trainable LLM backbone. Это CE на текстовых/JSON targets, не обучение action projector и не LoRA-рецепт. Appendix A Table4: AdamW β=.9/.999, wd0, clip1, cosine, warmup .05, global batch256, **4500 steps**,96 H200 GPU-hours. Основной текст округляет до~5000steps/10–15epochs. **LR в Table4 не указан** — не подставлять LR из Qwen3 release.
- Low-level **π0.5-DROID** обучается отдельно на `(current images, proprioception, subtask text) → action chunk`, не на predicted keyframe-index loss. Appendix A Table5: AdamW β=.9/.95, wd0, clip1, cosine, warmup .033, global batch128, **18,000 steps**,48 H200 GPU-hours. **LR, action horizon и полный freeze-mask не указаны в этой таблице**; LoRA для low-level не заявлена. Нет оснований выдавать эти неизвестные за полный unfreeze всех модулей.
- Данные:50 demonstrations и10–15 intervention examples на каждую из трёх задач; single multi-task policy каждого уровня (§3.3).
- После high-level fine-tune — weight interpolation `theta=.2*theta_pre+.8*theta_ft` (§3.3). Это model merging, не LoRA и не обучение проекции.

### Разметка памяти

§3.3: candidate keyframes берутся на границах подзадач; человек определяет правило first/last/no-frame для каждого subtask type, затем правило применяется автоматически ко всем demonstrations. Полученные keyframes — supervision targets. Одной общей инструкции на эпизод недостаточно для воспроизведения high-level SFT. Произвольная ручная выборка кадров не заменяет subtask boundaries и эти правила.

### Публичный Qwen3 recipe — другая версия, не параметры статьи

Repo pin **fb6dbd0e37a8dd76c965ff3f1344f9b448b61911**, [TRAINING_QWEN3VL.md:37–76](https://github.com/memer-policy/memer/blob/fb6dbd0e37a8dd76c965ff3f1344f9b448b61911/TRAINING_QWEN3VL.md#L37-L76): Qwen3-VL-4B-Instruct, две B200, per-device batch8 × accumulation8 ×2 GPUs = **effective128**, max_steps1500, LR6e-5, cosine/warmup .05, wd0, bf16, max length8192. Flags `tune_mm_vision=False,tune_mm_mlp=False,tune_mm_llm=True`: dense LLM tuning при frozen visual encoder и connector. Не заменять ими paper Qwen2.5-VL settings.

## 3. Следствия для Fetch — рекомендации, не авторские эксперименты

- Для faithful HAMLET сначала TCL tokens+`g`, потом backbone-specific freeze-карта. Для уже реализованного N1.6 single-stage не объявлять smoke test воспроизведением двухстадийной статьи. Если меняются state/action slots, включить action-head projector training; preserve internal padded widths/masks, где это позволяет checkpoint.
- Для MemER не добавлять «memory projection» в Fetch VLA по умолчанию: память представлена отобранными RGB и текстовыми subgoals. Сначала подготовить subtask/transition/keyframe supervision, затем отдельно high-level CE и low-level π0.5 action SFT. Existing high-level multimodal connector по авторскому рецепту **не дообучать**.
