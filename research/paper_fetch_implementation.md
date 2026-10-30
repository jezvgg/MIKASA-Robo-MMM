# Paper-faithful Fetch training: implementation and evidence

## 1. Final agreed scope

| Configuration | Outcome |
| --- | --- |
| RoboMME π0.5 | Dense paper path implemented; 3 optimizer updates, native checkpoint reload and real Fetch queries passed |
| FrameSamp+Modul | Same; trainable memory bridge/attention/modulation updated |
| TokenDrop+Modul | Same; causal token selection and real history/reset queries passed |
| HAMLET/CogACT | **Implementation cancelled by user** after the official Llama-2 dependency denied access; no CogACT implementation or smoke is claimed |
| MemoryVLA | **Deferred by user until licensed access** to the original Llama-2 dependency; no implemented Fetch model/smoke is claimed |
| PonderPounce | **Deferred by user until grounding labels exist**; no fake labels or action-only replacement |

The existing HAMLET/GR00T N1.6 working files are preserved, not relabelled as
HAMLET/CogACT. MemER remains out of scope. Reference descriptors for all six
methods remain in `utils/policies/recipes/author_recipes.json`; execution
cancellations/deferrals are explicitly recorded separately from scientific recipes.

## 2. What changed for RoboMME

Paper: [2603.04639v1](https://arxiv.org/html/2603.04639v1), Appendix B/Table7.
Backend: `RoboMME/robomme_policy_learning@ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b`.

- Default CLI profile is `--recipe paper`: **dense VLM + action expert + new
  memory**, frozen SigLIP vision. No hidden LoRA. The historical implementation
  remains available only via explicit `--recipe legacy-lora`.
- Paper optimizer: AdamW betas(.9,.95), LR5e-5, weight decay0, clip1,
  warmup4000 then constant, EMA.999; full reference80k updates. Baseline batch128
  follows the released baseline launcher because the paper does not separately
  identify that batch; perceptual variants use paper batch64. Full FSDP reference4
  GPUs is distinct from the one-GPU smoke.
- Current proprioceptive token conditioning is disabled in the paper profile,
  as specified in Table7; merely disabling historical state was insufficient.
  The real 12D dataset state is validated and retained in the adapter, not
  replaced with dummy state or a newly invented state projector.
- Native action width32 is retained for pretrained compatibility. The actual
  flow-matching objective reduces over **only the 13 valid Fetch channels**,
  normalized by13, not32. No extra EEF conversion or phantom-joint targets.
- New project-owned model adapter reproduces the native FM corruption/noise/
  attention path, and zero-initializes only conditional memory AdaLN projection
  so modulation starts as identity on normalized activations. It does not zero
  the memory-attention output projection.
- The paper memory path actually uses **max pooling**. Setting native
  `MemoryBuffer.pool_type` alone was ineffective because its implementation
  called the helper with the default mean. The project-owned buffer fixes the
  actual preprocessing call, including buffers rebuilt after serving reset.
- FrameSamp:32 frames ×16 tokens, budget512; TokenDrop:64 tokens/frame,
  RGB-difference stride8/threshold1e-4, budget512. History has one base camera;
  current observation retains all three Fetch cameras. Frozen SigLIP features
  are cached for one episode in RAM; trainable projection outputs are not cached
  or detached. Selection uses only the causal prefix.

Core code: `utils/policies/robomme.py`, `robomme_paper_model.py`,
`robomme_paper_memory.py`, `memory.py`, `parameter_audit.py`.

## 3. Real three-step evidence on gangway

Runs: `author_smoke_001`, batch1, FSDP1, **exactly3 optimizer steps** each.
Full history-token budget and the paper losses/freeze/gradient maps are retained.
Each run trained3 anchors, including both dataset parts. The deterministic
DiagnosticSampler yields global frame indices `[0,1,553533]` (two takeitback,
one veggie); these IDs are reconstructable from the sampler and recorded anchor
count, rather than a claim of an independently recorded per-step sample log.

| Variant | Last FM loss | Last raw gradient norm | Frozen vision | Changed required groups |
| --- | ---: | ---: | --- | --- |
| π0.5 | 0.5220641 | 4.2002439 | all23 leaves unchanged | VLM, action expert, action adapters |
| FrameSamp+Modul | 0.5227011 | 7.1221175 | all23 unchanged | above + bridge, memory attention, modulation |
| TokenDrop+Modul | 0.5220770 | 7.0502396 | all23 unchanged | above + bridge, memory attention, modulation |

Checksums compare actual raw pre/post parameter groups. All losses/gradients
are finite. Raw gradient norms above1 precede the configured clip1.
π0.5 peak device allocation was69,819,500,288 bytes; dense training genuinely
ran on the H100 rather than silently substituting LoRA.

Each native saved step2 checkpoint (zero-based step number) was reloaded in
its native server. Two queries in each of two real episodes per dataset part
returned finite `(20,13)` actions, with reset acknowledgements and intervening
10-Hz history ingestion. These are **real-data queries, not simulator rollouts or
success-rate evidence**. Three updates are not convergence or a quality claim.

Remote evidence under `~/policy-integration-20261019`:

- `logs/robomme_paper/{variant}-author_smoke_001-verification.json`
- `logs/robomme_paper/{variant}-author_smoke_001-parameters-before.json`
- `logs/robomme_paper/{variant}-author_smoke_001-parameters-after.json`
- `logs/robomme_paper/author_smoke_001-reload-verification.json`
- `logs/robomme_paper/{variant}-author_smoke_001-probe.json`
- `logs/robomme_{variant}_paper_smoke_001.log`
- `logs/robomme_paper_unit_003.log`:26 small CPU tests passed, including identity
  AdaLN, masked FM objective, pooling, recipe config and exact parameter audit.
- `logs/final_index_4c5aee2ec9ee_tests.log`: a fresh archive of the staged tree,
  excluding unrelated untracked dependencies, passed **103 baseline +44 policy/
  contract tests**. These include real dataset boundaries, deterministic sampling,
  causal future/episode guards, recipe config, numerical model and parameter-audit
  tests. Only the evidence/documentation summary was subsequently updated; tested
  implementation and test files were unchanged.
- `research/evidence/robomme_paper_smoke_20261020.json` is the small committed
  results/runtime/source-hash summary. It contains no weights or credentials.

## 4. Reproduce, without launching a full epoch

```bash
ssh gangway
export ORIGINAL_HOME="$HOME"
cd "$ORIGINAL_HOME/policy-integration-20261019"
export PYTHONPATH="$PWD/third_party/robomme/src:$PWD"
export BASE_PARAMS="$ORIGINAL_HOME/.cache/openpi/openpi-assets/checkpoints/pi05_base/params"

# Prepared .venv-robomme has pinned JAX0.5.3 / Flax0.10.2 / Optax0.2.4;
# it reads existing dependencies read-only. Package/version evidence is saved.
# Backend source must be the SHA above; environments are installed with uv.
for variant in pi05 framesamp tokendrop; do
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 OMP_NUM_THREADS=12 \
  .venv-eval/bin/python -m utils.policies.workspace --workspace "$PWD" \
    --reserve-gb 60 -- .venv-robomme/bin/python -m utils.policies.robomme train \
    --recipe paper --variant "$variant" --batch-size 1 --fsdp-devices 1 \
    --diagnostic-steps 3 --exp-name NEW_UNIQUE_NAME --root dataset \
    --features logs/features_bounded --output logs/robomme_paper --base-params "$BASE_PARAMS"
done

.venv-eval/bin/python -m utils.policies.workspace --workspace "$PWD" --reserve-gb 15 -- \
  .venv-eval/bin/python -m utils.policies.verify_robomme_paper \
  --base-params "$BASE_PARAMS" --output logs/robomme_paper --exp-name NEW_UNIQUE_NAME
```

Do not rerun an existing experiment name or start parallel GPU jobs. The full
80k-step commands are reference-only and are not executed by this smoke.
Native checkpoints are weights/assets, not an optimizer-resume implementation;
`--resume` is explicitly rejected rather than falsely resetting optimizer state.

## 5. Storage, licenses and provenance

All new server files/caches/checkpoints stay inside the designated workspace.
The conservative watchdog stops owned processes at350 GB, keeping headroom below
**400,000,000,000 bytes**; it is not an administrative filesystem quota. Existing
external dataset, assets, base weights and environment packages are read-only.
There is no full510GB feature cache. Workspace was approximately183 GB after the
three new dense checkpoints; final exact usage is in the synchronization evidence.

The original CogACT and MemoryVLA Llama-2 dependency check returned
`GatedRepoError` with the existing token. No token content was printed/committed,
no access restriction was bypassed, and no corresponding weights were downloaded.
PonderPounce needs transition/subgoal targets; coarse task text was not used as
fake grounding supervision.

The separate preparatory scene/planner commit is `4119aa488015d5c43ecc412c3cd6e0e51b807004`.
Canonical reference contracts were introduced in `5a17f848c51fdb3eb92690b2c94e103fa7aa3231`.
The final published HEAD/tracked-file hash comparison is recorded after the
implementation commit in remote `logs/` and the final delivery message. Existing
unrelated user edits remain dirty and are not silently included in commits.
