# OpenPI π0.5 on the Fetch LeRobot dataset

OpenPI is a pinned Git submodule at `third_party/openpi`. The root ManiSkill environment only installs the small OpenPI WebSocket client; model training and serving use OpenPI's own environment. No OpenPI fork or edits inside the submodule are needed.

## 1. Set up environments

From the repository root:

```bash
git submodule update --init --recursive
cd third_party/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
# OpenPI pins a LeRobot v2.1 loader; replace only that package with v0.4.0 for v3.0 datasets.
uv pip install --python .venv/bin/python \
  'lerobot[dataset] @ git+https://github.com/huggingface/lerobot@f25ac02e6c8fa9c467ab8462289e5f4aed3a2e85' \
  'numpy<2'
cd ../..

uv sync --group openpi-eval
```

The OpenPI virtualenv and lockfile stay inside the submodule. The LeRobot overlay is pinned to the official v0.4.0 commit, which reads v3.0 datasets; the compatibility shim is in `utils/openpi_compat.py`. Do not run `uv sync` in the OpenPI submodule again without repeating the overlay. Accept the gated PaliGemma model terms at [google/paligemma-3b-pt-224](https://huggingface.co/google/paligemma-3b-pt-224), then authenticate to Hugging Face (for example, `hf auth login`). The account must also have access to the dataset. OpenPI estimates >70 GB VRAM for full fine-tuning and >22.5 GB for LoRA; use `--lora` on a suitable smaller GPU.

## 2. Check the dataset and adapter

The current mapping expects these LeRobot v3.0 features:

- `observation.images.left_base_camera_link`, `observation.images.fetch_hand`, `observation.images.right_base_camera_link` (RGB)
- `observation.state`: 12 values
- `action`: 13 values
- a natural-language task string in LeRobot task metadata

The policy sees the 12D robot state (not `global_state`) and three images. Pi0.5's model dimension stays at 32: OpenPI pads state/actions during model preprocessing and the output transform returns the first 13 action values. The 13D vector mixes joint targets and base velocities, so this integration intentionally applies no delta-action conversion. If your HF dataset uses different feature names, edit the repack mapping in `utils/openpi_fetch.py`.

Run the transform check, then load a real dataset batch:

```bash
uv run --project third_party/openpi --no-sync python -m utils.openpi_fetch
uv run --project third_party/openpi --no-sync python -m utils.openpi_train \
  --mode check --repo-id YOUR_HF_USER/YOUR_DATASET
```

## 3. Train

Compute stats over the full dataset, then fine-tune:

```bash
uv run --project third_party/openpi --no-sync python -m utils.openpi_train \
  --mode stats --repo-id YOUR_HF_USER/YOUR_DATASET --batch-size 8

uv run --project third_party/openpi --no-sync python -m utils.openpi_train \
  --mode train --repo-id YOUR_HF_USER/YOUR_DATASET \
  --exp-name takeit --steps 30000 --batch-size 8
```

Or run check, full stats, and training together with `--mode all`. Add `--lora` to both commands if using LoRA. The default action horizon is 10 (10 Hz dataset cadence); actions are not converted to deltas. `--max-frames` is for quick stats experiments only, not final training.

Stats go under `logs/openpi/assets/pi05_fetch/`; checkpoints go under `logs/openpi/checkpoints/pi05_fetch/<exp-name>/<step>/`.

## 4. Evaluate in ManiSkill

Start the policy server in one terminal, using the same dataset ID, horizon, and LoRA setting as training:

```bash
uv run --project third_party/openpi --no-sync python -m utils.serve_openpi \
  --repo-id YOUR_HF_USER/YOUR_DATASET \
  --checkpoint logs/openpi/checkpoints/pi05_fetch/takeit/30000 \
  --action-horizon 10
```

In another terminal, run the ManiSkill-side client/evaluator:

```bash
uv run --group openpi-eval python -m utils.evaluate_openpi \
  --prompt "Pick up the cup and place it on the tray." \
  --num-episodes 10 --start-seed 1000
```

Pass exactly the task instruction used in the dataset. The evaluator repeats each predicted 10 Hz action for two 20 Hz simulator steps and checks the environment success metric. Default evaluation uses CPU physics/rendering; flags allow selecting another configured backend.

## No-fork detail

OpenPI's stock CLI only knows configs registered in its `_CONFIGS` list. These project-owned launchers instead construct the OpenPI `TrainConfig` in `utils/openpi_fetch.py` and call the pinned OpenPI training/policy APIs. OpenPI also pins the older LeRobot v2.1 API; setup overlays official LeRobot v0.4.0 (v3.0) in the OpenPI virtualenv and `utils/openpi_compat.py` bridges its old import path. This avoids modifying the submodule; pin both commits and repeat the overlay after OpenPI environment syncs.
