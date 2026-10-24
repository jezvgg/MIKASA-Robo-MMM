#!/usr/bin/env bash
# The environment of MemER's own scripts (memer-policy/memer): generate_sft_data.py (LeRobot v3 ->
# Qwen SFT data) and the offline rollout eval scripts/eval_subtask_rollout.py. Their pyproject's
# dependencies plus its `eval` extra, with transformers pinned to the trainer's 4.57.0 (the extra
# says >=4.57.0). Then patches/memer-*.patch; the submodule stays pinned:
#   memer-0001: a keyframe before a sample's first subsampled frame moves to that frame instead
#               of being dropped (vla/TRAINING_DEVIATIONS.md, "Early keyframes").
# usage: setup_memer.sh [venv dir]   (default ~/envs/memer)
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
ENV=${1:-$HOME/envs/memer}
for patch in "$REPO"/vla/memer/patches/memer-*.patch; do
    if git -C "$REPO/third_party/memer" apply --reverse --check "$patch" 2>/dev/null; then
        echo "already applied: $(basename "$patch")"
    else
        git -C "$REPO/third_party/memer" apply "$patch" && echo "applied: $(basename "$patch")"
    fi
done
[ -d "$ENV" ] || uv venv -q -p 3.11 "$ENV"
PY=$ENV/bin/python
uv pip install -q -p $PY -e "$REPO/third_party/memer[eval]" "transformers==4.57.0"
$PY -c "import torch, transformers, lerobot, qwen_vl_utils, memer_eval; print('torch', torch.__version__, 'gpu', torch.cuda.is_available(), '| transformers', transformers.__version__, '| lerobot', lerobot.__version__)"
