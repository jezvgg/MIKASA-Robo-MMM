#!/usr/bin/env bash
# MemoryVLA's environment as its README pins it (Python 3.10, torch 2.2.0 + cu121, the
# flash-attn 2.5.5 wheel, then `pip install -e .` of third_party/MemoryVLA), installed with uv.
# Two additions, neither part of training:
#  * tensorflow-metadata==1.14.0 and wandb==0.17.9: MemoryVLA leaves both unpinned. The newest
#    tensorflow-metadata (1.21.0) imports protobuf>=5 (`google.protobuf.runtime_version`) while
#    tensorflow 2.15 keeps protobuf<5, so `import tensorflow_datasets` fails; on Python 3.10
#    every tensorflow-metadata that tensorflow 2.15 accepts holds protobuf<4.21 (3.20.3), and the
#    newest wandb (0.28) no longer imports under protobuf 3, so train.py's `import wandb` fails;
#  * av, pandas, pyarrow for vla/memoryvla/rlds (LeRobot v3 -> RLDS).
# It also applies patches/*.patch to the submodule's working tree (the six Fetch datasets,
# their 13D action encoding and mixtures); the submodule stays pinned to the authors' commit.
# usage: setup_memvla.sh [venv dir]   (default ~/envs/memvla; run from anywhere)
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
ENV=${1:-$HOME/envs/memvla}
for patch in "$REPO"/vla/memoryvla/patches/*.patch; do
    if git -C "$REPO/third_party/MemoryVLA" apply --reverse --check "$patch" 2>/dev/null; then
        echo "already applied: $(basename "$patch")"
    else
        git -C "$REPO/third_party/MemoryVLA" apply "$patch" && echo "applied: $(basename "$patch")"
    fi
done
[ -d "$ENV" ] || uv venv -q -p 3.10 "$ENV"
PY=$ENV/bin/python
uv pip install -q -p $PY torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
uv pip install -q -p $PY "https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.5/flash_attn-2.5.5+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
uv pip install -q -p $PY -e "$REPO/third_party/MemoryVLA"
uv pip install -q -p $PY "tensorflow-metadata==1.14.0" "wandb==0.17.9" "av==12.3.0" "pandas==2.2.3" "pyarrow==17.0.0"
$PY -c "import torch, flash_attn, transformers, tensorflow, tensorflow_datasets, dlimp, av, wandb; print('torch', torch.__version__, 'gpu', torch.cuda.is_available(), '| flash_attn', flash_attn.__version__, '| transformers', transformers.__version__, '| tf', tensorflow.__version__, '| tfds', tensorflow_datasets.__version__)"
