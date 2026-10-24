#!/usr/bin/env bash
# The environment of MemER's high-level trainer: QwenLM/Qwen3-VL's qwen-vl-finetune at the
# versions its README lists (torch 2.6.0, torchvision 0.21.0, transformers 4.57 (the README's
# 4.57.0.dev0 is the pre-release of this one), deepspeed 0.17.1, flash_attn 2.7.4.post1,
# triton 3.2.0, accelerate 1.7.0, torchcodec 0.2, peft 0.17.1) plus `pip install -e
# ../qwen-vl-utils`, as memer/TRAINING_QWEN3VL.md says. Then patches/qwen3vl-*.patch registers
# the six Fetch exports (memer_<task>) in the trainer's dataset dict; the submodule stays pinned.
# usage: setup_qwen3vl.sh [venv dir]   (default ~/envs/qwen3vl)
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
ENV=${1:-$HOME/envs/qwen3vl}
for patch in "$REPO"/vla/memer/patches/qwen3vl-*.patch; do
    if git -C "$REPO/third_party/Qwen3-VL" apply --reverse --check "$patch" 2>/dev/null; then
        echo "already applied: $(basename "$patch")"
    else
        git -C "$REPO/third_party/Qwen3-VL" apply "$patch" && echo "applied: $(basename "$patch")"
    fi
done
[ -d "$ENV" ] || uv venv -q -p 3.11 "$ENV"
PY=$ENV/bin/python
uv pip install -q -p $PY torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install -q -p $PY "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
uv pip install -q -p $PY "transformers==4.57.0" "deepspeed==0.17.1" "triton==3.2.0" "accelerate==1.7.0" \
    "torchcodec==0.2" "peft==0.17.1" -e "$REPO/third_party/Qwen3-VL/qwen-vl-utils"
uv pip install -q -p $PY "setuptools<80"   # uv venvs ship none; deepspeed imports torch.utils.cpp_extension
$PY -c "import torch, flash_attn, transformers, deepspeed, qwen_vl_utils; from transformers import Qwen3VLForConditionalGeneration; print('torch', torch.__version__, 'gpu', torch.cuda.is_available(), '| flash_attn', flash_attn.__version__, '| transformers', transformers.__version__, '| deepspeed', deepspeed.__version__)"
