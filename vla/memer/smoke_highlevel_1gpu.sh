#!/usr/bin/env bash
# MemER high-level smoke on ONE GPU: the command of memer/TRAINING_QWEN3VL.md (the authors'
# Qwen3-VL-4B run on 2 B200) with only what one GPU and a smoke change: one process, per-device
# batch 1 with no accumulation (theirs: 8 x 8 x 2 GPUs = 128), max_steps 3, no checkpoint.
# The rendezvous is pinned to 127.0.0.1 (the lab container cannot use its host name).
# usage: smoke_highlevel_1gpu.sh <MEMER_SFT_ROOT holding <task>/train.json> <out dir> [dataset_use]
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
export MEMER_SFT_ROOT=$1; OUT=$2; DATA=${3:-memer_samedrawer}
export QWEN_VL_ATTN_IMPL=flash_attention_2
cd "$REPO/third_party/Qwen3-VL/qwen-vl-finetune"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} "${TORCHRUN:-$HOME/envs/qwen3vl/bin/torchrun}" \
  --nnodes=1 --nproc_per_node=1 --master_addr=127.0.0.1 --master_port="${MASTER_PORT:-29512}" \
  qwenvl/train/train_qwen.py \
  --model_name_or_path Qwen/Qwen3-VL-4B-Instruct \
  --dataset_use "$DATA" \
  --data_flatten True \
  --tune_mm_vision False \
  --tune_mm_mlp False \
  --tune_mm_llm True \
  --bf16 \
  --output_dir "$OUT" \
  --num_train_epochs 15 \
  --max_steps 3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --optim adamw_torch \
  --max_pixels 115200 \
  --min_pixels 50176 \
  --eval_strategy no \
  --save_strategy no \
  --learning_rate 6e-5 \
  --weight_decay 0 \
  --warmup_ratio 0.05 \
  --max_grad_norm 1 \
  --lr_scheduler_type cosine \
  --logging_steps 1 \
  --logging_nan_inf_filter False \
  --model_max_length 8192 \
  --gradient_checkpointing True \
  --dataloader_num_workers 2 \
  --report_to none
