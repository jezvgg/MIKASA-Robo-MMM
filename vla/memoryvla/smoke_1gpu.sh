#!/usr/bin/env bash
# MemoryVLA smoke test on ONE GPU: the authors' train.py with the arguments of their
# script/train/mikasa/train_mikasa.sh, except what one GPU forces and what a smoke is:
#   expected_world_size 1, per-device and global batch 2, max_steps 3, save_interval 3,
#   vision AND LLM frozen (their "align" stage) because the full 7B fine-tune does not fit
#   one 80 GB card (the full run: FSDP on 4 GPUs, 32 per GPU, global 128, no accumulation);
#   shuffle buffer 1000, jsonl tracker only.
# Ours on top of theirs: data_mix of a Fetch dataset (patches/0001), --action_dim 13 and
# --load_bi_wrist True (three cameras). Llama-2 resolves offline from the local HF cache
# (models/llama_cache.sh links the sha256-checked copy as meta-llama/Llama-2-7b-hf).
# The rendezvous is pinned to 127.0.0.1: on the lab's container `--standalone` resolves the
# cluster host name and fails to connect (errno 97).
# usage: smoke_1gpu.sh <data_root_dir with fetch_* RLDS> <run_root_dir> [data_mix]
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
DATA=$1; RUNS=$2; MIX=${3:-fetch_seasondish}
CKPT=${OPENVLA_CKPT:-$HOME/models/openvla-7b-prismatic/checkpoints/step-295000-epoch-40-loss=0.2200.pt}
cd "$REPO/third_party/MemoryVLA"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
"${PYTHON_BIN:-$HOME/envs/memvla/bin/torchrun}" --nnodes=1 --nproc_per_node=1 \
  --master_addr=127.0.0.1 --master_port="${MASTER_PORT:-29511}" train.py \
  --pretrained_checkpoint "$CKPT" \
  --vla.type prism-dinosiglip-224px+oxe+diffusion \
  --vla.data_mix "$MIX" \
  --vla.expected_world_size 1 \
  --vla.per_device_batch_size 2 \
  --vla.global_batch_size 2 \
  --vla.learning_rate 2e-5 \
  --vla.max_steps 3 \
  --vla.freeze_vision_backbone True \
  --vla.freeze_llm_backbone True \
  --vla.shuffle_buffer_size 1000 \
  --data_root_dir "$DATA" \
  --run_root_dir "$RUNS" \
  --run_id smoke_1gpu \
  --image_aug True \
  --save_interval 3 \
  --repeated_diffusion_steps 4 \
  --future_action_window_size 3 \
  --action_model_type DiT-L \
  --dataloader_type group \
  --is_resume False \
  --hf_token HF_TOKEN \
  --trackers '["jsonl"]' \
  --action_dim 13 \
  --load_bi_wrist True
