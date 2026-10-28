#!/usr/bin/env bash
# MemER high level (Qwen3-VL-4B) on $GPUS GPUs (default 4): the command of memer/TRAINING_QWEN3VL.md
# (the authors' run on 2 B200) with what our GPUs change: per-device batch $BS (theirs 8; an H100
# has 80 GB, a B200 180 GB) and their accumulation 8, so 4 GPUs x 4 x 8 = their global batch 128.
# Checkpoints as theirs, every SAVE steps with the newest 2 kept (~23 GB each with the optimizer),
# or weights only with SAVE_ONLY_MODEL=1 (~18 GB, fp32, cannot resume). An output dir that holds a
# checkpoint-* resumes from it: the authors' train_qwen.py does that itself. vla/mem_guard.py stops
# the run when the container's memory passes 80% of its limit (past it the pod is killed); each
# dataloader worker holds a private copy of the annotations (~21 GiB for four tasks), so WORKERS
# sets the memory: their 8 per GPU killed the pod on 2026-10-27; 2 (the owner's choice, 2026-10-29)
# fits six tasks with the most room (~380 GiB of the 582 the guard allows), at ~15% slower steps than 4.
# usage: train_high.sh <MEMER_SFT_ROOT holding <task>/train.json> <output dir> <max steps> [save steps]
# env: GPUS (4), BS (4), DATASETS (comma list; default the six memer_<task>), LR (6e-5),
#      SAVE_ONLY_MODEL (0), WORKERS (dataloader workers per GPU, 2; theirs 8), MASTER_PORT (29532)
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
export MEMER_SFT_ROOT=$1; OUT=$2; STEPS=$3; SAVE=${4:-500}
GPUS=${GPUS:-4}; BS=${BS:-4}; LR=${LR:-6e-5}
DATASETS=${DATASETS:-memer_samedrawer,memer_seasondish,memer_cabinetsearch,memer_depthrecall,memer_takeitback,memer_veggie}
EXTRA=(); [ "${SAVE_ONLY_MODEL:-0}" = 1 ] && EXTRA=(--save_only_model True)
export QWEN_VL_ATTN_IMPL=flash_attention_2 HF_HUB_OFFLINE=1
cd "$REPO/third_party/Qwen3-VL/qwen-vl-finetune"
python3 "$REPO/vla/mem_guard.py" $$ >&2 &   # $$ is torchrun after the exec
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((GPUS - 1)))} exec "$HOME/envs/qwen3vl/bin/torchrun" \
  --nnodes=1 --nproc_per_node="$GPUS" --master_addr=127.0.0.1 --master_port="${MASTER_PORT:-29532}" \
  qwenvl/train/train_qwen.py \
  --model_name_or_path Qwen/Qwen3-VL-4B-Instruct \
  --dataset_use "$DATASETS" \
  --data_flatten True \
  --tune_mm_vision False \
  --tune_mm_mlp False \
  --tune_mm_llm True \
  --bf16 \
  --output_dir "$OUT" \
  --num_train_epochs 15 \
  --max_steps "$STEPS" \
  --per_device_train_batch_size "$BS" \
  --gradient_accumulation_steps 8 \
  --optim adamw_torch \
  --max_pixels 115200 \
  --min_pixels 50176 \
  --eval_strategy no \
  --save_strategy steps \
  --save_steps "$SAVE" \
  --save_total_limit 2 \
  --learning_rate "$LR" \
  --weight_decay 0 \
  --warmup_ratio 0.05 \
  --max_grad_norm 1 \
  --lr_scheduler_type cosine \
  --logging_steps 1 \
  --logging_nan_inf_filter False \
  --model_max_length 8192 \
  --gradient_checkpointing True \
  --dataloader_num_workers "${WORKERS:-2}" \
  --report_to none \
  "${EXTRA[@]}"
