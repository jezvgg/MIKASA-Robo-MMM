#!/usr/bin/env bash
# MemoryVLA on ONE H100: does it learn our base commands with the normalization bounds of
# set_norm_bounds.py (base +-1, joints min/max), and does it hold the arm without joint state?
# A check, not the paper's recipe: one task, $EPISODES episodes, the authors' train.py with the
# vision and LLM backbones frozen (their "align" stage; the full 7B fine-tune does not fit one
# card), batch $BS, $STEPS steps at LR $LR; everything else as script/train/mikasa/train_mikasa.sh
# plus our --action_dim 13 --load_bi_wrist True. Stages (a finished stage is skipped on a re-run):
#   rlds   episodes 0..$EPISODES-1 of the task's LeRobot copy ($DATA_ROOT/<task>) -> $W/rlds;
#   stats  the authors' code computes dataset_statistics_<hash>.json on a first launch, which is
#          stopped as soon as the file exists; set_norm_bounds.py then rewrites its q01/q99;
#   train  the real run, which loads the edited file and saves it into its run dir;
#   check  check_action_error.py on held-out episodes $HELD_OUT.
# $WAIT_CMD (optional) is polled until it succeeds before the GPU is used.
# usage: nohup vla/memoryvla/debug_1gpu.sh [work dir, default ~/mvla-debug] > log 2>&1 < /dev/null &
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
W=${1:-$HOME/mvla-debug}
TASK=${TASK:-depthrecall}; EPISODES=${EPISODES:-40}; STEPS=${STEPS:-2500}; LR=${LR:-1e-4}; BS=${BS:-32}
HELD_OUT=${HELD_OUT:-990,991,992,993,994}
DATA_ROOT=${DATA_ROOT:-$HOME/datasets/memer}
PY=$HOME/envs/memvla/bin/python; TORCHRUN=$HOME/envs/memvla/bin/torchrun
CKPT=${OPENVLA_CKPT:-$HOME/models/openvla-7b-prismatic/checkpoints/step-295000-epoch-40-loss=0.2200.pt}
LOGS=$W/logs REPORT=$W/REPORT.txt
mkdir -p "$LOGS" "$W/src"
ln -sfn "$DATA_ROOT" "$W/src/memer"          # fetch_rlds reads <FETCH_LEROBOT_ROOT>/memer/<task>
say() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$REPORT"; }

train() {  # train RUN_ID MAX_STEPS BATCH -> runs in its own process group
  cd "$REPO/third_party/MemoryVLA" && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled \
  CUDA_VISIBLE_DEVICES=0 exec "$TORCHRUN" --nnodes=1 --nproc_per_node=1 --master_addr=127.0.0.1 \
    --master_port=29521 train.py --pretrained_checkpoint "$CKPT" \
    --vla.type prism-dinosiglip-224px+oxe+diffusion --vla.data_mix "fetch_$TASK" \
    --vla.expected_world_size 1 --vla.per_device_batch_size "$3" --vla.global_batch_size "$3" \
    --vla.learning_rate "$LR" --vla.max_steps "$2" --vla.freeze_vision_backbone True \
    --vla.freeze_llm_backbone True --vla.shuffle_buffer_size 1000 --data_root_dir "$W/rlds" \
    --run_root_dir "$W/runs" --run_id "$1" --image_aug True --save_interval "$2" \
    --repeated_diffusion_steps 4 --future_action_window_size 3 --action_model_type DiT-L \
    --dataloader_type group --is_resume False --hf_token HF_TOKEN --trackers '["jsonl"]' \
    --action_dim 13 --load_bi_wrist True
}
stats_file() { find "$W/rlds/fetch_$TASK" -name 'dataset_statistics_*.json' 2>/dev/null | head -1; }

say "MemoryVLA debug: task $TASK, $EPISODES episodes, $STEPS steps, batch $BS, LR $LR; repo $(git -C "$REPO" rev-parse --short HEAD)"
if [ ! -f "$W/rlds/.built_$TASK" ]; then
  say "rlds: building episodes 0-$((EPISODES - 1)) (log $LOGS/rlds.log)"
  FETCH_LEROBOT_ROOT=$W/src FETCH_RLDS_MAX_EPISODES=$EPISODES OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= \
    "$PY" "$REPO/vla/memoryvla/rlds/build_rlds.py" "fetch_$TASK" --data-dir "$W/rlds" > "$LOGS/rlds.log" 2>&1 \
    || { say "rlds: FAILED"; tail -n 20 "$LOGS/rlds.log" >> "$REPORT"; exit 1; }
  touch "$W/rlds/.built_$TASK"; say "rlds: $(du -sh "$W/rlds/fetch_$TASK" | cut -f1)"
fi
if [ -n "${WAIT_CMD:-}" ]; then
  say "waiting for: $WAIT_CMD"; until eval "$WAIT_CMD"; do sleep 60; done
fi

if [ -z "$(stats_file)" ]; then
  say "stats: first launch until the statistics file exists (log $LOGS/stats.log)"
  # setsid: the launch gets its own process group, which is stopped as a whole below.
  setsid bash -c "$(declare -f train); REPO='$REPO' TASK='$TASK' W='$W' LR='$LR' CKPT='$CKPT' TORCHRUN='$TORCHRUN' train stats 1000000 '$BS'" \
     > "$LOGS/stats.log" 2>&1 &
  STATS_PID=$!
  for _ in $(seq 1 240); do [ -n "$(stats_file)" ] && break; kill -0 "$STATS_PID" 2>/dev/null || break; sleep 30; done
  sleep 10; kill -TERM -- "-$STATS_PID" 2>/dev/null; sleep 30; kill -KILL -- "-$STATS_PID" 2>/dev/null
  rm -rf "$W/runs/stats--image_aug"
  [ -n "$(stats_file)" ] || { say "stats: FAILED"; tail -n 30 "$LOGS/stats.log" >> "$REPORT"; exit 1; }
fi
"$PY" "$REPO/vla/memoryvla/set_norm_bounds.py" "$(stats_file)" --check > /dev/null 2>&1 \
  || "$PY" "$REPO/vla/memoryvla/set_norm_bounds.py" "$(stats_file)" | tee -a "$REPORT"

RUN=$W/runs/${TASK}_bounds--image_aug
if [ -z "$(ls "$RUN/checkpoints/"*.pt 2>/dev/null)" ]; then
  say "train: ${TASK}_bounds (log $LOGS/train.log)"
  (train "${TASK}_bounds" "$STEPS" "$BS") > "$LOGS/train.log" 2>&1
  if [ -z "$(ls "$RUN/checkpoints/"*.pt 2>/dev/null)" ] && grep -qi "out of memory" "$LOGS/train.log"; then
    say "train: out of GPU memory at batch $BS; retrying at batch 16"; rm -rf "$RUN"
    (train "${TASK}_bounds" "$STEPS" 16) > "$LOGS/train.log" 2>&1
  fi
  [ -n "$(ls "$RUN/checkpoints/"*.pt 2>/dev/null)" ] || { say "train: FAILED"; tail -n 30 "$LOGS/train.log" >> "$REPORT"; exit 1; }
fi
"$PY" "$REPO/vla/memoryvla/set_norm_bounds.py" "$RUN/dataset_statistics.json" --check | sed 's/^/  run dir statistics: /' | tee -a "$REPORT"
"$PY" - "$RUN/${TASK}_bounds--image_aug.jsonl" <<'EOF' | tee -a "$REPORT"
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if '"VLA Train/Loss"' in l]
loss = [r["VLA Train/Loss"] for r in rows]; dt = sorted(r["VLA Train/Step Time"] for r in rows)
mean = lambda x: sum(x) / max(len(x), 1)
print(f"  train loss: first 50 steps {mean(loss[:50]):.4f}, last 50 {mean(loss[-50:]):.4f} ({len(loss)} steps); "
      f"median step {dt[len(dt) // 2]:.2f} s" if loss else "  train loss: not found")
EOF
say "check: held-out episodes $HELD_OUT (log $LOGS/check.log)"
"$PY" "$REPO/vla/memoryvla/check_action_error.py" "$RUN" "$W/src/memer/$TASK" "fetch_$TASK" \
  --episodes "$HELD_OUT" --every 4 --out "$W/check_$TASK.json" > "$LOGS/check.log" 2>&1 \
  && cat "$W/check_$TASK.json" >> "$REPORT" || { say "check: FAILED"; tail -n 30 "$LOGS/check.log" >> "$REPORT"; }
say "finished"
