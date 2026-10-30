#!/usr/bin/env bash
# MemoryVLA on $GPUS GPUs (default 4): the authors' train.py with the arguments of their
# script/train/mikasa/train_mikasa.sh (OpenVLA-7B prismatic init, full fine-tune with FSDP full
# shard, DiT-L, 4 diffusion repeats, 4-step window, LR 2e-5, image aug) and our
# changes, all in vla/TRAINING_DEVIATIONS.md: the six Fetch datasets (patches/0001, mixture
# $MIX), --action_dim 13 --load_bi_wrist True, the long-episode memory of their train_real.sh
# (--dataloader_type stream --mem_length 256), per-task min/max normalization bounds
# (set_norm_bounds.py), per-device batch $BS (theirs 32) with no accumulation: global = BS x GPUS,
# shuffle buffer $SHUFFLE episodes per rank (the stream dataloader buffers whole episodes, ~50 MB
# each of ours; their 64k from the group dataloader would be ~3 TB per rank).
# Stages (a finished stage is skipped on a re-run):
#   stats  a first launch computes every dataset's dataset_statistics_<hash>.json next to its RLDS
#          and is stopped as soon as all exist; set_norm_bounds.py then rewrites their q01/q99;
#   train  the run, which loads the edited files and saves them into its run dir: it is stopped at
#          once if the run dir's statistics are not the edited ones.
# Checkpoints: model weights only (the authors' save; 32 GB each, fp32), every $SAVE steps.
# vla/mem_guard.py stops a launch whose container memory passes 80% of its limit (past it the pod
# is killed; log <run id>.guard.log).
# usage: train.sh <RLDS root holding fetch_*> <run root> <run id> <max steps> [save interval]
# env: GPUS (4), BS (32), MIX (fetch_six), LR (2e-5), SHUFFLE (500), MASTER_PORT (29531),
#      BASE_BOUNDS (minmax | pm1, see set_norm_bounds.py), OPENVLA_CKPT
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
DATA=$1; RUNS=$2; RUN_ID=$3; STEPS=$4; SAVE=${5:-$STEPS}
GPUS=${GPUS:-4}; BS=${BS:-32}; MIX=${MIX:-fetch_six}; LR=${LR:-2e-5}; SHUFFLE=${SHUFFLE:-500}
BASE_BOUNDS=${BASE_BOUNDS:-minmax}; PORT=${MASTER_PORT:-29531}
CKPT=${OPENVLA_CKPT:-$HOME/models/openvla-7b-prismatic/checkpoints/step-295000-epoch-40-loss=0.2200.pt}
PY=$HOME/envs/memvla/bin/python; TORCHRUN=$HOME/envs/memvla/bin/torchrun
mkdir -p "$RUNS/logs"
say() { printf '[%s] memoryvla %s\n' "$(date '+%F %T')" "$*"; }
# Each launch runs in its own session (setsid); stopping this script stops the current one.
pid=""
trap '[ -n "$pid" ] && { kill -TERM -- "-$pid" 2>/dev/null; sleep 20; kill -KILL -- "-$pid" 2>/dev/null; }; exit 143' TERM INT

DATASETS=$("$PY" - "$REPO" "$MIX" <<'EOF'
import sys
sys.path.insert(0, sys.argv[1] + "/third_party/MemoryVLA")
from vla.datasets.rlds.oxe.mixtures import OXE_NAMED_MIXTURES
print(" ".join(name for name, _ in OXE_NAMED_MIXTURES[sys.argv[2]]))
EOF
) || { say "unknown mixture $MIX (is patches/0001 applied? run setup_memvla.sh)"; exit 1; }

launch() {  # launch RUN_ID MAX_STEPS SAVE_INTERVAL: the authors' command; runs in the foreground
  cd "$REPO/third_party/MemoryVLA" && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled \
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((GPUS - 1)))} exec "$TORCHRUN" --nnodes=1 \
    --nproc_per_node="$GPUS" --master_addr=127.0.0.1 --master_port="$PORT" train.py \
    --pretrained_checkpoint "$CKPT" --vla.type prism-dinosiglip-224px+oxe+diffusion --vla.data_mix "$MIX" \
    --vla.expected_world_size "$GPUS" --vla.per_device_batch_size "$BS" --vla.global_batch_size $((GPUS * BS)) \
    --vla.learning_rate "$LR" --vla.max_steps "$2" --vla.shuffle_buffer_size "$SHUFFLE" \
    --data_root_dir "$DATA" --run_root_dir "$RUNS" --run_id "$1" --image_aug True --save_interval "$3" \
    --repeated_diffusion_steps 4 --future_action_window_size 3 --action_model_type DiT-L \
    --dataloader_type stream --mem_length 256 --is_resume False --hf_token HF_TOKEN --trackers '["jsonl"]' \
    --action_dim 13 --load_bi_wrist True
}
stats_files() { for d in $DATASETS; do find "$DATA/$d" -name 'dataset_statistics_*.json' 2>/dev/null; done; }
all_stats() { [ "$(stats_files | wc -l)" -eq "$(wc -w <<< "$DATASETS")" ]; }

if [ ! -f "$DATA/.bounds_$MIX" ]; then
  # A cache file's name hashes the dataset path; a stale one from another path or run could be
  # edited while training computes and loads a fresh one, so start from none.
  for d in $DATASETS; do find "$DATA/$d" -name 'dataset_statistics_*.json' -delete 2>/dev/null; done
  say "stats: first launch until $(wc -w <<< "$DATASETS") statistics files exist (log $RUNS/logs/$RUN_ID.stats.log)"
  setsid bash -c "$(declare -f launch); REPO='$REPO' GPUS='$GPUS' BS='$BS' MIX='$MIX' LR='$LR' SHUFFLE=16 \
    DATA='$DATA' RUNS='$RUNS/.stats' PORT='$PORT' CKPT='$CKPT' TORCHRUN='$TORCHRUN' launch stats 1000000 1000000" \
    > "$RUNS/logs/$RUN_ID.stats.log" 2>&1 &
  pid=$!
  "$PY" "$REPO/vla/mem_guard.py" "$pid" >> "$RUNS/logs/$RUN_ID.guard.log" 2>&1 &
  for _ in $(seq 1 480); do all_stats && break; kill -0 "$pid" 2>/dev/null || break; sleep 30; done
  sleep 10; kill -TERM -- "-$pid" 2>/dev/null; sleep 30; kill -KILL -- "-$pid" 2>/dev/null
  rm -rf "$RUNS/.stats"
  all_stats || { say "stats: FAILED"; tail -n 30 "$RUNS/logs/$RUN_ID.stats.log"; exit 1; }
  for f in $(stats_files); do "$PY" "$REPO/vla/memoryvla/set_norm_bounds.py" "$f" --base "$BASE_BOUNDS" || exit 1; done
  touch "$DATA/.bounds_$MIX"
fi
for f in $(stats_files); do
  "$PY" "$REPO/vla/memoryvla/set_norm_bounds.py" "$f" --base "$BASE_BOUNDS" --check > /dev/null \
    || { say "stats: $f lost its bounds; delete $DATA/.bounds_$MIX to redo them"; exit 1; }
done

RUN=$RUNS/$RUN_ID--image_aug
say "train: $RUN_ID, $GPUS GPUs x batch $BS, $STEPS steps, checkpoints every $SAVE (log $RUNS/logs/$RUN_ID.log)"
setsid bash -c "$(declare -f launch); REPO='$REPO' GPUS='$GPUS' BS='$BS' MIX='$MIX' LR='$LR' SHUFFLE='$SHUFFLE' \
  DATA='$DATA' RUNS='$RUNS' PORT='$PORT' CKPT='$CKPT' TORCHRUN='$TORCHRUN' launch '$RUN_ID' '$STEPS' '$SAVE'" \
  > "$RUNS/logs/$RUN_ID.log" 2>&1 &
pid=$!
"$PY" "$REPO/vla/mem_guard.py" "$pid" >> "$RUNS/logs/$RUN_ID.guard.log" 2>&1 &
for _ in $(seq 1 240); do [ -f "$RUN/dataset_statistics.json" ] && break; kill -0 "$pid" 2>/dev/null || break; sleep 15; done
if ! "$PY" "$REPO/vla/memoryvla/set_norm_bounds.py" "$RUN/dataset_statistics.json" --base "$BASE_BOUNDS" --check > /dev/null 2>&1; then
  kill -TERM -- "-$pid" 2>/dev/null; say "train: STOPPED, the run's statistics are not the edited ones"; exit 1
fi
wait "$pid"; rc=$?
say "train: exit $rc; checkpoints: $(ls "$RUN/checkpoints" 2>/dev/null | tr '\n' ' ')"
grep -h GUARD "$RUNS/logs/$RUN_ID.guard.log" 2>/dev/null | tail -1
exit $rc
