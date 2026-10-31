#!/usr/bin/env bash
# MemER low level (pi0.5 from pi0.5 base, as RoboMME; openpi) on 4 GPUs: memer_lowlevel_4xh100
# of vla/pi05_first_frame/src/mikasa_pi05/configs.py (RoboMME's recipe for pi0.5 base: SigLIP
# frozen, 5e-5 constant after 10,000 warmup steps, EMA 0.999, 80,000 steps, chunks of 20; batch 128
# on 4 FSDP devices; ~1.5 s/step -> ~33 h, est.) on the six labelled copies
# under $MEMER_DATA_ROOT (vla/memer/label_subtasks.py). Stages, each skipped when already done:
#   check  every task's copy holds meta/subtasks.parquet;
#   stats  the norm stats ($PI05_WORK/assets/memer-lowlevel-h20/memer-lowlevel/norm_stats.json) are
#          computed on CPU when missing or older than any copy's subtasks.parquet: openpi loads
#          whatever file is there, so stale stats from older data would be used without a word;
#   train  the run; a re-run of the same command resumes from the last full checkpoint (every
#          6000 steps; params-only snapshots every 1000, the newest 3 kept, ~6.6 GB each; a full
#          checkpoint with the optimizer ~37 GB). vla/mem_guard.py stops it when the container's
#          memory passes 80% of its limit (past it the pod is killed whole).
# usage: . ~/mikasa-pi05/server.env; nohup vla/memer/train_low.sh <exp name> > log 2>&1 < /dev/null &
# env: MEMER_DATA_ROOT (~/datasets/memer), CONFIG (memer_lowlevel_4xh100), GPUS (0,1,2,3),
#      STAGES (check stats train), TRAIN_ARGS (extra flags for `mikasa_pi05 train`)
set -uo pipefail
: "${OPENPI_DIR:?source ~/mikasa-pi05/server.env first}" "${PI05_WORK:?}"
REPO=$(cd "$(dirname "$0")/../.." && pwd)
EXP=${1:?usage: train_low.sh <exp name>}
CONFIG=${CONFIG:-memer_lowlevel_4xh100}; GPUS=${GPUS:-0,1,2,3}; STAGES=${STAGES:-check stats train}
export MEMER_DATA_ROOT=${MEMER_DATA_ROOT:-$HOME/datasets/memer}
export PYTHONPATH=$REPO/vla/pi05_first_frame/src
PY=$OPENPI_DIR/.venv/bin/python
LOGS=$PI05_WORK/logs; REPORT=$LOGS/report-$EXP.txt
read -r -a TRAIN_EXTRA <<< "${TRAIN_ARGS:-}"
mkdir -p "$LOGS"
say() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$REPORT"; }
has() { [[ " $STAGES " == *" $1 "* ]]; }
cfg() { JAX_PLATFORMS=cpu "$PY" -c "from mikasa_pi05.configs import get_config as g; c = g('$CONFIG'); print($1)"; }

say "MemER low level $EXP: $CONFIG on GPUs $GPUS, data $MEMER_DATA_ROOT; repo $(git -C "$REPO" rev-parse --short HEAD)"
tasks=$(cfg "' '.join(c.data.tasks)") || { say "config $CONFIG not found"; exit 1; }
read -r -a TASKS <<< "$tasks"
STATS=$(cfg "f'{c.data.assets.assets_dir}/{c.data.repo_id}/norm_stats.json'")

if has check; then
  for t in "${TASKS[@]}"; do
    [ -f "$MEMER_DATA_ROOT/$t/meta/subtasks.parquet" ] || { say "check: no labelled copy $MEMER_DATA_ROOT/$t"; exit 1; }
  done
  say "check: ${#TASKS[*]} labelled copies (${TASKS[*]})"
fi

if has stats; then
  newer=()
  for t in "${TASKS[@]}"; do
    [ -f "$STATS" ] && [ "$MEMER_DATA_ROOT/$t/meta/subtasks.parquet" -ot "$STATS" ] || newer+=("$t")
  done
  if [ ${#newer[@]} -gt 0 ]; then
    say "stats: computing $STATS (missing or older than ${newer[*]}; log $LOGS/norm-stats-$EXP.log)"
    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= "$PY" -m mikasa_pi05 norm-stats "$CONFIG" > "$LOGS/norm-stats-$EXP.log" 2>&1 \
      || { say "stats: FAILED"; tail -n 30 "$LOGS/norm-stats-$EXP.log" | tee -a "$REPORT"; exit 1; }
    grep -E 'frames in|q01==q99' "$LOGS/norm-stats-$EXP.log" | sed 's/^/  /' | tee -a "$REPORT"
  else
    say "stats: $STATS is newer than every copy"
  fi
fi

if has train; then
  ckpt=$PI05_WORK/checkpoints/$CONFIG/$EXP
  last=$(cfg "c.num_train_steps - 1")
  if [ -d "$ckpt/$last" ]; then say "train: already finished ($ckpt/$last)"; exit 0; fi
  mode=()
  if ls -d "$ckpt"/[0-9]* > /dev/null 2>&1; then
    mode=(--resume); say "train: resuming from $(ls -d "$ckpt"/[0-9]* | sort -V | tail -1)"
  elif [ -d "$ckpt" ]; then
    mode=(--overwrite)   # a directory without a checkpoint holds nothing to keep
  fi
  say "train: started (log $LOGS/train-$EXP.log; memory guard $LOGS/train-$EXP.guard.log)"
  CUDA_VISIBLE_DEVICES=$GPUS setsid "$PY" -m mikasa_pi05 train "$CONFIG" --exp-name "$EXP" "${mode[@]}" "${TRAIN_EXTRA[@]}" \
    > "$LOGS/train-$EXP.log" 2>&1 &
  pid=$!
  trap 'kill -TERM -- "-$pid" 2>/dev/null; exit 143' TERM INT
  python3 "$REPO/vla/mem_guard.py" "$pid" >> "$LOGS/train-$EXP.guard.log" 2>&1 &
  wait "$pid"; rc=$?
  grep -h GUARD "$LOGS/train-$EXP.guard.log" 2>/dev/null | tail -1 | tee -a "$REPORT"
  if [ $rc -ne 0 ]; then say "train: FAILED (exit $rc)"; tail -n 40 "$LOGS/train-$EXP.log" | tee -a "$REPORT"; exit $rc; fi
  say "train: done; checkpoints $(ls "$ckpt" | tr '\n' ' ')"
fi
