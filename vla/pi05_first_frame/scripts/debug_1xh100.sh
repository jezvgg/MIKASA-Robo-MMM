#!/usr/bin/env bash
# The single-GPU debug, unattended. Source server.env first:
#
#   . ~/mikasa-pi05/server.env
#   nohup $REPO_DIR/vla/pi05_first_frame/scripts/debug_1xh100.sh > ~/mikasa-pi05/logs/debug.log 2>&1 < /dev/null &
#
# Stages ($STAGES, default "overfit"; "overfit debug" adds stage 2). A finished stage is skipped,
# so running the script again continues where it stopped:
#   0. render check (qd-gpucheck where it exists) and norm stats
#   1. overfit (~30 min): pi05_sd_ff_overfit on 4 episodes, one per cue drawer (500 steps), then
#      - open-loop: predicted vs recorded action chunks on those episodes,
#      - closed-loop: the simulator on those episodes' seeds.
#      The model has seen these exact episodes; failing here means a pipeline bug, not capacity.
#   2. debug (~2 h): pi05_sd_ff_1xh100 on all episodes (3k steps), then 20 validation seeds
# A full fine-tune that runs out of GPU memory is retried once with --batch-size 16.
# Stops before the disk passes $DISK_LIMIT_GB (default 90 of the 100 GB quota).
# The short report to send back: $PI05_WORK/logs/debug-report.txt
#
# For a quick dry run of the script itself: OVERFIT_CONFIG=pi05_sd_ff_dummy DEBUG_CONFIG=pi05_sd_ff_dummy
#   TRAIN_ARGS="--num-train-steps 6 --save-interval 3" EVAL_ARGS="--max-policy-steps 10"
set -euo pipefail
: "${OPENPI_DIR:?source server.env first}" "${SIM_VENV:?}" "${REPO_DIR:?}" "${PI05_WORK:?}"
PY="$OPENPI_DIR/.venv/bin/python"
SCRIPTS="$REPO_DIR/vla/pi05_first_frame/scripts"
LOGS="$PI05_WORK/logs"
REPORT="$LOGS/debug-report.txt"
OVERFIT_CONFIG="${OVERFIT_CONFIG:-pi05_sd_ff_overfit}"
DEBUG_CONFIG="${DEBUG_CONFIG:-pi05_sd_ff_1xh100}"
STAGES="${STAGES:-overfit}"
read -r -a TRAIN_EXTRA <<< "${TRAIN_ARGS:-}"
read -r -a EVAL_EXTRA <<< "${EVAL_ARGS:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p "$LOGS"

say() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$REPORT"; }

disk_guard() {
  local used
  used=$(du -s --block-size=1G "$HOME" 2>/dev/null | cut -f1)
  say "disk: ${used} GB used in $HOME"
  if (( used > ${DISK_LIMIT_GB:-90} )); then
    say "STOP: over ${DISK_LIMIT_GB:-90} GB; free space before continuing (old checkpoints in $PI05_WORK/checkpoints)"
    exit 1
  fi
}

latest_checkpoint() {  # latest_checkpoint CONFIG EXP
  find "$PI05_WORK/checkpoints/$1/$2" -mindepth 1 -maxdepth 1 -type d -regex '.*/[0-9]+' 2>/dev/null \
    | sort -t/ -k1 -V | tail -1
}

train() {  # train CONFIG EXP
  local config="$1" exp="$2" done="$PI05_WORK/checkpoints/$1/$2/.finished"
  if [[ -f "$done" ]]; then say "train $config/$exp: already finished"; return; fi
  disk_guard
  say "train $config/$exp: started (log $LOGS/train-$exp.log)"
  if ! "$PY" -m mikasa_pi05 train "$config" --exp-name "$exp" --overwrite "${TRAIN_EXTRA[@]}" \
      > "$LOGS/train-$exp.log" 2>&1; then
    if grep -qE 'RESOURCE_EXHAUSTED|Out of memory|out of memory' "$LOGS/train-$exp.log"; then
      say "train $config/$exp: out of GPU memory with the default batch; retrying with --batch-size 16"
      "$PY" -m mikasa_pi05 train "$config" --exp-name "$exp" --overwrite --batch-size 16 "${TRAIN_EXTRA[@]}" \
        > "$LOGS/train-$exp.log" 2>&1 || { say "train $config/$exp: FAILED"; tail -n 40 "$LOGS/train-$exp.log" | tee -a "$REPORT"; exit 1; }
      echo "batch_size=16" > "$PI05_WORK/checkpoints/$config/$exp/.batch"
    else
      say "train $config/$exp: FAILED"; tail -n 40 "$LOGS/train-$exp.log" | tee -a "$REPORT"; exit 1
    fi
  fi
  touch "$done"
  local rate first last
  rate=$(grep -oE '[0-9.]+(s/it|it/s)' "$LOGS/train-$exp.log" | tail -1 || true)
  first=$(grep -m1 -oE 'Step 0: .*' "$LOGS/train-$exp.log" || true)
  last=$(grep -oE 'Step [0-9]+: .*' "$LOGS/train-$exp.log" | tail -1 || true)
  say "train $config/$exp: done; ${rate:-?} per step; $(cat "$PI05_WORK/checkpoints/$config/$exp/.batch" 2>/dev/null || echo default batch)"
  say "  first: $first"
  say "  last:  $last"
}

evaluate() {  # evaluate CONFIG EXP OUT [eval args...]
  local config="$1" exp="$2" out="$3"; shift 3
  local ckpt; ckpt="$(latest_checkpoint "$config" "$exp")"
  [[ -n "$ckpt" ]] || { say "eval $exp: no checkpoint found"; exit 1; }
  # One results directory per finished training run: a retrained checkpoint never resumes old results.
  out="$out-$(date -r "$PI05_WORK/checkpoints/$config/$exp/.finished" +%m%d-%H%M)"
  say "eval $exp: checkpoint $ckpt -> $out"
  "$SCRIPTS/eval_parallel.sh" "$config" "$ckpt" "$out" 0 "$@" "${EVAL_EXTRA[@]}" > "$LOGS/eval-$exp.log" 2>&1 \
    || say "eval $exp: some episodes failed to run (see $LOGS/eval-$exp.log)"
  "$SIM_VENV/bin/python" - "$out/summary.json" <<'EOF' | tee -a "$REPORT"
import json, sys
s = json.load(open(sys.argv[1]))
print(f"  SR {s['success']}/{s['episodes']} = {s['success_rate']:.2f}, 95% CI {s['wilson95']}")
print(f"  stages {s['stages']}")
print(f"  by target drawer {s['by_target_drawer']}; status {s['status']}")
EOF
}

say "debug run on $(hostname); $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
say "branch $(git -C "$REPO_DIR" rev-parse --short HEAD)"
if command -v qd-gpucheck >/dev/null 2>&1; then
  verdict=$(qd-gpucheck 2>&1 | tee "$LOGS/qd-gpucheck.log" | grep -oE '"verdict": *"[a-z]+"' | head -1 || true)
  say "qd-gpucheck: ${verdict:-unknown}"
  if [[ "$verdict" == *wedged* ]]; then
    say "STOP: this node cannot render; ask the owner to restart the job on another node"; exit 1
  fi
fi
if [[ ! -f "$PI05_WORK/assets/samedrawer/nurtayev-d/samedrawer-1000ep/norm_stats.json" ]]; then
  say "norm stats: computing"
  "$PY" -m mikasa_pi05 norm-stats pi05_sd_ff_4xh100 > "$LOGS/norm-stats.log" 2>&1
fi

if [[ " $STAGES " == *" overfit "* ]]; then
  train "$OVERFIT_CONFIG" overfit1
  say "open-loop action error on the training episodes (log $LOGS/action-error-overfit1.log)"
  "$PY" "$REPO_DIR/vla/pi05_first_frame/tests/check_action_error.py" "$OVERFIT_CONFIG" \
      "$(latest_checkpoint "$OVERFIT_CONFIG" overfit1)" > "$LOGS/action-error-overfit1.log" 2>&1 \
    && grep -A8 'frames from episodes' "$LOGS/action-error-overfit1.log" | tee -a "$REPORT" \
    || { say "open-loop check failed"; tail -n 20 "$LOGS/action-error-overfit1.log" | tee -a "$REPORT"; }
  episodes=$("$PY" -c "from mikasa_pi05.configs import get_config as g; e = g('$OVERFIT_CONFIG').data.episodes; print(','.join(map(str, e)) if e else '')" 2>/dev/null)
  if [[ -n "$episodes" ]]; then seeds="episodes:$episodes"; else seeds="train:4"; fi
  evaluate "$OVERFIT_CONFIG" overfit1 "$PI05_WORK/results/overfit1" --seeds "$seeds" --video 4
fi

if [[ " $STAGES " == *" debug "* ]]; then
  train "$DEBUG_CONFIG" debug1
  evaluate "$DEBUG_CONFIG" debug1 "$PI05_WORK/results/debug1" --seeds validation --max-episodes 20 --video 5
fi

disk_guard
say "finished; send $REPORT"
