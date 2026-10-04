#!/usr/bin/env bash
# The baseline run on 4 GPUs, unattended: training with progress SR every 1000 steps, then the
# final evaluation on the 100 validation seeds and the black-first-frame control.
#
#   . ~/mikasa-pi05/server.env
#   nohup $REPO_DIR/vla/pi05_first_frame/scripts/run_4xh100.sh run1 > ~/mikasa-pi05/logs/run1.log 2>&1 < /dev/null &
#   tail -F ~/mikasa-pi05/logs/sr-run1.txt          # progress SR, one line per 1000 steps
#
# Re-running the same command after an interruption resumes from the last full checkpoint
# (every 5000 steps). Environment: GPUS (default 0,1,2,3), SR_GPU (3), CONFIG (pi05_sd_ff_4xh100),
# TRAIN_MEM_FRACTION (0.75: leaves room on each GPU for the progress evaluation), and
# TRAIN_ARGS for extra training flags, e.g. "--params-only-checkpoints" on a small disk (no resume);
# SR_EPISODES (20) and SR_EVAL_ARGS / FINAL_EVAL_ARGS to shorten evaluation in tests.
set -euo pipefail
: "${OPENPI_DIR:?source server.env first}" "${SIM_VENV:?}" "${REPO_DIR:?}" "${PI05_WORK:?}"
EXP="${1:-run1}"
CONFIG="${CONFIG:-pi05_sd_ff_4xh100}"
GPUS="${GPUS:-0,1,2,3}"
SR_GPU="${SR_GPU:-3}"
PY="$OPENPI_DIR/.venv/bin/python"
SCRIPTS="$REPO_DIR/vla/pi05_first_frame/scripts"
LOGS="$PI05_WORK/logs"
REPORT="$LOGS/report-$EXP.txt"
read -r -a TRAIN_EXTRA <<< "${TRAIN_ARGS:-}"
read -r -a FINAL_EXTRA <<< "${FINAL_EVAL_ARGS:-}"
mkdir -p "$LOGS"
say() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$REPORT"; }

say "run $EXP: $CONFIG on GPUs $GPUS, progress SR on GPU $SR_GPU; branch $(git -C "$REPO_DIR" rev-parse --short HEAD)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader | tee -a "$REPORT"
if [[ ! -f "$PI05_WORK/assets/samedrawer/nurtayev-d/samedrawer-1000ep/norm_stats.json" ]]; then
  say "norm stats: computing"; "$PY" -m mikasa_pi05 norm-stats pi05_sd_ff_4xh100 > "$LOGS/norm-stats.log" 2>&1
fi

ckpt_dir="$PI05_WORK/checkpoints/$CONFIG/$EXP"
last=$("$PY" -c "from mikasa_pi05.configs import get_config; print(get_config('$CONFIG').num_train_steps - 1)" 2>/dev/null)
if [[ -d "$ckpt_dir/$last" ]]; then
  say "training already finished ($ckpt_dir/$last)"
else
  mode=()
  if ls -d "$ckpt_dir"/[0-9]* >/dev/null 2>&1; then mode=(--resume); say "resuming from $(ls -d "$ckpt_dir"/[0-9]* | sort -V | tail -1)"; fi
  "$SCRIPTS/sr_monitor.sh" "$CONFIG" "$EXP" "$SR_GPU" "${SR_EPISODES:-20}" > "$LOGS/sr-monitor-$EXP.log" 2>&1 &
  monitor=$!
  say "training started (log $LOGS/train-$EXP.log)"
  if ! CUDA_VISIBLE_DEVICES="$GPUS" XLA_PYTHON_CLIENT_MEM_FRACTION="${TRAIN_MEM_FRACTION:-0.75}" \
      "$PY" -m mikasa_pi05 train "$CONFIG" --exp-name "$EXP" "${mode[@]}" "${TRAIN_EXTRA[@]}" \
      > "$LOGS/train-$EXP.log" 2>&1; then
    say "training FAILED"; tail -n 40 "$LOGS/train-$EXP.log" | tee -a "$REPORT"
    kill "$monitor" 2>/dev/null || true; exit 1
  fi
  say "training done; waiting for the progress SR of the last step"
  wait "$monitor" || true
fi
cat "$LOGS/sr-$EXP.txt" >> "$REPORT" 2>/dev/null || true

final="$ckpt_dir/$last"
for mode in reset black; do
  out="$PI05_WORK/results/$EXP-final-$mode"
  say "final evaluation, first frame: $mode -> $out"
  "$SCRIPTS/eval_parallel.sh" "$CONFIG" "$final" "$out" "$GPUS" --seeds validation --first-frame "$mode" \
      $( [[ $mode == reset ]] && echo --video 10 ) "${FINAL_EXTRA[@]}" > "$LOGS/eval-$EXP-final-$mode.log" 2>&1 \
    || say "some episodes failed to run (see $LOGS/eval-$EXP-final-$mode.log)"
  "$SIM_VENV/bin/python" - "$out/summary.json" <<'PY' | tee -a "$REPORT"
import json, sys
s = json.load(open(sys.argv[1]))
print(f"  SR {s['success']}/{s['episodes']} = {s['success_rate']:.2f}, 95% CI {s['wilson95']}")
print(f"  stages {s['stages']}")
print(f"  by target drawer {s['by_target_drawer']}")
PY
done
say "finished; send $REPORT"
